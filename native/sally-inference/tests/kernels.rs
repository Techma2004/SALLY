//! Kernel tests. Block bytes are constructed here from chosen scales and integer
//! codes; expected values are computed analytically from the ggml layout rules,
//! independently of the code under test.
#[path = "common/gguf_builder.rs"]
mod gguf_builder;

use gguf_builder::GgufBuilder;
use sally_inference::backend::{dequant_row, f16_to_f32, matvec};
use sally_inference::gguf::{GgmlType, GgufFile};
use sally_inference::tensor_store::{TensorStore, TensorView};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

struct TempFile(PathBuf);
impl TempFile {
    fn new(bytes: &[u8]) -> TempFile {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let p = std::env::temp_dir().join(format!("sally-kern-{}-{n}.gguf", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        TempFile(p)
    }
}
impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Writes a one-tensor GGUF ("w", dims `[cols, rows]`), opens it and runs `f`.
fn with_tensor<R>(
    dtype: GgmlType,
    cols: usize,
    rows: usize,
    bytes: Vec<u8>,
    f: impl FnOnce(&TensorView<'_>) -> R,
) -> R {
    let gguf = GgufBuilder::new()
        .kv_str("general.architecture", "test")
        .tensor("w", &[cols as u64, rows as u64], dtype, bytes)
        .build();
    let tmp = TempFile::new(&gguf);
    let file = GgufFile::open(&tmp.0).unwrap();
    let store = TensorStore::open(&file, &tmp.0).unwrap();
    let view = store.view("w").unwrap();
    f(&view)
}

// Exactly representable f16 scale values, as bit patterns.
const H_1: u16 = 0x3C00;
const H_HALF: u16 = 0x3800;
const H_2: u16 = 0x4000;
const H_QUARTER: u16 = 0x3400;
const H_NEG1: u16 = 0xBC00;
const H_1_5: u16 = 0x3E00;

fn h(bits: u16) -> f64 {
    match bits {
        H_1 => 1.0,
        H_HALF => 0.5,
        H_2 => 2.0,
        H_QUARTER => 0.25,
        H_NEG1 => -1.0,
        H_1_5 => 1.5,
        _ => panic!("unlisted test scale"),
    }
}

/// Deterministic pseudo-random sequence (LCG), values in `0..m`.
fn rnd(state: &mut u64, m: u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) % m
}

/// A vector of multiples of 0.25 in [-2, 2): exactly representable.
fn make_x(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| (rnd(&mut s, 16) as f32 - 8.0) * 0.25)
        .collect()
}

/// Reference result and the sum of absolute terms (tolerance scale).
fn reference(w: &[Vec<f64>], x: &[f32]) -> (Vec<f64>, Vec<f64>) {
    let mut res = Vec::new();
    let mut mag = Vec::new();
    for row in w {
        let mut s = 0.0;
        let mut m = 0.0;
        for (a, b) in row.iter().zip(x) {
            s += a * *b as f64;
            m += (a * *b as f64).abs();
        }
        res.push(s);
        mag.push(m);
    }
    (res, mag)
}

fn check(got: &[f32], expected: &(Vec<f64>, Vec<f64>), tol: f64) {
    assert_eq!(got.len(), expected.0.len());
    for (i, g) in got.iter().enumerate() {
        let err = (*g as f64 - expected.0[i]).abs();
        assert!(
            err <= tol * expected.1[i].max(1e-3),
            "row {i}: got {g}, expected {}, err {err}, scale {}",
            expected.0[i],
            expected.1[i]
        );
    }
}

fn run_matvec(w: &TensorView<'_>, x: &[f32], threads: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w.rows];
    matvec(w, x, &mut out, threads).unwrap();
    out
}

#[test]
fn f32_matches_reference() {
    let (cols, rows) = (8usize, 3usize);
    let mut s = 1;
    let vals: Vec<Vec<f64>> = (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| (rnd(&mut s, 64) as f64 - 32.0) / 8.0)
                .collect()
        })
        .collect();
    let bytes: Vec<u8> = vals
        .iter()
        .flatten()
        .flat_map(|v| (*v as f32).to_le_bytes())
        .collect();
    let x = make_x(cols, 7);
    let exp = reference(&vals, &x);
    with_tensor(GgmlType::F32, cols, rows, bytes, |w| {
        assert_eq!((w.rows, w.cols), (rows, cols));
        check(&run_matvec(w, &x, 1), &exp, 1e-5);
    });
}

#[test]
fn f16_matches_reference() {
    let (cols, rows) = (8usize, 2usize);
    let table = [H_1, H_HALF, H_2, H_QUARTER, H_NEG1, H_1_5];
    let mut s = 3;
    let bits: Vec<Vec<u16>> = (0..rows)
        .map(|_| (0..cols).map(|_| table[rnd(&mut s, 6) as usize]).collect())
        .collect();
    let vals: Vec<Vec<f64>> = bits
        .iter()
        .map(|r| r.iter().map(|b| h(*b)).collect())
        .collect();
    let bytes: Vec<u8> = bits
        .iter()
        .flatten()
        .flat_map(|b| b.to_le_bytes())
        .collect();
    let x = make_x(cols, 9);
    let exp = reference(&vals, &x);
    with_tensor(GgmlType::F16, cols, rows, bytes, |w| {
        check(&run_matvec(w, &x, 2), &exp, 1e-4);
    });
}

#[test]
fn q8_0_matches_reference() {
    let (cols, rows) = (64usize, 3usize);
    let scales = [H_HALF, H_2, H_1_5, H_QUARTER, H_NEG1, H_1];
    let mut s = 11;
    let mut bytes = Vec::new();
    let mut vals = vec![Vec::new(); rows];
    for row in vals.iter_mut() {
        for blk in 0..cols / 32 {
            let sc = scales[(rnd(&mut s, 6) as usize + blk) % 6];
            bytes.extend_from_slice(&sc.to_le_bytes());
            for _ in 0..32 {
                let q = rnd(&mut s, 256) as i64 - 128;
                bytes.push(q as i8 as u8);
                row.push(h(sc) * q as f64);
            }
        }
    }
    let x = make_x(cols, 13);
    let exp = reference(&vals, &x);
    with_tensor(GgmlType::Q8_0, cols, rows, bytes, |w| {
        check(&run_matvec(w, &x, 1), &exp, 1e-4);
    });
}

#[test]
fn q4_0_matches_reference() {
    let (cols, rows) = (64usize, 3usize);
    let scales = [H_HALF, H_2, H_1_5, H_QUARTER, H_NEG1, H_1];
    let mut s = 17;
    let mut bytes = Vec::new();
    let mut vals = vec![Vec::new(); rows];
    for row in vals.iter_mut() {
        for blk in 0..cols / 32 {
            let sc = scales[(rnd(&mut s, 6) as usize + blk) % 6];
            let codes: Vec<u8> = (0..32).map(|_| rnd(&mut s, 16) as u8).collect();
            bytes.extend_from_slice(&sc.to_le_bytes());
            // Elements 0..16 live in the low nibbles, 16..32 in the high nibbles.
            for j in 0..16 {
                bytes.push(codes[j] | (codes[j + 16] << 4));
            }
            for c in &codes {
                row.push(h(sc) * (*c as f64 - 8.0));
            }
        }
    }
    let x = make_x(cols, 19);
    let exp = reference(&vals, &x);
    with_tensor(GgmlType::Q4_0, cols, rows, bytes, |w| {
        check(&run_matvec(w, &x, 3), &exp, 1e-4);
    });
}

/// Packs eight 6-bit scales and eight 6-bit mins into ggml's 12 bytes.
fn pack_k4(sc: &[u8; 8], mn: &[u8; 8]) -> [u8; 12] {
    let mut p = [0u8; 12];
    for i in 0..4 {
        p[i] = sc[i] | ((sc[i + 4] >> 4) << 6);
        p[i + 4] = mn[i] | ((mn[i + 4] >> 4) << 6);
        p[i + 8] = (sc[i + 4] & 0xF) | ((mn[i + 4] & 0xF) << 4);
    }
    p
}

#[test]
fn q4_k_matches_reference() {
    let (cols, rows) = (512usize, 3usize); // two super-blocks per row
    let mut s = 23;
    let mut bytes = Vec::new();
    let mut vals = vec![Vec::new(); rows];
    for row in vals.iter_mut() {
        for _ in 0..cols / 256 {
            let d = [H_HALF, H_QUARTER, H_1][rnd(&mut s, 3) as usize];
            let dmin = [H_QUARTER, H_HALF, H_2][rnd(&mut s, 3) as usize];
            let mut sc = [0u8; 8];
            let mut mn = [0u8; 8];
            for j in 0..8 {
                sc[j] = rnd(&mut s, 64) as u8; // full 6-bit range
                mn[j] = rnd(&mut s, 64) as u8;
            }
            let codes: Vec<u8> = (0..256).map(|_| rnd(&mut s, 16) as u8).collect();
            bytes.extend_from_slice(&d.to_le_bytes());
            bytes.extend_from_slice(&dmin.to_le_bytes());
            bytes.extend_from_slice(&pack_k4(&sc, &mn));
            // Each 64-element chunk uses 32 bytes: elements 0..32 low nibbles,
            // elements 32..64 high nibbles.
            for chunk in 0..4 {
                for l in 0..32 {
                    bytes.push(codes[chunk * 64 + l] | (codes[chunk * 64 + 32 + l] << 4));
                }
            }
            for (e, c) in codes.iter().enumerate() {
                let g = e / 32;
                row.push(h(d) * sc[g] as f64 * *c as f64 - h(dmin) * mn[g] as f64);
            }
        }
    }
    let x = make_x(cols, 29);
    let exp = reference(&vals, &x);
    with_tensor(GgmlType::Q4_K, cols, rows, bytes, |w| {
        check(&run_matvec(w, &x, 2), &exp, 1e-4);
    });
}

#[test]
fn q6_k_matches_reference() {
    let (cols, rows) = (512usize, 3usize);
    let mut s = 31;
    let mut bytes = Vec::new();
    let mut vals = vec![Vec::new(); rows];
    for row in vals.iter_mut() {
        for _ in 0..cols / 256 {
            let d = [H_HALF, H_QUARTER, H_1_5][rnd(&mut s, 3) as usize];
            let sc: Vec<i8> = (0..16).map(|_| rnd(&mut s, 256) as i64 as i8).collect();
            let v: Vec<u8> = (0..256).map(|_| rnd(&mut s, 64) as u8).collect();
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            for half in 0..2 {
                for l in 0..32 {
                    let (v1, v2, v3, v4) = (
                        v[half * 128 + l],
                        v[half * 128 + 32 + l],
                        v[half * 128 + 64 + l],
                        v[half * 128 + 96 + l],
                    );
                    ql[half * 64 + l] = (v1 & 15) | ((v3 & 15) << 4);
                    ql[half * 64 + 32 + l] = (v2 & 15) | ((v4 & 15) << 4);
                    qh[half * 32 + l] =
                        (v1 >> 4) | ((v2 >> 4) << 2) | ((v3 >> 4) << 4) | ((v4 >> 4) << 6);
                }
            }
            bytes.extend_from_slice(&ql);
            bytes.extend_from_slice(&qh);
            bytes.extend(sc.iter().map(|b| *b as u8));
            bytes.extend_from_slice(&d.to_le_bytes());
            for (e, c) in v.iter().enumerate() {
                row.push(h(d) * sc[e / 16] as f64 * (*c as f64 - 32.0));
            }
        }
    }
    let x = make_x(cols, 37);
    let exp = reference(&vals, &x);
    with_tensor(GgmlType::Q6_K, cols, rows, bytes, |w| {
        check(&run_matvec(w, &x, 1), &exp, 1e-4);
    });
}

#[test]
fn dequant_row_returns_expected_values() {
    // Q8_0, 2 rows x 32: scale 0.5, codes = element index and negated index.
    let mut bytes = Vec::new();
    for sign in [1i32, -1] {
        bytes.extend_from_slice(&H_HALF.to_le_bytes());
        for i in 0..32i32 {
            bytes.push((sign * i) as i8 as u8);
        }
    }
    with_tensor(GgmlType::Q8_0, 32, 2, bytes, |w| {
        let mut out = vec![0.0f32; 32];
        dequant_row(w, 1, &mut out).unwrap();
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, -0.5 * i as f32);
        }
    });
}

#[test]
fn dequant_row_out_of_range_errors() {
    let bytes: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
    with_tensor(GgmlType::F32, 4, 2, bytes, |w| {
        let mut out = vec![0.0f32; 4];
        assert!(dequant_row(w, 1, &mut out).is_ok());
        let e = dequant_row(w, 2, &mut out).unwrap_err();
        assert_eq!(e.code(), "invalid_model");
        let e = dequant_row(w, usize::MAX, &mut out).unwrap_err();
        assert_eq!(e.code(), "invalid_model");
        let mut short = vec![0.0f32; 3];
        assert_eq!(
            dequant_row(w, 0, &mut short).unwrap_err().code(),
            "invalid_model"
        );
    });
}

#[test]
fn unsupported_dtype_errors() {
    let view = TensorView {
        dtype: GgmlType::Other(99),
        rows: 1,
        cols: 4,
        data: &[],
    };
    let mut out = vec![0.0f32; 1];
    let e = matvec(&view, &[0.0; 4], &mut out, 1).unwrap_err();
    assert_eq!(e.code(), "unsupported_architecture");
    assert!(e.to_string().contains("99"), "message: {e}");
    let mut row = vec![0.0f32; 4];
    let e = dequant_row(&view, 0, &mut row).unwrap_err();
    assert_eq!(e.code(), "unsupported_architecture");

    // Also through a real file: an unknown-type tensor can be viewed but not used.
    with_tensor(GgmlType::Other(30), 4, 1, vec![], |w| {
        let mut out = vec![0.0f32; 1];
        let e = matvec(w, &[0.0; 4], &mut out, 1).unwrap_err();
        assert_eq!(e.code(), "unsupported_architecture");
        assert!(e.to_string().contains("30"));
    });
    // A known type that is not implemented (Q5_0 = 6) is also rejected.
    let view = TensorView {
        dtype: GgmlType::from_id(6),
        rows: 1,
        cols: 32,
        data: &[],
    };
    let e = matvec(&view, &[0.0; 32], &mut out, 1).unwrap_err();
    assert_eq!(e.code(), "unsupported_architecture");
}

#[test]
fn view_rejects_unknown_tensor() {
    let gguf = GgufBuilder::new()
        .tensor("w", &[4], GgmlType::F32, vec![0; 16])
        .build();
    let tmp = TempFile::new(&gguf);
    let file = GgufFile::open(&tmp.0).unwrap();
    let store = TensorStore::open(&file, &tmp.0).unwrap();
    assert_eq!(store.view("nope").unwrap_err().code(), "invalid_model");
    let v = store.view("w").unwrap();
    assert_eq!((v.rows, v.cols, v.data.len()), (1, 4, 16));
}

#[test]
fn matvec_rejects_wrong_input_len() {
    let bytes: Vec<u8> = (0..8).flat_map(|i| (i as f32).to_le_bytes()).collect();
    with_tensor(GgmlType::F32, 4, 2, bytes, |w| {
        let mut out = vec![0.0f32; 2];
        let e = matvec(w, &[1.0; 3], &mut out, 1).unwrap_err();
        assert_eq!(e.code(), "invalid_model");
        let mut bad_out = vec![0.0f32; 3];
        let e = matvec(w, &[1.0; 4], &mut bad_out, 1).unwrap_err();
        assert_eq!(e.code(), "invalid_model");
        // threads == 0 is treated as 1.
        matvec(w, &[1.0; 4], &mut out, 0).unwrap();
        assert_eq!(out, vec![0.0 + 1.0 + 2.0 + 3.0, 4.0 + 5.0 + 6.0 + 7.0]);
    });
}

#[test]
fn threads_do_not_change_result() {
    let (cols, rows) = (64usize, 37usize);
    let mut s = 41;
    let mut bytes = Vec::new();
    for _ in 0..rows {
        for _ in 0..cols / 32 {
            bytes.extend_from_slice(&H_QUARTER.to_le_bytes());
            for _ in 0..32 {
                bytes.push(rnd(&mut s, 256) as u8);
            }
        }
    }
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.37).sin()).collect();
    with_tensor(GgmlType::Q8_0, cols, rows, bytes, |w| {
        let one = run_matvec(w, &x, 1);
        for t in [0usize, 2, 4, 8, 100] {
            let many = run_matvec(w, &x, t);
            let a: Vec<u32> = one.iter().map(|v| v.to_bits()).collect();
            let b: Vec<u32> = many.iter().map(|v| v.to_bits()).collect();
            assert_eq!(a, b, "threads = {t}");
        }
    });
}

#[test]
fn f16_edge_values() {
    assert_eq!(f16_to_f32(0x0000).to_bits(), 0.0f32.to_bits());
    assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
    assert_eq!(f16_to_f32(0x3C00), 1.0);
    assert_eq!(f16_to_f32(0xC000), -2.0);
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);
    // Smallest subnormal 2^-24, largest subnormal (1023/1024) * 2^-14, smallest normal 2^-14.
    assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
    assert_eq!(f16_to_f32(0x8001), -(2f32.powi(-24)));
    assert_eq!(f16_to_f32(0x03FF), 1023.0 / 1024.0 * 2f32.powi(-14));
    assert_eq!(f16_to_f32(0x0400), 2f32.powi(-14));
    assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
    assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
    assert!(f16_to_f32(0x7E00).is_nan());
    assert!(f16_to_f32(0x7C01).is_nan());
    assert!(f16_to_f32(0xFFFF).is_nan());

    // Through a kernel: a subnormal Q8_0 scale.
    let mut bytes = 0x0001u16.to_le_bytes().to_vec();
    bytes.extend(std::iter::repeat_n(64u8, 32));
    with_tensor(GgmlType::Q8_0, 32, 1, bytes, |w| {
        let mut out = vec![0.0f32; 32];
        dequant_row(w, 0, &mut out).unwrap();
        assert!(out.iter().all(|v| *v == 64.0 * 2f32.powi(-24)));
    });
}

#[test]
fn truncated_file_is_rejected() {
    let gguf = GgufBuilder::new()
        .tensor("w", &[8], GgmlType::F32, vec![1; 32])
        .build();
    let tmp = TempFile::new(&gguf);
    let file = GgufFile::open(&tmp.0).unwrap();
    // Truncate after parsing so the parsed ranges now exceed the file.
    let f = std::fs::OpenOptions::new()
        .write(true)
        .open(&tmp.0)
        .unwrap();
    f.set_len(file.file_len - 8).unwrap();
    drop(f);
    match TensorStore::open(&file, &tmp.0) {
        Err(e) => assert_eq!(e.code(), "invalid_model"),
        Ok(store) => assert_eq!(store.view("w").unwrap_err().code(), "invalid_model"),
    }
}

#[test]
fn view_revalidates_ranges_and_sizes() {
    let gguf = GgufBuilder::new()
        .tensor("w", &[8], GgmlType::F32, vec![1; 32])
        .build();
    let tmp = TempFile::new(&gguf);
    let good = GgufFile::open(&tmp.0).unwrap();

    // Offset pushed past the end of the file.
    let mut bad = good.clone();
    bad.tensors[0].offset = 1 << 40;
    let store = TensorStore::open(&bad, &tmp.0).unwrap();
    assert_eq!(store.view("w").unwrap_err().code(), "invalid_model");

    // Offset + size overflow.
    let mut bad = good.clone();
    bad.tensors[0].offset = u64::MAX - 4;
    let store = TensorStore::open(&bad, &tmp.0).unwrap();
    assert_eq!(store.view("w").unwrap_err().code(), "invalid_model");

    // Recorded byte size disagrees with dims * row size.
    let mut bad = good.clone();
    bad.tensors[0].n_bytes = 16;
    let store = TensorStore::open(&bad, &tmp.0).unwrap();
    assert_eq!(store.view("w").unwrap_err().code(), "invalid_model");

    // Data section start past the end of the file.
    let mut bad = good.clone();
    bad.data_offset = u64::MAX;
    let store = TensorStore::open(&bad, &tmp.0).unwrap();
    assert_eq!(store.view("w").unwrap_err().code(), "invalid_model");
}

#[test]
fn multi_dim_tensor_rows_are_product_of_outer_dims() {
    let bytes: Vec<u8> = (0..24).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let gguf = GgufBuilder::new()
        .tensor("w", &[4, 3, 2], GgmlType::F32, bytes)
        .build();
    let tmp = TempFile::new(&gguf);
    let file = GgufFile::open(&tmp.0).unwrap();
    let store = TensorStore::open(&file, &tmp.0).unwrap();
    let v = store.view("w").unwrap();
    assert_eq!((v.rows, v.cols), (6, 4));
}
