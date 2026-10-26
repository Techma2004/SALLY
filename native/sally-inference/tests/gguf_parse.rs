#[path = "common/gguf_builder.rs"]
mod gguf_builder;

use gguf_builder::{GgufBuilder, T_STRING};
use sally_inference::error::EngineError;
use sally_inference::gguf::{GgmlType, GgufFile};
use std::io::Cursor;

fn parse(bytes: &[u8]) -> Result<GgufFile, EngineError> {
    GgufFile::from_reader(Cursor::new(bytes), bytes.len() as u64)
}

fn valid_file() -> Vec<u8> {
    GgufBuilder::new()
        .kv_str("general.architecture", "llama")
        .kv_u32("llama.context_length", 4096)
        .tensor("token_embd.weight", &[4, 2], GgmlType::F32, vec![0u8; 32])
        .build()
}

fn assert_invalid(r: Result<GgufFile, EngineError>) {
    match r {
        Err(e) => assert_eq!(e.code(), "invalid_model", "unexpected error: {e}"),
        Ok(_) => panic!("expected invalid_model, got Ok"),
    }
}

#[test]
fn parses_valid_minimal_file() {
    let bytes = valid_file();
    let g = parse(&bytes).unwrap();
    assert_eq!(g.version, 3);
    assert_eq!(g.metadata.get_str("general.architecture"), Some("llama"));
    assert_eq!(g.metadata.get_u32("llama.context_length"), Some(4096));
    assert_eq!(g.tensors.len(), 1);
    let t = &g.tensors[0];
    assert_eq!(t.name, "token_embd.weight");
    assert_eq!(t.dims, vec![4, 2]);
    assert_eq!(t.dtype, GgmlType::F32);
    assert_eq!(t.offset, 0);
    assert_eq!(t.n_bytes, 32);
    assert_eq!(g.data_offset % 32, 0);
    assert_eq!(g.data_offset + 32, bytes.len() as u64);
}

#[test]
fn rejects_bad_magic() {
    let bytes = GgufBuilder::new().magic(*b"GGUX").build();
    assert_invalid(parse(&bytes));
}

#[test]
fn rejects_truncated_header() {
    let bytes = valid_file();
    assert_invalid(parse(&bytes[..10]));
    assert_invalid(parse(&bytes[..30]));
    assert_invalid(parse(&bytes[..60]));
    assert_invalid(parse(&[]));
}

#[test]
fn rejects_tensor_range_past_eof() {
    let bytes = GgufBuilder::new()
        .tensor_raw("t", &[1024], GgmlType::F32.id(), 0)
        .build();
    assert_invalid(parse(&bytes));
    // Offset itself far beyond the file.
    let bytes = GgufBuilder::new()
        .tensor_raw("t", &[1], GgmlType::F32.id(), u64::MAX - 31)
        .build();
    assert_invalid(parse(&bytes));
}

#[test]
fn rejects_impossible_tensor_count() {
    let bytes = GgufBuilder::new().tensor_count(u64::MAX).build();
    assert_invalid(parse(&bytes));
    let bytes = GgufBuilder::new().kv_count(u64::MAX).build();
    assert_invalid(parse(&bytes));
}

#[test]
fn rejects_dim_product_overflow() {
    let bytes = GgufBuilder::new()
        .tensor_raw("t", &[u64::MAX, 2], GgmlType::F32.id(), 0)
        .build();
    assert_invalid(parse(&bytes));
    let bytes = GgufBuilder::new()
        .tensor_raw("t", &[1 << 40, 1 << 40], GgmlType::F32.id(), 0)
        .build();
    assert_invalid(parse(&bytes));
}

#[test]
fn rejects_unsupported_version() {
    let bytes = GgufBuilder::new().version(1).build();
    assert_invalid(parse(&bytes));
    let bytes = GgufBuilder::new().version(4).build();
    assert_invalid(parse(&bytes));
}

#[test]
fn huge_metadata_string_length_rejected_before_alloc() {
    // A string claiming u64::MAX, 1 TiB and 20 MiB bytes inside a tiny file:
    // must fail without attempting the allocation.
    for len in [u64::MAX, 1u64 << 40, 20 << 20] {
        let bytes = GgufBuilder::new()
            .kv_raw("general.name", T_STRING, len.to_le_bytes().to_vec())
            .build();
        assert_invalid(parse(&bytes));
    }
    // Same for a huge key length.
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&u64::MAX.to_le_bytes());
    assert_invalid(parse(&bytes));
}

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

#[test]
fn random_bytes_never_panic() {
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let valid = valid_file();
    for i in 0..2000u32 {
        let len = (rng.next() % 512) as usize;
        let mut buf: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        match i % 3 {
            // Valid magic + version, small counts, random body: reaches metadata/tensor parsing.
            0 => {
                let mut head = b"GGUF".to_vec();
                head.extend_from_slice(&(2 + (rng.next() % 2) as u32).to_le_bytes());
                head.extend_from_slice(&(rng.next() % 4).to_le_bytes());
                head.extend_from_slice(&(rng.next() % 4).to_le_bytes());
                head.extend_from_slice(&buf);
                buf = head;
            }
            // Mutated valid file.
            1 => {
                buf = valid.clone();
                for _ in 0..(1 + rng.next() % 6) {
                    let idx = (rng.next() as usize) % buf.len();
                    buf[idx] = rng.next() as u8;
                }
            }
            // Pure random bytes.
            _ => {}
        }
        // Any Ok/Err is fine; a panic fails the test.
        let _ = parse(&buf);
    }
}
