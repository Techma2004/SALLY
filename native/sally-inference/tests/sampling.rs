use sally_inference::sampling::Sampler;

fn logits() -> Vec<f32> {
    vec![0.1, 2.0, -1.0, 1.9, 0.5, 2.0, 1.0, -3.0]
}

#[test]
fn argmax_when_temperature_zero() {
    let mut s = Sampler::new(0.0, 0, 1.0, 1);
    // tie between index 1 and 5 -> lowest index
    assert_eq!(s.sample(&mut logits()), 1);
    let mut s = Sampler::new(-1.0, 5, 0.1, 1);
    assert_eq!(s.sample(&mut [0.0, 3.0, 1.0]), 1);
}

#[test]
fn same_seed_same_sequence() {
    let run = |seed| {
        let mut s = Sampler::new(1.0, 0, 1.0, seed);
        (0..64).map(|_| s.sample(&mut logits())).collect::<Vec<_>>()
    };
    assert_eq!(run(42), run(42));
    // seed 0 is remapped but still deterministic
    assert_eq!(run(0), run(0));
}

#[test]
fn different_seed_can_differ() {
    let run = |seed| {
        let mut s = Sampler::new(1.5, 0, 1.0, seed);
        (0..64).map(|_| s.sample(&mut logits())).collect::<Vec<_>>()
    };
    assert_ne!(run(1), run(2));
}

#[test]
fn top_k_one_equals_argmax() {
    let mut s = Sampler::new(1.0, 1, 1.0, 7);
    for _ in 0..20 {
        assert_eq!(s.sample(&mut logits()), 1);
    }
}

#[test]
fn top_k_limits_candidates() {
    let mut s = Sampler::new(5.0, 3, 1.0, 9);
    for _ in 0..200 {
        let id = s.sample(&mut logits());
        assert!([1, 5, 3].contains(&id), "got {id}");
    }
}

#[test]
fn top_p_keeps_at_least_one() {
    for p in [0.0, 1e-9, 0.01, f32::NAN] {
        let mut s = Sampler::new(1.0, 0, p, 3);
        for _ in 0..20 {
            let id = s.sample(&mut [0.0, 10.0, 0.0]);
            assert_eq!(id, 1, "p={p}");
        }
    }
    // p=0.5 over a flat-ish distribution still returns a valid index
    let mut s = Sampler::new(1.0, 0, 0.5, 3);
    assert!(s.sample(&mut logits()) < 8);
}

#[test]
fn top_p_truncates_tail() {
    // probs ~ [0.9, 0.05, 0.05]: p=0.8 keeps only index 0
    let l = [(0.9f32).ln(), (0.05f32).ln(), (0.05f32).ln()];
    let mut s = Sampler::new(1.0, 0, 0.8, 11);
    for _ in 0..100 {
        assert_eq!(s.sample(&mut l.clone()), 0);
    }
}

#[test]
fn nan_logits_do_not_panic() {
    let mut s = Sampler::new(0.0, 0, 1.0, 1);
    assert_eq!(s.sample(&mut [f32::NAN, 1.0, f32::NAN, 2.0]), 3);
    assert_eq!(s.sample(&mut [f32::NAN, f32::NAN]), 0);
    let mut s = Sampler::new(1.0, 2, 0.9, 1);
    let id = s.sample(&mut [f32::NAN, 1.0, f32::NAN, 2.0]);
    assert!(id == 1 || id == 3);
    assert_eq!(s.sample(&mut [f32::NAN, f32::NAN]), 0);
    assert_eq!(s.sample(&mut []), 0);
    // infinities
    assert_eq!(s.sample(&mut [f32::NEG_INFINITY, f32::INFINITY, 0.0]), 1);
    let id = s.sample(&mut [f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0]);
    assert_eq!(id, 2);
}

#[test]
fn distribution_roughly_follows_probabilities() {
    let mut s = Sampler::new(1.0, 0, 1.0, 5);
    let mut counts = [0u32; 2];
    for _ in 0..4000 {
        counts[s.sample(&mut [(0.75f32).ln(), (0.25f32).ln()]) as usize] += 1;
    }
    let frac = counts[0] as f32 / 4000.0;
    assert!((frac - 0.75).abs() < 0.05, "frac {frac}");
}
