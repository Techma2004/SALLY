//! Token sampling: argmax, temperature, top-k, top-p, with a seeded xorshift64* RNG.
//! No external crates; output is fully determined by (parameters, seed, logits).

const SEED_FALLBACK: u64 = 0x9E37_79B9_7F4A_7C15;

pub struct Sampler {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    state: u64,
}

impl Sampler {
    /// `temperature <= 0` (or NaN) selects argmax. `top_k == 0` disables top-k;
    /// `top_p >= 1.0` disables top-p. Seed 0 is remapped to a nonzero constant
    /// because xorshift has an all-zero fixed point.
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Sampler {
        Sampler {
            temperature,
            top_k,
            top_p,
            state: if seed == 0 { SEED_FALLBACK } else { seed },
        }
    }

    /// Uniform in [0, 1).
    fn next_f64(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        let r = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (r >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Picks a token id. Returns 0 for an empty or all-NaN slice.
    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        let greedy = argmax(logits);
        let Some(greedy) = greedy else { return 0 };
        if self.temperature.is_nan() || self.temperature <= 0.0 {
            return greedy;
        }
        // An infinite maximum makes softmax undefined (inf - inf); the +inf token wins.
        if logits[greedy as usize] == f32::INFINITY {
            return greedy;
        }

        // Candidates: non-NaN logits, ordered by logit descending then index ascending.
        let mut cand: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .filter(|(_, l)| !l.is_nan())
            .map(|(i, &l)| (i, l))
            .collect();
        let order = |a: &(usize, f32), b: &(usize, f32)| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        };
        if self.top_k > 0 && self.top_k < cand.len() {
            cand.select_nth_unstable_by(self.top_k - 1, order);
            cand.truncate(self.top_k);
        }
        cand.sort_by(order);

        // Stable softmax with temperature; max is cand[0].
        let max = cand[0].1;
        let mut probs: Vec<f64> = cand
            .iter()
            .map(|&(_, l)| (f64::from(l - max) / f64::from(self.temperature)).exp())
            .collect();
        let mut sum: f64 = probs.iter().sum();
        if !(sum.is_finite() && sum > 0.0) {
            return greedy;
        }

        if self.top_p < 1.0 {
            // Keep the smallest prefix whose cumulative probability reaches top_p (at least one).
            let mut cum = 0.0;
            let mut keep = probs.len();
            for (i, p) in probs.iter().enumerate() {
                cum += p / sum;
                if cum >= f64::from(self.top_p) {
                    keep = i + 1;
                    break;
                }
            }
            // NaN top_p: the comparison is false everywhere; keep exactly one.
            if self.top_p.is_nan() {
                keep = 1;
            }
            probs.truncate(keep.max(1));
            sum = probs.iter().sum();
        }

        let target = self.next_f64() * sum;
        let mut acc = 0.0;
        for (i, p) in probs.iter().enumerate() {
            acc += p;
            if target < acc {
                return cand[i].0 as u32;
            }
        }
        cand[probs.len() - 1].0 as u32
    }
}

/// Index of the largest non-NaN value; ties go to the lowest index.
fn argmax(logits: &[f32]) -> Option<u32> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &l) in logits.iter().enumerate() {
        if l.is_nan() {
            continue;
        }
        match best {
            Some((_, b)) if l <= b => {}
            _ => best = Some((i, l)),
        }
    }
    best.map(|(i, _)| i as u32)
}
