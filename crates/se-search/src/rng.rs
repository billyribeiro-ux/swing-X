//! A tiny deterministic PRNG (SplitMix64) seeded explicitly — never from the clock.
//!
//! The search must be reproducible: the same `(base_seed, generation)` yields the same
//! population, mutations, and threshold draws. This mirrors the generator used in
//! `se_validation::fixtures` so the whole workspace shares one cheap, well-distributed,
//! seed-controlled source of randomness.

/// SplitMix64. Cheap, deterministic, good enough for genome sampling.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// Seed the generator. Combine a run-level base seed with the generation index so each
    /// generation draws an independent (but reproducible) stream.
    pub fn seeded(base_seed: u64, generation: u32) -> Self {
        let mixed = base_seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(u64::from(generation).wrapping_mul(0xD1B5_4A32_D192_ED03))
            .wrapping_add(0x9E37_79B9_7F4A_7C15);
        Rng(mixed)
    }

    /// Construct from a raw seed (no generation mixing).
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` with a 53-bit mantissa.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in `[0, n)`. Returns 0 if `n == 0`.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }

    /// Pick a random element of `xs`, or `None` if empty.
    pub fn choice<'a, T>(&mut self, xs: &'a [T]) -> Option<&'a T> {
        if xs.is_empty() {
            None
        } else {
            Some(&xs[self.below(xs.len())])
        }
    }

    /// True with probability `p`.
    pub fn chance(&mut self, p: f64) -> bool {
        self.uniform() < p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_per_seed_and_generation() {
        let a: Vec<u64> = {
            let mut r = Rng::seeded(42, 1);
            (0..8).map(|_| r.next_u64()).collect()
        };
        let b: Vec<u64> = {
            let mut r = Rng::seeded(42, 1);
            (0..8).map(|_| r.next_u64()).collect()
        };
        assert_eq!(a, b, "same seed+gen must be reproducible");

        let c: Vec<u64> = {
            let mut r = Rng::seeded(42, 2);
            (0..8).map(|_| r.next_u64()).collect()
        };
        assert_ne!(a, c, "different generation must differ");
    }

    #[test]
    fn uniform_in_range_and_below_bounded() {
        let mut r = Rng::seeded(7, 0);
        for _ in 0..1000 {
            let u = r.uniform();
            assert!((0.0..1.0).contains(&u));
            assert!(r.below(5) < 5);
        }
        assert_eq!(r.below(0), 0);
    }

    #[test]
    fn choice_empty_is_none() {
        let mut r = Rng::seeded(1, 0);
        let empty: [u8; 0] = [];
        assert!(r.choice(&empty).is_none());
        assert!(r.choice(&[9u8]).is_some());
    }
}
