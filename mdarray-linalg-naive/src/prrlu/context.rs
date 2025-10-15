use num_traits::{Float, One, Signed};

use mdarray::{DSlice, Layout};

use mdarray_linalg::prrlu::{PRRLU, PRRLUDecomp};

use super::simple::{eye, prrlu};

use crate::Naive;

impl<T: Default + Clone + One + Float + Signed + std::fmt::Debug> PRRLU<T> for Naive {
    fn prrlu<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> PRRLUDecomp<T> {
        let (m, n) = *a.shape();
        let mut p = eye::<T>(m);
        let mut q = eye::<T>(n);
        let mut l = eye::<T>(m);
        let k = m.max(n);
        let epsilon = T::from(1e-12).unwrap();
        let rank = prrlu(a, &mut p, &mut q, &mut l, k, epsilon);
        PRRLUDecomp {
            p,
            l,
            u: a.as_ref().to_owned().into(),
            q,
            rank,
        }
    }

    fn prrlu_rank<L: Layout>(&self, a: &mut DSlice<T, 2, L>, target_rank: usize) -> PRRLUDecomp<T> {
        let (m, n) = *a.shape();
        let mut p = eye::<T>(m);
        let mut q = eye::<T>(n);
        let mut l = eye::<T>(m);
        let epsilon = T::from(1e-12).unwrap();

        let rank = prrlu(a, &mut p, &mut q, &mut l, target_rank, epsilon);
        PRRLUDecomp {
            p,
            l,
            u: a.as_ref().to_owned().into(),
            q,
            rank,
        }
    }

    fn prrlu_overwrite<L: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        p: &mut DSlice<T, 2>,
        l: &mut DSlice<T, 2>,
        q: &mut DSlice<T, 2>,
    ) -> usize {
        let (m, n) = *a.shape();
        let k = m.max(n);

        let epsilon = T::from(1e-12).unwrap();
        prrlu(a, p, q, l, k, epsilon)
    }

    fn rank<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> usize {
        let (m, n) = *a.shape();
        let mut p = eye::<T>(m);
        let mut q = eye::<T>(n);
        let mut l = eye::<T>(m);
        let k = m.max(n);

        let epsilon = T::from(1e-12).unwrap();

        prrlu(a, &mut p, &mut q, &mut l, k, epsilon)
    }
}
