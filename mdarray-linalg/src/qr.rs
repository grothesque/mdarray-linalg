use mdarray::{DSlice, DTensor, Layout};

pub trait QR<T> {
    fn qr_overwrite<L: Layout, Lq: Layout, Lr: Layout>(
        &self,
        a: &mut DSlice<T, 2, L>,
        q: &mut DSlice<T, 2, Lq>,
        r: &mut DSlice<T, 2, Lr>,
    );

    fn qr<L: Layout>(&self, a: &mut DSlice<T, 2, L>) -> (DTensor<T, 2>, DTensor<T, 2>);
}
