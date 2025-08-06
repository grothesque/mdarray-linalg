use mdarray::{DSlice, DTensor, Layout};

pub trait QR<T> {
    fn qr<'a, L: Layout>(&self, a: &'a mut DSlice<T, 2, L>) -> impl QRBuilder<'a, T, L>;
}

pub trait QRBuilder<'a, T, L> {
    fn overwrite<Lq: Layout, Lr: Layout>(
        &mut self,
        q: &'a mut DSlice<T, 2, Lq>,
        r: &'a mut DSlice<T, 2, Lr>,
    );

    fn eval<Lq: Layout, Lr: Layout>(&mut self) -> (DTensor<T, 2>, DTensor<T, 2>);
}
