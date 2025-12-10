pub mod matmul;
pub mod matvec;
pub mod qr;

/// Simple backend, mostly for demonstratration purposes
#[derive(Default)]
pub struct Naive;
