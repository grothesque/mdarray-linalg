mod context;
mod scalar;
mod simple;

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub enum SVDConfig {
    #[default]
    Auto,
    DivideConquer,
    Jacobi,
}
