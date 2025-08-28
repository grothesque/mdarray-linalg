mod context;
mod scalar;
mod simple;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SVDConfig {
    Auto,
    DivideConquer,
    Jacobi,
}

impl Default for SVDConfig {
    fn default() -> Self {
        SVDConfig::Auto
    }
}
