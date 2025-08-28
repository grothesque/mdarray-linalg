mod context;
mod scalar;
mod simple;

#[derive(Debug, Clone)]
pub enum LapackQRConfig {
    Full,
    Pivoting,
    TallSkinny,
}

impl Default for LapackQRConfig {
    fn default() -> Self {
        LapackQRConfig::Full
    }
}
