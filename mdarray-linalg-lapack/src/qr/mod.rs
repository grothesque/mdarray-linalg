mod context;
mod scalar;
mod simple;

#[derive(Default, Debug, Clone)]
pub enum LapackQRConfig {
    #[default]
    Full,
    Pivoting,
    TallSkinny,
}
