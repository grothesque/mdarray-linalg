pub mod eig;
pub mod lu;
pub mod qr;
pub mod solve;
pub mod svd;

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub enum SVDConfig {
    #[default]
    Auto,
    DivideConquer,
    Jacobi,
}

#[derive(Debug, Default, Clone)]
pub struct Lapack {
    svd_config: SVDConfig,
    qr_config: LapackQRConfig,
}

#[derive(Default, Debug, Clone)]
pub enum LapackQRConfig {
    #[default]
    Full,
    Pivoting,
    TallSkinny,
}

impl Lapack {
    pub fn new() -> Self {
        Self {
            svd_config: SVDConfig::default(),
            qr_config: LapackQRConfig::default(),
        }
    }

    pub fn config_svd(mut self, config: SVDConfig) -> Self {
        self.svd_config = config;
        self
    }

    pub fn config_qr(mut self, _config: LapackQRConfig) -> Self {
        todo!()
    }
}
