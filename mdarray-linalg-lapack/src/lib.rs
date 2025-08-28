pub mod qr;
pub mod svd;

#[derive(Debug, Default, Clone)]
pub struct Lapack {
    svd_config: svd::SVDConfig,
    qr_config: qr::LapackQRConfig,
}

impl Lapack {
    pub fn new() -> Self {
        Self {
            svd_config: svd::SVDConfig::default(),
            qr_config: qr::LapackQRConfig::default(),
        }
    }

    pub fn config_svd(mut self, config: svd::SVDConfig) -> Self {
        self.svd_config = config;
        self
    }

    pub fn config_qr(mut self, config: qr::LapackQRConfig) -> Self {
        self.qr_config = config;
        self
    }
}
