pub struct ObfConfig {
    pub seed: u64,
    pub string_encryption: bool,
    pub text_encryption: bool,
    pub numeric_split: bool,
    pub test_literal_decompose: bool,
    pub mba: bool,
    pub scheduling: bool,
    pub control_flow_flatten: bool,
    pub opaque_predicates: bool,
    pub dead_code: bool,
}

impl Default for ObfConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            string_encryption: true,
            text_encryption: true,
            numeric_split: true,
            test_literal_decompose: true,
            mba: true,
            scheduling: true,
            control_flow_flatten: true,
            opaque_predicates: true,
            dead_code: true,
        }
    }
}
