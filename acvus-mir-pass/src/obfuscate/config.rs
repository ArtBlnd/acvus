pub struct ObfConfig {
    pub seed: u64,
    pub text_encryption: bool,
    pub scheduling: bool,
    pub control_flow_flatten: bool,
    pub opaque_predicates: bool,
    pub hash_predicate: bool,
}

impl Default for ObfConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            text_encryption: true,
            scheduling: true,
            control_flow_flatten: true,
            opaque_predicates: true,
            hash_predicate: true,
        }
    }
}
