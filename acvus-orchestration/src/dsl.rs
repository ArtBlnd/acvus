use std::collections::HashMap;

use serde::Deserialize;

/// Node specification parsed from TOML.
#[derive(Debug, Clone, Deserialize)]
pub struct NodeSpec {
    pub name: String,
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub tools: Vec<ToolDecl>,
    pub messages: Vec<MessageSpec>,
}

/// A single message entry referencing an external template file.
#[derive(Debug, Clone, Deserialize)]
pub struct MessageSpec {
    pub role: String,
    pub template: String,
}

/// Tool declaration.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolDecl {
    pub name: String,
    #[serde(default)]
    pub params: HashMap<String, String>,
}
