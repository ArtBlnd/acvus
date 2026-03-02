/// Parsed DSL file representation.
#[derive(Debug, Clone)]
pub struct DslFile {
    pub config: ConfigBlock,
    pub blocks: Vec<MessageBlock>,
}

/// Configuration section from `#![configs]`.
#[derive(Debug, Clone)]
pub struct ConfigBlock {
    pub name: String,
    pub model: String,
    pub inputs: Vec<(String, String)>,
    pub tools: Vec<ToolDecl>,
}

/// A single message block.
#[derive(Debug, Clone)]
pub struct MessageBlock {
    pub format: String,
    pub attrs: BlockAttrs,
    pub template_source: String,
}

/// Attributes of a message block.
#[derive(Debug, Clone)]
pub struct BlockAttrs {
    pub role: RoleSpec,
    pub bind: Option<String>,
}

/// Role specification: literal string or context reference.
#[derive(Debug, Clone, PartialEq)]
pub enum RoleSpec {
    /// Literal role name, e.g. `"system"`.
    Literal(String),
    /// Context reference, e.g. `"@type"` → Ref("type").
    Ref(String),
}

/// Tool declaration in config.
#[derive(Debug, Clone)]
pub struct ToolDecl {
    pub name: String,
    pub params: Vec<(String, String)>,
}
