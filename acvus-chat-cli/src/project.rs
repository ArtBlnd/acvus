use acvus_interpreter::Value;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;
use rustc_hash::FxHashMap;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ExprDef {
    pub name: String,
    pub source: Option<String>,
    pub inline_source: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ProjectSpec {
    pub name: String,
    #[serde(default = "default_fuel_limit")]
    pub fuel_limit: u64,
    pub nodes: Vec<String>,
    pub entrypoint: String,
    #[serde(default)]
    pub providers: FxHashMap<String, ProviderConfig>,
    #[serde(default)]
    pub context: toml::Table,
    #[serde(default)]
    pub expr: Vec<ExprDef>,
}

fn default_fuel_limit() -> u64 {
    50
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProviderConfig {
    pub api: ApiKind,
    pub endpoint: String,
    pub api_key_env: Option<String>,
    pub api_key: Option<String>,
}

/// Provider API kind — local to acvus-chat-cli.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApiKind {
    OpenAI,
    Anthropic,
    Google,
}

/// Resolve a `Ty` from a TOML value.
///
/// If the value is a type name string ("string", "int", "float", "bool"),
/// returns that type directly. Otherwise infers from the value.
pub fn toml_to_ty(interner: &Interner, value: &toml::Value) -> Ty {
    match value {
        toml::Value::String(s) => match s.as_str() {
            "string" => Ty::String,
            "int" => Ty::Int,
            "float" => Ty::Float,
            "bool" => Ty::Bool,
            _ => Ty::String,
        },
        toml::Value::Integer(_) => Ty::Int,
        toml::Value::Float(_) => Ty::Float,
        toml::Value::Boolean(_) => Ty::Bool,
        toml::Value::Array(arr) => {
            let elem_ty = arr
                .first()
                .map(|v| toml_to_ty(interner, v))
                .unwrap_or(Ty::Unit);
            Ty::List(Box::new(elem_ty))
        }
        toml::Value::Table(table) => {
            let fields = table
                .iter()
                .map(|(k, v)| (interner.intern(k), toml_to_ty(interner, v)))
                .collect();
            Ty::Object(fields)
        }
        toml::Value::Datetime(_) => Ty::String,
    }
}

/// Context entry: type + optional default value.
pub struct ContextEntry {
    pub ty: Ty,
    pub default: Option<Value>,
}

/// Parse a context entry from TOML.
///
/// - String ("int", "string"...) → type only
/// - Table { type = "...", value = ... } → type + default
pub fn parse_context_entry(interner: &Interner, value: &toml::Value) -> ContextEntry {
    if let toml::Value::Table(table) = value
        && let Some(ty_val) = table.get("type")
    {
        let ty = toml_to_ty(interner, ty_val);
        let default = table.get("value").map(toml_to_value);
        return ContextEntry { ty, default };
    }
    ContextEntry {
        ty: toml_to_ty(interner, value),
        default: None,
    }
}

/// Convert a TOML value to a runtime Value.
pub fn toml_to_value(value: &toml::Value) -> Value {
    match value {
        toml::Value::String(s) => Value::string(s.clone()),
        toml::Value::Integer(n) => Value::int(*n),
        toml::Value::Float(f) => Value::float(*f),
        toml::Value::Boolean(b) => Value::bool_(*b),
        _ => Value::string(value.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toml_to_ty_primitives() {
        let interner = Interner::new();
        assert_eq!(
            toml_to_ty(&interner, &toml::Value::String("hi".into())),
            Ty::String
        );
        assert_eq!(toml_to_ty(&interner, &toml::Value::Integer(42)), Ty::Int);
        assert_eq!(toml_to_ty(&interner, &toml::Value::Float(3.14)), Ty::Float);
        assert_eq!(toml_to_ty(&interner, &toml::Value::Boolean(true)), Ty::Bool);
    }

    #[test]
    fn toml_to_ty_array() {
        let interner = Interner::new();
        let arr = toml::Value::Array(vec![toml::Value::Integer(1), toml::Value::Integer(2)]);
        assert_eq!(toml_to_ty(&interner, &arr), Ty::List(Box::new(Ty::Int)));
    }

    #[test]
    fn toml_to_ty_empty_array() {
        let interner = Interner::new();
        let arr = toml::Value::Array(vec![]);
        assert_eq!(toml_to_ty(&interner, &arr), Ty::List(Box::new(Ty::Unit)));
    }

    #[test]
    fn toml_to_ty_table() {
        let interner = Interner::new();
        let mut table = toml::Table::new();
        table.insert("name".into(), toml::Value::String("alice".into()));
        table.insert("age".into(), toml::Value::Integer(30));
        let ty = toml_to_ty(&interner, &toml::Value::Table(table));
        let expected = Ty::Object(FxHashMap::from_iter([
            (interner.intern("age"), Ty::Int),
            (interner.intern("name"), Ty::String),
        ]));
        assert_eq!(ty, expected);
    }

    #[test]
    fn parse_project_spec() {
        let toml_str = r#"
name = "test-pipeline"
fuel_limit = 10
nodes = ["a.toml", "b.toml"]
entrypoint = "a"

[providers.openai]
api = "openai"
endpoint = "https://api.openai.com"
api_key_env = "OPENAI_API_KEY"

[context]
topic = "Rust"
count = 5
"#;
        let spec: ProjectSpec = toml::from_str(toml_str).unwrap();
        assert_eq!(spec.name, "test-pipeline");
        assert_eq!(spec.fuel_limit, 10);
        assert_eq!(spec.nodes.len(), 2);
        assert!(spec.providers.contains_key("openai"));
        assert_eq!(spec.context.get("topic").unwrap().as_str(), Some("Rust"));
    }

    #[test]
    fn parse_project_spec_defaults() {
        let toml_str = r#"
name = "minimal"
nodes = []
entrypoint = "main"
"#;
        let spec: ProjectSpec = toml::from_str(toml_str).unwrap();
        assert_eq!(spec.fuel_limit, 50);
        assert!(spec.providers.is_empty());
        assert!(spec.context.is_empty());
    }
}
