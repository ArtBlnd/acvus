use crate::dsl::{BlockAttrs, ConfigBlock, DslFile, MessageBlock, RoleSpec, ToolDecl};
use crate::error::{OrchError, OrchErrorKind};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn parse_dsl(input: &str) -> Result<DslFile, Vec<OrchError>> {
    let sections = split_sections(input).map_err(|e| vec![e])?;

    if sections.is_empty() {
        return Err(vec![OrchError::new(OrchErrorKind::UnexpectedEof)]);
    }

    if sections[0].name != "configs" {
        return Err(vec![OrchError::new(OrchErrorKind::InvalidConfig(
            "first block must be #![configs]".into(),
        ))]);
    }

    let config = parse_config(&sections[0].body).map_err(|e| vec![e])?;

    let mut blocks = Vec::new();
    for section in &sections[1..] {
        let attrs = parse_block_attrs(&section.attrs).map_err(|e| vec![e])?;
        blocks.push(MessageBlock {
            format: section.name.clone(),
            attrs,
            template_source: section.body.clone(),
        });
    }

    Ok(DslFile { config, blocks })
}

// ---------------------------------------------------------------------------
// Section splitting
// ---------------------------------------------------------------------------

struct RawSection {
    name: String,
    attrs: String,
    body: String,
}

fn split_sections(input: &str) -> Result<Vec<RawSection>, OrchError> {
    let mut sections = Vec::new();
    let mut current: Option<(String, String)> = None; // (name, attrs)
    let mut body_lines: Vec<&str> = Vec::new();

    for line in input.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("#![") {
            // Finish previous section
            if let Some((name, attrs)) = current.take() {
                sections.push(RawSection { name, attrs, body: trim_body(&body_lines) });
                body_lines.clear();
            }

            let header_content = extract_header_content(trimmed)?;
            let (name, attrs) = parse_header(&header_content);
            current = Some((name, attrs));
        } else if current.is_some() {
            if !is_separator(trimmed) {
                body_lines.push(line);
            }
        }
    }

    // Finish last section
    if let Some((name, attrs)) = current {
        sections.push(RawSection { name, attrs, body: trim_body(&body_lines) });
    }

    Ok(sections)
}

fn extract_header_content(line: &str) -> Result<String, OrchError> {
    let start = 3; // skip "#!["
    let mut in_string = false;
    for (i, ch) in line[start..].char_indices() {
        match ch {
            '"' => in_string = !in_string,
            ']' if !in_string => return Ok(line[start..start + i].to_string()),
            _ => {}
        }
    }
    Err(OrchError::new(OrchErrorKind::InvalidBlockAttr(
        "unclosed #![".into(),
    )))
}

fn parse_header(content: &str) -> (String, String) {
    if let Some(paren) = content.find('(') {
        let name = content[..paren].trim().to_string();
        // Find matching ')' respecting quotes
        let inner = &content[paren + 1..];
        let mut in_string = false;
        for (i, ch) in inner.char_indices() {
            match ch {
                '"' => in_string = !in_string,
                ')' if !in_string => return (name, inner[..i].to_string()),
                _ => {}
            }
        }
        (name, inner.to_string())
    } else {
        (content.trim().to_string(), String::new())
    }
}

fn is_separator(line: &str) -> bool {
    !line.is_empty() && (line.chars().all(|c| c == '=') || line.chars().all(|c| c == '+'))
}

fn trim_body(lines: &[&str]) -> String {
    let start = lines.iter().position(|l| !l.trim().is_empty()).unwrap_or(lines.len());
    let end = lines
        .iter()
        .rposition(|l| !l.trim().is_empty())
        .map_or(0, |p| p + 1);
    if start >= end {
        String::new()
    } else {
        lines[start..end].join("\n")
    }
}

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------

fn parse_config(body: &str) -> Result<ConfigBlock, OrchError> {
    let mut scanner = Scanner::new(body);
    let mut name = None;
    let mut model = None;
    let mut inputs = Vec::new();
    let mut tools = Vec::new();

    scanner.skip_ws_nl();
    while !scanner.at_end() {
        let key = scanner.read_ident()?;
        scanner.skip_ws();
        scanner.expect('=')?;
        scanner.skip_ws();
        let value = scanner.read_value()?;

        match key.as_str() {
            "name" => name = Some(value.into_string()?),
            "model" => model = Some(value.into_string()?),
            "tools" => tools = parse_tools(value)?,
            _ => inputs.push((key, value.into_string()?)),
        }

        scanner.skip_ws_nl();
    }

    Ok(ConfigBlock {
        name: name.ok_or_else(|| {
            OrchError::new(OrchErrorKind::InvalidConfig("missing 'name'".into()))
        })?,
        model: model.ok_or_else(|| {
            OrchError::new(OrchErrorKind::InvalidConfig("missing 'model'".into()))
        })?,
        inputs,
        tools,
    })
}

fn parse_tools(value: ConfigValue) -> Result<Vec<ToolDecl>, OrchError> {
    let arr = match value {
        ConfigValue::Array(arr) => arr,
        _ => {
            return Err(OrchError::new(OrchErrorKind::InvalidConfig(
                "tools must be an array".into(),
            )));
        }
    };

    arr.into_iter()
        .map(|item| {
            let entries = match item {
                ConfigValue::Object(entries) => entries,
                _ => {
                    return Err(OrchError::new(OrchErrorKind::InvalidConfig(
                        "tool must be an object".into(),
                    )));
                }
            };

            let mut name = None;
            let mut params = Vec::new();

            for (key, value) in entries {
                match key.as_str() {
                    "name" => name = Some(value.into_string()?),
                    "type" => {
                        let type_entries = match value {
                            ConfigValue::Object(entries) => entries,
                            _ => {
                                return Err(OrchError::new(OrchErrorKind::InvalidConfig(
                                    "tool type must be an object".into(),
                                )));
                            }
                        };
                        for (param_name, param_type) in type_entries {
                            params.push((param_name, param_type.into_string()?));
                        }
                    }
                    _ => {
                        return Err(OrchError::new(OrchErrorKind::InvalidConfig(format!(
                            "unknown tool field: {key}"
                        ))));
                    }
                }
            }

            Ok(ToolDecl {
                name: name.ok_or_else(|| {
                    OrchError::new(OrchErrorKind::InvalidConfig("tool missing 'name'".into()))
                })?,
                params,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Block attrs parsing
// ---------------------------------------------------------------------------

fn parse_block_attrs(attrs_str: &str) -> Result<BlockAttrs, OrchError> {
    if attrs_str.is_empty() {
        return Err(OrchError::new(OrchErrorKind::InvalidBlockAttr(
            "missing attributes".into(),
        )));
    }

    let mut scanner = Scanner::new(attrs_str);
    let mut role = None;
    let mut bind = None;

    scanner.skip_ws();
    while !scanner.at_end() {
        let key = scanner.read_ident()?;
        scanner.skip_ws();
        scanner.expect('=')?;
        scanner.skip_ws();
        let value = scanner.read_quoted_string()?;

        match key.as_str() {
            "type" => {
                role = Some(if let Some(stripped) = value.strip_prefix('@') {
                    RoleSpec::Ref(stripped.to_string())
                } else {
                    RoleSpec::Literal(value)
                });
            }
            "bind" => bind = Some(value),
            _ => {
                return Err(OrchError::new(OrchErrorKind::InvalidBlockAttr(format!(
                    "unknown attribute: {key}"
                ))));
            }
        }

        scanner.skip_ws();
        if scanner.peek() == Some(',') {
            scanner.advance();
            scanner.skip_ws();
        }
    }

    Ok(BlockAttrs {
        role: role.ok_or_else(|| {
            OrchError::new(OrchErrorKind::InvalidBlockAttr(
                "missing 'type' attribute".into(),
            ))
        })?,
        bind,
    })
}

// ---------------------------------------------------------------------------
// Config value type
// ---------------------------------------------------------------------------

enum ConfigValue {
    String(String),
    Array(Vec<ConfigValue>),
    Object(Vec<(String, ConfigValue)>),
}

impl ConfigValue {
    fn into_string(self) -> Result<String, OrchError> {
        match self {
            ConfigValue::String(s) => Ok(s),
            _ => Err(OrchError::new(OrchErrorKind::InvalidConfig(
                "expected string".into(),
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Scanner — character-by-character parser
// ---------------------------------------------------------------------------

struct Scanner {
    chars: Vec<char>,
    pos: usize,
}

impl Scanner {
    fn new(input: &str) -> Self {
        Self { chars: input.chars().collect(), pos: 0 }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.chars.len()
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.chars.get(self.pos).copied();
        if ch.is_some() {
            self.pos += 1;
        }
        ch
    }

    fn skip_ws(&mut self) {
        while let Some(ch) = self.peek() {
            if ch == ' ' || ch == '\t' {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn skip_ws_nl(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, expected: char) -> Result<(), OrchError> {
        match self.advance() {
            Some(ch) if ch == expected => Ok(()),
            Some(ch) => Err(OrchError::new(OrchErrorKind::InvalidConfig(format!(
                "expected '{expected}', got '{ch}'"
            )))),
            None => Err(OrchError::new(OrchErrorKind::UnexpectedEof)),
        }
    }

    fn read_ident(&mut self) -> Result<String, OrchError> {
        let mut ident = String::new();
        while let Some(ch) = self.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                ident.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        if ident.is_empty() {
            Err(OrchError::new(OrchErrorKind::InvalidConfig(
                format!("expected identifier, got {:?}", self.peek()),
            )))
        } else {
            Ok(ident)
        }
    }

    fn read_quoted_string(&mut self) -> Result<String, OrchError> {
        self.expect('"')?;
        let mut s = String::new();
        loop {
            match self.advance() {
                Some('\\') => match self.advance() {
                    Some('"') => s.push('"'),
                    Some('\\') => s.push('\\'),
                    Some('n') => s.push('\n'),
                    Some(ch) => {
                        s.push('\\');
                        s.push(ch);
                    }
                    None => return Err(OrchError::new(OrchErrorKind::UnexpectedEof)),
                },
                Some('"') => return Ok(s),
                Some(ch) => s.push(ch),
                None => return Err(OrchError::new(OrchErrorKind::UnexpectedEof)),
            }
        }
    }

    fn read_value(&mut self) -> Result<ConfigValue, OrchError> {
        match self.peek() {
            Some('"') => self.read_quoted_string().map(ConfigValue::String),
            Some('[') => self.read_array(),
            Some('{') => self.read_object(),
            other => Err(OrchError::new(OrchErrorKind::InvalidConfig(format!(
                "expected value, got {other:?}"
            )))),
        }
    }

    fn read_array(&mut self) -> Result<ConfigValue, OrchError> {
        self.expect('[')?;
        let mut elements = Vec::new();
        self.skip_ws_nl();
        if self.peek() == Some(']') {
            self.advance();
            return Ok(ConfigValue::Array(elements));
        }
        loop {
            self.skip_ws_nl();
            elements.push(self.read_value()?);
            self.skip_ws_nl();
            match self.peek() {
                Some(',') => {
                    self.advance();
                }
                Some(']') => {
                    self.advance();
                    return Ok(ConfigValue::Array(elements));
                }
                other => {
                    return Err(OrchError::new(OrchErrorKind::InvalidConfig(format!(
                        "expected ',' or ']', got {other:?}"
                    ))));
                }
            }
        }
    }

    fn read_object(&mut self) -> Result<ConfigValue, OrchError> {
        self.expect('{')?;
        let mut entries = Vec::new();
        self.skip_ws_nl();
        if self.peek() == Some('}') {
            self.advance();
            return Ok(ConfigValue::Object(entries));
        }
        loop {
            self.skip_ws_nl();
            let key = if self.peek() == Some('"') {
                self.read_quoted_string()?
            } else {
                self.read_ident()?
            };
            self.skip_ws_nl();
            // Accept both '=' and ':' as key-value separator
            match self.peek() {
                Some('=') | Some(':') => {
                    self.advance();
                }
                other => {
                    return Err(OrchError::new(OrchErrorKind::InvalidConfig(format!(
                        "expected '=' or ':', got {other:?}"
                    ))));
                }
            }
            self.skip_ws_nl();
            let value = self.read_value()?;
            entries.push((key, value));
            self.skip_ws_nl();
            match self.peek() {
                Some(',') => {
                    self.advance();
                }
                Some('}') => {
                    self.advance();
                    return Ok(ConfigValue::Object(entries));
                }
                other => {
                    return Err(OrchError::new(OrchErrorKind::InvalidConfig(format!(
                        "expected ',' or '}}', got {other:?}"
                    ))));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal() {
        let input = r#"
#![configs]
name = "output"
model = "gpt-4"

#![openai(type = "system")]
Hello!
"#;
        let dsl = parse_dsl(input).unwrap();
        assert_eq!(dsl.config.name, "output");
        assert_eq!(dsl.config.model, "gpt-4");
        assert!(dsl.config.inputs.is_empty());
        assert!(dsl.config.tools.is_empty());
        assert_eq!(dsl.blocks.len(), 1);
        assert_eq!(dsl.blocks[0].format, "openai");
        assert_eq!(dsl.blocks[0].attrs.role, RoleSpec::Literal("system".into()));
        assert_eq!(dsl.blocks[0].template_source, "Hello!");
    }

    #[test]
    fn parse_with_inputs() {
        let input = r#"
#![configs]
name = "@result"
text = "@text"
model = "gemini"

#![gemini(type = "user")]
{{ @text }}
"#;
        let dsl = parse_dsl(input).unwrap();
        assert_eq!(dsl.config.inputs, vec![("text".into(), "@text".into())]);
    }

    #[test]
    fn parse_with_tools() {
        let input = r#"
#![configs]
name = "agent"
model = "gpt-4"
tools = [
    {
        name = "search",
        type = {
            "@query": "string",
            "@limit": "integer"
        }
    }
]

#![openai(type = "system")]
You are helpful.
"#;
        let dsl = parse_dsl(input).unwrap();
        assert_eq!(dsl.config.tools.len(), 1);
        assert_eq!(dsl.config.tools[0].name, "search");
        assert_eq!(
            dsl.config.tools[0].params,
            vec![("@query".into(), "string".into()), ("@limit".into(), "integer".into())]
        );
    }

    #[test]
    fn parse_role_ref() {
        let input = r#"
#![configs]
name = "out"
model = "m"

#![fmt(type = "@role")]
body
"#;
        let dsl = parse_dsl(input).unwrap();
        assert_eq!(dsl.blocks[0].attrs.role, RoleSpec::Ref("role".into()));
    }

    #[test]
    fn parse_bind() {
        let input = r#"
#![configs]
name = "out"
model = "m"

#![fmt(bind = "@text | enumerate | x -> { index: x.0, value: x.1 }", type = "@type")]
{{@index}}: {{@value}}
"#;
        let dsl = parse_dsl(input).unwrap();
        let block = &dsl.blocks[0];
        assert_eq!(
            block.attrs.bind.as_deref(),
            Some("@text | enumerate | x -> { index: x.0, value: x.1 }")
        );
        assert_eq!(block.attrs.role, RoleSpec::Ref("type".into()));
    }

    #[test]
    fn parse_multiple_blocks() {
        let input = r#"
#![configs]
name = "chat"
model = "gpt-4"

#![openai(type = "system")]
You are a helpful assistant.

#![openai(type = "user")]
Hello!

#![openai(type = "assistant")]
Hi there!
"#;
        let dsl = parse_dsl(input).unwrap();
        assert_eq!(dsl.blocks.len(), 3);
        assert_eq!(dsl.blocks[0].attrs.role, RoleSpec::Literal("system".into()));
        assert_eq!(dsl.blocks[1].attrs.role, RoleSpec::Literal("user".into()));
        assert_eq!(dsl.blocks[2].attrs.role, RoleSpec::Literal("assistant".into()));
    }

    #[test]
    fn parse_separator_lines() {
        let input = r#"
#![configs]
name = "out"
model = "m"
=====
#![fmt(type = "user")]
hello
"#;
        let dsl = parse_dsl(input).unwrap();
        assert_eq!(dsl.blocks[0].template_source, "hello");
    }

    #[test]
    fn parse_new_format_sample() {
        let input = r#"#![configs]
name = "@sample_storage_name_that_can_be_accessed"
text = "@text"
model = "gemini"
tools = [
    {
        name = "callable_name_blabla",
        type = {
            "@args1": "string",
            "@args2": "integer"
        }
    }
]

#![gemini(type = "system")]
Hello there!

#![gemini(type = "model")]
Hello there! {{ @other_orchestration_template_that_can_be_accessed }}

#![gemini(bind = "@text | enumerate | x -> { index: x.0, value: x.1 }", type = "@type")]
{{@index}}: {{@value}}"#;

        let dsl = parse_dsl(input).unwrap();
        assert_eq!(dsl.config.name, "@sample_storage_name_that_can_be_accessed");
        assert_eq!(dsl.config.model, "gemini");
        assert_eq!(dsl.config.inputs, vec![("text".into(), "@text".into())]);
        assert_eq!(dsl.config.tools.len(), 1);
        assert_eq!(dsl.blocks.len(), 3);
    }

    #[test]
    fn error_missing_configs() {
        let input = "#![openai(type = \"user\")]\nhello";
        let err = parse_dsl(input).unwrap_err();
        assert!(matches!(err[0].kind, OrchErrorKind::InvalidConfig(_)));
    }

    #[test]
    fn error_empty_input() {
        let err = parse_dsl("").unwrap_err();
        assert!(matches!(err[0].kind, OrchErrorKind::UnexpectedEof));
    }

    #[test]
    fn error_missing_name() {
        let input = "#![configs]\nmodel = \"m\"";
        let err = parse_dsl(input).unwrap_err();
        assert!(matches!(err[0].kind, OrchErrorKind::InvalidConfig(_)));
    }
}
