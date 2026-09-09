//! Shared Value extraction helpers for LLM provider modules.

use acvus_interpreter::{RuntimeError, Value};
use acvus_utils::{Astr, Interner};
use rust_decimal::Decimal;
use rustc_hash::FxHashMap;

use crate::message::{Content, Message};

pub fn obj_get_str(obj: &FxHashMap<Astr, Value>, key: Astr) -> Option<String> {
    match obj.get(&key)? {
        Value::String(s) => Some(s.to_string()),
        _ => None,
    }
}

pub fn obj_get_decimal(obj: &FxHashMap<Astr, Value>, key: Astr) -> Option<Decimal> {
    match obj.get(&key)? {
        Value::Float(f) => Decimal::try_from(*f).ok(),
        Value::Int(i) => Some(Decimal::from(*i)),
        _ => None,
    }
}

pub fn obj_get_u32(obj: &FxHashMap<Astr, Value>, key: Astr) -> Option<u32> {
    match obj.get(&key)? {
        Value::Int(i) => u32::try_from(*i).ok(),
        _ => None,
    }
}

/// Convert message Objects (role/content) into `Vec<Message>`.
pub fn values_to_messages(
    list: &[Value],
    interner: &Interner,
    provider: &str,
) -> Result<Vec<Message>, RuntimeError> {
    let role_key = interner.intern("role");
    let content_key = interner.intern("content");

    let mut messages = Vec::with_capacity(list.len());
    for item in list {
        let obj = match item {
            Value::Object(o) => o,
            other => {
                return Err(RuntimeError::fetch(format!(
                    "{provider}: expected Object in messages list, got {:?}",
                    other.kind()
                )));
            }
        };

        let role = obj_get_str(obj, role_key).ok_or_else(|| {
            RuntimeError::fetch(format!("{provider}: missing 'role' field in message"))
        })?;
        let content_str = obj_get_str(obj, content_key).ok_or_else(|| {
            RuntimeError::fetch(format!("{provider}: missing 'content' field in message"))
        })?;

        messages.push(Message::Content {
            role,
            content: Content::Text(content_str),
        });
    }
    Ok(messages)
}

/// Split the first system-role message out of the list.
///
/// Some APIs (Anthropic, Gemini) take system as a separate top-level field.
pub fn split_system(messages: &[Message]) -> (Option<String>, Vec<&Message>) {
    let mut system = None;
    let mut rest = Vec::new();
    for m in messages {
        if let Message::Content {
            role,
            content: Content::Text(text),
        } = m
            && role == "system"
            && system.is_none()
        {
            system = Some(text.clone());
            continue;
        }
        rest.push(m);
    }
    (system, rest)
}
