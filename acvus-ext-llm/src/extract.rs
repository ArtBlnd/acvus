//! Shared message helpers for LLM provider modules.

use crate::message::{Content, InputMessage, Message};

pub fn input_messages(list: Vec<InputMessage>) -> Vec<Message> {
    list.into_iter()
        .map(|m| Message::Content {
            role: m.role,
            content: Content::Text(m.content),
        })
        .collect()
}

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
