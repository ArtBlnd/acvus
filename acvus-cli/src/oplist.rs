//! What the interpreter prepares from a script: `main`'s blocks and every
//! closure body's, in the shape the bench oplist dumps print.

use acvus_interpreter::Prepared;
use acvus_interpreter::listing::{CodeText, code_text};

/// Text or JSON, chosen by `--json`: one walk, two renderings.
pub enum Form {
    Text,
    Json,
}

/// One prepared body under the name a reader sees it by.
struct Named {
    name: String,
    code: CodeText,
}

/// `main` first, then the closures by ascending label: the map that holds them
/// has no order a reader could rely on.
fn bodies(prepared: &Prepared) -> Vec<Named> {
    let mut closures: Vec<_> = prepared.closures.iter().collect();
    closures.sort_by_key(|(label, _)| label.0);
    let mut bodies = vec![Named {
        name: "main".to_string(),
        code: code_text(&prepared.main),
    }];
    bodies.extend(closures.into_iter().map(|(label, code)| Named {
        name: format!("closure L{}", label.0),
        code: code_text(code),
    }));
    bodies
}

pub fn dump(prepared: &Prepared, form: Form) -> Result<String, serde_json::Error> {
    let bodies = bodies(prepared);
    match form {
        Form::Text => Ok(bodies
            .iter()
            .map(|body| format!("{}:\n{}", body.name, body.code))
            .collect()),
        // An array, not an object: the bodies are in the order the text form
        // prints them, and a JSON object's keys are not.
        Form::Json => {
            let array: Vec<serde_json::Value> = bodies
                .iter()
                .map(|body| {
                    Ok(serde_json::json!({
                        "name": body.name,
                        "code": serde_json::to_value(&body.code)?,
                    }))
                })
                .collect::<Result<_, serde_json::Error>>()?;
            let mut text = serde_json::to_string_pretty(&serde_json::Value::Array(array))?;
            text.push('\n');
            Ok(text)
        }
    }
}
