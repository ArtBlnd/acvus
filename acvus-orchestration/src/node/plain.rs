use std::collections::HashMap;
use std::sync::Arc;

use acvus_interpreter::{Coroutine, ExternFnRegistry, Interpreter, ResumeKey, Stepped, Value};

use super::Node;

pub struct PlainNode {
    module: acvus_mir::ir::MirModule,
    extern_fns: ExternFnRegistry,
}

impl PlainNode {
    pub fn new(module: acvus_mir::ir::MirModule, extern_fns: &ExternFnRegistry) -> Self {
        Self {
            module,
            extern_fns: extern_fns.clone(),
        }
    }
}

impl Node for PlainNode {
    fn spawn(&self, local: HashMap<String, Arc<Value>>) -> (Coroutine<Value>, ResumeKey<Value>) {
        let interp = Interpreter::new(self.module.clone(), &self.extern_fns);
        let (mut inner, mut key) = interp.execute();
        acvus_coroutine::coroutine(move |handle| async move {
            let mut output = String::new();
            loop {
                match inner.resume(key).await {
                    Stepped::Emit(emit) => {
                        let (value, next_key) = emit.into_parts();
                        let Value::String(s) = value else {
                            panic!("PlainNode: expected String emit, got {value:?}");
                        };
                        output.push_str(&s);
                        key = next_key;
                    }
                    Stepped::NeedContext(need) => {
                        let name = need.name().to_string();
                        if let Some(arc) = local.get(&name) {
                            key = need.into_key(Arc::clone(arc));
                        } else {
                            let bindings = need.bindings().clone();
                            let value = if bindings.is_empty() {
                                handle.request_context(name).await
                            } else {
                                handle.request_context_with(name, bindings).await
                            };
                            key = need.into_key(value);
                        }
                    }
                    Stepped::Done => break,
                }
            }
            handle.yield_val(Value::String(output)).await;
        })
    }
}
