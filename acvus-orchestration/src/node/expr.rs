use std::collections::HashMap;
use std::sync::Arc;

use acvus_interpreter::{Coroutine, ExternFnRegistry, Interpreter, ResumeKey, Stepped, Value};

use super::Node;

pub struct ExprNode {
    module: acvus_mir::ir::MirModule,
    extern_fns: ExternFnRegistry,
}

impl ExprNode {
    pub fn new(module: acvus_mir::ir::MirModule, extern_fns: &ExternFnRegistry) -> Self {
        Self {
            module,
            extern_fns: extern_fns.clone(),
        }
    }
}

impl Node for ExprNode {
    fn spawn(&self, local: HashMap<String, Arc<Value>>) -> (Coroutine<Value>, ResumeKey<Value>) {
        let interp = Interpreter::new(self.module.clone(), &self.extern_fns);
        let (mut inner, mut key) = interp.execute();
        acvus_coroutine::coroutine(move |handle| async move {
            loop {
                match inner.resume(key).await {
                    Stepped::Emit(emit) => {
                        let (value, _) = emit.into_parts();
                        handle.yield_val(value).await;
                        return;
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
                    Stepped::Done => {
                        handle.yield_val(Value::Unit).await;
                        return;
                    }
                }
            }
        })
    }
}
