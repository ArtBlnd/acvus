use std::sync::Arc;

use acvus_interpreter::{
    Coroutine, ExternFnRegistry, Interpreter, ResumeKey, RuntimeError, Stepped, Value,
};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use super::Node;

pub struct ExprNode {
    module: acvus_mir::ir::MirModule,
    extern_fns: ExternFnRegistry,
    interner: Interner,
}

impl ExprNode {
    pub fn new(
        module: acvus_mir::ir::MirModule,
        extern_fns: &ExternFnRegistry,
        interner: &Interner,
    ) -> Self {
        Self {
            module,
            extern_fns: extern_fns.clone(),
            interner: interner.clone(),
        }
    }
}

impl Node for ExprNode {
    fn spawn(
        &self,
        local: FxHashMap<Astr, Arc<Value>>,
    ) -> (Coroutine<Value, RuntimeError>, ResumeKey<Value>) {
        let interp = Interpreter::new(&self.interner, self.module.clone(), &self.extern_fns);
        let (mut inner, mut key) = interp.execute();
        acvus_utils::coroutine(move |handle| async move {
            loop {
                match inner.resume(key).await {
                    Stepped::Emit(emit) => {
                        let (value, _) = emit.into_parts();
                        handle.yield_val(value).await;
                        return Ok(());
                    }
                    Stepped::NeedContext(need) => {
                        let name = need.name();
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
                        return Ok(());
                    }
                    Stepped::Error(e) => return Err(e),
                }
            }
        })
    }
}
