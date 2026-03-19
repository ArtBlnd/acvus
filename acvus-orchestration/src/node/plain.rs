use acvus_interpreter::{RuntimeError, TypedValue};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use super::Node;

pub struct PlainNode {
    module: acvus_mir::ir::MirModule,
    interner: Interner,
}

impl PlainNode {
    pub fn new(
        module: acvus_mir::ir::MirModule,
        interner: &Interner,
    ) -> Self {
        Self {
            module,
            interner: interner.clone(),
        }
    }
}

impl Node for PlainNode {
    fn spawn(
        &self,
        local: FxHashMap<Astr, TypedValue>,
    ) -> acvus_utils::Coroutine<TypedValue, RuntimeError> {
        let interner = self.interner.clone();
        let module = self.module.clone();
        acvus_utils::coroutine(move |handle| async move {
            let output = super::helpers::render_block_in_coroutine(
                &interner, &module, &local, &handle,
            ).await?;
            handle.yield_val(TypedValue::string(output)).await;
            Ok(())
        })
    }
}
