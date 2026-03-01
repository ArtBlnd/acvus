use std::collections::HashMap;

use futures::future::BoxFuture;

use acvus_mir::extern_module::{ExternModule, ExternRegistry};
use acvus_mir::ty::Ty;

use crate::value::Value;

pub struct ExternFnSig {
    pub params: Vec<Ty>,
    pub ret: Ty,
    pub effectful: bool,
}

pub struct ExternFnBody(
    Box<dyn Fn(Vec<Value>) -> BoxFuture<'static, Value> + Send + Sync>,
);

impl ExternFnBody {
    pub fn new<F, Fut>(f: F) -> Self
    where
        F: Fn(Vec<Value>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Value> + Send + 'static,
    {
        Self(Box::new(move |args| Box::pin(f(args))))
    }

    pub async fn call(&self, args: Vec<Value>) -> Value {
        (self.0)(args).await
    }
}

pub trait ExternFn: Send + Sync + 'static {
    fn name(&self) -> &str;
    fn sig(&self) -> ExternFnSig;
    fn into_body(self) -> ExternFnBody;
}

pub struct ExternFnRegistry {
    fns: HashMap<String, (ExternFnSig, ExternFnBody)>,
}

impl ExternFnRegistry {
    pub fn new() -> Self {
        Self {
            fns: HashMap::new(),
        }
    }

    pub fn register(&mut self, f: impl ExternFn) {
        let name = f.name().to_owned();
        let sig = f.sig();
        let body = f.into_body();
        assert!(
            !self.fns.contains_key(&name),
            "duplicate extern function: {name}",
        );
        self.fns.insert(name, (sig, body));
    }

    pub fn get(&self, name: &str) -> Option<&ExternFnBody> {
        self.fns.get(name).map(|(_, body)| body)
    }

    /// Extract compile-time type information for the MIR compiler.
    pub fn to_mir_registry(&self) -> ExternRegistry {
        let mut module = ExternModule::new("extern");
        for (name, (sig, _)) in &self.fns {
            module.add_fn(name.clone(), sig.params.clone(), sig.ret.clone(), sig.effectful);
        }
        let mut registry = ExternRegistry::new();
        registry.register(&module);
        registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AddOne;

    impl ExternFn for AddOne {
        fn name(&self) -> &str {
            "add_one"
        }
        fn sig(&self) -> ExternFnSig {
            ExternFnSig {
                params: vec![Ty::Int],
                ret: Ty::Int,
                effectful: false,
            }
        }
        fn into_body(self) -> ExternFnBody {
            ExternFnBody::new(|args| async move {
                match &args[0] {
                    Value::Int(n) => Value::Int(n + 1),
                    _ => panic!("expected Int"),
                }
            })
        }
    }

    #[tokio::test]
    async fn register_and_call() {
        let mut registry = ExternFnRegistry::new();
        registry.register(AddOne);

        let body = registry.get("add_one").unwrap();
        let result = body.call(vec![Value::Int(41)]).await;
        match result {
            Value::Int(42) => {}
            _ => panic!("expected Int(42)"),
        }
    }

    #[test]
    fn to_mir_registry_extracts_types() {
        let mut registry = ExternFnRegistry::new();
        registry.register(AddOne);

        let mir_registry = registry.to_mir_registry();
        let def = mir_registry.get("add_one").unwrap();
        assert_eq!(def.params, vec![Ty::Int]);
        assert_eq!(def.ret, Ty::Int);
        assert!(!def.effectful);
    }

    #[test]
    #[should_panic(expected = "duplicate extern function")]
    fn duplicate_panics() {
        let mut registry = ExternFnRegistry::new();
        registry.register(AddOne);
        registry.register(AddOne);
    }
}
