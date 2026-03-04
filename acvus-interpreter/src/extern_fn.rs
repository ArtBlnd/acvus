use std::collections::HashMap;
use std::sync::Arc;

use futures::future::BoxFuture;

use acvus_mir::extern_module::{ExternFnId, ExternModule, ExternRegistry};
use acvus_mir::ty::Ty;

use crate::builtins::{FromValue, IntoValue};
use crate::value::Value;

#[derive(Clone)]
pub struct ExternFnSig {
    pub params: Vec<Ty>,
    pub ret: Ty,
    pub effectful: bool,
}

#[derive(Clone)]
pub struct ExternFnBody(Arc<dyn Fn(Vec<Value>) -> BoxFuture<'static, Value> + Send + Sync>);

impl ExternFnBody {
    pub fn new<F, Fut>(f: F) -> Self
    where
        F: Fn(Vec<Value>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Value> + Send + 'static,
    {
        Self(Arc::new(move |args| Box::pin(f(args))))
    }

    pub fn from_fn<Args, F>(f: F) -> Self
    where
        F: IntoExternFnBody<Args>,
    {
        f.into_extern_body()
    }

    pub async fn call(&self, args: Vec<Value>) -> Value {
        (self.0)(args).await
    }
}

// -- IntoExternFnBody (typed constructor) -------------------------------------

pub trait IntoExternFnBody<Args> {
    fn into_extern_body(self) -> ExternFnBody;
}

impl<F, Fut, R, A> IntoExternFnBody<(A,)> for F
where
    F: Fn(A) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = R> + Send + 'static,
    A: FromValue + Send + 'static,
    R: IntoValue + Send + 'static,
{
    fn into_extern_body(self) -> ExternFnBody {
        ExternFnBody(Arc::new(move |args| {
            let mut it = args.into_iter();
            let a = A::from_value(it.next().unwrap());
            let fut = self(a);
            Box::pin(async move { fut.await.into_value() })
        }))
    }
}

impl<F, Fut, R, A, B> IntoExternFnBody<(A, B)> for F
where
    F: Fn(A, B) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = R> + Send + 'static,
    A: FromValue + Send + 'static,
    B: FromValue + Send + 'static,
    R: IntoValue + Send + 'static,
{
    fn into_extern_body(self) -> ExternFnBody {
        ExternFnBody(Arc::new(move |args| {
            let mut it = args.into_iter();
            let a = A::from_value(it.next().unwrap());
            let b = B::from_value(it.next().unwrap());
            let fut = self(a, b);
            Box::pin(async move { fut.await.into_value() })
        }))
    }
}

impl<F, Fut, R, A, B, C> IntoExternFnBody<(A, B, C)> for F
where
    F: Fn(A, B, C) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = R> + Send + 'static,
    A: FromValue + Send + 'static,
    B: FromValue + Send + 'static,
    C: FromValue + Send + 'static,
    R: IntoValue + Send + 'static,
{
    fn into_extern_body(self) -> ExternFnBody {
        ExternFnBody(Arc::new(move |args| {
            let mut it = args.into_iter();
            let a = A::from_value(it.next().unwrap());
            let b = B::from_value(it.next().unwrap());
            let c = C::from_value(it.next().unwrap());
            let fut = self(a, b, c);
            Box::pin(async move { fut.await.into_value() })
        }))
    }
}

pub trait ExternFn: Send + Sync + 'static {
    fn name(&self) -> &str;
    fn sig(&self) -> ExternFnSig;
    fn into_body(self) -> ExternFnBody;
}

#[derive(Clone)]
pub struct ExternFnRegistry {
    fns: HashMap<String, (ExternFnSig, ExternFnBody)>,
}

impl Default for ExternFnRegistry {
    fn default() -> Self {
        Self::new()
    }
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

    /// Build an ID-indexed table for fast runtime lookup.
    /// Call after `to_mir_registry()` with the same registry's name table.
    pub fn build_id_table(&self, extern_names: &HashMap<ExternFnId, String>) -> Vec<ExternFnBody> {
        let mut table = vec![None; extern_names.len()];
        for (&id, name) in extern_names {
            let body = self
                .fns
                .get(name)
                .unwrap_or_else(|| panic!("extern fn not found: {name}"));
            table[id.0 as usize] = Some(body.1.clone());
        }
        table.into_iter().map(|b| b.unwrap()).collect()
    }

    /// Extract compile-time type information for the MIR compiler.
    pub fn to_mir_registry(&self) -> ExternRegistry {
        let mut module = ExternModule::new("extern");
        for (name, (sig, _)) in &self.fns {
            module.add_fn(
                name.clone(),
                sig.params.clone(),
                sig.ret.clone(),
                sig.effectful,
            );
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
            ExternFnBody::from_fn(|n: i64| async move { n + 1 })
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
