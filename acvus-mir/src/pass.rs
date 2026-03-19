use std::any::{Any, TypeId};

use std::marker::PhantomData;

use crate::ir::{MirBody, MirModule};
use rustc_hash::FxHashMap;

/// Analysis result store. Each analysis deposits its output here, keyed by TypeId.
pub struct PassContext {
    results: FxHashMap<TypeId, Box<dyn Any>>,
}

impl Default for PassContext {
    fn default() -> Self {
        Self::new()
    }
}

impl PassContext {
    pub fn new() -> Self {
        Self {
            results: FxHashMap::default(),
        }
    }

    pub fn insert<T>(&mut self, value: T)
    where
        T: 'static,
    {
        self.results.insert(TypeId::of::<T>(), Box::new(value));
    }

    pub fn get<T>(&self) -> &T
    where
        T: 'static,
    {
        self.results
            .get(&TypeId::of::<T>())
            .and_then(|b| b.downcast_ref::<T>())
            .expect("analysis result not found in PassContext — missing dependency?")
    }
}

/// Extract required analysis results from PassContext as references.
/// Implemented for () and tuples of references up to 4 elements.
pub trait FromContext<'a> {
    fn from_context(ctx: &'a PassContext) -> Self;
    fn required_type_ids() -> Vec<TypeId>;
}

impl<'a> FromContext<'a> for () {
    fn from_context(_ctx: &'a PassContext) -> Self {}
    fn required_type_ids() -> Vec<TypeId> {
        vec![]
    }
}

macro_rules! impl_from_context {
    ($($T:ident),+) => {
        impl<'a, $($T),+> FromContext<'a> for ($(&'a $T,)+)
        where
            $($T: 'static),+
        {
            fn from_context(ctx: &'a PassContext) -> Self {
                ($(ctx.get::<$T>(),)+)
            }
            fn required_type_ids() -> Vec<TypeId> {
                vec![$(TypeId::of::<$T>()),+]
            }
        }
    };
}

impl_from_context!(A);
impl_from_context!(A, B);
impl_from_context!(A, B, C);
impl_from_context!(A, B, C, D);

/// An analysis pass that reads MirModule + dependencies and produces a result.
pub trait AnalysisPass {
    type Required<'a>: FromContext<'a>;
    type Output: 'static;
    fn run(&self, module: &MirModule, deps: Self::Required<'_>) -> Self::Output;
}

/// A transform pass that rewrites MirModule in-place within the pipeline.
pub trait TransformPass {
    type Required<'a>: FromContext<'a>;
    fn transform(&self, module: MirModule, deps: Self::Required<'_>) -> MirModule;
}

/// Wrapper to place a TransformPass into a Chain.
pub struct Transform<P>(pub P);

/// Marker stored in PassContext after a TransformPass runs.
/// Downstream passes can depend on this to enforce ordering.
pub struct TransformMarker<P: 'static>(PhantomData<P>);

/// Heterogeneous linked list of passes.
pub struct Chain<H, T>(pub H, pub T);

/// Trait for collecting and dispatching passes.
pub trait PassSet {
    /// Collect (output_type_id, dep_type_ids, analysis_type_id) for each pass.
    fn collect_deps(&self, out: &mut Vec<(TypeId, Vec<TypeId>, TypeId)>);

    /// Run the pass whose TypeId matches `target`, storing result in ctx.
    /// TransformPass may mutate module in-place.
    fn run_targeted(&self, target: TypeId, module: &mut MirModule, ctx: &mut PassContext);
}

impl PassSet for () {
    fn collect_deps(&self, _out: &mut Vec<(TypeId, Vec<TypeId>, TypeId)>) {}
    fn run_targeted(&self, _target: TypeId, _module: &mut MirModule, _ctx: &mut PassContext) {}
}

impl<P, Rest> PassSet for Chain<P, Rest>
where
    P: AnalysisPass + 'static,
    Rest: PassSet,
{
    fn collect_deps(&self, out: &mut Vec<(TypeId, Vec<TypeId>, TypeId)>) {
        out.push((
            TypeId::of::<P::Output>(),
            <P::Required<'static>>::required_type_ids(),
            TypeId::of::<P>(),
        ));
        self.1.collect_deps(out);
    }

    fn run_targeted(&self, target: TypeId, module: &mut MirModule, ctx: &mut PassContext) {
        if target == TypeId::of::<P>() {
            let deps = <P::Required<'_>>::from_context(ctx);
            let output = self.0.run(module, deps);
            ctx.insert(output);
        } else {
            self.1.run_targeted(target, module, ctx);
        }
    }
}

impl<P, Rest> PassSet for Chain<Transform<P>, Rest>
where
    P: TransformPass + 'static,
    Rest: PassSet,
{
    fn collect_deps(&self, out: &mut Vec<(TypeId, Vec<TypeId>, TypeId)>) {
        out.push((
            TypeId::of::<TransformMarker<P>>(),
            <P::Required<'static>>::required_type_ids(),
            TypeId::of::<Transform<P>>(),
        ));
        self.1.collect_deps(out);
    }

    fn run_targeted(&self, target: TypeId, module: &mut MirModule, ctx: &mut PassContext) {
        if target == TypeId::of::<Transform<P>>() {
            let deps = <P::Required<'_>>::from_context(ctx);
            let old = std::mem::replace(
                module,
                MirModule {
                    main: MirBody::new(),
                    closures: FxHashMap::default(),
                },
            );
            *module = self.0.0.transform(old, deps);
            ctx.insert(TransformMarker::<P>(PhantomData));
        } else {
            self.1.run_targeted(target, module, ctx);
        }
    }
}

/// Manages a set of analysis passes, running them in dependency order.
pub struct PassManager<S> {
    passes: S,
    /// Topologically sorted analysis TypeIds.
    order: Vec<TypeId>,
}

impl<S> PassManager<S>
where
    S: PassSet,
{
    pub fn new(passes: S) -> Self {
        let mut entries = Vec::new();
        passes.collect_deps(&mut entries);

        let output_to_analysis: FxHashMap<TypeId, TypeId> = entries
            .iter()
            .map(|&(output, _, analysis)| (output, analysis))
            .collect();

        let all_analyses: Vec<TypeId> = entries.iter().map(|&(_, _, a)| a).collect();
        let mut in_degree: FxHashMap<TypeId, usize> =
            all_analyses.iter().map(|&a| (a, 0)).collect();
        let mut adj: FxHashMap<TypeId, Vec<TypeId>> =
            all_analyses.iter().map(|&a| (a, Vec::new())).collect();

        for &(_, ref dep_outputs, analysis_id) in &entries {
            for dep_out in dep_outputs {
                if let Some(&dep_analysis) = output_to_analysis.get(dep_out) {
                    adj.get_mut(&dep_analysis).unwrap().push(analysis_id);
                    *in_degree.get_mut(&analysis_id).unwrap() += 1;
                }
            }
        }

        // Kahn's topological sort
        let mut queue: Vec<TypeId> = in_degree
            .iter()
            .filter(|&(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();
        let mut order = Vec::with_capacity(all_analyses.len());

        while let Some(node) = queue.pop() {
            order.push(node);
            for &next in &adj[&node] {
                let deg = in_degree.get_mut(&next).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push(next);
                }
            }
        }

        assert_eq!(
            order.len(),
            all_analyses.len(),
            "cycle detected in pass dependencies"
        );

        Self { passes, order }
    }

    pub fn run(&self, mut module: MirModule) -> (MirModule, PassContext) {
        let mut ctx = PassContext::new();
        for &analysis_id in &self.order {
            self.passes.run_targeted(analysis_id, &mut module, &mut ctx);
        }
        (module, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::MirModule;

    // -- Dummy analyses for testing --

    struct ResultA(i32);
    struct PassA;
    impl AnalysisPass for PassA {
        type Required<'a> = ();
        type Output = ResultA;
        fn run(&self, _module: &MirModule, _: ()) -> ResultA {
            ResultA(42)
        }
    }

    struct ResultB(i32);
    struct PassB;
    impl AnalysisPass for PassB {
        type Required<'a> = (&'a ResultA,);
        type Output = ResultB;
        fn run(&self, _module: &MirModule, (a,): (&ResultA,)) -> ResultB {
            ResultB(a.0 + 1)
        }
    }

    struct ResultC(i32);
    struct PassC;
    impl AnalysisPass for PassC {
        type Required<'a> = (&'a ResultA, &'a ResultB);
        type Output = ResultC;
        fn run(&self, _module: &MirModule, (a, b): (&ResultA, &ResultB)) -> ResultC {
            ResultC(a.0 + b.0)
        }
    }

    fn empty_module() -> MirModule {
        use crate::ir::MirBody;

        MirModule {
            main: MirBody::new(),
            closures: FxHashMap::default(),

        }
    }

    #[test]
    fn single_pass_no_deps() {
        let manager = PassManager::new(Chain(PassA, ()));
        let (_, ctx) = manager.run(empty_module());
        assert_eq!(ctx.get::<ResultA>().0, 42);
    }

    #[test]
    fn two_passes_with_dependency() {
        let manager = PassManager::new(Chain(PassB, Chain(PassA, ())));
        let (_, ctx) = manager.run(empty_module());
        assert_eq!(ctx.get::<ResultA>().0, 42);
        assert_eq!(ctx.get::<ResultB>().0, 43);
    }

    #[test]
    fn three_passes_dag() {
        let manager = PassManager::new(Chain(PassC, Chain(PassB, Chain(PassA, ()))));
        let (_, ctx) = manager.run(empty_module());
        assert_eq!(ctx.get::<ResultA>().0, 42);
        assert_eq!(ctx.get::<ResultB>().0, 43);
        assert_eq!(ctx.get::<ResultC>().0, 85);
    }

    #[test]
    fn registration_order_independent() {
        let manager = PassManager::new(Chain(PassA, Chain(PassB, ())));
        let (_, ctx) = manager.run(empty_module());
        assert_eq!(ctx.get::<ResultB>().0, 43);
    }

    #[test]
    fn from_context_unit() {
        let ctx = PassContext::new();
        let () = <()>::from_context(&ctx);
        assert_eq!(<() as FromContext>::required_type_ids().len(), 0);
    }

    #[test]
    fn from_context_single() {
        let mut ctx = PassContext::new();
        ctx.insert(ResultA(10));
        let (a,) = <(&ResultA,)>::from_context(&ctx);
        assert_eq!(a.0, 10);
    }

    #[test]
    fn from_context_pair() {
        let mut ctx = PassContext::new();
        ctx.insert(ResultA(1));
        ctx.insert(ResultB(2));
        let (a, b) = <(&ResultA, &ResultB)>::from_context(&ctx);
        assert_eq!(a.0, 1);
        assert_eq!(b.0, 2);
    }

    // -- Cycle detection --

    struct ResultX(i32);
    struct PassX;
    impl AnalysisPass for PassX {
        type Required<'a> = (&'a ResultY,);
        type Output = ResultX;
        fn run(&self, _module: &MirModule, (y,): (&ResultY,)) -> ResultX {
            ResultX(y.0)
        }
    }

    struct ResultY(i32);
    struct PassY;
    impl AnalysisPass for PassY {
        type Required<'a> = (&'a ResultX,);
        type Output = ResultY;
        fn run(&self, _module: &MirModule, (x,): (&ResultX,)) -> ResultY {
            ResultY(x.0)
        }
    }

    #[test]
    #[should_panic(expected = "cycle detected")]
    fn cycle_detected() {
        PassManager::new(Chain(PassX, Chain(PassY, ())));
    }
}
