//! The ceiling RFC-0047 buys, before either half of the compiler lands:
//! one inner iteration, three ways, over the same data.
//!
//! | shape | the inner loop |
//! |---|---|
//! | `as_slice in loop` | an `AsSlice`, an `Index` and a `Drop` per element, twice |
//! | `as_slice hoisted` | the two `AsSlice`s above the header, two `Index` in the loop |
//! | `unchecked` | the same with the bound check removed |
//!
//! Nothing lowers `a[i]` yet, so each shape is a body built by hand. The
//! last shape runs a handler `prepare` never emits: the probe substitutes
//! it into the prepared body, which is the only way to reach it until the
//! interval pass carries its own proof (RFC-0047 rule 7).
//!
//! These timings hold only under one pinned core and a fixed load base;
//! `benches/README.md` states the protocol.

use std::collections::HashMap;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use acvus_extern::{Externs, Owned};
use acvus_interpreter::code::{Body, Code, Op, substitute};
use acvus_interpreter::{
    AcvusRuntime, Executable, InMemoryContext, Interpreter, InterpreterContext, PrepareCtx,
    SequentialExecutor, Value, prepare_module,
};
use acvus_mir::ir::{
    DebugInfo, ExternInstance, IndexMode, Inst, InstKind, Label, MirBody, MirModule, RefTarget,
    ValueId,
};
use acvus_mir::ty::{Mutability, Task, Ty, TypeArg};
use acvus_utils::{Astr, Interner, LocalFactory, LocalIdOps, QualifiedRef};
use rustc_hash::FxHashMap;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    SliceInLoop,
    SliceHoisted,
    Unchecked,
}

impl Shape {
    fn name(self) -> &'static str {
        match self {
            Shape::SliceInLoop => "as_slice in loop",
            Shape::SliceHoisted => "as_slice hoisted",
            Shape::Unchecked => "unchecked",
        }
    }

    fn hoists_the_slice(self) -> bool {
        matches!(self, Shape::SliceHoisted | Shape::Unchecked)
    }
}

/// The type the loop counts at: an `Index` takes `u64` and nothing else
/// (RFC-0047 rule 4).
const INDEX_TY: Ty = Ty::U64;

const QUERY: &str = "q";
const KEYS: &str = "k";
const LENGTH: &str = "n";
const N: usize = 1_000_000;
const REPS: NonZeroUsize = reps(6);

const fn reps(n: usize) -> NonZeroUsize {
    match NonZeroUsize::new(n) {
        Some(reps) => reps,
        None => panic!("a size measures at least one rep"),
    }
}

// -- One body, built by hand -----------------------------------------

/// One container's registers: the element read of one operand of the
/// product, by whichever of the three routes the shape takes.
#[derive(Clone, Copy)]
struct Operand {
    context: &'static str,
    storage: ValueId,
    container_ref: ValueId,
    hoisted_slice: ValueId,
    loop_slice: ValueId,
    element_ref: ValueId,
    element: ValueId,
}

/// The loop's own registers, and the constants it enters with.
#[derive(Clone, Copy)]
struct Counter {
    zero: ValueId,
    first_index: ValueId,
    one: ValueId,
    length: ValueId,
    accumulator: ValueId,
    index: ValueId,
    test: ValueId,
    product: ValueId,
    sum: ValueId,
    next_index: ValueId,
    result: ValueId,
}

struct Build {
    insts: Vec<Inst>,
    val_types: FxHashMap<ValueId, Ty>,
}

impl Build {
    fn new() -> Self {
        Self {
            insts: Vec::new(),
            val_types: FxHashMap::default(),
        }
    }

    fn typed(&mut self, id: ValueId, ty: Ty) -> &mut Self {
        self.val_types.insert(id, ty);
        self
    }

    fn push(&mut self, kind: InstKind) -> &mut Self {
        self.insts.push(Inst {
            span: acvus_ast::Span::ZERO,
            kind,
        });
        self
    }

    fn mir(self) -> MirBody {
        let mut factory = LocalFactory::<ValueId>::new();
        for _ in 0..64 {
            factory.next();
        }
        MirBody {
            demoted_diamonds: Default::default(),
            insts: self.insts,
            val_types: self.val_types,
            params: Vec::new(),
            captures: Vec::new(),
            order_param: None,
            task: Task::Sync,
            debug: DebugInfo::new(),
            val_factory: factory,
            label_count: 3,
        }
    }
}

fn v(n: usize) -> ValueId {
    ValueId::from_raw(n)
}

fn reference(mutability: Mutability, target: Ty) -> Ty {
    Ty::Ref(mutability, Box::new(TypeArg::uniform(target)))
}

fn root(interner: &Interner, name: &str) -> QualifiedRef {
    QualifiedRef::root(interner.intern(name))
}

fn in_vec(interner: &Interner, name: &str) -> QualifiedRef {
    QualifiedRef::qualified(interner.intern("vec"), interner.intern(name))
}

fn operands() -> [Operand; 2] {
    [
        Operand {
            context: QUERY,
            storage: v(0),
            container_ref: v(1),
            hoisted_slice: v(2),
            loop_slice: v(3),
            element_ref: v(4),
            element: v(5),
        },
        Operand {
            context: KEYS,
            storage: v(6),
            container_ref: v(7),
            hoisted_slice: v(8),
            loop_slice: v(9),
            element_ref: v(10),
            element: v(11),
        },
    ]
}

fn counter() -> Counter {
    Counter {
        zero: v(20),
        first_index: v(21),
        one: v(22),
        length: v(23),
        accumulator: v(24),
        index: v(25),
        test: v(26),
        product: v(27),
        sum: v(28),
        next_index: v(29),
        result: v(30),
    }
}

fn body_of(interner: &Interner, shape: Shape) -> MirBody {
    let element = Ty::Float;
    let container = acvus_extern::vec_ty(interner, element.clone());
    let container_ref = reference(Mutability::Shared, container.clone());
    let element_ref = reference(Mutability::Shared, element.clone());
    let slice_ty = reference(Mutability::Shared, Ty::Slice(Box::new(element.clone())));
    let index_ty = INDEX_TY;
    let as_slice = ExternInstance {
        id: in_vec(interner, "as_slice"),
        instance: 0,
    };
    let c = counter();
    let mut b = Build::new();
    for id in [c.zero, c.accumulator, c.product, c.sum, c.result] {
        b.typed(id, element.clone());
    }
    for id in [c.first_index, c.one, c.length, c.index, c.next_index] {
        b.typed(id, index_ty.clone());
    }
    b.typed(c.test, Ty::Bool);
    for operand in operands() {
        b.typed(operand.storage, container.clone())
            .typed(operand.container_ref, container_ref.clone())
            .typed(operand.hoisted_slice, slice_ty.clone())
            .typed(operand.loop_slice, slice_ty.clone())
            .typed(operand.element_ref, element_ref.clone())
            .typed(operand.element, element.clone());
    }

    // The entry: the containers, the loop's constants, and the borrows
    // `code_motion` lifts here.
    for operand in operands() {
        b.push(InstKind::Fetch {
            dst: operand.storage,
            context: root(interner, operand.context),
        });
    }
    b.push(InstKind::Const {
        dst: c.zero,
        value: acvus_ast::Literal::Float(0.0),
    })
    .push(InstKind::Const {
        dst: c.first_index,
        value: acvus_ast::Literal::Int(0),
    })
    .push(InstKind::Const {
        dst: c.one,
        value: acvus_ast::Literal::Int(1),
    })
    .push(InstKind::Fetch {
        dst: c.length,
        context: root(interner, LENGTH),
    });
    for operand in operands() {
        b.push(InstKind::Ref {
            dst: operand.container_ref,
            target: RefTarget::Var(operand.storage),
            path: Vec::new(),
            mutability: Mutability::Shared,
        });
    }
    if shape.hoists_the_slice() {
        for operand in operands() {
            b.push(InstKind::AsSlice {
                dst: operand.hoisted_slice,
                container: operand.container_ref,
                mutability: Mutability::Shared,
                instance: as_slice,
            });
        }
    }

    b.push(InstKind::Jump {
        label: Label(0),
        args: vec![c.zero, c.first_index],
    })
    .push(InstKind::BlockLabel {
        label: Label(0),
        params: vec![c.accumulator, c.index],
    })
    .push(InstKind::BinOp {
        dst: c.test,
        op: acvus_ast::BinOp::Lt,
        left: c.index,
        right: c.length,
    })
    .push(InstKind::JumpIf {
        cond: c.test,
        then_label: Label(1),
        then_args: Vec::new(),
        else_label: Label(2),
        else_args: vec![c.accumulator],
    })
    .push(InstKind::BlockLabel {
        label: Label(1),
        params: Vec::new(),
    });

    for operand in operands() {
        match shape {
            Shape::SliceInLoop => {
                b.push(InstKind::AsSlice {
                    dst: operand.loop_slice,
                    container: operand.container_ref,
                    mutability: Mutability::Shared,
                    instance: as_slice,
                })
                .push(InstKind::Index {
                    dst: operand.element,
                    slice: operand.loop_slice,
                    index: c.index,
                    mode: IndexMode::Copy,
                })
                .push(InstKind::Drop {
                    src: operand.loop_slice,
                });
            }
            Shape::SliceHoisted | Shape::Unchecked => {
                b.push(InstKind::Index {
                    dst: operand.element,
                    slice: operand.hoisted_slice,
                    index: c.index,
                    mode: IndexMode::Copy,
                });
            }
        }
    }

    let [query, keys] = operands();
    b.push(InstKind::BinOp {
        dst: c.product,
        op: acvus_ast::BinOp::Mul,
        left: query.element,
        right: keys.element,
    })
    .push(InstKind::BinOp {
        dst: c.sum,
        op: acvus_ast::BinOp::Add,
        left: c.accumulator,
        right: c.product,
    })
    .push(InstKind::BinOp {
        dst: c.next_index,
        op: acvus_ast::BinOp::Add,
        left: c.index,
        right: c.one,
    })
    .push(InstKind::Jump {
        label: Label(0),
        args: vec![c.sum, c.next_index],
    })
    .push(InstKind::BlockLabel {
        label: Label(2),
        params: vec![c.result],
    });

    if shape.hoists_the_slice() {
        for operand in operands() {
            b.push(InstKind::Drop {
                src: operand.hoisted_slice,
            });
        }
    }
    b.push(InstKind::Return {
        value: c.result,
        order: None,
    });

    b.mir()
}

// -- Running one shape -----------------------------------------------

/// Every `Index` handler in a prepared body, swapped for the unchecked
/// form of the same mode (RFC-0047 rule 7).
fn drop_the_bound_check(body: &mut Body) {
    let swapped: usize = body.heads.iter_mut().map(swap_in_chain).sum();
    assert_eq!(swapped, 2, "both element reads lost their bound check");
}

/// Every checked `Index` of this chain and of the chains its regions hold,
/// replaced by the unchecked form of the same read, carrying the successor
/// the node it replaces held.
fn swap_in_chain(head: &mut Box<dyn Op>) -> usize {
    let mut swapped = 0;
    let mut at = head;
    loop {
        if let Some(read) = at.index_read() {
            substitute(at, |next| {
                acvus_interpreter::index_handlers::unchecked(IndexMode::Copy, read, next)
            });
            swapped += 1;
        }
        for owned in at.owns_mut() {
            swapped += swap_in_chain(owned);
        }
        match at.successor_mut() {
            Some(next) => at = next,
            None => return swapped,
        }
    }
}

fn page(n: usize) -> HashMap<String, Owned<AcvusRuntime>> {
    let run = |offset: f64| {
        let values: Vec<Value> = (0..n).map(|i| Value::float(i as f64 + offset)).collect();
        // SAFETY: read back only as this same `Vec<Value>`, which is what
        // `vec::get` and `vec::as_slice` deref.
        unsafe { Value::erase(values) }
    };
    [
        (QUERY.to_string(), run(1.0)),
        (KEYS.to_string(), run(2.0)),
        (LENGTH.to_string(), Value::int(n as i64)),
    ]
    .into_iter()
    .map(|(name, value)| (name, Owned::from_value(value)))
    .collect()
}

struct Timing {
    elapsed: Duration,
    value: f64,
    dispatches: usize,
}

/// The operations one iteration runs: the loop's head and its body, which
/// is what a band is built from.
fn dispatches_per_iteration(body: &Body) -> usize {
    body.heads
        .iter()
        .flat_map(|head| chain_of(head.as_ref()))
        .flat_map(|op| op.owns())
        .map(|owned| ops_in(owned.head))
        .sum()
}

/// The operations of one chain, its terminator excluded.
fn chain_of(head: &dyn Op) -> Vec<&dyn Op> {
    let mut ops = Vec::new();
    let mut at = head;
    while let Some(next) = at.successor() {
        ops.push(at);
        at = next;
    }
    ops
}

/// The operations a region's part runs, its nested regions included.
fn ops_in(head: &dyn Op) -> usize {
    chain_of(head)
        .into_iter()
        .map(|op| {
            1 + op
                .owns()
                .into_iter()
                .map(|owned| ops_in(owned.head))
                .sum::<usize>()
        })
        .sum()
}

fn run_shape(shape: Shape, n: usize) -> Timing {
    let interner = Interner::new();
    let combined = Externs::combine(acvus_ext::std_registries::<AcvusRuntime>(), &interner)
        .expect("the standard registries combine");
    let mut functions: FxHashMap<QualifiedRef, Executable> = combined
        .handlers
        .into_iter()
        .map(|(qref, handlers)| (qref, Executable::Extern(handlers)))
        .collect();

    let entry = root(&interner, "ceiling");
    let context_names: FxHashMap<QualifiedRef, Astr> = [QUERY, KEYS, LENGTH]
        .into_iter()
        .map(|name| root(&interner, name))
        .map(|qref| (qref, qref.name))
        .collect();
    let module = MirModule {
        main: body_of(&interner, shape),
        closures: FxHashMap::default(),
        ret: Ty::Float,
    };
    let mut prepared = prepare_module(
        &module,
        &PrepareCtx {
            interner: &interner,
            externs: &functions,
            context_names: &context_names,
            instances: &acvus_extern::NoInstances,
        },
    );
    if shape == Shape::Unchecked {
        let code = Arc::get_mut(&mut prepared.main).expect("the prepared body is not yet shared");
        drop_the_bound_check(code);
    }
    let dispatches = dispatches_per_iteration(&prepared.main);
    functions.insert(entry, Executable::Module(Arc::new(prepared)));

    let shared = InterpreterContext::new(&interner, functions, Arc::new(SequentialExecutor))
        .with_context_names(context_names);
    let mut interpreter = Interpreter::new(shared, entry, InMemoryContext::new(page(n)));

    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a current-thread tokio runtime");
    let start = Instant::now();
    let value = runtime.block_on(interpreter.execute());
    Timing {
        elapsed: start.elapsed(),
        value: black_box(value.as_float()),
        dispatches,
    }
}

/// The same sum in Rust: a shape that does not produce it is a defect.
fn expected(n: usize) -> f64 {
    (0..n).map(|i| (i as f64 + 1.0) * (i as f64 + 2.0)).sum()
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn main() {
    let only = std::env::var("CEILING_SHAPE").ok();
    let shapes: Vec<Shape> = [Shape::SliceInLoop, Shape::SliceHoisted, Shape::Unchecked]
        .into_iter()
        .filter(|shape| only.as_deref().is_none_or(|name| shape.name() == name))
        .collect();
    let checked_shape = |shape: Shape| {
        let Timing {
            elapsed,
            value,
            dispatches,
        } = run_shape(shape, N);
        assert!(
            value == expected(N),
            "{}: produced {value}, expected {}",
            shape.name(),
            expected(N)
        );
        (dispatches, elapsed)
    };

    for shape in &shapes {
        let (_dispatches, _warm_up) = checked_shape(*shape);
    }
    let mut samples: Vec<Vec<Duration>> = vec![Vec::with_capacity(REPS.get()); shapes.len()];
    let mut dispatches = vec![0usize; shapes.len()];
    for _ in 0..REPS.get() {
        for (at, shape) in shapes.iter().enumerate() {
            let (per_iteration, elapsed) = checked_shape(*shape);
            dispatches[at] = per_iteration;
            samples[at].push(elapsed);
        }
    }
    println!(
        "{:>18} {:>12} {:>14} {:>16}",
        "shape", "dispatches", "execute/us", "ns/element"
    );
    for (at, shape) in shapes.iter().enumerate() {
        let d = median(std::mem::take(&mut samples[at]));
        println!(
            "{:>18} {:>12} {:>14.1} {:>16.3}",
            shape.name(),
            dispatches[at],
            d.as_secs_f64() * 1e6,
            d.as_secs_f64() * 1e9 / N as f64
        );
    }
}
