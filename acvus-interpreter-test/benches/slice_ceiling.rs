//! The ceiling RFC-0047 buys, before either half of the compiler lands:
//! one inner iteration, four ways, over the same data.
//!
//! | shape | the inner loop |
//! |---|---|
//! | `get` | today: two `Ref`s above the loop, two fused `get` runs with a deref tail |
//! | `as_slice in loop` | an `AsSlice`, an `Index` and a `Drop` per element, twice |
//! | `as_slice hoisted` | the two `AsSlice`s above the header, two `Index` in the loop |
//! | `unchecked` | the same with the bound check removed |
//!
//! Nothing lowers `a[i]` yet, so each shape is a body built by hand. The
//! last shape runs a handler `prepare` never emits: the probe substitutes
//! it into the prepared body, which is the only way to reach it until the
//! interval pass carries its own proof (RFC-0047 §7).

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use acvus_extern::Externs;
use acvus_interpreter::code::{Code, Op, Payload};
use acvus_interpreter::{
    AcvusRuntime, Executable, InMemoryContext, Interpreter, InterpreterContext, PrepareCtx,
    SequentialExecutor, Value, VtableRegistry, prepare_module,
};
use acvus_mir::ir::{
    Callee, DebugInfo, ExternInstance, IndexMode, Inst, InstKind, Label, MirBody, MirModule,
    RefTarget, ValueId,
};
use acvus_mir::ty::{Effect, Mutability, Task, Ty, TypeArg};
use acvus_utils::{Astr, Interner, LocalFactory, LocalIdOps, QualifiedRef};
use rustc_hash::FxHashMap;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    Get,
    SliceInLoop,
    SliceHoisted,
    Unchecked,
}

impl Shape {
    fn name(self) -> &'static str {
        match self {
            Shape::Get => "get",
            Shape::SliceInLoop => "as_slice in loop",
            Shape::SliceHoisted => "as_slice hoisted",
            Shape::Unchecked => "unchecked",
        }
    }

    fn hoists_the_slice(self) -> bool {
        matches!(self, Shape::SliceHoisted | Shape::Unchecked)
    }

    /// The type the loop counts at. An `Index` takes `u64` and nothing
    /// else while `vec::get` takes `i64` (RFC-0047 §4), so the test and the
    /// increment run at one width in the `get` shape and the other in the
    /// three slice shapes. Each is one dispatch either way.
    fn index_ty(self) -> Ty {
        match self {
            Shape::Get => Ty::I64,
            Shape::SliceInLoop | Shape::SliceHoisted | Shape::Unchecked => Ty::U64,
        }
    }
}

const QUERY: &str = "q";
const KEYS: &str = "k";
const LENGTH: &str = "n";
const N: usize = 1_000_000;
const REPS: usize = 7;

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
    let index_ty = shape.index_ty();
    let as_slice = ExternInstance {
        id: in_vec(interner, "as_slice"),
        instance: 0,
    };
    let get = Callee::Extern {
        id: in_vec(interner, "get"),
        instance: 0,
    };
    let get_ty = Ty::Fn {
        params: Vec::new(),
        ret: Box::new(element_ref.clone()),
        captures: Vec::new(),
        effect: Effect::PURE.into(),
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
        merge_of: None,
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
        merge_of: None,
    });

    for operand in operands() {
        match shape {
            Shape::Get => {
                b.push(InstKind::FunctionCall {
                    dst: operand.element_ref,
                    callee: get,
                    callee_ty: get_ty.clone(),
                    args: vec![operand.container_ref, c.index],
                    order: None,
                })
                .push(InstKind::Take {
                    dst: operand.element,
                    target: RefTarget::Through(operand.element_ref),
                    path: Vec::new(),
                });
            }
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
        merge_of: None,
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
/// form of the same mode (RFC-0047 §7).
fn drop_the_bound_check(code: &mut Code) {
    let Code::Body(body) = code else {
        panic!("a body with a loop prepares as a Body")
    };
    let checked = acvus_interpreter::index_handlers::checked(IndexMode::Copy);
    let unchecked = acvus_interpreter::index_handlers::unchecked(IndexMode::Copy);
    let swap = |ops: &mut [Op]| {
        ops.iter_mut()
            .filter(|op| std::ptr::fn_addr_eq(op.f, checked))
            .map(|op| op.f = unchecked)
            .count()
    };
    let swapped: usize = body
        .payloads
        .iter_mut()
        .filter_map(|payload| match payload {
            Payload::Loop(region) => Some(region),
            _ => None,
        })
        .map(|region| swap(region.head.ops_mut()) + swap(region.body.ops_mut()))
        .sum();
    assert_eq!(swapped, 2, "both element reads lost their bound check");
}

fn page(table: &VtableRegistry, n: usize) -> HashMap<String, Value> {
    let run = |offset: f64| {
        let values: Vec<Value> = (0..n).map(|i| Value::float(i as f64 + offset)).collect();
        // SAFETY: read back only as this same `Vec<Value>`, which is what
        // `vec::get` and `vec::as_slice` deref.
        unsafe { Value::erase(table, values) }
    };
    [
        (QUERY.to_string(), run(1.0)),
        (KEYS.to_string(), run(2.0)),
        (LENGTH.to_string(), Value::int(n as i64)),
    ]
    .into_iter()
    .collect()
}

struct Timing {
    elapsed: Duration,
    value: f64,
    dispatches: usize,
}

/// The operations one iteration runs: the loop's head and its body, which
/// is what a band is built from.
fn dispatches_per_iteration(code: &Code) -> usize {
    let Code::Body(body) = code else {
        panic!("a body with a loop prepares as a Body")
    };
    body.payloads
        .iter()
        .filter_map(|payload| match payload {
            Payload::Loop(region) => Some(region),
            _ => None,
        })
        .map(|region| region.head.iter().count() + region.body.iter().count())
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
    };
    let mut prepared = prepare_module(
        &module,
        &PrepareCtx {
            interner: &interner,
            externs: &functions,
            context_names: &context_names,
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
    let table = VtableRegistry::default();
    let mut interpreter = Interpreter::new(shared, entry, InMemoryContext::new(page(&table, n)));

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
    let shapes: Vec<Shape> = [
        Shape::Get,
        Shape::SliceInLoop,
        Shape::SliceHoisted,
        Shape::Unchecked,
    ]
    .into_iter()
    .filter(|shape| only.as_deref().is_none_or(|name| shape.name() == name))
    .collect();
    let mut samples: Vec<Vec<Duration>> = vec![Vec::new(); shapes.len()];
    let mut dispatches = vec![0usize; shapes.len()];
    for rep in 0..REPS {
        for (at, shape) in shapes.iter().enumerate() {
            let Timing {
                elapsed,
                value,
                dispatches: per_iteration,
            } = run_shape(*shape, N);
            dispatches[at] = per_iteration;
            assert!(
                value == expected(N),
                "{}: produced {value}, expected {}",
                shape.name(),
                expected(N)
            );
            if rep > 0 {
                samples[at].push(elapsed);
            }
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
