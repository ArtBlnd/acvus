//! A runtime that runs extern fns and nothing else. It holds every value in a
//! slot it checks on each access, so a misuse panics, and it writes no
//! `unsafe`: it is the evidence that the safe kinds are implementable
//! without it.
//!
//! Its slots are thread-local because `Runtime::release` takes no runtime.

#![forbid(unsafe_code)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use acvus_extern::{
    Arguments, ExternFn, FnKind, OptionKind, Registry, ResultKind, Runtime, ScalarKind, StringKind, TupleKind,
};
use acvus_mir::decl::Scheme;
use acvus_mir::ty::{Poly, Types};
use acvus_utils::{Interner, QualifiedRef};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Value(usize);

type Body = Box<dyn Fn(&Safe, &[Value]) -> Value>;

enum Held {
    I8(RefCell<i8>),
    I16(RefCell<i16>),
    I32(RefCell<i32>),
    I64(RefCell<i64>),
    U8(RefCell<u8>),
    U16(RefCell<u16>),
    U32(RefCell<u32>),
    U64(RefCell<u64>),
    F64(RefCell<f64>),
    Bool(RefCell<bool>),
    Char(RefCell<char>),
    Unit(RefCell<()>),
    String(RefCell<String>),
    Tuple(RefCell<Box<[Value]>>),
    Option(RefCell<Option<Value>>),
    Result(RefCell<Result<Value, Value>>),
    Fn(Body),
}

#[derive(Default)]
struct Slots {
    held: Vec<Option<Rc<Held>>>,
    free: Vec<usize>,
}

thread_local! {
    static SLOTS: RefCell<Slots> = RefCell::default();
}

fn hold(held: Held) -> Value {
    SLOTS.with_borrow_mut(|slots| {
        let held = Some(Rc::new(held));
        match slots.free.pop() {
            Some(at) => {
                slots.held[at] = held;
                Value(at)
            }
            None => {
                slots.held.push(held);
                Value(slots.held.len() - 1)
            }
        }
    })
}

/// The value's slot, shared for a view; the slot table is not borrowed while
/// the view runs, so a view may open other values.
fn lend(value: Value) -> Rc<Held> {
    SLOTS.with_borrow(|slots| slots.held[value.0].clone()).expect("safe runtime: a released value is used")
}

/// Ends the value's slot and gives back what it held.
fn unhold(value: Value) -> Held {
    let held = SLOTS.with_borrow_mut(|slots| {
        let held = slots.held[value.0].take();
        slots.free.push(value.0);
        held
    });
    let held = held.expect("safe runtime: a released value is released or taken again");
    Rc::into_inner(held).expect("safe runtime: a value is released or taken while it is lent")
}

/// The number of values held, so that a test can show none is leaked.
pub fn live() -> usize {
    SLOTS.with_borrow(|slots| slots.held.iter().filter(|held| held.is_some()).count())
}

fn mismatch(kind: &str) -> ! {
    panic!("safe runtime: a value of another kind is used as {kind}")
}

pub struct Safe {
    names: Interner,
    types: Types<Poly>,
    declared: Vec<(QualifiedRef, Scheme)>,
    functions: HashMap<QualifiedRef, Body>,
}

impl Safe {
    pub fn new(names: Interner) -> Self {
        Safe { names, types: Types::new(), declared: Vec::new(), functions: HashMap::new() }
    }

    pub fn declared(&self) -> &[(QualifiedRef, Scheme)] {
        &self.declared
    }

    pub fn types(&self) -> &Types<Poly> {
        &self.types
    }

    pub fn call(&self, name: &str, args: &[Value]) -> Value {
        let name = QualifiedRef::root(self.names.intern(name));
        let function = self.functions.get(&name).expect("safe runtime: no extern fn is registered at this name");
        function(self, args)
    }
}

impl Registry for Safe {
    fn register<F>(&mut self)
    where
        F: ExternFn<Self>,
    {
        let (name, scheme) = F::declare(&mut self.types, &self.names);
        self.declared.push((name, scheme));
        let body: Body = Box::new(|rt, args| F::call(rt, F::Args::from_fn(|at| args[at])));
        self.functions.insert(name, body);
    }
}

pub struct SafeScalar;
pub struct SafeString;
pub struct SafeTuple;
pub struct SafeOption;
pub struct SafeResult;
pub struct SafeFn;

impl Runtime for Safe {
    type Value = Value;

    type Scalar = SafeScalar;
    type String = SafeString;
    type Tuple = SafeTuple;
    type Option = SafeOption;
    type Result = SafeResult;
    type Fn = SafeFn;

    fn release(value: Value) {
        match unhold(value) {
            Held::Tuple(parts) => {
                for part in parts.into_inner() {
                    Safe::release(part);
                }
            }
            Held::Option(option) => {
                if let Some(payload) = option.into_inner() {
                    Safe::release(payload);
                }
            }
            Held::Result(result) => match result.into_inner() {
                Ok(payload) | Err(payload) => Safe::release(payload),
            },
            Held::I8(_)
            | Held::I16(_)
            | Held::I32(_)
            | Held::I64(_)
            | Held::U8(_)
            | Held::U16(_)
            | Held::U32(_)
            | Held::U64(_)
            | Held::F64(_)
            | Held::Bool(_)
            | Held::Char(_)
            | Held::Unit(_)
            | Held::String(_)
            | Held::Fn(_) => {}
        }
    }
}

macro_rules! scalars {
    ($($scalar:ty => $variant:ident),* $(,)?) => {$(
        impl ScalarKind<Safe, $scalar> for SafeScalar {
            fn make(_rt: &Safe, scalar: $scalar) -> Value {
                hold(Held::$variant(RefCell::new(scalar)))
            }

            fn take(_rt: &Safe, value: Value) -> $scalar {
                match unhold(value) {
                    Held::$variant(scalar) => scalar.into_inner(),
                    _ => mismatch(stringify!($scalar)),
                }
            }

            fn view<O>(_rt: &Safe, value: &Value, lent: impl FnOnce(&$scalar) -> O) -> O {
                match &*lend(*value) {
                    Held::$variant(scalar) => lent(&scalar.borrow()),
                    _ => mismatch(stringify!($scalar)),
                }
            }

            fn view_mut<O>(_rt: &Safe, value: &mut Value, lent: impl FnOnce(&mut $scalar) -> O) -> O {
                match &*lend(*value) {
                    Held::$variant(scalar) => lent(&mut scalar.borrow_mut()),
                    _ => mismatch(stringify!($scalar)),
                }
            }
        }
    )*};
}

scalars! {
    i8 => I8,
    i16 => I16,
    i32 => I32,
    i64 => I64,
    u8 => U8,
    u16 => U16,
    u32 => U32,
    u64 => U64,
    f64 => F64,
    bool => Bool,
    char => Char,
    () => Unit,
}

impl StringKind<Safe> for SafeString {
    fn make(_rt: &Safe, string: String) -> Value {
        hold(Held::String(RefCell::new(string)))
    }

    fn take(_rt: &Safe, value: Value) -> String {
        match unhold(value) {
            Held::String(string) => string.into_inner(),
            _ => mismatch("a string"),
        }
    }

    fn view<O>(_rt: &Safe, value: &Value, lent: impl FnOnce(&String) -> O) -> O {
        match &*lend(*value) {
            Held::String(string) => lent(&string.borrow()),
            _ => mismatch("a string"),
        }
    }

    fn view_mut<O>(_rt: &Safe, value: &mut Value, lent: impl FnOnce(&mut String) -> O) -> O {
        match &*lend(*value) {
            Held::String(string) => lent(&mut string.borrow_mut()),
            _ => mismatch("a string"),
        }
    }
}

impl TupleKind<Safe> for SafeTuple {
    fn make<const N: usize>(_rt: &Safe, parts: [Value; N]) -> Value {
        hold(Held::Tuple(RefCell::new(Box::new(parts))))
    }

    fn take<const N: usize>(_rt: &Safe, value: Value) -> [Value; N] {
        match unhold(value) {
            Held::Tuple(parts) => {
                let parts: Box<[Value; N]> = parts.into_inner().try_into().expect("safe runtime: a tuple is taken at another arity");
                *parts
            }
            _ => mismatch("a tuple"),
        }
    }

    fn view<const N: usize, O>(_rt: &Safe, value: &Value, lent: impl FnOnce([&Value; N]) -> O) -> O {
        match &*lend(*value) {
            Held::Tuple(parts) => {
                let parts = parts.borrow();
                let parts: &[Value; N] = (&**parts).try_into().expect("safe runtime: a tuple is lent at another arity");
                lent(parts.each_ref())
            }
            _ => mismatch("a tuple"),
        }
    }

    fn view_mut<const N: usize, O>(_rt: &Safe, value: &mut Value, lent: impl FnOnce([&mut Value; N]) -> O) -> O {
        match &*lend(*value) {
            Held::Tuple(parts) => {
                let mut parts = parts.borrow_mut();
                let parts: &mut [Value; N] = (&mut **parts).try_into().expect("safe runtime: a tuple is lent at another arity");
                lent(parts.each_mut())
            }
            _ => mismatch("a tuple"),
        }
    }
}

impl OptionKind<Safe> for SafeOption {
    fn make(_rt: &Safe, option: Option<Value>) -> Value {
        hold(Held::Option(RefCell::new(option)))
    }

    fn take(_rt: &Safe, value: Value) -> Option<Value> {
        match unhold(value) {
            Held::Option(option) => option.into_inner(),
            _ => mismatch("an option"),
        }
    }

    fn view<O>(_rt: &Safe, value: &Value, lent: impl FnOnce(Option<&Value>) -> O) -> O {
        match &*lend(*value) {
            Held::Option(option) => lent(option.borrow().as_ref()),
            _ => mismatch("an option"),
        }
    }

    fn view_mut<O>(_rt: &Safe, value: &mut Value, lent: impl FnOnce(Option<&mut Value>) -> O) -> O {
        match &*lend(*value) {
            Held::Option(option) => lent(option.borrow_mut().as_mut()),
            _ => mismatch("an option"),
        }
    }

    fn replace(_rt: &Safe, value: &mut Value, option: Option<Value>) -> Option<Value> {
        match &*lend(*value) {
            Held::Option(held) => held.replace(option),
            _ => mismatch("an option"),
        }
    }
}

impl ResultKind<Safe> for SafeResult {
    fn make(_rt: &Safe, result: Result<Value, Value>) -> Value {
        hold(Held::Result(RefCell::new(result)))
    }

    fn take(_rt: &Safe, value: Value) -> Result<Value, Value> {
        match unhold(value) {
            Held::Result(result) => result.into_inner(),
            _ => mismatch("a result"),
        }
    }

    fn view<O>(_rt: &Safe, value: &Value, lent: impl FnOnce(Result<&Value, &Value>) -> O) -> O {
        match &*lend(*value) {
            Held::Result(result) => lent(result.borrow().as_ref()),
            _ => mismatch("a result"),
        }
    }

    fn view_mut<O>(_rt: &Safe, value: &mut Value, lent: impl FnOnce(Result<&mut Value, &mut Value>) -> O) -> O {
        match &*lend(*value) {
            Held::Result(result) => lent(result.borrow_mut().as_mut()),
            _ => mismatch("a result"),
        }
    }

    fn replace(_rt: &Safe, value: &mut Value, result: Result<Value, Value>) -> Result<Value, Value> {
        match &*lend(*value) {
            Held::Result(held) => held.replace(result),
            _ => mismatch("a result"),
        }
    }
}

impl FnKind<Safe> for SafeFn {
    fn make<const N: usize>(_rt: &Safe, body: impl Fn(&Safe, [Value; N]) -> Value + 'static) -> Value {
        let body: Body = Box::new(move |rt, args| {
            let args: [Value; N] = args.try_into().expect("safe runtime: a function is called at another arity");
            body(rt, args)
        });
        hold(Held::Fn(body))
    }

    fn call<const N: usize>(rt: &Safe, function: &Value, args: [Value; N]) -> Value {
        match &*lend(*function) {
            Held::Fn(body) => body(rt, &args),
            _ => mismatch("a function"),
        }
    }
}
