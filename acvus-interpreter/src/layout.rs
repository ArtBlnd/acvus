//! A value as canonical bytes by its type (RFC-0033): no tags, the type is
//! the schema. The language's shapes are laid out here; an extension type
//! goes through the hooks its registry declared, and a nested extension
//! value is written as the head of its own log, which the space supplies.
//!
//! Decoding treats its bytes as untrusted (RFC-0033 rule 2): every byte
//! pattern either decodes to a value of the type or is a `SpaceError`, and a
//! count is bounded before any element is read, by the remaining bytes or,
//! for zero-width elements, by the `ZeroWidth` allowance the caller gives.
//! A nested extension value is read by the space through its type's
//! `Journaled` hooks, which are the type's own promise (RFC-0080 rule 4);
//! what the space checks around them is in `space.rs`.

use std::cell::Cell;

use acvus_extern::repr::{self, Word};
use acvus_extern::{NodeHash, ObjectShape, Owned, SpaceError, SpaceHooks, SpaceResult};
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ty::{IntTy, LenTerm, Ty, TypeArg};
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::runtime::AcvusRuntime;
use crate::value::Value;

pub type Hooks = FxHashMap<QualifiedRef, SpaceHooks<AcvusRuntime>>;

/// What a layout does at a nested extension value: the space commits or
/// loads it and the layout writes or reads its head.
pub trait Nested {
    fn commit(&self, rt: &AcvusRuntime, ty: &Ty, value: &Value) -> SpaceResult<NodeHash>;
    /// The value `head` names; everything it reads zero-width is taken
    /// from `zero_width`, the allowance of the decode that reached it.
    fn load(
        &self,
        rt: &AcvusRuntime,
        ty: &Ty,
        head: NodeHash,
        zero_width: &ZeroWidth,
    ) -> SpaceResult<Value>;
}

/// How many zero-width elements one decode may read in all (RFC-0033 rule
/// 2). A zero-width element (`()`, `{}`, a tuple or object of them) lays no
/// byte, so the remaining bytes bound no count of them; this does.
///
/// The caller constructs it with its bound, and everything one decode reads
/// takes from the same allowance: each array count of zero-width elements,
/// each zero-width part an extension's hooks read through the space
/// (`decode_part`), and every nested value the decode loads. A count above
/// what is left is refused before any element is read, and nesting does not
/// multiply the bound: `[[()]]` reads at most the bound's units in all, not
/// the bound's units per inner array.
pub struct ZeroWidth {
    left: Cell<usize>,
}

impl ZeroWidth {
    pub fn allowing(bound: usize) -> Self {
        Self {
            left: Cell::new(bound),
        }
    }

    fn take(&self, count: usize) -> SpaceResult<()> {
        let left = self.left.get();
        if count > left {
            return Err(SpaceError::new(format!(
                "{count} zero-width elements, where {left} are left of the decode's allowance"
            )));
        }
        self.left.set(left - count);
        Ok(())
    }
}

/// The fewest bytes a value of `ty` lays: zero exactly for the zero-width
/// types, whose counts `ZeroWidth` bounds. A type no space holds lays
/// nothing and is refused where a value of it is read, so it counts as zero
/// width: a count of its elements takes from the allowance, and the first
/// element is refused.
pub(crate) fn least_width(ty: &Ty) -> usize {
    match ty {
        Ty::Int(k) => k.bytes(),
        Ty::Float => 8,
        Ty::Char => 4,
        Ty::Bool => 1,
        Ty::Unit => 0,
        Ty::String | Ty::Array(..) | Ty::Enum { .. } => 8,
        Ty::Tuple(elems) => elems
            .iter()
            .fold(0usize, |sum, t| sum.saturating_add(least_width(t))),
        Ty::Object(fields) => fields
            .values()
            .fold(0usize, |sum, t| sum.saturating_add(least_width(t))),
        Ty::Option(_) | Ty::Result(..) => 1,
        Ty::UserDefined { .. } => NodeHash::LEN,
        Ty::Fn { .. }
        | Ty::Handle(_)
        | Ty::Ref(..)
        | Ty::Slice(_)
        | Ty::Str
        | Ty::Order
        | Ty::Never
        | Ty::Error(_) => 0,
        Ty::Var(v) => match *v {},
    }
}

fn hooks_of<'a>(
    rt: &'a AcvusRuntime,
    ty: &Ty,
    id: &QualifiedRef,
) -> SpaceResult<&'a SpaceHooks<AcvusRuntime>> {
    rt.shared.space.get(id).ok_or_else(|| {
        SpaceError::new(format!(
            "{} declares no space hooks",
            ty.display(&rt.shared.interner)
        ))
    })
}

fn not_held(rt: &AcvusRuntime, ty: &Ty) -> SpaceError {
    SpaceError::new(format!(
        "a value of type {} is not held by a space",
        ty.display(&rt.shared.interner)
    ))
}

/// A structural object's field order: its field names as strings (RFC-0050
/// rule 8).
///
/// `prepare::runs::lay` calls this for a run's layout and `encode`/`decode`
/// call it for the canonical bytes, so a committed object and the registers it
/// is read from cannot disagree. An interned symbol's own `Ord` is the order of
/// first interning, which differs between programs and across a source change,
/// and that is why the comparison is on the resolved string.
pub(crate) fn sorted_fields<'a>(
    interner: &Interner,
    fields: &'a FxHashMap<Astr, Ty>,
) -> Vec<(&'a Astr, &'a Ty)> {
    let mut out: Vec<_> = fields.iter().collect();
    out.sort_by(|(a, _), (b, _)| interner.resolve(**a).cmp(interner.resolve(**b)));
    out
}

fn sorted_variants<'a>(
    interner: &Interner,
    variants: &'a FxHashMap<Astr, Option<Box<Ty>>>,
) -> Vec<(&'a Astr, &'a Option<Box<Ty>>)> {
    let mut out: Vec<_> = variants.iter().collect();
    out.sort_by(|(a, _), (b, _)| interner.resolve(**a).cmp(interner.resolve(**b)));
    out
}

/// Which side of a `Result` the tag word names: `Ok`'s type, `Err`'s, and the
/// byte the canonical form spells it with. RFC-0050 rule 8 leaves a `Result` a
/// variant holding one of two tags, and this is where a reader that has the
/// type turns that word back into the side.
pub fn result_side<'t>(
    rt: &AcvusRuntime,
    tag: Astr,
    ok: &'t Ty,
    err: &'t Ty,
) -> SpaceResult<(u8, &'t Ty)> {
    if tag == rt.shared.interner.intern("Ok") {
        return Ok((0, ok));
    }
    if tag == rt.shared.interner.intern("Err") {
        return Ok((1, err));
    }
    Err(SpaceError::new(format!(
        "a Result holds the tag `{}`",
        tag.display(&rt.shared.interner)
    )))
}

pub fn encode(
    rt: &AcvusRuntime,
    nested: &dyn Nested,
    ty: &Ty,
    value: &Value,
    out: &mut Vec<u8>,
) -> SpaceResult<()> {
    match ty {
        Ty::Int(k) => {
            let word = value.bits();
            // SAFETY (each arm): a value of type `Int(k)` holds `into_word` of
            // an integer of that width.
            match k {
                IntTy::I8 => out.extend_from_slice(&unsafe { i8::from_word(word) }.to_le_bytes()),
                IntTy::I16 => out.extend_from_slice(&unsafe { i16::from_word(word) }.to_le_bytes()),
                IntTy::I32 => out.extend_from_slice(&unsafe { i32::from_word(word) }.to_le_bytes()),
                IntTy::I64 => out.extend_from_slice(&unsafe { i64::from_word(word) }.to_le_bytes()),
                IntTy::U8 => out.extend_from_slice(&unsafe { u8::from_word(word) }.to_le_bytes()),
                IntTy::U16 => out.extend_from_slice(&unsafe { u16::from_word(word) }.to_le_bytes()),
                IntTy::U32 => out.extend_from_slice(&unsafe { u32::from_word(word) }.to_le_bytes()),
                IntTy::U64 => out.extend_from_slice(&unsafe { u64::from_word(word) }.to_le_bytes()),
            }
        }
        Ty::Float => out.extend_from_slice(&value.as_float().to_bits().to_le_bytes()),
        Ty::Char => out.extend_from_slice(&value.as_char().to_le_bytes()),
        Ty::Bool => out.push(value.as_bool() as u8),
        Ty::Unit => {}
        // SAFETY (each composite): the type is the runtime's witness of the
        // value's shape; the checker admits no other value in a place of
        // this type.
        Ty::String => {
            let s = unsafe { value.as_str() };
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        Ty::Array(elem, _) => {
            let items = unsafe { value.as_array() };
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for v in items {
                encode(rt, nested, elem, v, out)?;
            }
        }
        Ty::Tuple(elems) => {
            for (v, t) in unsafe { value.as_tuple() }.iter().zip(elems) {
                encode(rt, nested, t, v, out)?;
            }
        }
        Ty::Object(fields) => {
            let laid = sorted_fields(&rt.shared.interner, fields);
            let values = unsafe { value.as_object() };
            if values.len() != laid.len() {
                return Err(SpaceError::new(format!(
                    "an object of {} fields committed as {}, which lays {}",
                    values.len(),
                    ty.display(&rt.shared.interner),
                    laid.len()
                )));
            }
            for ((_, t), v) in laid.iter().zip(values) {
                encode(rt, nested, t, v, out)?;
            }
        }
        Ty::Option(inner) => match value.option_payload() {
            Some(v) => {
                out.push(1);
                encode(rt, nested, inner, &v, out)?;
            }
            None => out.push(0),
        },
        Ty::Result(ok, err) => {
            let variant = unsafe { value.as_variant() };
            // SAFETY: the same witness — a variant's first register is its tag.
            let tag = unsafe { variant.tag().as_tag() };
            let (byte, held) = result_side(rt, tag, ok, err)?;
            out.push(byte);
            encode(rt, nested, held, variant.payload(), out)?;
        }
        Ty::Enum { variants, .. } => {
            let variant = unsafe { value.as_variant() };
            // SAFETY: the same witness — a variant's first register is its tag.
            let tag = unsafe { variant.tag().as_tag() };
            let sorted = sorted_variants(&rt.shared.interner, variants);
            let index = sorted
                .iter()
                .position(|(k, _)| **k == tag)
                .ok_or_else(|| SpaceError::new("variant not in its enum type"))?;
            out.extend_from_slice(&(index as u64).to_le_bytes());
            if let Some(t) = sorted[index].1 {
                encode(rt, nested, t, variant.payload(), out)?;
            }
        }
        Ty::UserDefined { .. } => {
            let head = nested.commit(rt, ty, value)?;
            out.extend_from_slice(&head.0);
        }
        Ty::Fn { .. }
        | Ty::Handle(_)
        | Ty::Ref(..)
        | Ty::Slice(_)
        | Ty::Str
        | Ty::Order
        | Ty::Never
        | Ty::Error(_) => {
            return Err(not_held(rt, ty));
        }
        Ty::Var(v) => match *v {},
    }
    Ok(())
}

fn take<'a>(input: &mut &'a [u8], n: usize) -> SpaceResult<&'a [u8]> {
    if input.len() < n {
        return Err(SpaceError::new("truncated layout"));
    }
    let (head, rest) = input.split_at(n);
    *input = rest;
    Ok(head)
}

fn take_array<const N: usize>(input: &mut &[u8]) -> SpaceResult<[u8; N]> {
    let (head, rest) = input
        .split_first_chunk::<N>()
        .ok_or_else(|| SpaceError::new("truncated layout"))?;
    *input = rest;
    Ok(*head)
}

fn take_u64(input: &mut &[u8]) -> SpaceResult<u64> {
    Ok(u64::from_le_bytes(take_array(input)?))
}

fn take_byte(input: &mut &[u8]) -> SpaceResult<u8> {
    let [byte] = take_array(input)?;
    Ok(byte)
}

/// A length, a count or an index laid as eight bytes, refused where this
/// target's `usize` cannot hold it: a space written on a 64-bit target and
/// read on a 32-bit one names numbers the reader has no `usize` for.
fn take_len(input: &mut &[u8], what: &str) -> SpaceResult<usize> {
    let word = take_u64(input)?;
    repr::len_of_word(word).map_err(|e| SpaceError::new(format!("{what}: {e}")))
}

/// A part `decode` reads, in a holder of its own.
pub(crate) fn decode_owned(
    rt: &AcvusRuntime,
    nested: &dyn Nested,
    ty: &Ty,
    input: &mut &[u8],
    zero_width: &ZeroWidth,
) -> SpaceResult<Owned<AcvusRuntime>> {
    let value = decode(rt, nested, ty, input, zero_width)?;
    // SAFETY: `decode` made the word, and no other holder owns it.
    Ok(unsafe { Owned::from_value(acvus_extern::Holding::new(), value) })
}

/// A part an extension's hooks read through the space, one at a time: a
/// zero-width part takes one from the allowance, so a count the hooks read
/// themselves is bounded as an array's count is.
pub(crate) fn decode_part(
    rt: &AcvusRuntime,
    nested: &dyn Nested,
    ty: &Ty,
    input: &mut &[u8],
    zero_width: &ZeroWidth,
) -> SpaceResult<Owned<AcvusRuntime>> {
    if least_width(ty) == 0 {
        zero_width.take(1)?;
    }
    decode_owned(rt, nested, ty, input, zero_width)
}

/// Refuses the bytes a decode left unread: a value decoded from bytes
/// leaves none (RFC-0033 rule 2).
pub(crate) fn all_read(rest: &[u8], what: &dyn std::fmt::Display) -> SpaceResult<()> {
    match rest.len() {
        0 => Ok(()),
        n => Err(SpaceError::new(format!("{n} bytes left unread after {what}"))),
    }
}

/// A whole value of `ty` from `bytes`, refused when any byte is left
/// unread. `decode` reads a part and leaves the rest for its caller; this
/// is the entry point for bytes that are one value.
pub fn decode_all(
    rt: &AcvusRuntime,
    nested: &dyn Nested,
    ty: &Ty,
    bytes: &[u8],
    zero_width: &ZeroWidth,
) -> SpaceResult<Value> {
    let mut input = bytes;
    // In a holder, so a value refused for its trailing bytes is released.
    let value = decode_owned(rt, nested, ty, &mut input, zero_width)?;
    all_read(input, &ty.display(&rt.shared.interner))?;
    // SAFETY: the word moves to the caller, which owns it from then on.
    Ok(value.into_value(unsafe { acvus_extern::Holding::new() }))
}

/// A value of `ty` read from the front of `input`, which is left at the
/// first byte the value does not use. Each count is bounded before its
/// elements are read: by what the remaining bytes can hold, and for
/// zero-width elements by `zero_width`.
pub fn decode(
    rt: &AcvusRuntime,
    nested: &dyn Nested,
    ty: &Ty,
    input: &mut &[u8],
    zero_width: &ZeroWidth,
) -> SpaceResult<Value> {
    Ok(match ty {
        Ty::Int(k) => {
            let mut word = [0u8; 8];
            word[..k.bytes()].copy_from_slice(take(input, k.bytes())?);
            Value::from_bits(*k, k.read(u64::from_le_bytes(word)) as u64)
        }
        Ty::Float => Value::float(f64::from_bits(take_u64(input)?)),
        Ty::Char => {
            let code = u32::from_le_bytes(take_array(input)?);
            let c = char::from_u32(code).ok_or_else(|| {
                SpaceError::new(format!("{code:#x} is not a Unicode scalar value"))
            })?;
            Value::char_(c)
        }
        Ty::Bool => match take_byte(input)? {
            0 => Value::bool_(false),
            1 => Value::bool_(true),
            other => return Err(SpaceError::new(format!("bool: byte {other}"))),
        },
        Ty::Unit => Value::unit(),
        Ty::String => {
            let len = take_len(input, "String length")?;
            let bytes = take(input, len)?;
            Value::string(
                std::str::from_utf8(bytes).map_err(|e| SpaceError::new(format!("String: {e}")))?,
            )
        }
        Ty::Array(elem, len) => {
            let count = take_len(input, "Array length")?;
            if let LenTerm::Known(n) = len
                && *n != count
            {
                return Err(SpaceError::new(format!(
                    "expected {n} elements for {}, got {count}",
                    ty.display(&rt.shared.interner)
                )));
            }
            match least_width(elem) {
                0 => zero_width.take(count)?,
                width if count > input.len() / width => {
                    return Err(SpaceError::new(format!(
                        "{count} elements of {} laid in {} bytes, each at least {width}",
                        elem.display(&rt.shared.interner),
                        input.len()
                    )));
                }
                _ => {}
            }
            Value::array(
                (0..count)
                    .map(|_| decode_owned(rt, nested, elem, input, zero_width))
                    .collect::<SpaceResult<_>>()?,
            )
        }
        Ty::Tuple(elems) => Value::tuple(
            elems
                .iter()
                .map(|t| decode_owned(rt, nested, t, input, zero_width))
                .collect::<SpaceResult<_>>()?,
        ),
        Ty::Object(fields) => {
            let laid = sorted_fields(&rt.shared.interner, fields);
            let shape = ObjectShape::in_order(laid.iter().map(|(name, _)| **name).collect());
            let values: Box<[Owned<AcvusRuntime>]> = laid
                .iter()
                .map(|(_, t)| decode_owned(rt, nested, t, input, zero_width))
                .collect::<SpaceResult<_>>()?;
            Value::object(shape, values)
        }
        Ty::Option(inner) => match take_byte(input)? {
            0 => Value::NONE,
            1 => Value::some(decode(rt, nested, inner, input, zero_width)?),
            other => return Err(SpaceError::new(format!("Option: tag {other}"))),
        },
        Ty::Result(ok, err) => {
            let (tag, held) = match take_byte(input)? {
                0 => ("Ok", ok),
                1 => ("Err", err),
                other => return Err(SpaceError::new(format!("Result: tag {other}"))),
            };
            let payload = decode_owned(rt, nested, held, input, zero_width)?;
            Value::variant(rt.shared.interner.intern(tag), Some(payload))
        }
        Ty::Enum { variants, .. } => {
            let sorted = sorted_variants(&rt.shared.interner, variants);
            let index = take_len(input, "variant index")?;
            let (tag, payload_ty) = sorted
                .get(index)
                .ok_or_else(|| SpaceError::new("variant index out of its enum type"))?;
            let payload = match payload_ty {
                Some(t) => Some(decode_owned(rt, nested, t, input, zero_width)?),
                None => None,
            };
            Value::variant(**tag, payload)
        }
        // The head names a node of the space; what the space reads there
        // is checked in `Space::load_at`, and the value's own bytes by its
        // type's hooks (RFC-0080 rule 4).
        Ty::UserDefined { .. } => nested.load(rt, ty, NodeHash(take_array(input)?), zero_width)?,
        Ty::Fn { .. }
        | Ty::Handle(_)
        | Ty::Ref(..)
        | Ty::Slice(_)
        | Ty::Str
        | Ty::Order
        | Ty::Never
        | Ty::Error(_) => {
            return Err(not_held(rt, ty));
        }
        Ty::Var(v) => match *v {},
    })
}

/// Whether a value of `ty` may hold an extension value anywhere inside:
/// what a space has to walk before it encodes.
pub fn holds_extension(ty: &Ty) -> bool {
    match ty {
        Ty::UserDefined { .. } => true,
        Ty::Array(inner, _) | Ty::Option(inner) => holds_extension(inner),
        Ty::Result(ok, err) => holds_extension(ok) || holds_extension(err),
        Ty::Tuple(elems) => elems.iter().any(holds_extension),
        Ty::Object(fields) => fields.values().any(holds_extension),
        Ty::Enum { variants, .. } => variants.values().flatten().any(|t| holds_extension(t)),
        _ => false,
    }
}

/// The hooks of an extension type, with the type's arguments. A
/// specialized slot (`#τ`, hash-types.md) has no space layout yet and is
/// refused here.
pub fn extension<'a>(
    rt: &'a AcvusRuntime,
    ty: &'a Ty,
) -> SpaceResult<(&'a SpaceHooks<AcvusRuntime>, Vec<Ty>)> {
    let Ty::UserDefined { id, type_args, .. } = ty else {
        return Err(SpaceError::new(format!(
            "{} is not an extension type",
            ty.display(&rt.shared.interner)
        )));
    };
    if type_args
        .iter()
        .any(|a| matches!(a, TypeArg::Specialized(_)))
    {
        return Err(SpaceError::new(format!(
            "{} has a specialized slot, which has no space layout",
            ty.display(&rt.shared.interner)
        )));
    }
    let args = type_args.iter().map(|a| a.ty().into_owned()).collect();
    Ok((hooks_of(rt, ty, id)?, args))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use acvus_extern::Holding;
    use acvus_mir::ty::{Home, ObjectTy};

    use super::*;
    use crate::executor::SequentialExecutor;
    use crate::interpreter::InterpreterContext;
    use crate::space::{Head, MemoryStore, Plain, Space, Store};

    fn runtime(i: &Interner) -> AcvusRuntime {
        InterpreterContext::new(i, FxHashMap::default(), Arc::new(SequentialExecutor))
            .runtime_over_an_empty_page()
    }

    /// The layout of a program with no extension types: a nested head is
    /// never reached by these values, and is refused if it is.
    struct NoExtensions;

    impl Nested for NoExtensions {
        fn commit(&self, _: &AcvusRuntime, _: &Ty, _: &Value) -> SpaceResult<NodeHash> {
            Err(SpaceError::new("no extension types"))
        }

        fn load(&self, _: &AcvusRuntime, _: &Ty, _: NodeHash, _: &ZeroWidth) -> SpaceResult<Value> {
            Err(SpaceError::new("no extension types"))
        }
    }

    /// The allowance the space's own loads use.
    fn allowance() -> ZeroWidth {
        ZeroWidth::allowing(Space::ZERO_WIDTH_ELEMENTS)
    }

    fn whole(rt: &AcvusRuntime, ty: &Ty, bytes: &[u8]) -> SpaceResult<Value> {
        decode_all(rt, &NoExtensions, ty, bytes, &allowance())
    }

    fn laid(rt: &AcvusRuntime, ty: &Ty, value: &Value) -> Vec<u8> {
        let mut out = Vec::new();
        encode(rt, &NoExtensions, ty, value, &mut out).expect("the value encodes");
        out
    }

    fn own(value: Value) -> Owned<AcvusRuntime> {
        // SAFETY: the word was made for this holder and moved in; no other
        // holder owns it.
        unsafe { Owned::from_value(Holding::new(), value) }
    }

    fn refused(rt: &AcvusRuntime, ty: &Ty, bytes: &[u8], says: &str) {
        match whole(rt, ty, bytes) {
            Ok(_) => panic!("{} decoded from {bytes:?}", ty.display(&rt.shared.interner)),
            Err(e) => assert!(e.0.contains(says), "{:?} does not say {says:?}", e.0),
        }
    }

    /// A settled array type states its length, so an array's count is its
    /// type's; a count that reaches the zero-width bound or the remaining
    /// bytes is one the type states too, as a head's recorded type may.
    fn array(elem: Ty, n: usize) -> Ty {
        Ty::Array(Box::new(elem), LenTerm::Known(n))
    }

    fn unit_array(n: usize) -> Ty {
        array(Ty::Unit, n)
    }

    fn count(n: u64) -> Vec<u8> {
        n.to_le_bytes().to_vec()
    }

    fn with(mut bytes: Vec<u8>, more: &[u8]) -> Vec<u8> {
        bytes.extend_from_slice(more);
        bytes
    }

    #[test]
    fn a_bool_is_the_byte_0_or_1() {
        let i = Interner::new();
        let rt = runtime(&i);
        assert_eq!(whole(&rt, &Ty::Bool, &[0]).ok(), Some(Value::bool_(false)));
        assert_eq!(whole(&rt, &Ty::Bool, &[1]).ok(), Some(Value::bool_(true)));
        for byte in [2u8, 0x80, 0xff] {
            refused(&rt, &Ty::Bool, &[byte], &format!("bool: byte {byte}"));
        }
    }

    /// A count of zero-width elements takes from the decode's allowance
    /// before any element is read, so `u64::MAX` of them is refused at
    /// once, whatever the zero-width type.
    #[test]
    fn a_zero_width_count_above_the_allowance_is_refused_at_once() {
        let i = Interner::new();
        let rt = runtime(&i);
        let zero_width = [
            Ty::Unit,
            Ty::Object(ObjectTy::written(FxHashMap::default())),
            Ty::Tuple(vec![Ty::Unit, Ty::Tuple(vec![])]),
        ];
        for elem in zero_width {
            assert_eq!(least_width(&elem), 0);
            let ty = array(elem, usize::MAX);
            let started = Instant::now();
            refused(&rt, &ty, &count(u64::MAX), "zero-width elements");
            assert!(
                started.elapsed() < Duration::from_secs(1),
                "refused after {:?}",
                started.elapsed()
            );
        }
    }

    /// The allowance is the caller's: a count up to it decodes, one past it
    /// is refused.
    #[test]
    fn a_zero_width_count_is_bounded_by_the_allowance_given() {
        let i = Interner::new();
        let rt = runtime(&i);
        let at = |bound, n: usize| {
            decode_all(&rt, &NoExtensions, &unit_array(n), &count(n as u64), &ZeroWidth::allowing(bound))
        };
        let four = at(4, 4).expect("four units within four");
        assert_eq!(unsafe { four.as_array() }.len(), 4);
        drop(own(four));
        assert!(at(4, 5).is_err_and(|e| e.0.contains("zero-width elements")));
        assert!(at(0, 0).is_ok());
    }

    /// Every zero-width count of one decode takes from one allowance, so
    /// nesting does not multiply it.
    #[test]
    fn nested_zero_width_counts_share_one_allowance() {
        let i = Interner::new();
        let rt = runtime(&i);
        let ty = array(unit_array(3), 2);
        let bytes = [count(2), count(3), count(3)].concat();
        let decoded = |bound| decode_all(&rt, &NoExtensions, &ty, &bytes, &ZeroWidth::allowing(bound));
        drop(own(decoded(6).expect("six units within six")));
        assert!(decoded(5).is_err_and(|e| e.0.contains("zero-width elements")));
    }

    /// A count of elements that lay bytes is at most what the remaining
    /// bytes can hold, refused before any element is read.
    #[test]
    fn a_count_above_what_the_bytes_can_hold_is_refused_at_once() {
        let i = Interner::new();
        let rt = runtime(&i);
        let started = Instant::now();
        for elem in [Ty::Bool, Ty::I64, Ty::String, unit_array(0), Ty::Option(Box::new(Ty::Unit))] {
            let ty = array(elem, usize::MAX);
            refused(&rt, &ty, &with(count(u64::MAX), &[0; 64]), "elements of");
        }
        // two bools need two bytes
        refused(&rt, &array(Ty::Bool, 2), &with(count(2), &[1]), "elements of");
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn a_byte_left_after_a_value_is_refused() {
        let i = Interner::new();
        let rt = runtime(&i);
        let ty = Ty::Tuple(vec![Ty::Bool, Ty::Int(IntTy::U16)]);
        let bytes = [1, 7, 0];
        drop(own(whole(&rt, &ty, &bytes).expect("the value alone")));
        refused(&rt, &ty, &with(bytes.to_vec(), &[0]), "1 bytes left unread");
        refused(&rt, &Ty::Unit, &[0], "1 bytes left unread");
        // `decode` reads a part and leaves the rest for its caller.
        let mut input = &with(bytes.to_vec(), &[9])[..];
        drop(own(decode(&rt, &NoExtensions, &ty, &mut input, &allowance()).expect("a part")));
        assert_eq!(input, [9]);
    }

    /// The refusals that were already there, each pinned.
    #[test]
    fn each_malformed_layout_is_refused() {
        let i = Interner::new();
        let rt = runtime(&i);
        // a `char` is a Unicode scalar value
        for code in [0xD800u32, 0xDFFF, 0x11_0000, u32::MAX] {
            refused(&rt, &Ty::Char, &code.to_le_bytes(), "is not a Unicode scalar value");
        }
        // a string is valid UTF-8, and its length fits the bytes
        refused(&rt, &Ty::String, &with(count(2), &[0xC3, 0x28]), "String:");
        refused(&rt, &Ty::String, &with(count(1), &[0x80]), "String:");
        refused(&rt, &Ty::String, &with(count(5), b"abc"), "truncated");
        refused(&rt, &Ty::String, &with(count(u64::MAX), b"abc"), "");
        // an option's and a result's tag is 0 or 1
        refused(&rt, &Ty::Option(Box::new(Ty::Unit)), &[2], "Option: tag 2");
        refused(&rt, &Ty::Result(Box::new(Ty::Unit), Box::new(Ty::Unit)), &[2], "Result: tag 2");
        // an enum's index is one of its variants
        let enumeration = Ty::Enum {
            name: QualifiedRef::root(i.intern("E")),
            variants: [(i.intern("A"), None), (i.intern("B"), Some(Box::new(Ty::Bool)))]
                .into_iter()
                .collect(),
            home: Home::NONE,
        };
        refused(&rt, &enumeration, &count(2), "variant index out of its enum type");
        refused(&rt, &enumeration, &count(u64::MAX), "");
        refused(&rt, &enumeration, &with(count(1), &[2]), "bool: byte 2");
        // an array of a known length holds that many
        let pair = Ty::Array(Box::new(Ty::Bool), LenTerm::Known(2));
        refused(&rt, &pair, &with(count(3), &[0, 0, 0]), "expected 2 elements");
        // a type no space holds is refused, not read
        refused(&rt, &Ty::Order, &[], "is not held by a space");
        refused(&rt, &array(Ty::Order, 1), &count(1), "is not held by a space");
        // an extension value's head names a node, 32 bytes
        let ext = Ty::UserDefined {
            id: QualifiedRef::root(i.intern("X")),
            type_args: vec![],
            effect_args: vec![],
            identity_args: vec![],
            region_params: 0,
        };
        refused(&rt, &ext, &[0; 31], "truncated");
    }

    /// Values of each shape, with their types.
    fn samples(i: &Interner) -> Vec<(Ty, Value)> {
        let int = |k: IntTy, n: i128| (Ty::Int(k), Value::from_bits(k, n as u64));
        let mut out = vec![
            int(IntTy::I8, -128),
            int(IntTy::I16, -2),
            int(IntTy::I32, i32::MIN as i128),
            int(IntTy::I64, i64::MAX as i128),
            int(IntTy::U8, 255),
            int(IntTy::U16, 0xBEEF),
            int(IntTy::U32, u32::MAX as i128),
            int(IntTy::U64, u64::MAX as i128),
            (Ty::Float, Value::float(-0.0)),
            (Ty::Float, Value::float(f64::from_bits(0x7FF8_0000_DEAD_BEEF))),
            (Ty::Float, Value::float(1.5e300)),
            (Ty::Char, Value::char_('\u{10FFFF}')),
            (Ty::Char, Value::char_('한')),
            (Ty::Bool, Value::bool_(true)),
            (Ty::Bool, Value::bool_(false)),
            (Ty::Unit, Value::unit()),
            (Ty::String, Value::string("")),
            (Ty::String, Value::string("acvus 한글 \u{0}")),
            (
                array(Ty::I64, 2),
                Value::array(vec![own(Value::int(3)), own(Value::int(-1))]),
            ),
            (unit_array(0), Value::array(Vec::new())),
            (unit_array(3), Value::array((0..3).map(|_| own(Value::unit())).collect())),
            (
                Ty::Tuple(vec![Ty::Bool, Ty::String]),
                Value::tuple(vec![own(Value::bool_(true)), own(Value::string("t"))]),
            ),
            (Ty::Option(Box::new(Ty::I64)), Value::NONE),
            (Ty::Option(Box::new(Ty::I64)), Value::some(Value::int(9))),
            (
                Ty::Result(Box::new(Ty::I64), Box::new(Ty::String)),
                Value::variant(i.intern("Err"), Some(own(Value::string("no")))),
            ),
        ];
        let object = Ty::Object(ObjectTy::written(
            [
                (i.intern("zeta"), Ty::Bool),
                (i.intern("alpha"), Ty::Array(Box::new(Ty::Char), LenTerm::Known(2))),
            ]
            .into_iter()
            .collect(),
        ));
        let object_value = || {
            Value::object_by_name(
                i,
                [
                    (i.intern("zeta"), own(Value::bool_(false))),
                    (
                        i.intern("alpha"),
                        own(Value::array(vec![own(Value::char_('a')), own(Value::char_('z'))])),
                    ),
                ],
            )
        };
        out.push((object.clone(), object_value()));
        let enumeration = Ty::Enum {
            name: QualifiedRef::root(i.intern("E")),
            variants: [
                (i.intern("Leaf"), None),
                (i.intern("Node"), Some(Box::new(object.clone()))),
            ]
            .into_iter()
            .collect(),
            home: Home::NONE,
        };
        out.push((enumeration.clone(), Value::variant(i.intern("Leaf"), None)));
        let nested = array(Ty::Option(Box::new(enumeration.clone())), 3);
        out.push((
            nested,
            Value::array(vec![
                own(Value::some(Value::variant(i.intern("Node"), Some(own(object_value()))))),
                own(Value::NONE),
                own(Value::some(Value::variant(i.intern("Leaf"), None))),
            ]),
        ));
        out.push((enumeration, Value::variant(i.intern("Node"), Some(own(object_value())))));
        out
    }

    /// A value decodes from its own encoding: an inline value to the same
    /// word, and any value to one whose encoding is the same bytes, which
    /// for a layout with no tags is the same value. Every strict prefix of
    /// the encoding is refused.
    #[test]
    fn a_value_comes_back_from_its_encoding_and_no_prefix_decodes() {
        let i = Interner::new();
        let rt = runtime(&i);
        for (ty, value) in samples(&i) {
            let shown = ty.display(&i).to_string();
            let value = own(value);
            let bytes = laid(&rt, &ty, &*value);
            let back = own(whole(&rt, &ty, &bytes).unwrap_or_else(|e| panic!("{shown}: {e}")));
            if back.kind().is_inline() {
                assert!(*back == *value, "{shown}: {:?} is not {:?}", *back, *value);
            }
            assert_eq!(laid(&rt, &ty, &back), bytes, "{shown}");
            for end in 0..bytes.len() {
                assert!(
                    whole(&rt, &ty, &bytes[..end]).is_err(),
                    "{shown}: the prefix {:?} of {bytes:?} decodes",
                    &bytes[..end]
                );
            }
        }
    }

    /// The space refuses a node whose stored bytes do not hash to the
    /// address they are stored at.
    #[test]
    fn a_node_that_does_not_hash_to_its_address_is_refused() {
        let i = Interner::new();
        let rt = runtime(&i);
        let store = MemoryStore::default();
        let address = NodeHash([7; NodeHash::LEN]);
        // a state node, no parent, one `true`
        store.put(address, &[0, 0, 1]).unwrap();
        let head = Head {
            hash: address,
            ty: Ty::Bool.to_ser(&i),
        };
        store.cmpxchg("b", None, head).unwrap().unwrap();
        let space = Space::over(Plain, Box::new(store));
        let refusal = space.load(&rt, "b", &Ty::Bool).expect_err("a node at another address");
        assert!(refusal.0.contains("hashes elsewhere"), "{}", refusal.0);
    }

    /// A length laid by a 64-bit writer is refused, not truncated, where this
    /// target's `usize` is narrower; on a 64-bit target every length fits.
    #[test]
    fn a_length_is_read_whole_or_refused() {
        let laid = u64::MAX.to_le_bytes();
        let read = take_len(&mut laid.as_slice(), "String length");
        match usize::BITS {
            64 => assert_eq!(read.ok(), Some(usize::MAX)),
            _ => assert!(read.is_err_and(|e| e.0.starts_with("String length:"))),
        }
        let laid = 7u64.to_le_bytes();
        assert_eq!(take_len(&mut laid.as_slice(), "String length").ok(), Some(7));
        assert!(take_len(&mut [0u8; 4].as_slice(), "String length").is_err(), "four bytes are no length");
    }
}
