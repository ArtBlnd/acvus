//! A value as canonical bytes by its type (RFC-0033): no tags, the type is
//! the schema. The language's shapes are laid out here; an extension type
//! goes through the hooks its registry declared, and a nested extension
//! value is written as the head of its own log, which the space supplies.

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
    fn load(&self, rt: &AcvusRuntime, ty: &Ty, head: NodeHash) -> SpaceResult<Value>;
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

fn take_u64(input: &mut &[u8]) -> SpaceResult<u64> {
    let bytes: [u8; 8] = take(input, 8)?.try_into().expect("eight bytes");
    Ok(u64::from_le_bytes(bytes))
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
) -> SpaceResult<Owned<AcvusRuntime>> {
    let value = decode(rt, nested, ty, input)?;
    // SAFETY: `decode` made the word, and no other holder owns it.
    Ok(unsafe { Owned::from_value(acvus_extern::Holding::new(), value) })
}

pub fn decode(
    rt: &AcvusRuntime,
    nested: &dyn Nested,
    ty: &Ty,
    input: &mut &[u8],
) -> SpaceResult<Value> {
    Ok(match ty {
        Ty::Int(k) => {
            let mut word = [0u8; 8];
            word[..k.bytes()].copy_from_slice(take(input, k.bytes())?);
            Value::from_bits(*k, k.read(u64::from_le_bytes(word)) as u64)
        }
        Ty::Float => Value::float(f64::from_bits(take_u64(input)?)),
        Ty::Char => {
            let mut word = [0u8; 4];
            word.copy_from_slice(take(input, 4)?);
            let code = u32::from_le_bytes(word);
            let c = char::from_u32(code).ok_or_else(|| {
                SpaceError::new(format!("{code:#x} is not a Unicode scalar value"))
            })?;
            Value::char_(c)
        }
        Ty::Bool => Value::bool_(take(input, 1)?[0] != 0),
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
            Value::array(
                (0..count)
                    .map(|_| decode_owned(rt, nested, elem, input))
                    .collect::<SpaceResult<_>>()?,
            )
        }
        Ty::Tuple(elems) => Value::tuple(
            elems
                .iter()
                .map(|t| decode_owned(rt, nested, t, input))
                .collect::<SpaceResult<_>>()?,
        ),
        Ty::Object(fields) => {
            let laid = sorted_fields(&rt.shared.interner, fields);
            let shape = ObjectShape::in_order(laid.iter().map(|(name, _)| **name).collect());
            let values: Box<[Owned<AcvusRuntime>]> = laid
                .iter()
                .map(|(_, t)| decode_owned(rt, nested, t, input))
                .collect::<SpaceResult<_>>()?;
            Value::object(shape, values)
        }
        Ty::Option(inner) => match take(input, 1)?[0] {
            0 => Value::NONE,
            1 => Value::some(decode(rt, nested, inner, input)?),
            other => return Err(SpaceError::new(format!("Option: tag {other}"))),
        },
        Ty::Result(ok, err) => {
            let (tag, held) = match take(input, 1)?[0] {
                0 => ("Ok", ok),
                1 => ("Err", err),
                other => return Err(SpaceError::new(format!("Result: tag {other}"))),
            };
            let payload = decode_owned(rt, nested, held, input)?;
            Value::variant(rt.shared.interner.intern(tag), Some(payload))
        }
        Ty::Enum { variants, .. } => {
            let sorted = sorted_variants(&rt.shared.interner, variants);
            let index = take_len(input, "variant index")?;
            let (tag, payload_ty) = sorted
                .get(index)
                .ok_or_else(|| SpaceError::new("variant index out of its enum type"))?;
            let payload = match payload_ty {
                Some(t) => Some(decode_owned(rt, nested, t, input)?),
                None => None,
            };
            Value::variant(**tag, payload)
        }
        Ty::UserDefined { .. } => {
            let bytes: [u8; NodeHash::LEN] =
                take(input, NodeHash::LEN)?.try_into().expect("32 bytes");
            nested.load(rt, ty, NodeHash(bytes))?
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
    use super::*;

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
