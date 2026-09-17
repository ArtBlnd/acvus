//! A value as canonical bytes by its type (RFC-0033): no tags, the type is
//! the schema. The language's shapes are laid out here; an extension type
//! goes through the hooks its registry declared, and a nested extension
//! value is written as the head of its own log, which the space supplies.

use acvus_extern::{NodeHash, SpaceError, SpaceHooks, SpaceResult};
use acvus_mir::graph::QualifiedRef;
use acvus_mir::ty::{LenTerm, Repr, Ty};
use acvus_utils::Astr;
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
    rt.0.space.get(id).ok_or_else(|| {
        SpaceError::new(format!(
            "{} declares no space hooks",
            ty.display(&rt.0.interner)
        ))
    })
}

fn not_held(rt: &AcvusRuntime, ty: &Ty) -> SpaceError {
    SpaceError::new(format!(
        "a value of type {} is not held by a space",
        ty.display(&rt.0.interner)
    ))
}

fn sorted_fields<'a>(
    rt: &AcvusRuntime,
    fields: &'a FxHashMap<Astr, Ty>,
) -> Vec<(&'a Astr, &'a Ty)> {
    let mut out: Vec<_> = fields.iter().collect();
    out.sort_by_key(|(k, _)| rt.0.interner.resolve(**k).to_string());
    out
}

fn sorted_variants<'a>(
    rt: &AcvusRuntime,
    variants: &'a FxHashMap<Astr, Option<Box<Ty>>>,
) -> Vec<(&'a Astr, &'a Option<Box<Ty>>)> {
    let mut out: Vec<_> = variants.iter().collect();
    out.sort_by_key(|(k, _)| rt.0.interner.resolve(**k).to_string());
    out
}

pub fn encode(
    rt: &AcvusRuntime,
    nested: &dyn Nested,
    ty: &Ty,
    value: &Value,
    out: &mut Vec<u8>,
) -> SpaceResult<()> {
    match ty {
        Ty::Int(k) => out.extend_from_slice(&value.bits().to_le_bytes()[..k.bytes()]),
        Ty::Float => out.extend_from_slice(&value.as_float().to_bits().to_le_bytes()),
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
            let values = unsafe { value.as_object() };
            for (k, t) in sorted_fields(rt, fields) {
                let v = values.get(k).ok_or_else(|| {
                    SpaceError::new(format!(
                        "object lacks field `{}`",
                        rt.0.interner.resolve(*k)
                    ))
                })?;
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
        Ty::Result(ok, err) => match unsafe { value.as_result() } {
            Ok(v) => {
                out.push(0);
                encode(rt, nested, ok, v, out)?;
            }
            Err(e) => {
                out.push(1);
                encode(rt, nested, err, e, out)?;
            }
        },
        Ty::Enum { variants, .. } => {
            let variant = unsafe { value.as_variant() };
            let sorted = sorted_variants(rt, variants);
            let index = sorted
                .iter()
                .position(|(k, _)| **k == variant.tag)
                .ok_or_else(|| SpaceError::new("variant not in its enum type"))?;
            out.extend_from_slice(&(index as u64).to_le_bytes());
            if let (Some(payload), Some(t)) = (&variant.payload, sorted[index].1) {
                encode(rt, nested, t, payload, out)?;
            }
        }
        Ty::UserDefined { .. } => {
            let head = nested.commit(rt, ty, value)?;
            out.extend_from_slice(&head.0);
        }
        Ty::Fn { .. } | Ty::Handle(_) | Ty::Ref(..) | Ty::Order | Ty::Never | Ty::Error(_) => {
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
        Ty::Bool => Value::bool_(take(input, 1)?[0] != 0),
        Ty::Unit => Value::unit(),
        Ty::String => {
            let len = take_u64(input)? as usize;
            let bytes = take(input, len)?;
            Value::string(
                std::str::from_utf8(bytes).map_err(|e| SpaceError::new(format!("String: {e}")))?,
            )
        }
        Ty::Array(elem, len) => {
            let count = take_u64(input)? as usize;
            if let LenTerm::Known(n) = len
                && *n != count
            {
                return Err(SpaceError::new(format!(
                    "expected {n} elements for {}, got {count}",
                    ty.display(&rt.0.interner)
                )));
            }
            Value::array(
                (0..count)
                    .map(|_| decode(rt, nested, elem, input))
                    .collect::<SpaceResult<_>>()?,
            )
        }
        Ty::Tuple(elems) => Value::tuple(
            elems
                .iter()
                .map(|t| decode(rt, nested, t, input))
                .collect::<SpaceResult<_>>()?,
        ),
        Ty::Object(fields) => {
            let mut values = FxHashMap::default();
            for (k, t) in sorted_fields(rt, fields) {
                values.insert(*k, decode(rt, nested, t, input)?);
            }
            Value::object(values)
        }
        Ty::Option(inner) => match take(input, 1)?[0] {
            0 => Value::NONE,
            1 => Value::some(decode(rt, nested, inner, input)?),
            other => return Err(SpaceError::new(format!("Option: tag {other}"))),
        },
        Ty::Result(ok, err) => Value::result(match take(input, 1)?[0] {
            0 => Ok(decode(rt, nested, ok, input)?),
            1 => Err(decode(rt, nested, err, input)?),
            other => return Err(SpaceError::new(format!("Result: tag {other}"))),
        }),
        Ty::Enum { variants, .. } => {
            let sorted = sorted_variants(rt, variants);
            let index = take_u64(input)? as usize;
            let (tag, payload_ty) = sorted
                .get(index)
                .ok_or_else(|| SpaceError::new("variant index out of its enum type"))?;
            let payload = match payload_ty {
                Some(t) => Some(decode(rt, nested, t, input)?),
                None => None,
            };
            Value::variant(**tag, payload)
        }
        Ty::UserDefined { .. } => {
            let bytes: [u8; NodeHash::LEN] =
                take(input, NodeHash::LEN)?.try_into().expect("32 bytes");
            nested.load(rt, ty, NodeHash(bytes))?
        }
        Ty::Fn { .. } | Ty::Handle(_) | Ty::Ref(..) | Ty::Order | Ty::Never | Ty::Error(_) => {
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
            ty.display(&rt.0.interner)
        )));
    };
    if type_args.iter().any(|a| a.repr == Repr::Specialized) {
        return Err(SpaceError::new(format!(
            "{} has a specialized slot, which has no space layout",
            ty.display(&rt.0.interner)
        )));
    }
    let args = type_args.iter().map(|a| a.ty.clone()).collect();
    Ok((hooks_of(rt, ty, id)?, args))
}
