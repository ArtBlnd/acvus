//! A bound `$` is a constant (RFC-0071 rule 5).
//!
//! Obligation across passes: the type written here is the type
//! `graph::types::bound_input_ty` gave the checker for the same name, and the
//! parameter this drops is one `graph::optimize` will no longer report as an
//! input the host must supply.

use acvus_ast::{Literal, Span};
use acvus_utils::Interner;

use crate::error::{MirError, MirErrorKind};
use crate::ir::{Inst, InstKind, MirBody, RefTarget, ValueId};
use crate::ty::{Mutability, Ty, TypeArg};

use super::types::Bindings;

pub fn substitute(interner: &Interner, body: &mut MirBody, bindings: &Bindings) -> Vec<MirError> {
    let mut errors = Vec::new();
    for (name, value) in bindings.iter() {
        let Some(at) = body.params.iter().position(|(held, _)| *held == name) else {
            continue;
        };
        let (_, slot) = body.params[at];
        let ty = body
            .val_types
            .get(&slot)
            .cloned()
            .expect("lowering gives every parameter of a body its type");
        match constant(body, &ty, value) {
            Ok(written) => {
                body.params.remove(at);
                read_as_local(body, slot);
                let assign = Inst {
                    span: Span::ZERO,
                    kind: InstKind::Assign {
                        target: RefTarget::Var(slot),
                        path: Vec::new(),
                        value: written.value,
                    },
                };
                body.insts.splice(
                    0..0,
                    written.insts.into_iter().chain(std::iter::once(assign)),
                );
            }
            Err(Mismatch) => errors.push(MirError {
                kind: MirErrorKind::BindingTypeMismatch {
                    name: interner.resolve(name).to_string(),
                    value: value.clone(),
                    ty,
                },
                span: Span::ZERO,
                labels: Vec::new(),
            }),
        }
    }
    errors
}

struct Written {
    value: ValueId,
    insts: Vec<Inst>,
}

struct Mismatch;

fn constant(body: &mut MirBody, ty: &Ty, value: &Literal) -> Result<Written, Mismatch> {
    let dst = body.val_factory.next();
    body.val_types.insert(dst, ty.clone());
    let const_str = |dst: ValueId, text: String| Inst {
        span: Span::ZERO,
        kind: InstKind::ConstStr { dst, text },
    };
    let scalar = |value: Literal| Inst {
        span: Span::ZERO,
        kind: InstKind::Const { dst, value },
    };
    let written = match (ty, value.desugared()) {
        (Ty::Int(_), held @ Literal::Int(_))
        | (Ty::Float, held @ Literal::Float(_))
        | (Ty::Bool, held @ Literal::Bool(_))
        | (Ty::Char, held @ Literal::Char(_)) => vec![scalar(held)],
        (Ty::Str, Literal::String(held)) => vec![const_str(dst, held)],
        (Ty::Ref(_, inner), Literal::String(held)) if matches!(inner.ty, Ty::Str | Ty::String) => {
            vec![const_str(dst, held)]
        }
        (Ty::String, Literal::String(held)) => {
            let borrowed = body.val_factory.next();
            body.val_types.insert(
                borrowed,
                Ty::Ref(Mutability::Shared, Box::new(TypeArg::uniform(Ty::Str))),
            );
            vec![
                const_str(borrowed, held),
                Inst {
                    span: Span::ZERO,
                    kind: InstKind::StringClone { dst, src: borrowed },
                },
            ]
        }
        _ => return Err(Mismatch),
    };
    Ok(Written {
        value: dst,
        insts: written,
    })
}

fn read_as_local(body: &mut MirBody, slot: ValueId) {
    let targets = body
        .insts
        .iter_mut()
        .filter_map(|inst| match &mut inst.kind {
            InstKind::Ref { target, .. }
            | InstKind::Take { target, .. }
            | InstKind::Assign { target, .. } => Some(target),
            _ => None,
        });
    for target in targets {
        if let RefTarget::Param(held) = target
            && *held == slot
        {
            *target = RefTarget::Var(slot);
        }
    }
}
