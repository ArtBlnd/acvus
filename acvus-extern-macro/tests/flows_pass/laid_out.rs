//! Whether `#[extern_fn]` reads a generic type position by position is the
//! type's own statement (RFC-0096 rule 2), never how its name is written.
use std::marker::PhantomData;

use acvus_extern::{
    Alignment, Ctx, ExternFn, ExternType, Flow, FlowEnd, Flows, InPlaceElement, Interner, Laid,
    Labelled, Payload, PolyTy, Ref, Runtime, Shared, Take, TransparentOver, TypesOnly, Var,
    extern_fn, kind,
};

mod shadow {
    use super::*;

    #[derive(Payload)]
    pub struct Body<'a, C, Rt>
    where
        C: Var<kind::Type>,
        Rt: Runtime,
    {
        pub items: Ref<'a, C, Shared, Rt>,
    }

    /// Named as the language's own map is, which the macro once read as
    /// unread by that name; it is derived, so it states it is laid out.
    #[derive(ExternType)]
    #[extern_type(name = "Cursor")]
    #[repr(transparent)]
    pub struct HashMap<'a, C, I, Rt>(pub Body<'a, C, Rt>, PhantomData<I>)
    where
        C: Var<kind::Type>,
        I: Var<kind::Identity>,
        Rt: Runtime;
}

type Cursor<'a, C, I, Rt> = shadow::HashMap<'a, C, I, Rt>;

#[extern_fn(effect = pure)]
fn peek<'a, T, I, Rt>(_ctx: &mut Ctx<'_, Rt>, _it: &mut shadow::HashMap<'a, Vec<T>, I, Rt>) -> Option<&'a T>
where
    T: Var<kind::Type> + TransparentOver<Rt> + InPlaceElement<Rt>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    None
}

#[extern_fn(effect = pure)]
fn peek_aliased<'a, T, I, Rt>(_ctx: &mut Ctx<'_, Rt>, _it: &mut Cursor<'a, Vec<T>, I, Rt>) -> Option<&'a T>
where
    T: Var<kind::Type> + TransparentOver<Rt> + InPlaceElement<Rt>,
    I: Var<kind::Identity>,
    Rt: Runtime,
{
    None
}

#[extern_fn(effect = pure)]
fn pop<T>(c: &mut Vec<T>) -> Option<T>
where
    T: Var<kind::Type>,
{
    c.pop()
}

fn flows_of(declared: Vec<ExternFn<TypesOnly>>) -> Flows {
    let declared = declared.first().expect("#[extern_fn] declares one function");
    match &declared.decl.ty {
        PolyTy::Fn { flows, .. } => flows.get().clone(),
        other => panic!("a declaration's type is a function type, not {other:?}"),
    }
}

fn main() {
    let i = Interner::new();
    let element_from_the_collection = Flows::of([
        Flow {
            to: FlowEnd::Result,
            from: FlowEnd::Param(0),
            alignment: Alignment::Labelled(Box::new(Labelled {
                to: Laid::Option(Box::new(Laid::Ref(Box::new(Laid::Var)))),
                from: Laid::Ref(Box::new(Laid::User {
                    regions: 1,
                    args: vec![Laid::Unread],
                })),
                takes: vec![
                    vec![Take {
                        segment: 1,
                        aligned: false,
                    }],
                    vec![Take {
                        segment: 2,
                        aligned: false,
                    }],
                ],
            })),
        },
        Flow {
            to: FlowEnd::Param(0),
            from: FlowEnd::Param(0),
            alignment: Alignment::Any,
        },
    ]);
    assert_eq!(flows_of(__extern_fn_peek::<TypesOnly>(&i, None)), element_from_the_collection);
    assert_eq!(
        flows_of(__extern_fn_peek_aliased::<TypesOnly>(&i, None)),
        element_from_the_collection
    );
    assert_eq!(
        flows_of(__extern_fn_pop::<TypesOnly>(&i, None)),
        Flows::of([
            Flow {
                to: FlowEnd::Result,
                from: FlowEnd::Param(0),
                alignment: Alignment::Any,
            },
            Flow {
                to: FlowEnd::Param(0),
                from: FlowEnd::Param(0),
                alignment: Alignment::Any,
            },
        ])
    );
}
