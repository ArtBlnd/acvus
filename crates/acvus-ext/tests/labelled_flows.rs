use acvus_ext::std_registries;
use acvus_extern::{Alignment, Externs, FnKind, FlowEnd, Flows, Laid, Labelled, PolyTy, Take};
use acvus_interpreter::AcvusRuntime;
use acvus_utils::Interner;

fn declared_types(i: &Interner) -> Vec<(String, PolyTy)> {
    let externs = Externs::<AcvusRuntime>::combine(std_registries::<AcvusRuntime>(), i)
        .expect("the std registries combine");
    let mut found = Vec::new();
    for f in externs.functions {
        let name = i.resolve(f.qref.name).to_string();
        if let FnKind::Extern { instances, .. } = &f.kind {
            found.extend(instances.concrete.iter().map(|inst| (name.clone(), inst.ty.clone())));
        }
        found.push((name, f.ty));
    }
    found
}

fn flows_of(ty: &PolyTy) -> Option<&Flows> {
    match ty {
        PolyTy::Fn { flows, .. } => Some(flows.get()),
        _ => None,
    }
}

fn mut_receiver_type_name(i: &Interner, ty: &PolyTy) -> Option<String> {
    let PolyTy::Fn { params, .. } = ty else {
        return None;
    };
    let PolyTy::Ref(acvus_extern::Mutability::Mut, pointee) = &params.first()?.ty else {
        return None;
    };
    match pointee.ty().as_ref() {
        PolyTy::UserDefined { id, .. } => Some(i.resolve(id.name).to_string()),
        _ => None,
    }
}

#[test]
fn a_borrowing_next_takes_its_element_from_the_collection_and_not_the_iterator() {
    let i = Interner::new();
    let borrowing: Vec<(String, PolyTy)> = declared_types(&i)
        .into_iter()
        .filter(|(name, ty)| {
            name == "next"
                && mut_receiver_type_name(&i, ty).is_some_and(|r| ["Refs", "Keys", "Values"].contains(&r.as_str()))
        })
        .collect();
    assert_eq!(
        borrowing.len(),
        6,
        "vec, array, deque and set over `Refs`, and a map's `Keys` and `Values`"
    );
    for (_, ty) in &borrowing {
        let flows = flows_of(ty).expect("a declaration's type is a function type");
        let into_result = flows.into_end(FlowEnd::Result, 1);
        let [source] = into_result.as_slice() else {
            panic!("one flow into the result: {flows:?}");
        };
        let Alignment::Labelled(map) = &source.alignment else {
            panic!("the result's flow is labelled: {flows:?}");
        };
        let Labelled { to, from, takes } = map.as_ref();
        assert!(matches!(to, Laid::Option(inner) if matches!(inner.as_ref(), Laid::Ref(_))));
        let Laid::Ref(receiver) = from else {
            panic!("the input is the iterator's `&mut`: {from:?}");
        };
        assert!(matches!(receiver.as_ref(), Laid::User { regions: 1, .. }));
        assert_eq!(
            takes[0],
            [Take {
                segment: 1,
                aligned: false
            }],
            "the element's reference takes the iterator's region parameter, its loan on the collection"
        );
        assert!(
            takes.iter().flatten().all(|take| take.segment != 0),
            "no segment of the result takes the `&mut` of the iterator: {takes:?}"
        );
    }
}

#[test]
fn every_labelled_flow_lays_out_its_declared_ends() {
    let i = Interner::new();
    let mut labelled = 0;
    for (name, ty) in declared_types(&i) {
        let PolyTy::Fn { params, ret, flows, .. } = &ty else {
            continue;
        };
        let Flows::Listed(listed) = flows.get() else {
            continue;
        };
        for flow in listed {
            let Alignment::Labelled(map) = &flow.alignment else {
                continue;
            };
            labelled += 1;
            let to_ty = match flow.to {
                FlowEnd::Result => ret.as_ref(),
                other => panic!("{name}: a labelled flow is into the result, not {other:?}"),
            };
            let from_ty = match flow.from {
                FlowEnd::Param(index) => &params[index].ty,
                other => panic!("{name}: a labelled flow is from a parameter, not {other:?}"),
            };
            assert!(map.to.fits(to_ty), "{name}: {:?} is the result's shape", map.to);
            assert!(map.from.fits(from_ty), "{name}: {:?} is {:?}'s shape", map.from, flow.from);
            assert_eq!(map.takes.len(), map.to.segment_count(), "{name}: one entry per output segment");
            let inputs = map.from.segment_count();
            assert!(
                map.takes.iter().flatten().all(|take| take.segment < inputs),
                "{name}: every take names an input segment"
            );
        }
    }
    assert!(labelled > 0, "the std registries declare labelled flows");
}
