//! `#[extern_fn(ensures(..))]` (RFC-0082 rules 4 and 5): the registry
//! carries each postcondition beside the laws, and a debug build checks it
//! where the function returns.

use acvus_extern::{
    Interner, PostTerm, Postcondition, Relation, Subject, TypesOnly, Var, extern_fn, kind,
};

#[extern_fn(effect = pure, ensures(ret < n))]
fn below(n: u64) -> u64 {
    n - 1
}

#[extern_fn(effect = pure, ensures(ret < n))]
fn not_below(n: u64) -> u64 {
    n
}

#[extern_fn(effect = pure, ensures(ret = len(v)))]
fn counted<T>(v: &Vec<T>) -> u64
where
    T: Var<kind::Type>,
{
    v.len() as u64
}

#[extern_fn(effect = pure, ensures(ret = len(v)))]
fn miscounted<T>(v: &Vec<T>) -> u64
where
    T: Var<kind::Type>,
{
    if v.is_empty() {
        return 1;
    }
    v.len() as u64
}

#[extern_fn(effect = pure, ensures(len(v) = old(len(v)) + 1))]
fn grown<T>(v: &mut Vec<T>, item: T)
where
    T: Var<kind::Type>,
{
    v.push(item);
}

#[extern_fn(effect = pure, ensures(len(v) = old(len(v)) + 1))]
fn ungrown<T>(v: &mut Vec<T>, item: T)
where
    T: Var<kind::Type>,
{
    let _ = v.len();
    drop(item);
}

#[test]
fn the_registry_carries_the_postcondition_beside_the_laws() {
    let interner = Interner::new();
    let declared = __extern_fn_counted::<TypesOnly>(&interner, None);
    let [counted] = &declared[..] else {
        panic!("one declaration, found {}", declared.len());
    };
    assert_eq!(
        counted.decl.ensures,
        vec![Postcondition {
            left: PostTerm::Ret,
            relation: Relation::Eq,
            right: PostTerm::Len(Subject::Param(0)),
        }]
    );
}

#[test]
fn a_postcondition_that_holds_returns_the_result() {
    assert_eq!(below(3), 2);
    assert_eq!(counted::<i64>(&vec![1, 2, 3]), 3);
}

#[test]
#[should_panic(expected = "`not_below` broke its postcondition `ret < n`: the left side is 3 and \
                           the right side is 3")]
fn a_broken_postcondition_panics_at_the_return() {
    not_below(3);
}

#[test]
#[should_panic(expected = "`miscounted` broke its postcondition `ret = len(v)`: the left side is \
                           1 and the right side is 0")]
fn a_return_statement_is_checked_as_the_tail_is() {
    miscounted::<i64>(&Vec::new());
}

#[test]
fn the_registry_carries_a_term_as_it_stood_when_the_call_began() {
    let interner = Interner::new();
    let declared = __extern_fn_grown::<TypesOnly>(&interner, None);
    let [grown] = &declared[..] else {
        panic!("one declaration, found {}", declared.len());
    };
    assert_eq!(
        grown.decl.ensures,
        vec![Postcondition {
            left: PostTerm::Len(Subject::Param(0)),
            relation: Relation::Eq,
            right: PostTerm::Add(
                Box::new(PostTerm::Old(Box::new(PostTerm::Len(Subject::Param(0))))),
                Box::new(PostTerm::Const(1)),
            ),
        }]
    );
}

/// `old(len(v))` is the length the call began with, not the one it
/// returns with: a push onto every length keeps `len(v) = old(len(v)) + 1`.
#[test]
fn a_postcondition_over_the_state_before_the_call_holds_where_it_is_true() {
    let mut v: Vec<i64> = Vec::new();
    for at in 0..16 {
        grown(&mut v, at);
    }
    assert_eq!(v.len(), 16);
}

#[test]
#[should_panic(expected = "`ungrown` broke its postcondition `len(v) = (old(len(v)) + 1)`: the \
                           left side is 0 and the right side is 1")]
fn a_postcondition_over_the_state_before_the_call_is_read_before_the_body_runs() {
    ungrown::<i64>(&mut Vec::new(), 7);
}
