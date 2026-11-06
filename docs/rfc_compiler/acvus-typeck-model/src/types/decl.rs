//! A program's declarations, by the name a call writes, and the tree its
//! declared casts form.

use std::collections::BTreeMap;

use crate::sig::{DeclaredTypes, SigArgs, SigTy, SigVar, Signature};
use crate::ty::{Constructor, HasTerm, Mutability, Name, Term, Ty};

/// One declaration among those of its name, by the order they are declared in.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct DeclId {
    pub name: Name,
    pub overload: usize,
}

/// What a declaration is besides a function a call reaches by its name: a cast
/// is a step up the cast tree, and a conversion is no part of it.
/// A view impl, an impl of `core::as_slice`, `core::as_slice_mut` or
/// `core::as_str`, is a cast. The language's view of an array, from a
/// reference to it to one to the slice of its element, is a cast the language
/// gives, from a type no declaration writes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Fn,
    Cast,
    ArrayView,
    Conversion,
}

/// The names of the language's views of an array, shared and exclusive.
pub const ARRAY_VIEW: &str = "core::array_view";
pub const ARRAY_VIEW_MUT: &str = "core::array_view_mut";

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Decl {
    pub id: DeclId,
    pub sig: Signature,
    pub role: Role,
}

/// A node of the cast tree: a constructor, and for a reference its mutability
/// and its target's constructor.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Node {
    Owned(Constructor),
    Ref(Mutability, Constructor),
}

impl Node {
    pub fn of(ty: &Ty) -> Node {
        Node::of_decided(ty).expect("every position of a closed type is decided")
    }

    /// The node of a type whose outermost term, and a reference's target there,
    /// are decided.
    pub fn of_decided<T>(ty: &T) -> Option<Node>
    where
        T: HasTerm,
    {
        match ty.decided()? {
            Term::Ref(mutability, target) => {
                Some(Node::Ref(*mutability, target.decided()?.constructor()))
            }
            term => Some(Node::Owned(term.constructor())),
        }
    }

    /// The node of a type a signature writes, where its outermost term, and a
    /// reference's target, are not variables.
    fn of_sig(ty: &SigTy) -> Option<Node> {
        let SigTy::Term(term) = ty else {
            return None;
        };
        match &**term {
            Term::Ref(mutability, SigTy::Term(target)) => {
                Some(Node::Ref(*mutability, target.constructor()))
            }
            Term::Ref(_, SigTy::Var(_)) => None,
            term => Some(Node::Owned(term.constructor())),
        }
    }

    fn is_extern_defined(&self) -> bool {
        matches!(
            self,
            Node::Owned(Constructor::Extern(_)) | Node::Ref(_, Constructor::Extern(_))
        )
    }

    fn mode(&self) -> Option<Mutability> {
        match self {
            Node::Owned(_) => None,
            Node::Ref(mutability, _) => Some(*mutability),
        }
    }
}

/// Why the declared casts are not a tree, or why a declaration is not
/// one of its role.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum TreeError {
    /// A cast takes one parameter.
    CastArity(DeclId),
    /// A cast goes from an extern-defined type or a reference to one.
    NotFromExtern(DeclId),
    /// A cast's result has no constructor of its own at its outermost term, or a
    /// reference's target there is a variable.
    NoTargetNode(DeclId),
    /// A cast changes whether it is a reference, or its mutability.
    ChangesMode(DeclId),
    /// A cast's parameter writes its node other than with distinct type
    /// parameters as its parts, so that some type of the node goes up by no
    /// cast.
    NarrowerThanNode(DeclId),
    /// A cast's result writes a type parameter its parameter does not, so
    /// that where a type goes up is not decided.
    ResultUnbound(DeclId),
    /// A second cast goes up from one node.
    TwoUp {
        first: DeclId,
        second: DeclId,
    },
    Cycle(DeclId),
    /// A cast goes from `&mut X` to `&mut Y`, and the cast going up from `&X`
    /// is absent or goes elsewhere than `&Y`.
    NotCommuting(DeclId),
    /// A conversion takes one parameter.
    ConversionArity(DeclId),
}

/// One step up the cast tree.
#[derive(Clone, PartialEq, Eq, Debug)]
struct Step {
    decl: DeclId,
    to: Node,
}

/// The declarations, as they are declared, before the cast tree is checked.
#[derive(Clone, Default, Debug)]
pub struct DeclsBuilder {
    types: DeclaredTypes,
    by_name: BTreeMap<Name, Vec<Decl>>,
}

/// The name of `e?`'s use, whose two candidates the language gives.
pub const TRY: &str = "?";

impl DeclsBuilder {
    /// The declarations of the program, with the language's two candidates of
    /// `e?`: `?(Option⟨T⟩) -> T` and `?(Result⟨T, E⟩) -> T`.
    pub fn new(types: DeclaredTypes) -> DeclsBuilder {
        let mut builder = DeclsBuilder {
            types,
            by_name: BTreeMap::new(),
        };
        let var = |i| SigTy::Var(SigVar::Type(i));
        let option = Signature::new(
            vec![Name::new("T")],
            false,
            vec![SigTy::term(Term::Option(var(0)))],
            var(0),
            &builder.types,
        )
        .expect("`?(Option⟨T⟩) -> T` is a signature");
        let result = Signature::new(
            vec![Name::new("T"), Name::new("E")],
            false,
            vec![SigTy::term(Term::Result {
                ok: var(0),
                err: var(1),
            })],
            var(0),
            &builder.types,
        )
        .expect("`?(Result⟨T, E⟩) -> T` is a signature");
        builder.declare(Name::new(TRY), option, Role::Fn);
        builder.declare(Name::new(TRY), result, Role::Fn);
        builder
    }

    pub fn types(&self) -> &DeclaredTypes {
        &self.types
    }

    /// The language's views of the arrays of length `len`, `&[T; len] -> &[T]`
    /// and `&mut [T; len] -> &mut [T]`, each a cast of the tree.
    pub fn array_views(&mut self, len: u64) {
        for (name, mutability) in [
            (ARRAY_VIEW, Mutability::Shared),
            (ARRAY_VIEW_MUT, Mutability::Exclusive),
        ] {
            let element = || SigTy::Var(SigVar::Type(0));
            let sig = Signature::new(
                vec![Name::new("T")],
                false,
                vec![SigTy::term(Term::Ref(
                    mutability,
                    SigTy::term(Term::Array(len, element())),
                ))],
                SigTy::term(Term::Ref(mutability, SigTy::term(Term::Slice(element())))),
                &self.types,
            )
            .expect("an array's view is a signature");
            self.declare(Name::new(name), sig, Role::ArrayView);
        }
    }

    pub fn declare(&mut self, name: Name, sig: Signature, role: Role) -> DeclId {
        let overloads = self.by_name.entry(name.clone()).or_default();
        let id = DeclId {
            name,
            overload: overloads.len(),
        };
        overloads.push(Decl {
            id: id.clone(),
            sig,
            role,
        });
        id
    }

    /// The declarations, once their casts form a tree.
    pub fn finish(self) -> Result<Decls, TreeError> {
        let mut up: BTreeMap<Node, Step> = BTreeMap::new();
        for decl in self.by_name.values().flatten() {
            match decl.role {
                Role::Fn => {}
                Role::Conversion => {
                    if decl.sig.params().len() != 1 {
                        return Err(TreeError::ConversionArity(decl.id.clone()));
                    }
                }
                Role::Cast | Role::ArrayView => {
                    let [param] = decl.sig.params() else {
                        return Err(TreeError::CastArity(decl.id.clone()));
                    };
                    let from = Node::of_sig(param)
                        .filter(|from| decl.role == Role::ArrayView || from.is_extern_defined())
                        .ok_or_else(|| TreeError::NotFromExtern(decl.id.clone()))?;
                    let to = Node::of_sig(decl.sig.result())
                        .ok_or_else(|| TreeError::NoTargetNode(decl.id.clone()))?;
                    if from.mode() != to.mode() {
                        return Err(TreeError::ChangesMode(decl.id.clone()));
                    }
                    if !writes_node_generally(param) {
                        return Err(TreeError::NarrowerThanNode(decl.id.clone()));
                    }
                    let unbound = [SigVar::Identity]
                        .into_iter()
                        .chain((0..decl.sig.type_params().len()).map(SigVar::Type))
                        .any(|var| decl.sig.result().mentions(var) && !param.mentions(var));
                    if unbound {
                        return Err(TreeError::ResultUnbound(decl.id.clone()));
                    }
                    if let Some(first) = up.get(&from) {
                        return Err(TreeError::TwoUp {
                            first: first.decl.clone(),
                            second: decl.id.clone(),
                        });
                    }
                    up.insert(
                        from,
                        Step {
                            decl: decl.id.clone(),
                            to,
                        },
                    );
                }
            }
        }
        let tree = CastTree { up };
        tree.check_acyclic()?;
        tree.check_commuting()?;
        Ok(Decls {
            types: self.types,
            by_name: self.by_name,
            tree,
        })
    }
}

/// Whether a cast's parameter writes its node, through a reference's target
/// where it is one, with distinct type parameters as its parts, its identity
/// the signature's where it has one.
fn writes_node_generally(param: &SigTy) -> bool {
    let SigTy::Term(term) = param else {
        return false;
    };
    let term = match &**term {
        Term::Ref(_, SigTy::Term(target)) => &**target,
        Term::Ref(..) => return false,
        term => term,
    };
    let mut seen = Vec::new();
    term.parts().into_iter().all(|(_, part)| match part {
        SigTy::Var(var) if !seen.contains(var) => {
            seen.push(*var);
            true
        }
        _ => false,
    })
}

/// The casts that go up from each node.
#[derive(Clone, Default, Debug)]
pub struct CastTree {
    up: BTreeMap<Node, Step>,
}

impl CastTree {
    fn check_acyclic(&self) -> Result<(), TreeError> {
        for (from, step) in &self.up {
            let mut node = step.to.clone();
            for _ in 0..=self.up.len() {
                if &node == from {
                    return Err(TreeError::Cycle(step.decl.clone()));
                }
                match self.up.get(&node) {
                    Some(next) => node = next.to.clone(),
                    None => break,
                }
            }
        }
        Ok(())
    }

    fn check_commuting(&self) -> Result<(), TreeError> {
        for (from, step) in &self.up {
            let (Node::Ref(Mutability::Exclusive, x), Node::Ref(Mutability::Exclusive, y)) =
                (from, &step.to)
            else {
                continue;
            };
            let shared_up = self.up.get(&Node::Ref(Mutability::Shared, x.clone()));
            if shared_up.map(|up| &up.to) != Some(&Node::Ref(Mutability::Shared, y.clone())) {
                return Err(TreeError::NotCommuting(step.decl.clone()));
            }
        }
        Ok(())
    }

    /// The node itself, then each node up the tree from it.
    pub fn ancestors(&self, node: &Node) -> Vec<Node> {
        let mut ancestors = vec![node.clone()];
        while let Some(step) = self
            .up
            .get(ancestors.last().expect("ancestors starts at the node"))
        {
            ancestors.push(step.to.clone());
        }
        ancestors
    }

    /// The nearest node that both nodes are, or are below.
    pub fn nca(&self, a: &Node, b: &Node) -> Option<Node> {
        let above_b = self.ancestors(b);
        self.ancestors(a)
            .into_iter()
            .find(|node| above_b.contains(node))
    }

    /// The cast that goes up from the node.
    pub fn up(&self, node: &Node) -> Option<&DeclId> {
        self.up.get(node).map(|step| &step.decl)
    }

    /// The casts from one node up to another, none where they are one node.
    pub fn path(&self, from: &Node, to: &Node) -> Option<Vec<DeclId>> {
        let mut path = Vec::new();
        let mut node = from.clone();
        while &node != to {
            let step = self.up.get(&node)?;
            path.push(step.decl.clone());
            node = step.to.clone();
        }
        Some(path)
    }
}

/// A program's declarations, their casts forming a tree.
#[derive(Clone, Debug)]
pub struct Decls {
    types: DeclaredTypes,
    by_name: BTreeMap<Name, Vec<Decl>>,
    tree: CastTree,
}

impl Decls {
    pub fn types(&self) -> &DeclaredTypes {
        &self.types
    }

    pub fn tree(&self) -> &CastTree {
        &self.tree
    }

    /// The declarations a name reaches.
    pub fn candidates(&self, name: &Name) -> &[Decl] {
        self.by_name.get(name).map_or(&[], Vec::as_slice)
    }

    pub fn decl(&self, id: &DeclId) -> &Decl {
        &self.by_name[&id.name][id.overload]
    }

    pub fn conversions(&self) -> impl Iterator<Item = &Decl> {
        self.by_name
            .values()
            .flatten()
            .filter(|decl| decl.role == Role::Conversion)
    }

    /// `ty` one step up the tree, by the cast that goes up from its node, where
    /// that cast's parameter matches it.
    pub fn step_up<T>(&self, ty: &T) -> Option<(DeclId, T)>
    where
        T: HasTerm,
    {
        let id = self.tree.up(&Node::of_decided(ty)?)?;
        let applied = apply(&self.decl(id).sig, ty)?;
        Some((id.clone(), applied))
    }

    /// `ty` raised up the tree to `node`, with each cast taken.
    pub fn raise<T>(&self, ty: &T, node: &Node) -> Option<(Vec<DeclId>, T)>
    where
        T: HasTerm,
    {
        let mut casts = Vec::new();
        let mut ty = ty.clone();
        while Node::of_decided(&ty).as_ref() != Some(node) {
            let (cast, up) = self.step_up(&ty)?;
            casts.push(cast);
            ty = up;
        }
        Some((casts, ty))
    }

    /// The declared conversions from `from` to `to`: those whose parameter
    /// matches `from` and whose result matches `to`, by one binding of their
    /// variables, since the slot's type decides a conversion.
    pub fn conversions_between(&self, from: &Ty, to: &Ty) -> Vec<DeclId> {
        self.conversions()
            .filter(|decl| {
                let [param] = decl.sig.params() else {
                    return false;
                };
                let mut args = decl.sig.unbound();
                param.match_ty(from, &mut args) && decl.sig.result().match_ty(to, &mut args)
            })
            .map(|decl| decl.id.clone())
            .collect()
    }
}

/// The result of a one-parameter signature at the type its parameter matches,
/// where every variable of the result is bound by that match.
fn apply<T>(sig: &Signature, ty: &T) -> Option<T>
where
    T: HasTerm,
{
    let [param] = sig.params() else {
        return None;
    };
    let mut args: SigArgs<T> = sig.unbound();
    if !param.match_ty(ty, &mut args) {
        return None;
    }
    sig.result().substitute(&args).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sig::{ExternDecl, SigVar};
    use crate::ty::{IntTy, Scalar};

    fn name(text: &str) -> Name {
        Name::new(text)
    }

    fn ty(term: Term<Ty>) -> Ty {
        Ty::new(term).expect("the term is a closed type")
    }

    fn int(int: IntTy) -> Ty {
        ty(Term::Scalar(Scalar::Int(int)))
    }

    fn closed_extern(text: &str, args: Vec<Ty>) -> Ty {
        ty(Term::Extern {
            name: name(text),
            args,
            identity: None,
        })
    }

    fn sig_extern(text: &str, args: Vec<SigTy>) -> SigTy {
        SigTy::term(Term::Extern {
            name: name(text),
            args,
            identity: None,
        })
    }

    fn sig_ref(mutability: Mutability, target: SigTy) -> SigTy {
        SigTy::term(Term::Ref(mutability, target))
    }

    fn var(i: usize) -> SigTy {
        SigTy::Var(SigVar::Type(i))
    }

    /// Extern-defined types of no argument `A` to `E`, and `Vec`, `Box`, `Seq` and
    /// `Bytes` of one argument but `Bytes`.
    fn new_builder() -> DeclsBuilder {
        let mut types = DeclaredTypes::default();
        for text in ["A", "B", "C", "D", "E", "Bytes"] {
            types
                .declare_extern(
                    name(text),
                    ExternDecl {
                        arity: 0,
                        identity: false,
                    },
                )
                .unwrap();
        }
        for text in ["Vec", "Box", "Seq"] {
            types
                .declare_extern(
                    name(text),
                    ExternDecl {
                        arity: 1,
                        identity: false,
                    },
                )
                .unwrap();
        }
        DeclsBuilder::new(types)
    }

    fn cast(
        builder: &mut DeclsBuilder,
        text: &str,
        params: Vec<Name>,
        from: SigTy,
        to: SigTy,
    ) -> DeclId {
        let sig = Signature::new(params, false, vec![from], to, builder.types()).unwrap();
        builder.declare(name(text), sig, Role::Cast)
    }

    fn plain(text: &str) -> SigTy {
        sig_extern(text, vec![])
    }

    #[test]
    fn casts_that_step_up_once_from_each_node_form_a_tree() {
        let mut builder = new_builder();
        cast(&mut builder, "ab", vec![], plain("A"), plain("B"));
        cast(&mut builder, "cb", vec![], plain("C"), plain("B"));
        cast(&mut builder, "bd", vec![], plain("B"), plain("D"));
        let decls = builder.finish().unwrap();
        let node = |text: &str| Node::Owned(Constructor::Extern(name(text)));
        let tree = decls.tree();
        assert_eq!(
            tree.ancestors(&node("A")),
            vec![node("A"), node("B"), node("D")]
        );
        assert_eq!(tree.nca(&node("A"), &node("C")), Some(node("B")));
        assert_eq!(tree.nca(&node("A"), &node("D")), Some(node("D")));
        assert_eq!(tree.nca(&node("A"), &node("E")), None);
        let path = tree.path(&node("A"), &node("D")).unwrap();
        assert_eq!(
            path.iter().map(|id| id.name.as_str()).collect::<Vec<_>>(),
            ["ab", "bd"]
        );
        assert_eq!(tree.path(&node("A"), &node("A")), Some(vec![]));
        assert_eq!(tree.path(&node("D"), &node("A")), None);
    }

    #[test]
    fn a_second_cast_up_from_one_node_is_refused() {
        let mut builder = new_builder();
        let first = cast(&mut builder, "ab", vec![], plain("A"), plain("B"));
        let second = cast(&mut builder, "ac", vec![], plain("A"), plain("C"));
        assert_eq!(
            builder.finish().err(),
            Some(TreeError::TwoUp { first, second })
        );
    }

    #[test]
    fn casts_forming_a_cycle_are_refused() {
        let mut builder = new_builder();
        cast(&mut builder, "ab", vec![], plain("A"), plain("B"));
        cast(&mut builder, "ba", vec![], plain("B"), plain("A"));
        assert!(matches!(builder.finish(), Err(TreeError::Cycle(_))));
    }

    #[test]
    fn a_cast_goes_from_an_extern_defined_type_and_keeps_its_mode() {
        let mut builder = new_builder();
        let from_int = cast(
            &mut builder,
            "int",
            vec![],
            SigTy::term(Term::Scalar(Scalar::Int(IntTy::U8))),
            plain("A"),
        );
        assert_eq!(
            builder.finish().err(),
            Some(TreeError::NotFromExtern(from_int))
        );
        let mut builder = new_builder();
        let to_ref = cast(
            &mut builder,
            "to_ref",
            vec![],
            plain("A"),
            sig_ref(Mutability::Shared, plain("B")),
        );
        assert_eq!(builder.finish().err(), Some(TreeError::ChangesMode(to_ref)));
        let mut builder = new_builder();
        let weaker = cast(
            &mut builder,
            "weaker",
            vec![],
            sig_ref(Mutability::Exclusive, plain("A")),
            sig_ref(Mutability::Shared, plain("B")),
        );
        assert_eq!(builder.finish().err(), Some(TreeError::ChangesMode(weaker)));
        let mut builder = new_builder();
        let to_var = cast(&mut builder, "to_var", vec![name("T")], plain("A"), var(0));
        assert_eq!(
            builder.finish().err(),
            Some(TreeError::NoTargetNode(to_var))
        );
    }

    #[test]
    fn a_cast_of_exclusive_references_has_its_shared_counterpart() {
        let mut builder = new_builder();
        cast(
            &mut builder,
            "mut_ab",
            vec![],
            sig_ref(Mutability::Exclusive, plain("A")),
            sig_ref(Mutability::Exclusive, plain("B")),
        );
        cast(
            &mut builder,
            "ab",
            vec![],
            sig_ref(Mutability::Shared, plain("A")),
            sig_ref(Mutability::Shared, plain("B")),
        );
        assert!(builder.finish().is_ok());

        let mut builder = new_builder();
        let alone = cast(
            &mut builder,
            "mut_ab",
            vec![],
            sig_ref(Mutability::Exclusive, plain("A")),
            sig_ref(Mutability::Exclusive, plain("B")),
        );
        assert_eq!(
            builder.finish().err(),
            Some(TreeError::NotCommuting(alone.clone()))
        );

        let mut builder = new_builder();
        let elsewhere = cast(
            &mut builder,
            "mut_ab",
            vec![],
            sig_ref(Mutability::Exclusive, plain("A")),
            sig_ref(Mutability::Exclusive, plain("B")),
        );
        cast(
            &mut builder,
            "ac",
            vec![],
            sig_ref(Mutability::Shared, plain("A")),
            sig_ref(Mutability::Shared, plain("C")),
        );
        assert_eq!(
            builder.finish().err(),
            Some(TreeError::NotCommuting(elsewhere))
        );
    }

    #[test]
    fn a_view_steps_a_reference_to_a_container_up_to_a_slice_of_its_elements() {
        let mut builder = new_builder();
        cast(
            &mut builder,
            "core::as_slice",
            vec![name("T")],
            sig_ref(Mutability::Shared, sig_extern("Vec", vec![var(0)])),
            sig_ref(Mutability::Shared, SigTy::term(Term::Slice(var(0)))),
        );
        let decls = builder.finish().unwrap();
        let vec_u8 = ty(Term::Ref(
            Mutability::Shared,
            closed_extern("Vec", vec![int(IntTy::U8)]),
        ));
        let (cast, up) = decls.step_up(&vec_u8).unwrap();
        assert_eq!(cast.name, name("core::as_slice"));
        assert_eq!(
            up,
            ty(Term::Ref(
                Mutability::Shared,
                ty(Term::Slice(int(IntTy::U8)))
            ))
        );
        assert_eq!(
            Node::of(&up),
            Node::Ref(Mutability::Shared, Constructor::Slice)
        );
    }

    #[test]
    fn raising_takes_each_cast_up_to_the_node() {
        let mut builder = new_builder();
        cast(
            &mut builder,
            "box_seq",
            vec![name("T")],
            sig_extern("Box", vec![var(0)]),
            sig_extern("Seq", vec![var(0)]),
        );
        let decls = builder.finish().unwrap();
        let box_u16 = closed_extern("Box", vec![int(IntTy::U16)]);
        let seq = Node::Owned(Constructor::Extern(name("Seq")));
        let (casts, raised) = decls.raise(&box_u16, &seq).unwrap();
        assert_eq!(casts.len(), 1);
        assert_eq!(raised, closed_extern("Seq", vec![int(IntTy::U16)]));
    }

    #[test]
    fn a_cast_writes_its_node_generally_and_its_result_by_its_parameter() {
        let mut narrow = new_builder();
        let bytes = cast(
            &mut narrow,
            "bytes",
            vec![],
            sig_extern(
                "Vec",
                vec![SigTy::term(Term::Scalar(Scalar::Int(IntTy::U8)))],
            ),
            plain("Bytes"),
        );
        assert!(
            matches!(narrow.finish(), Err(TreeError::NarrowerThanNode(id)) if id == bytes),
            "Vec<u8> is not Vec's every type"
        );
        let mut unbound = new_builder();
        let boxed = cast(
            &mut unbound,
            "boxed",
            vec![name("T")],
            plain("A"),
            sig_extern("Box", vec![var(0)]),
        );
        assert!(
            matches!(unbound.finish(), Err(TreeError::ResultUnbound(id)) if id == boxed),
            "where A goes up is not decided"
        );
    }

    #[test]
    fn conversions_between_two_types_are_found_by_their_signatures() {
        let mut builder = new_builder();
        let u16_to_u8 = Signature::new(
            vec![],
            false,
            vec![SigTy::term(Term::Scalar(Scalar::Int(IntTy::U16)))],
            SigTy::term(Term::Scalar(Scalar::Int(IntTy::U8))),
            builder.types(),
        )
        .unwrap();
        let first = builder.declare(name("narrow"), u16_to_u8.clone(), Role::Conversion);
        let second = builder.declare(name("truncate"), u16_to_u8, Role::Conversion);
        let decls = builder.finish().unwrap();
        assert_eq!(
            decls.conversions_between(&int(IntTy::U16), &int(IntTy::U8)),
            vec![first, second]
        );
        assert!(
            decls
                .conversions_between(&int(IntTy::U8), &int(IntTy::U16))
                .is_empty()
        );
    }

    #[test]
    fn the_slots_type_binds_what_a_conversions_parameter_leaves_unbound() {
        let mut builder = new_builder();
        let a_to_box = Signature::new(
            vec![name("T")],
            false,
            vec![plain("A")],
            sig_extern("Box", vec![var(0)]),
            builder.types(),
        )
        .unwrap();
        let boxed = builder.declare(name("boxed"), a_to_box, Role::Conversion);
        let decls = builder.finish().unwrap();
        let a = closed_extern("A", vec![]);
        assert_eq!(
            decls.conversions_between(&a, &closed_extern("Box", vec![int(IntTy::U8)])),
            vec![boxed]
        );
        assert!(
            decls
                .conversions_between(&a, &closed_extern("Seq", vec![int(IntTy::U8)]))
                .is_empty()
        );
    }

    #[test]
    fn a_name_reaches_each_of_its_declarations_in_the_order_declared() {
        let mut builder = new_builder();
        let unit = SigTy::term(Term::Unit);
        let take_u8 = Signature::new(
            vec![],
            false,
            vec![SigTy::term(Term::Scalar(Scalar::Int(IntTy::U8)))],
            unit.clone(),
            builder.types(),
        )
        .unwrap();
        let take_u16 = Signature::new(
            vec![],
            false,
            vec![SigTy::term(Term::Scalar(Scalar::Int(IntTy::U16)))],
            unit,
            builder.types(),
        )
        .unwrap();
        builder.declare(name("take"), take_u8.clone(), Role::Fn);
        builder.declare(name("take"), take_u16.clone(), Role::Fn);
        let decls = builder.finish().unwrap();
        let candidates = decls.candidates(&name("take"));
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].sig, take_u8);
        assert_eq!(candidates[1].id.overload, 1);
        assert!(decls.candidates(&name("other")).is_empty());
    }
}
