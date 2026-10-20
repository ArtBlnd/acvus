//! A program's declarations, by the name a call writes, and the tree its
//! declared casts form (l.76–78, extern fn §5, §7).

use std::collections::BTreeMap;

use crate::sig::{DeclaredTypes, SigArgs, SigTy, Signature};
use crate::ty::{Constructor, Mutability, Name, Term, Ty};

#[cfg(test)]
mod tests;

/// One declaration among those of its name, by the order they are declared in.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct DeclId {
    pub name: Name,
    pub overload: usize,
}

/// What a declaration is besides a function a call reaches by its name: a cast
/// is a step up the cast tree, and a conversion is no part of it (extern fn §7).
/// A view impl, an impl of `core::as_slice`, `core::as_slice_mut` or
/// `core::as_str`, is a cast (l.78).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Fn,
    Cast,
    Conversion,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Decl {
    pub id: DeclId,
    pub sig: Signature,
    pub role: Role,
}

/// A node of the cast tree: a constructor, and for a reference its mutability
/// and its target's constructor (l.78).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Node {
    Owned(Constructor),
    Ref(Mutability, Constructor),
}

impl Node {
    pub fn of(ty: &Ty) -> Node {
        match ty.term() {
            Term::Ref(mutability, target) => Node::Ref(*mutability, target.term().constructor()),
            term => Node::Owned(term.constructor()),
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

/// Why the declared casts are not a tree (l.78), or why a declaration is not
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

impl DeclsBuilder {
    pub fn new(types: DeclaredTypes) -> DeclsBuilder {
        DeclsBuilder {
            types,
            by_name: BTreeMap::new(),
        }
    }

    pub fn types(&self) -> &DeclaredTypes {
        &self.types
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
                Role::Cast => {
                    let [param] = decl.sig.params() else {
                        return Err(TreeError::CastArity(decl.id.clone()));
                    };
                    let from = Node::of_sig(param)
                        .filter(Node::is_extern_defined)
                        .ok_or_else(|| TreeError::NotFromExtern(decl.id.clone()))?;
                    let to = Node::of_sig(decl.sig.result())
                        .ok_or_else(|| TreeError::NoTargetNode(decl.id.clone()))?;
                    if from.mode() != to.mode() {
                        return Err(TreeError::ChangesMode(decl.id.clone()));
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

    /// The declarations a name reaches (l.331).
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
    pub fn step_up(&self, ty: &Ty) -> Option<(DeclId, Ty)> {
        let id = self.tree.up(&Node::of(ty))?;
        let applied = apply(&self.decl(id).sig, ty)?;
        Some((id.clone(), applied))
    }

    /// `ty` raised up the tree to `node`, with each cast taken.
    pub fn raise(&self, ty: &Ty, node: &Node) -> Option<(Vec<DeclId>, Ty)> {
        let mut casts = Vec::new();
        let mut ty = ty.clone();
        while &Node::of(&ty) != node {
            let (cast, up) = self.step_up(&ty)?;
            casts.push(cast);
            ty = up;
        }
        Some((casts, ty))
    }

    /// The declared conversions from `from` to `to`: those whose parameter
    /// matches `from` and whose result matches `to`, by one binding of their
    /// variables, since the slot's type decides a conversion (types §7).
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

/// The result of a one-parameter signature at the closed type its parameter
/// matches, where every variable of the result is bound by that match.
fn apply(sig: &Signature, ty: &Ty) -> Option<Ty> {
    let [param] = sig.params() else {
        return None;
    };
    let mut args: SigArgs = sig.unbound();
    if !param.match_ty(ty, &mut args) {
        return None;
    }
    sig.result().substitute(&args).ok()
}
