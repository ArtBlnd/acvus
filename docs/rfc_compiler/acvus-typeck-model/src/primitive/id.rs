//! Identifiers, and the arena that issues them.

use std::marker::PhantomData;

/// What an arena hands an id type to make an id. Its field is private, so
/// only an arena makes one, and so only an arena makes an id.
pub struct Mint(());

/// An identifier an [`Arena`] issues: an index into it.
pub trait Id: Copy {
    fn mint(index: u32, mint: Mint) -> Self;
    fn index(self) -> usize;
}

macro_rules! id {
    ($($(#[$doc:meta])* $name:ident),* $(,)?) => { $(
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        pub struct $name(u32);

        impl Id for $name {
            fn mint(index: u32, _: Mint) -> Self {
                Self(index)
            }

            fn index(self) -> usize {
                self.0 as usize
            }
        }
    )* };
}

id! {
    /// A live class, one that is solved. Once classes are unified, several
    /// ids name one class; `store::classes::Classes::find` gives the one that
    /// holds it.
    ClassId,
    /// A class of a template: recorded once, never solved, and only copied.
    TemplateId,
    /// What recorded something: a place in the text, or solving itself.
    ItemId,
    /// A lambda together with its captures.
    ClosureId,
    /// A lambda as the text writes it: its template.
    LambdaId,
    /// A name a function is called by.
    NameId,
    /// The text of a body or a lambda, or one iteration of a loop.
    ScopeId,
}

/// A sequence of `T`, each issued an id `I` when it is pushed.
#[derive(Clone, Debug)]
pub struct Arena<I, T> {
    items: Vec<T>,
    id: PhantomData<fn() -> I>,
}

impl<I, T> Default for Arena<I, T> {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            id: PhantomData,
        }
    }
}

impl<I, T> Arena<I, T>
where
    I: Id,
{
    pub fn push(&mut self, item: T) -> I {
        let id = Self::id_at(self.items.len());
        self.items.push(item);
        id
    }

    pub fn get(&self, id: I) -> &T {
        &self.items[id.index()]
    }

    pub fn get_mut(&mut self, id: I) -> &mut T {
        &mut self.items[id.index()]
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Each item with its id, in the order they were pushed.
    pub fn iter(&self) -> impl Iterator<Item = (I, &T)> {
        self.items
            .iter()
            .enumerate()
            .map(|(index, item)| (Self::id_at(index), item))
    }

    fn id_at(index: usize) -> I {
        let index = u32::try_from(index).expect("an arena holds fewer than 2³² items");
        I::mint(index, Mint(()))
    }
}
