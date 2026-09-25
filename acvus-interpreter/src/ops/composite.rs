//! Composite constructors: array, object, tuple.

use std::sync::Arc;

use acvus_extern::{ObjectShape, Owned};

use crate::code::{Exit, Marked, Off, Op, successor};
use crate::machine::Machine;
use crate::runtime::AcvusRuntime;
use crate::value::Value;

pub struct Elements {
    pub slots: Box<[Off]>,
    /// Bit `i` is "slot `i` of `slots` owns a `Large`", which this operation
    /// takes over in one mask store (RFC-0048 rule 5).
    pub owns_large: u64,
}

impl Elements {
    #[inline]
    fn take(&self, m: &mut Machine<'_>) -> Vec<Owned<AcvusRuntime>> {
        let regs = m.regs();
        let items = self
            .slots
            .iter()
            // SAFETY: every register read here that owns a word is in the mask
            // `take_mask` clears below, so its word moves here; the others
            // hold words that own nothing (`prepare` builds the mask from the
            // operands that own).
            .map(|slot| unsafe { Owned::from_value(acvus_extern::Holding::new(), regs.read(*slot)) })
            .collect();
        regs.take_mask(self.owns_large);
        items
    }
}

pub struct ArrayBegin {
    pub dst: Marked,
    pub capacity: usize,
    pub next: Box<dyn Op>,
}

impl Op for ArrayBegin {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let value = Value::array(Vec::with_capacity(self.capacity));
        m.regs().define::<true>(self.dst, value);
        self.next.run(m, r0)
    }
}

pub struct ArrayPush {
    pub dst: Marked,
    pub array: Off,
    pub value: Off,
    /// The claims of `array` and of `value` where it owns a `Large`, as
    /// `Elements::owns_large` (RFC-0048 rule 5).
    pub owns_large: u64,
    pub next: Box<dyn Op>,
}

impl Op for ArrayPush {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let mut array = regs.read(self.array);
        let value = regs.read(self.value);
        regs.take_mask(self.owns_large);
        // SAFETY: `validate::type_check` holds `array` to an array type, so
        // its register holds an `Array`, whose claim this operation took
        // above; `value` moves in as `Elements::take`'s operands do.
        unsafe {
            let value = Owned::from_value(acvus_extern::Holding::new(), value);
            array.as_array_mut().0.push(value);
        }
        m.regs().define::<true>(self.dst, array);
        self.next.run(m, r0)
    }
}

pub struct MakeTuple {
    pub dst: Marked,
    pub elements: Elements,
    pub next: Box<dyn Op>,
}

impl Op for MakeTuple {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let value = Value::tuple_with(|| self.elements.take(m));
        m.regs().define::<true>(self.dst, value);
        self.next.run(m, r0)
    }
}

pub struct MakeObject {
    pub dst: Marked,
    pub shape: Arc<ObjectShape>,
    /// One per field of `shape`, in its order: the register that field's value
    /// is read from, and `None` for a field the settled type has and this
    /// construction does not write — `Undef` at its position (RFC-0050 rule 8).
    pub fields: Box<[Option<Off>]>,
    /// Bit `i` is "the slot of the `i`th written field owns a `Large`", as
    /// `Elements`; the mask counts the registers, so a field with no register
    /// takes no bit.
    pub owns_large: u64,
    pub next: Box<dyn Op>,
}

impl Op for MakeObject {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let object = Value::object_filled(&self.shape, |at| {
            let word = match self.fields[at.index()] {
                Some(at) => regs.read(at),
                None => Value::UNDEF,
            };
            // SAFETY: as `Elements::take`'s, with `owns_large`; `UNDEF` owns
            // nothing.
            unsafe { Owned::from_value(acvus_extern::Holding::new(), word) }
        });
        regs.take_mask(self.owns_large);
        m.regs().define::<true>(self.dst, object);
        self.next.run(m, r0)
    }
}
