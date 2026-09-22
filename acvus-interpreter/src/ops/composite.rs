//! Composite constructors: array, object, tuple.

use std::sync::Arc;

use acvus_extern::{ObjectShape, Owned};

use crate::code::{Exit, Marked, Next, Off, Op, successor};
use crate::machine::Machine;
use crate::regs::Cell;
use crate::runtime::AcvusRuntime;
use crate::value::Value;

pub struct Elements {
    pub slots: Box<[Off]>,
    /// Bit `i` is "slot `i` of `slots` owns a `Large`", which this operation
    /// takes over in one mask store (RFC-0048 §5).
    pub owns_large: u64,
}

impl Elements {
    #[inline]
    fn take(&self, m: &mut Machine<'_>) -> Vec<Owned<AcvusRuntime>> {
        let regs = m.regs();
        let items = self
            .slots
            .iter()
            .map(|slot| Owned::from_value(regs.read(*slot)))
            .collect();
        regs.take_mask(self.owns_large);
        items
    }
}

pub struct MakeArray {
    pub dst: Marked,
    pub elements: Elements,
    pub next: Next,
}

impl Op for MakeArray {
    successor!();

    fn run(&self, m: &mut Machine<'_>, regs: *mut Cell, r0: u64) -> Exit {
        let items = self.elements.take(m);
        m.regs().define::<true>(self.dst, Value::array(items));
        self.next.run(m, regs, r0)
    }
}

pub struct MakeTuple {
    pub dst: Marked,
    pub elements: Elements,
    pub next: Next,
}

impl Op for MakeTuple {
    successor!();

    fn run(&self, m: &mut Machine<'_>, regs: *mut Cell, r0: u64) -> Exit {
        let items = self.elements.take(m);
        m.regs().define::<true>(self.dst, Value::tuple(items));
        self.next.run(m, regs, r0)
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
    pub next: Next,
}

impl Op for MakeObject {
    successor!();

    fn run(&self, m: &mut Machine<'_>, regs: *mut Cell, r0: u64) -> Exit {
        let frame = m.regs();
        let object = Value::object_filled(Arc::clone(&self.shape), |at| {
            Owned::from_value(match self.fields[at.index()] {
                Some(at) => frame.read(at),
                None => Value::UNDEF,
            })
        });
        frame.take_mask(self.owns_large);
        m.regs().define::<true>(self.dst, object);
        self.next.run(m, regs, r0)
    }
}
