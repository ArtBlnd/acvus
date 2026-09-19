//! Composite constructors: array, object, tuple.

use acvus_extern::Owned;
use rustc_hash::FxHashMap;

use crate::code::{Exit, FieldSlot, Marked, Off, Op, successor};
use crate::machine::Machine;
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
    pub next: Box<dyn Op>,
}

impl Op for MakeArray {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let items = self.elements.take(m);
        m.regs().define::<true>(self.dst, Value::array(items));
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
        let items = self.elements.take(m);
        m.regs().define::<true>(self.dst, Value::tuple(items));
        self.next.run(m, r0)
    }
}

pub struct MakeObject {
    pub dst: Marked,
    pub fields: Box<[FieldSlot]>,
    /// Bit `i` is "the slot of field `i` owns a `Large`", as `Elements`.
    pub owns_large: u64,
    pub next: Box<dyn Op>,
}

impl Op for MakeObject {
    successor!();

    fn run(&self, m: &mut Machine<'_>, r0: u64) -> Exit {
        let regs = m.regs();
        let object: FxHashMap<_, Owned<AcvusRuntime>> = self
            .fields
            .iter()
            .map(|field| (field.key, Owned::from_value(regs.read(field.slot))))
            .collect();
        regs.take_mask(self.owns_large);
        m.regs().define::<true>(self.dst, Value::object(object));
        self.next.run(m, r0)
    }
}
