//! A declared object or enum held as its Rust type lays it out: one block,
//! every field a `Value` at the offset its table states. The runtime reads
//! and writes the fields through the table alone, the extern side through
//! the Rust type, and both see the same memory. A field is found by its
//! name, so no order of the fields is assumed.

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

use crate::contract::{EnumTable, ObjectTable, VariantTable};

/// The table a block is laid out by.
#[derive(Clone, Copy)]
pub(crate) enum Table {
    Object(&'static ObjectTable),
    Enum(&'static EnumTable),
}

impl Table {
    fn layout(self) -> Layout {
        let (size, align) = match self {
            Table::Object(table) => (table.size, table.align),
            Table::Enum(table) => (table.size, table.align),
        };
        Layout::from_size_align(size, align).expect("a table states a valid layout")
    }
}

fn allocate(layout: Layout) -> NonNull<u8> {
    match layout.size() {
        0 => NonNull::<u8>::dangling().with_addr(std::num::NonZero::new(layout.align()).expect("an alignment is not zero")),
        // SAFETY: the layout's size is not zero.
        _ => NonNull::new(unsafe { alloc(layout) }).unwrap_or_else(|| handle_alloc_error(layout)),
    }
}

/// # Safety
/// `block` was allocated by `allocate` at `layout`, and is not used again.
unsafe fn free(block: NonNull<u8>, layout: Layout) {
    if layout.size() != 0 {
        // SAFETY: the caller's premise.
        unsafe { dealloc(block.as_ptr(), layout) }
    }
}

fn offset_in(fields: &[(&'static str, usize)], name: &str) -> usize {
    fields
        .iter()
        .find(|(field, _)| *field == name)
        .map(|(_, offset)| *offset)
        .unwrap_or_else(|| panic!("no field named {name}"))
}

/// A block laid out by its table, holding values of the runtime's `V`.
pub(crate) struct Laid<V>
where
    V: Copy,
{
    block: NonNull<u8>,
    table: Table,
    /// The runtime's release, which the block's drop calls on each field.
    release: unsafe fn(V),
}

impl<V> Laid<V>
where
    V: Copy,
{
    /// A block holding `value`, which `table` lays out.
    ///
    /// # Safety
    /// `table` states `X`'s layout and fields, each field a `V`.
    pub(crate) unsafe fn put<X>(value: X, table: Table, release: unsafe fn(V)) -> Self {
        assert!(table.layout() == Layout::new::<X>(), "a structure put in at another layout than its table states");
        let block = allocate(table.layout());
        // SAFETY: the block is fresh and laid out for `X`.
        unsafe { block.cast::<X>().write(value) };
        Laid { block, table, release }
    }

    /// A block of `table`'s layout holding `fields`, each at its offset.
    ///
    /// # Safety
    /// Each field is a live `V` the caller owns and gives up, of its field's
    /// type.
    pub(crate) unsafe fn make_object(table: &'static ObjectTable, fields: &[(&str, V)], release: unsafe fn(V)) -> Self {
        let block = allocate(Table::Object(table).layout());
        // SAFETY: the caller's premise; the block is fresh.
        unsafe { write_fields(block, table.fields, fields) };
        Laid { block, table: Table::Object(table), release }
    }

    /// A block of `table`'s layout holding `variant`'s tag and its `fields`.
    ///
    /// # Safety
    /// As `make_object`.
    pub(crate) unsafe fn make_enum(table: &'static EnumTable, variant: &str, fields: &[(&str, V)], release: unsafe fn(V)) -> Self {
        let block = allocate(Table::Enum(table).layout());
        let variant = table.variants.iter().find(|stated| stated.name == variant).unwrap_or_else(|| panic!("no variant named {variant}"));
        // SAFETY: a `repr(u32)` enum stores its tag at its start.
        unsafe { block.cast::<u32>().write(variant.tag) };
        // SAFETY: the caller's premise; the block is fresh.
        unsafe { write_fields(block, variant.fields, fields) };
        Laid { block, table: Table::Enum(table), release }
    }

    /// Gives the value back as `X`, its fields with it: the block is freed
    /// and no field released.
    ///
    /// # Safety
    /// The block holds an `X`, or an instantiation of its type of the same layout.
    pub(crate) unsafe fn take<X>(self) -> X {
        let laid = ManuallyDrop::new(self);
        // SAFETY: the caller's premise; the block is not read again.
        let value = unsafe { laid.block.cast::<X>().read() };
        // SAFETY: allocated at this layout, and given up here.
        unsafe { free(laid.block, laid.table.layout()) };
        value
    }

    /// # Safety
    /// As `take`.
    pub(crate) unsafe fn project<X>(&self) -> &X {
        // SAFETY: the caller's premise.
        unsafe { self.block.cast::<X>().as_ref() }
    }

    /// # Safety
    /// As `take`.
    pub(crate) unsafe fn project_mut<X>(&mut self) -> &mut X {
        // SAFETY: the caller's premise.
        unsafe { self.block.cast::<X>().as_mut() }
    }

    fn current(&self, table: &'static EnumTable) -> &'static VariantTable {
        // SAFETY: a `repr(u32)` enum stores its tag at its start.
        let tag = unsafe { self.block.cast::<u32>().read() };
        table.variants.iter().find(|variant| variant.tag == tag).unwrap_or_else(|| panic!("no variant of tag {tag}"))
    }

    /// The fields the block holds: the object's, or its current variant's.
    fn fields(&self) -> &'static [(&'static str, usize)] {
        match self.table {
            Table::Object(table) => table.fields,
            Table::Enum(table) => self.current(table).fields,
        }
    }

    /// The name of the variant an enum's block holds.
    pub(crate) fn variant(&self) -> &'static str {
        match self.table {
            Table::Enum(table) => self.current(table).name,
            Table::Object(_) => panic!("an object has no variant"),
        }
    }

    /// The place of the named field, a `V`.
    pub(crate) fn place(&self, name: &str) -> NonNull<V> {
        // SAFETY: the table states each field's offset inside the block.
        unsafe { self.block.add(offset_in(self.fields(), name)) }.cast::<V>()
    }
}

/// # Safety
/// `block` is fresh and laid out with `stated` fields, each a `V`; each of
/// `fields` is a live `V` given up to it.
unsafe fn write_fields<V>(block: NonNull<u8>, stated: &'static [(&'static str, usize)], fields: &[(&str, V)])
where
    V: Copy,
{
    assert_eq!(stated.len(), fields.len(), "every field given once");
    fields.iter().for_each(|(name, value)| {
        // SAFETY: the caller's premise.
        unsafe { block.add(offset_in(stated, name)).cast::<V>().write(*value) }
    });
}

impl<V> Drop for Laid<V>
where
    V: Copy,
{
    fn drop(&mut self) {
        self.fields().iter().for_each(|(_, offset)| {
            // SAFETY: each field is a live `V` the block owns, released once.
            unsafe { (self.release)(self.block.add(*offset).cast::<V>().read()) }
        });
        // SAFETY: allocated at this layout, and given up here.
        unsafe { free(self.block, self.table.layout()) };
    }
}
