//! Builtin function implementations.
//!
//! All builtins have been migrated to ExternFn registries in acvus-ext.
//! This module only provides the registration bridge for backwards compatibility.

use acvus_mir::graph::QualifiedRef;
use acvus_utils::{Astr, Interner};
use rustc_hash::FxHashMap;

use crate::interpreter::BuiltinHandler;

/// Build all builtin handlers as a HashMap.
/// With all builtins migrated to ExternFn, this returns an empty map.
pub fn build_builtins(
    _builtin_ids: &FxHashMap<Astr, QualifiedRef>,
    _interner: &Interner,
) -> FxHashMap<QualifiedRef, BuiltinHandler> {
    FxHashMap::default()
}
