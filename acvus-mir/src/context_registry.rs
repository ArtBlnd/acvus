use std::fmt;

use acvus_utils::Astr;
use rustc_hash::FxHashMap;

use crate::ty::Ty;

/// Error when two tiers in a [`ContextTypeRegistry`] contain the same key.
#[derive(Debug, Clone)]
pub struct RegistryConflictError {
    pub key: Astr,
    pub tier_a: &'static str,
    pub tier_b: &'static str,
}

impl fmt::Display for RegistryConflictError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "context type registry conflict: key appears in both '{}' and '{}'",
            self.tier_a, self.tier_b,
        )
    }
}

/// Partial registry holding extern, system, and user context types.
///
/// - **extern**: extern fns (regex etc.) — only the typechecker needs these.
/// - **system**: orchestration-provided values (@turn, node names) — visible to frontend.
/// - **user**: frontend-injected params (@input, @char etc.) — visible to frontend.
///
/// Call [`with_scoped`](PartialContextTypeRegistry::with_scoped) to produce a
/// full [`ContextTypeRegistry`] for compilation.
#[derive(Debug, Clone, Default)]
pub struct PartialContextTypeRegistry {
    extern_fns: FxHashMap<Astr, Ty>,
    system: FxHashMap<Astr, Ty>,
    user: FxHashMap<Astr, Ty>,
    merged: FxHashMap<Astr, Ty>,
}

impl PartialContextTypeRegistry {
    /// Create a new partial registry.
    ///
    /// Returns an error if any key appears in more than one tier.
    pub fn new(
        extern_fns: FxHashMap<Astr, Ty>,
        system: FxHashMap<Astr, Ty>,
        user: FxHashMap<Astr, Ty>,
    ) -> Result<Self, RegistryConflictError> {
        check_disjoint(&extern_fns, "extern", &system, "system")?;
        check_disjoint(&extern_fns, "extern", &user, "user")?;
        check_disjoint(&system, "system", &user, "user")?;
        let merged = build_merged([&extern_fns, &system, &user]);
        Ok(Self { extern_fns, system, user, merged })
    }

    /// Create a partial registry with only system types.
    pub fn system_only(system: FxHashMap<Astr, Ty>) -> Self {
        let merged = system.clone();
        Self {
            extern_fns: FxHashMap::default(),
            system,
            user: FxHashMap::default(),
            merged,
        }
    }

    /// Create a partial registry with only user types.
    pub fn user_only(user: FxHashMap<Astr, Ty>) -> Self {
        let merged = user.clone();
        Self {
            extern_fns: FxHashMap::default(),
            system: FxHashMap::default(),
            user,
            merged,
        }
    }

    /// Promote to a full [`ContextTypeRegistry`] by adding scoped types.
    ///
    /// Returns an error if any scoped key conflicts with other tiers.
    pub fn with_scoped(
        self,
        scoped: FxHashMap<Astr, Ty>,
    ) -> Result<ContextTypeRegistry, RegistryConflictError> {
        check_disjoint(&scoped, "scoped", &self.extern_fns, "extern")?;
        check_disjoint(&scoped, "scoped", &self.system, "system")?;
        check_disjoint(&scoped, "scoped", &self.user, "user")?;
        let mut merged = self.merged;
        merged.extend(scoped.iter().map(|(k, v)| (*k, v.clone())));
        Ok(ContextTypeRegistry {
            extern_fns: self.extern_fns,
            system: self.system,
            scoped,
            user: self.user,
            merged,
        })
    }

    /// Promote to a full [`ContextTypeRegistry`] with an empty scoped tier.
    pub fn without_scoped(self) -> ContextTypeRegistry {
        ContextTypeRegistry {
            extern_fns: self.extern_fns,
            system: self.system,
            scoped: FxHashMap::default(),
            user: self.user,
            merged: self.merged,
        }
    }

    /// Merged view of all types (extern + system + user). For type checking.
    pub fn merged(&self) -> &FxHashMap<Astr, Ty> {
        &self.merged
    }

    /// The extern fn types.
    pub fn extern_fns(&self) -> &FxHashMap<Astr, Ty> {
        &self.extern_fns
    }

    /// The system-provided types.
    pub fn system(&self) -> &FxHashMap<Astr, Ty> {
        &self.system
    }

    /// The user-provided types.
    pub fn user(&self) -> &FxHashMap<Astr, Ty> {
        &self.user
    }

    /// Types visible to the frontend (system + user, excludes extern).
    pub fn visible(&self) -> FxHashMap<Astr, Ty> {
        build_merged([&self.system, &self.user])
    }

    /// Insert a single key into the system tier.
    ///
    /// Returns an error if the key already exists in extern or user tier.
    /// Panics if the key already exists in the system tier (programming bug).
    pub fn insert_system(
        &mut self,
        key: Astr,
        ty: Ty,
    ) -> Result<(), RegistryConflictError> {
        if self.extern_fns.contains_key(&key) {
            return Err(RegistryConflictError {
                key,
                tier_a: "system",
                tier_b: "extern",
            });
        }
        if self.user.contains_key(&key) {
            return Err(RegistryConflictError {
                key,
                tier_a: "system",
                tier_b: "user",
            });
        }
        assert!(
            !self.system.contains_key(&key),
            "insert_system: duplicate key in system tier",
        );
        self.system.insert(key, ty.clone());
        self.merged.insert(key, ty);
        Ok(())
    }


    /// Extend the system tier with multiple entries.
    ///
    /// Returns an error on the first key that conflicts with extern or user tier.
    pub fn extend_system(
        &mut self,
        types: impl IntoIterator<Item = (Astr, Ty)>,
    ) -> Result<(), RegistryConflictError> {
        for (key, ty) in types {
            self.insert_system(key, ty)?;
        }
        Ok(())
    }

    /// Returns true if the key is a user-provided key
    /// (not in extern or system tier).
    pub fn is_user_key(&self, key: &Astr) -> bool {
        !self.extern_fns.contains_key(key) && !self.system.contains_key(key)
    }
}

/// Full context type registry with extern, system, scoped, and user tiers.
///
/// Invariant: no key appears in more than one tier.
/// This is enforced at construction time.
#[derive(Debug, Clone)]
pub struct ContextTypeRegistry {
    extern_fns: FxHashMap<Astr, Ty>,
    system: FxHashMap<Astr, Ty>,
    scoped: FxHashMap<Astr, Ty>,
    user: FxHashMap<Astr, Ty>,
    merged: FxHashMap<Astr, Ty>,
}

impl ContextTypeRegistry {
    /// Create a full registry from four tiers.
    ///
    /// Returns an error if any key appears in more than one tier.
    pub fn new(
        extern_fns: FxHashMap<Astr, Ty>,
        system: FxHashMap<Astr, Ty>,
        scoped: FxHashMap<Astr, Ty>,
        user: FxHashMap<Astr, Ty>,
    ) -> Result<Self, RegistryConflictError> {
        check_disjoint(&extern_fns, "extern", &system, "system")?;
        check_disjoint(&extern_fns, "extern", &scoped, "scoped")?;
        check_disjoint(&extern_fns, "extern", &user, "user")?;
        check_disjoint(&system, "system", &scoped, "scoped")?;
        check_disjoint(&system, "system", &user, "user")?;
        check_disjoint(&scoped, "scoped", &user, "user")?;
        let merged = build_merged_vec(&[&extern_fns, &system, &scoped, &user]);
        Ok(Self { extern_fns, system, scoped, user, merged })
    }

    /// Convenience: treat all types as system (for backward compat / tests).
    pub fn all_system(types: FxHashMap<Astr, Ty>) -> Self {
        let merged = types.clone();
        Self {
            extern_fns: FxHashMap::default(),
            system: types,
            scoped: FxHashMap::default(),
            user: FxHashMap::default(),
            merged,
        }
    }

    /// Merged view of all types. For type checking.
    pub fn merged(&self) -> &FxHashMap<Astr, Ty> {
        &self.merged
    }

    /// The extern fn types.
    pub fn extern_fns(&self) -> &FxHashMap<Astr, Ty> {
        &self.extern_fns
    }

    /// The system-provided types.
    pub fn system(&self) -> &FxHashMap<Astr, Ty> {
        &self.system
    }

    /// The scoped types.
    pub fn scoped(&self) -> &FxHashMap<Astr, Ty> {
        &self.scoped
    }

    /// The user-provided types.
    pub fn user(&self) -> &FxHashMap<Astr, Ty> {
        &self.user
    }

    /// Iterator over user-provided key names.
    pub fn user_keys(&self) -> impl Iterator<Item = &Astr> {
        self.user.keys()
    }

    /// Returns true if the key is provided (extern, system, or scoped)
    /// — does NOT need user resolution.
    pub fn is_provided(&self, key: &Astr) -> bool {
        self.extern_fns.contains_key(key)
            || self.system.contains_key(key)
            || self.scoped.contains_key(key)
    }

    /// Returns true if the key is NOT provided
    /// — needs external (user) resolution.
    pub fn is_user_key(&self, key: &Astr) -> bool {
        !self.is_provided(key)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Check that two maps have no keys in common.
fn check_disjoint(
    a: &FxHashMap<Astr, Ty>,
    a_name: &'static str,
    b: &FxHashMap<Astr, Ty>,
    b_name: &'static str,
) -> Result<(), RegistryConflictError> {
    // Iterate the smaller map for efficiency.
    let (small, small_name, big, big_name) = if a.len() <= b.len() {
        (a, a_name, b, b_name)
    } else {
        (b, b_name, a, a_name)
    };
    for key in small.keys() {
        if big.contains_key(key) {
            return Err(RegistryConflictError {
                key: *key,
                tier_a: small_name,
                tier_b: big_name,
            });
        }
    }
    Ok(())
}

/// Build a merged HashMap from an array of maps (const size).
fn build_merged<const N: usize>(maps: [&FxHashMap<Astr, Ty>; N]) -> FxHashMap<Astr, Ty> {
    let total: usize = maps.iter().map(|m| m.len()).sum();
    let mut merged = FxHashMap::with_capacity_and_hasher(total, Default::default());
    for map in maps {
        merged.extend(map.iter().map(|(k, v)| (*k, v.clone())));
    }
    merged
}

/// Build a merged HashMap from a slice of maps (dynamic size).
fn build_merged_vec(maps: &[&FxHashMap<Astr, Ty>]) -> FxHashMap<Astr, Ty> {
    let total: usize = maps.iter().map(|m| m.len()).sum();
    let mut merged = FxHashMap::with_capacity_and_hasher(total, Default::default());
    for map in maps {
        merged.extend(map.iter().map(|(k, v)| (*k, v.clone())));
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use acvus_utils::Interner;

    fn setup() -> (Interner, Astr, Astr, Astr, Astr) {
        let i = Interner::new();
        let a = i.intern("a");
        let b = i.intern("b");
        let c = i.intern("c");
        let d = i.intern("d");
        (i, a, b, c, d)
    }

    #[test]
    fn partial_no_conflict() {
        let (_i, a, b, c, _d) = setup();
        let extern_fns = FxHashMap::from_iter([(a, Ty::Int)]);
        let system = FxHashMap::from_iter([(b, Ty::String)]);
        let user = FxHashMap::from_iter([(c, Ty::Bool)]);
        let reg = PartialContextTypeRegistry::new(extern_fns, system, user).unwrap();
        assert_eq!(reg.merged().len(), 3);
        assert!(!reg.is_user_key(&a)); // extern
        assert!(!reg.is_user_key(&b)); // system
        assert!(reg.is_user_key(&c));  // user
    }

    #[test]
    fn partial_extern_system_conflict() {
        let (_i, a, _b, _c, _d) = setup();
        let extern_fns = FxHashMap::from_iter([(a, Ty::Int)]);
        let system = FxHashMap::from_iter([(a, Ty::String)]);
        let user = FxHashMap::default();
        let err = PartialContextTypeRegistry::new(extern_fns, system, user).unwrap_err();
        assert_eq!(err.key, a);
    }

    #[test]
    fn partial_extern_user_conflict() {
        let (_i, a, _b, _c, _d) = setup();
        let extern_fns = FxHashMap::from_iter([(a, Ty::Int)]);
        let system = FxHashMap::default();
        let user = FxHashMap::from_iter([(a, Ty::String)]);
        let err = PartialContextTypeRegistry::new(extern_fns, system, user).unwrap_err();
        assert_eq!(err.key, a);
    }

    #[test]
    fn partial_system_user_conflict() {
        let (_i, a, _b, _c, _d) = setup();
        let extern_fns = FxHashMap::default();
        let system = FxHashMap::from_iter([(a, Ty::Int)]);
        let user = FxHashMap::from_iter([(a, Ty::String)]);
        let err = PartialContextTypeRegistry::new(extern_fns, system, user).unwrap_err();
        assert_eq!(err.key, a);
    }

    #[test]
    fn visible_excludes_extern() {
        let (_i, a, b, c, _d) = setup();
        let extern_fns = FxHashMap::from_iter([(a, Ty::Int)]);
        let system = FxHashMap::from_iter([(b, Ty::String)]);
        let user = FxHashMap::from_iter([(c, Ty::Bool)]);
        let reg = PartialContextTypeRegistry::new(extern_fns, system, user).unwrap();
        let vis = reg.visible();
        assert_eq!(vis.len(), 2);
        assert!(!vis.contains_key(&a));
        assert!(vis.contains_key(&b));
        assert!(vis.contains_key(&c));
    }

    #[test]
    fn with_scoped_no_conflict() {
        let (_i, a, b, c, d) = setup();
        let partial = PartialContextTypeRegistry::new(
            FxHashMap::from_iter([(a, Ty::Int)]),
            FxHashMap::from_iter([(b, Ty::String)]),
            FxHashMap::from_iter([(c, Ty::Bool)]),
        )
        .unwrap();
        let scoped = FxHashMap::from_iter([(d, Ty::Float)]);
        let full = partial.with_scoped(scoped).unwrap();
        assert_eq!(full.merged().len(), 4);
        assert!(full.is_provided(&a)); // extern
        assert!(full.is_provided(&b)); // system
        assert!(full.is_provided(&d)); // scoped
        assert!(full.is_user_key(&c)); // user
    }

    #[test]
    fn with_scoped_conflict_with_extern() {
        let (_i, a, b, c, _d) = setup();
        let partial = PartialContextTypeRegistry::new(
            FxHashMap::from_iter([(a, Ty::Int)]),
            FxHashMap::from_iter([(b, Ty::String)]),
            FxHashMap::from_iter([(c, Ty::Bool)]),
        )
        .unwrap();
        let scoped = FxHashMap::from_iter([(a, Ty::Float)]);
        let err = partial.with_scoped(scoped).unwrap_err();
        assert_eq!(err.key, a);
    }

    #[test]
    fn all_system_bridge() {
        let (_i, a, b, _c, _d) = setup();
        let types = FxHashMap::from_iter([(a, Ty::Int), (b, Ty::String)]);
        let reg = ContextTypeRegistry::all_system(types);
        assert_eq!(reg.merged().len(), 2);
        assert!(reg.is_provided(&a));
        assert!(reg.is_provided(&b));
    }

    #[test]
    fn without_scoped() {
        let (_i, a, b, c, _d) = setup();
        let partial = PartialContextTypeRegistry::new(
            FxHashMap::from_iter([(a, Ty::Int)]),
            FxHashMap::from_iter([(b, Ty::String)]),
            FxHashMap::from_iter([(c, Ty::Bool)]),
        )
        .unwrap();
        let full = partial.without_scoped();
        assert_eq!(full.merged().len(), 3);
        assert!(full.scoped().is_empty());
    }

    #[test]
    fn insert_system_conflict_with_extern() {
        let (_i, a, b, _c, _d) = setup();
        let mut reg = PartialContextTypeRegistry::new(
            FxHashMap::from_iter([(a, Ty::Int)]),
            FxHashMap::default(),
            FxHashMap::from_iter([(b, Ty::String)]),
        )
        .unwrap();
        let err = reg.insert_system(a, Ty::Float).unwrap_err();
        assert_eq!(err.key, a);
    }
}
