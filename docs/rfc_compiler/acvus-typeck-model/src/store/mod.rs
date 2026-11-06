//! The store of classes.
//!
//! It knows identity and structure alone: which ids are one class, which
//! parts have one name, which classes a bound relates, and whether two shapes
//! have one constructor and the same members. Every other judgment, a join, a
//! change, a domain, an instance, is its caller's, and the store returns what
//! needs one.

pub mod classes;
pub mod verify;
