//! The operations a prepared body runs: plain functions of the machine and
//! their own `Op`, one per operation the preparation can choose, grouped by
//! the IR section they come from.

pub mod arith;
pub mod call;
pub mod cast;
pub mod chain;
pub mod composite;
pub mod constant;
pub mod control;
pub mod index;
pub mod pattern;
pub mod place;
pub mod select;
pub mod storage;
pub mod string;
pub mod switch;
pub mod variant;
