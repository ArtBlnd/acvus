//! The operations a prepared body runs: plain functions of the machine and
//! their own `Op`, one per operation the preparation can choose, grouped by
//! the IR section they come from.

pub mod arith;
pub mod call;
pub mod composite;
pub mod constant;
pub mod control;
pub mod pattern;
pub mod storage;
pub mod string;
pub mod variant;

/// The payload an operation was prepared with, at the variant it names.
macro_rules! payload {
    ($m:expr, $op:expr, $variant:ident) => {
        match $m.payload($op) {
            crate::code::Payload::$variant(inner) => inner,
            other => panic!(
                "operation wants a {} payload, found {}",
                stringify!($variant),
                crate::code::payload_name(other)
            ),
        }
    };
}

pub(crate) use payload;
