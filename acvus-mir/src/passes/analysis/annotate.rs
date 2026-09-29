//! Annotations of the IR printer (`ir::cfg::print::Annotate`) from the
//! analyses: each says, in notation, what its analysis holds at a point.
//! - `LiveAnnotations`, key `live`: the values `Liveness` holds live at a
//!   block's start, `{v1, v2}`.
//! - `InitAnnotations`, key `init`: before each instruction, each unit of
//!   each place `Initialization` tracks, `slot0.a=W` written on every path
//!   that reaches, `N` left unwritten by some path, `M@start..end` moved out
//!   at that span by some path, several joined by `|`.
//! - `LoanAnnotations`, key `loans`: each value defined by the instruction
//!   and holding loans, `v2=[slot0.b mut @25..33]`, a loan by its place, its
//!   mode and where it was made, `~` after the path where it is not exact.
//!
//! Every set prints sorted; a value, a place and a span as the printer
//! prints them.
