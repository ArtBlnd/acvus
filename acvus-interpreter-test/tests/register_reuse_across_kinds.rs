//! RFC-0052 rule 5: a word-typed slot's kind is written once, when the frame is
//! made, and a word operation stores the word only.
//!
//! That is only sound while the register selector keeps a slot in one kind
//! class for the whole frame. A `String` dying and an `i64` being defined
//! is the case that would break it: reusing the string's register for the
//! integer would leave `Kind::Large` under a `set_word`, which is the
//! incident this file is the test of.
//!
//! Two readings, because one alone would not catch it. `Body::slot_kinds`
//! is what `open_frame` writes, so the slots it names are the word class;
//! the frame is longer than that list, and the registers outside it are
//! where the `Large` and the reference to it live. Then the script runs,
//! and in this build `Regs::set_word` reaches `Value::bits_mut`, whose
//! `debug_assert!` is what a violated class would trip.

use acvus_interpreter_test::listing::{main_body, prepared_script};
use acvus_interpreter_test::*;
use acvus_mir::ty::Ty;
use acvus_utils::Interner;

/// A `String` whose last read is `len(&s)`, and four word values defined
/// after it — enough that a selector willing to reuse across kinds would
/// have taken the string's register for one of them.
const A_STRING_DIES_THEN_INTEGERS: &str =
    "let s = \"abcd\".to_string(); let k = len(&s); let t = k * 2; let u = t + 1; u";

#[tokio::test]
async fn the_word_class_is_a_proper_part_of_the_frame() {
    let interner = Interner::new();
    let prepared = prepared_script(
        &interner,
        A_STRING_DIES_THEN_INTEGERS,
        Context::default(),
        Ty::U64,
    );
    let body = main_body(&prepared);

    let word_slots: Vec<usize> = body.slot_kinds.iter().map(|k| k.slot.index()).collect();
    assert_eq!(
        word_slots,
        [0, 1, 4, 5, 6],
        "the registers `open_frame` fixes a kind on, in a frame of {}",
        body.frame_len
    );
    assert!(
        body.slot_kinds.iter().all(|k| k.kind.is_inline()),
        "every slot the frame opens is opened at an inline kind"
    );
    assert!(
        usize::from(body.frame_len) > body.slot_kinds.len(),
        "the frame keeps registers outside the word class — slots 2 and 3, \
         where the `String` and the borrow of it live — rather than reusing \
         them for the integers"
    );
    assert!(
        !word_slots.contains(&2) && !word_slots.contains(&3),
        "neither register a `Large` occupies is opened at a word kind"
    );
}

#[tokio::test]
async fn a_string_dying_before_integers_runs_to_the_value() {
    let interner = Interner::new();
    let value = run_script_mode(
        &interner,
        A_STRING_DIES_THEN_INTEGERS,
        Context::default(),
        Ty::U64,
    )
    .await;
    assert_eq!(
        value.as_int(),
        ("abcd".len() as i64) * 2 + 1,
        "len, doubled, plus one — every step of it a `set_word` on a slot \
         the string did not take"
    );
}
