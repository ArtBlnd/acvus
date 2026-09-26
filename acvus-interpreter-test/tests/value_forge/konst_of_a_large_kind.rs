//! A constant's `Word` names any kind, so the value it makes is the
//! runtime's alone.
use acvus_interpreter::Kind;
use acvus_interpreter::code::Konst;

fn main() {
    let _ = Konst::Word(Kind::Large, 8).value();
}
