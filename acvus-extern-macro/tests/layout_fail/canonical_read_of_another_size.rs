//! Every read of a box at a type other than its canonical form carries an
//! inline `const` assert that the two layouts agree in size and alignment
//! (RFC-0076). A `Canonical` impl whose form is another size is the
//! author's broken promise, and the read at it does not compile.
use acvus_extern::derive::canonical;
use acvus_extern::{Canonical, TypesOnly, kind};

struct Wide(u64);

// SAFETY: deliberately false: `u8` is not `Wide`'s canonical form.
unsafe impl Canonical<kind::Type> for Wide {
    type Canon = u8;
}

fn main() {
    let _erase: fn(&TypesOnly, Wide) -> () = canonical::erase::<Wide, TypesOnly>;
}
