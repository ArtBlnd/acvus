//! The strategy a signature's result crosses by is acvus-extern's, and
//! RFC-0080 rule 2 puts its one `unsafe impl` there.
use acvus_extern::ret::{Concrete, Returned};
use acvus_extern::{Crossing, Runtime};

struct Forged<Rt>(Rt::Value)
where
    Rt: Runtime;

unsafe impl<Rt> Returned<Concrete<i64>, Rt> for Forged<Rt>
where
    Rt: Runtime,
{
    fn cross(_: Crossing<'_, Rt>, _: Self) -> i64 {
        0
    }
}

fn main() {}
