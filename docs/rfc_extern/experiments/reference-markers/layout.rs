//! Field offsets of author-defined containers under each element view, to be
//! compiled with and without `-Zrandomize-layout`. Prints one line per view:
//! whether its layout equals the generic view's, `Uniform<Never>`.

use std::marker::PhantomData;
use std::mem::offset_of;
use std::ptr::NonNull;

use model::{Exclusive, Mut, Never, Ref, Shared, Uniform};

#[allow(dead_code)]
struct OwnVec<T> {
    ptr: NonNull<T>,
    cap: usize,
    len: usize,
    marker: PhantomData<T>,
}

#[allow(dead_code)]
struct Stack<T> {
    flag: u8,
    items: OwnVec<T>,
    top: Option<T>,
    count: u32,
    tag: u16,
}

macro_rules! layout {
    ($t:ty) => {
        [
            offset_of!(Stack<$t>, flag),
            offset_of!(Stack<$t>, items),
            offset_of!(Stack<$t>, top),
            offset_of!(Stack<$t>, count),
            offset_of!(Stack<$t>, tag),
            offset_of!(OwnVec<$t>, ptr),
            offset_of!(OwnVec<$t>, cap),
            offset_of!(OwnVec<$t>, len),
            size_of::<Stack<$t>>(),
        ]
    };
}

fn main() {
    let generic = layout!(Uniform<Never>);
    let views: [(&str, [usize; 9]); 5] = [
        ("Uniform<String>", layout!(Uniform<String>)),
        ("Uniform<Shared<'static, i64>>", layout!(Uniform<Shared<'static, i64>>)),
        ("Uniform<Exclusive<'static, i64>>", layout!(Uniform<Exclusive<'static, i64>>)),
        ("Ref<'static, i64>", layout!(Ref<'static, i64>)),
        ("Mut<'static, i64>", layout!(Mut<'static, i64>)),
    ];
    for (name, layout) in views {
        let verdict = if layout == generic { "same" } else { "APART" };
        println!("{name:<34} {verdict}");
    }
}
