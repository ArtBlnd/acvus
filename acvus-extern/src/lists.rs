//! The closed lists the crossing is implemented over, each written once:
//! every trait implemented per tuple arity or per primitive takes its list
//! from here, by a macro of its own called with the list.

/// Calls `$m!` with every tuple arity the crossing admits, one to twelve, the
/// arity the standard library's own tuple impls stop at: each as its arity,
/// and its parts, each as a type parameter, a binding and an index.
macro_rules! tuples {
    ($m:ident) => {
        $m! {
            1 (A a 0)
            2 (A a 0, B b 1)
            3 (A a 0, B b 1, C c 2)
            4 (A a 0, B b 1, C c 2, D d 3)
            5 (A a 0, B b 1, C c 2, D d 3, E e 4)
            6 (A a 0, B b 1, C c 2, D d 3, E e 4, F f 5)
            7 (A a 0, B b 1, C c 2, D d 3, E e 4, F f 5, G g 6)
            8 (A a 0, B b 1, C c 2, D d 3, E e 4, F f 5, G g 6, H h 7)
            9 (A a 0, B b 1, C c 2, D d 3, E e 4, F f 5, G g 6, H h 7, I i 8)
            10 (A a 0, B b 1, C c 2, D d 3, E e 4, F f 5, G g 6, H h 7, I i 8, J j 9)
            11 (A a 0, B b 1, C c 2, D d 3, E e 4, F f 5, G g 6, H h 7, I i 8, J j 9, K k 10)
            12 (A a 0, B b 1, C c 2, D d 3, E e 4, F f 5, G g 6, H h 7, I i 8, J j 9, K k 10, L l 11)
        }
    };
}

/// Calls `$m!` with the language's primitives, each with its declared type.
macro_rules! primitives {
    ($m:ident) => {
        $m! {
            i8 => DeclTy::Int(IntTy::I8),
            i16 => DeclTy::Int(IntTy::I16),
            i32 => DeclTy::Int(IntTy::I32),
            i64 => DeclTy::Int(IntTy::I64),
            u8 => DeclTy::Int(IntTy::U8),
            u16 => DeclTy::Int(IntTy::U16),
            u32 => DeclTy::Int(IntTy::U32),
            u64 => DeclTy::Int(IntTy::U64),
            f64 => DeclTy::F64,
            bool => DeclTy::Bool,
            char => DeclTy::Char,
            () => DeclTy::Unit,
        }
    };
}
