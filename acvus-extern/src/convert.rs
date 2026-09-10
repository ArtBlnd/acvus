//! Crossing between Rust values and a runtime's values.

use acvus_utils::Interner;

use crate::error::ExternError;
use crate::runtime::Runtime;

pub trait FromValue<R: Runtime>: Sized {
    fn from_value(value: R::Value, interner: &Interner) -> Result<Self, R::Error>;
}

pub trait IntoValue<R: Runtime> {
    fn into_value(self, interner: &Interner) -> R::Value;
}

/// A tuple of Rust values into the values a call gives back for the
/// places it borrowed, in order.
pub trait IntoValues<R: Runtime> {
    fn into_values(self, interner: &Interner) -> Vec<R::Value>;
}

impl<R: Runtime> IntoValues<R> for () {
    fn into_values(self, _: &Interner) -> Vec<R::Value> {
        Vec::new()
    }
}

/// Call arguments into a tuple of Rust values.
pub trait FromValues<R: Runtime>: Sized {
    fn from_values(values: Vec<R::Value>, interner: &Interner) -> Result<Self, R::Error>;
}

macro_rules! impl_scalar {
    ($T:ty, $build:ident, $open:ident) => {
        impl<R: Runtime> FromValue<R> for $T {
            fn from_value(value: R::Value, _: &Interner) -> Result<Self, R::Error> {
                R::$open(value)
            }
        }

        impl<R: Runtime> IntoValue<R> for $T {
            fn into_value(self, _: &Interner) -> R::Value {
                R::$build(self)
            }
        }
    };
}

impl_scalar!(i64, int, into_int);
impl_scalar!(f64, float, into_float);
impl_scalar!(bool, bool, into_bool);
impl_scalar!(u8, byte, into_byte);
impl_scalar!(String, string, into_string);

impl<R: Runtime> FromValue<R> for () {
    fn from_value(value: R::Value, _: &Interner) -> Result<Self, R::Error> {
        R::into_unit(value)
    }
}

impl<R: Runtime> IntoValue<R> for () {
    fn into_value(self, _: &Interner) -> R::Value {
        R::unit()
    }
}

impl<R, T> FromValue<R> for Option<T>
where
    R: Runtime,
    T: FromValue<R>,
{
    fn from_value(value: R::Value, interner: &Interner) -> Result<Self, R::Error> {
        match R::into_option(interner, value)? {
            Some(inner) => Ok(Some(T::from_value(inner, interner)?)),
            None => Ok(None),
        }
    }
}

impl<R, T> IntoValue<R> for Option<T>
where
    R: Runtime,
    T: IntoValue<R>,
{
    fn into_value(self, interner: &Interner) -> R::Value {
        match self {
            Some(v) => R::some(interner, v.into_value(interner)),
            None => R::none(interner),
        }
    }
}

impl<R, T, const N: usize> FromValue<R> for [T; N]
where
    R: Runtime,
    T: FromValue<R>,
{
    fn from_value(value: R::Value, interner: &Interner) -> Result<Self, R::Error> {
        let items = R::into_array(value)?;
        if items.len() != N {
            return Err(ExternError::internal(format!(
                "array of length {N} expected, got {}",
                items.len()
            ))
            .into());
        }
        let mut out = Vec::with_capacity(N);
        for item in items {
            out.push(T::from_value(item, interner)?);
        }
        out.try_into()
            .map_err(|_| ExternError::internal("array length changed during conversion").into())
    }
}

impl<R, T, const N: usize> IntoValue<R> for [T; N]
where
    R: Runtime,
    T: IntoValue<R>,
{
    fn into_value(self, interner: &Interner) -> R::Value {
        R::array(self.into_iter().map(|v| v.into_value(interner)).collect())
    }
}

impl<R: Runtime> FromValues<R> for () {
    fn from_values(values: Vec<R::Value>, _: &Interner) -> Result<Self, R::Error> {
        if !values.is_empty() {
            return Err(ExternError::internal(format!(
                "expected 0 arguments, got {}",
                values.len()
            ))
            .into());
        }
        Ok(())
    }
}

macro_rules! impl_tuple {
    ($n:literal; $($T:ident : $idx:tt),+) => {
        impl<R, $($T),+> IntoValues<R> for ($($T,)+)
        where
            R: Runtime,
            $($T: IntoValue<R>,)+
        {
            fn into_values(self, interner: &Interner) -> Vec<R::Value> {
                vec![$( self.$idx.into_value(interner), )+]
            }
        }

        impl<R, $($T),+> FromValues<R> for ($($T,)+)
        where
            R: Runtime,
            $($T: FromValue<R>,)+
        {
            fn from_values(values: Vec<R::Value>, interner: &Interner) -> Result<Self, R::Error> {
                if values.len() != $n {
                    return Err(ExternError::internal(format!(
                        "expected {} arguments, got {}", $n, values.len()
                    )).into());
                }
                let mut iter = values.into_iter();
                Ok(($( $T::from_value(iter.next().expect("length checked"), interner)?, )+))
            }
        }

        impl<R, $($T),+> FromValue<R> for ($($T,)+)
        where
            R: Runtime,
            $($T: FromValue<R>,)+
        {
            fn from_value(value: R::Value, interner: &Interner) -> Result<Self, R::Error> {
                let items = R::into_tuple(value)?;
                if items.len() != $n {
                    return Err(ExternError::internal(format!(
                        "tuple of arity {} expected, got {}", $n, items.len()
                    )).into());
                }
                let mut iter = items.into_iter();
                Ok(($( $T::from_value(iter.next().expect("length checked"), interner)?, )+))
            }
        }

        impl<R, $($T),+> IntoValue<R> for ($($T,)+)
        where
            R: Runtime,
            $($T: IntoValue<R>,)+
        {
            fn into_value(self, interner: &Interner) -> R::Value {
                R::tuple(vec![$(self.$idx.into_value(interner),)+])
            }
        }
    };
}

impl_tuple!(1; T0: 0);
impl_tuple!(2; T0: 0, T1: 1);
impl_tuple!(3; T0: 0, T1: 1, T2: 2);
impl_tuple!(4; T0: 0, T1: 1, T2: 2, T3: 3);
impl_tuple!(5; T0: 0, T1: 1, T2: 2, T3: 3, T4: 4);
impl_tuple!(6; T0: 0, T1: 1, T2: 2, T3: 3, T4: 4, T5: 5);
