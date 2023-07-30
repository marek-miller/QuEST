#[cfg(not(feature = "f32"))]
mod _precision {
    #[allow(non_camel_case_types)]
    pub type qreal = std::ffi::c_double;
    pub type Qreal = f64;
    pub use std::f64::consts::{
        LN_10,
        LN_2,
        PI,
        SQRT_2,
        TAU,
    };
    /// Machine epsilon value for [`Qreal`](crate::Qreal)
    pub const EPSILON: Qreal = f64::EPSILON;
}

#[cfg(feature = "f32")]
mod _precision {
    #[allow(non_camel_case_types)]
    pub type qreal = std::ffi::c_float;
    pub type Qreal = f32;
    pub use std::f32::consts::{
        LN_10,
        LN_2,
        PI,
        SQRT_2,
        TAU,
    };
    /// Machine epsilon value for [`Qreal`](crate::Qreal)
    pub const EPSILON: Qreal = f32::EPSILON;
}

pub use _precision::{
    qreal,
    Qreal,
    EPSILON,
    LN_10,
    LN_2,
    PI,
    SQRT_2,
    TAU,
};

use crate::ffi;

pub type Qcomplex = num::Complex<Qreal>;

impl From<Qcomplex> for ffi::Complex {
    fn from(value: Qcomplex) -> Self {
        ffi::Complex {
            real: value.re,
            imag: value.im,
        }
    }
}

impl From<ffi::Complex> for Qcomplex {
    fn from(value: ffi::Complex) -> Self {
        Self::new(value.real, value.imag)
    }
}
