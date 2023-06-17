#![allow(clippy::missing_errors_doc)]

use std::ffi::CString;

use exceptions::catch_quest_exception;

mod exceptions;
mod ffi;

#[cfg(feature = "f32")]
pub use std::f32::consts::PI;
#[cfg(feature = "f32")]
pub use std::f32::consts::SQRT_2;
#[cfg(feature = "f32")]
pub use std::f32::EPSILON;
#[cfg(not(feature = "f32"))]
pub use std::f64::consts::PI;
#[cfg(not(feature = "f32"))]
pub use std::f64::consts::SQRT_2;
#[cfg(not(feature = "f32"))]
pub use std::f64::EPSILON;

pub use ffi::{
    bitEncoding as BitEncoding,
    pauliOpType as PauliOpType,
    phaseFunc as PhaseFunc,
    phaseGateType as PhaseGateType,
};

#[cfg(not(feature = "f32"))]
pub type Qreal = f64;

#[cfg(feature = "f32")]
pub type Qreal = f32;

#[derive(Debug, PartialEq, Clone)]
pub enum QuestError {
    /// An exception thrown by the C library.  From QuEST documentation:
    ///
    /// > An internal function is called when invalid arguments are passed to a
    /// > QuEST API call, which the user can optionally override by
    /// > redefining. This function is a weak symbol, so that users can
    /// > choose how input errors are handled, by redefining it in their own
    /// > code. Users must ensure that the triggered API call
    /// > does not continue (e.g. the user exits or throws an exception), else
    /// > QuEST will continue with the valid input and likely trigger a
    /// > seg-fault. This function is triggered before any internal
    /// > state-change, hence it is safe to interrupt with exceptions.
    ///
    /// See also [`invalidQuESTInputError()`][1].
    ///
    /// [1]: https://quest-kit.github.io/QuEST/group__debug.html#ga51a64b05d31ef9bcf6a63ce26c0092db
    InvalidQuESTInputError {
        err_msg:  String,
        err_func: String,
    },
    NulError(std::ffi::NulError),
    IntoStringError(std::ffi::IntoStringError),
    ArrayLengthError,
    QubitIndexError,
}

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

#[derive(Debug, Clone, Copy)]
pub struct ComplexMatrix2(ffi::ComplexMatrix2);

impl ComplexMatrix2 {
    #[must_use]
    pub fn new(
        real: [[Qreal; 2]; 2],
        imag: [[Qreal; 2]; 2],
    ) -> Self {
        Self(ffi::ComplexMatrix2 {
            real,
            imag,
        })
    }
}

#[derive(Debug)]
pub struct ComplexMatrix4(ffi::ComplexMatrix4);

impl ComplexMatrix4 {
    #[must_use]
    pub fn new(
        real: [[Qreal; 4]; 4],
        imag: [[Qreal; 4]; 4],
    ) -> Self {
        Self(ffi::ComplexMatrix4 {
            real,
            imag,
        })
    }
}

#[derive(Debug)]
pub struct ComplexMatrixN(ffi::ComplexMatrixN);

impl ComplexMatrixN {
    /// Allocate dynamic memory for a square complex matrix of any size.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let mtr = ComplexMatrixN::try_new(3).unwrap();
    /// ```
    ///
    /// See [QuEST API][1] for more information.
    ///
    /// # Errors
    ///
    /// Returns [`QuestError::InvalidQuESTInputError`](crate::QuestError::InvalidQuESTInputError)
    /// on failure.  This is an exception thrown by `QuEST`.
    ///
    /// [1]: https://quest-kit.github.io/QuEST/modules.html
    pub fn try_new(num_qubits: i32) -> Result<Self, QuestError> {
        catch_quest_exception(|| {
            Self(unsafe { ffi::createComplexMatrixN(num_qubits) })
        })
    }
}

impl Drop for ComplexMatrixN {
    fn drop(&mut self) {
        catch_quest_exception(|| unsafe { ffi::destroyComplexMatrixN(self.0) })
            .unwrap();
    }
}

#[derive(Debug)]
pub struct Vector(ffi::Vector);

impl Vector {
    #[must_use]
    pub fn new(
        x: Qreal,
        y: Qreal,
        z: Qreal,
    ) -> Self {
        Self(ffi::Vector {
            x,
            y,
            z,
        })
    }
}

#[derive(Debug)]
pub struct PauliHamil(ffi::PauliHamil);

impl PauliHamil {
    /// Dynamically allocates a Hamiltonian
    ///
    /// The Hamiltonian is expressed as a real-weighted sum of products of
    /// Pauli operators.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let hamil = PauliHamil::try_new(2, 3).unwrap();
    /// ```
    ///
    /// See [QuEST API][1] for more information.
    ///
    /// # Errors
    ///
    /// Returns [`QuestError::InvalidQuESTInputError`](crate::QuestError::InvalidQuESTInputError) on
    /// failure. This is an exception thrown by `QuEST`.
    ///
    /// [1]: https://quest-kit.github.io/QuEST/modules.html
    pub fn try_new(
        num_qubits: i32,
        num_sum_terms: i32,
    ) -> Result<Self, QuestError> {
        catch_quest_exception(|| {
            Self(unsafe { ffi::createPauliHamil(num_qubits, num_sum_terms) })
        })
    }

    /// Creates a [`PauliHamil`] instance
    /// populated with the data in filename `fn_`.
    ///
    /// # Bugs
    ///
    /// This function calls its C equivalent which unfortunately behaves
    /// erratically when the file specified is incorrectly formatted or
    /// inaccessible, often leading to seg-faults.  Use at your own risk.
    pub fn try_new_from_file(fn_: &str) -> Result<Self, QuestError> {
        let filename = CString::new(fn_).map_err(QuestError::NulError)?;
        catch_quest_exception(|| {
            Self(unsafe { ffi::createPauliHamilFromFile((*filename).as_ptr()) })
        })
    }
}

impl Drop for PauliHamil {
    fn drop(&mut self) {
        unsafe { ffi::destroyPauliHamil(self.0) }
    }
}

#[derive(Debug)]
pub struct DiagonalOp<'a> {
    env: &'a QuestEnv,
    op:  ffi::DiagonalOp,
}

impl<'a> DiagonalOp<'a> {
    pub fn try_new(
        num_qubits: i32,
        env: &'a QuestEnv,
    ) -> Result<Self, QuestError> {
        Ok(Self {
            env,
            op: catch_quest_exception(|| unsafe {
                ffi::createDiagonalOp(num_qubits, env.0)
            })?,
        })
    }

    pub fn try_new_from_file(
        fn_: &str,
        env: &'a QuestEnv,
    ) -> Result<Self, QuestError> {
        let filename = CString::new(fn_).map_err(QuestError::NulError)?;

        Ok(Self {
            env,
            op: catch_quest_exception(|| unsafe {
                ffi::createDiagonalOpFromPauliHamilFile(
                    (*filename).as_ptr(),
                    env.0,
                )
            })?,
        })
    }
}

impl<'a> Drop for DiagonalOp<'a> {
    fn drop(&mut self) {
        unsafe {
            ffi::destroyDiagonalOp(self.op, self.env.0);
        }
    }
}

#[derive(Debug)]
pub struct Qureg<'a> {
    env: &'a QuestEnv,
    reg: ffi::Qureg,
}

impl<'a> Qureg<'a> {
    /// Creates a state-vector Qureg object.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = Qureg::try_new(2, env).unwrap();
    /// ```
    ///
    /// See [QuEST API][1] for more information.
    ///
    /// # Errors
    ///
    /// Returns [`QuestError::InvalidQuESTInputError`](crate::QuestError::InvalidQuESTInputError)
    /// on failure.  This is an exception thrown by `QuEST`.
    ///
    /// [1]: https://quest-kit.github.io/QuEST/modules.html
    pub fn try_new(
        num_qubits: i32,
        env: &'a QuestEnv,
    ) -> Result<Self, QuestError> {
        Ok(Self {
            env,
            reg: catch_quest_exception(|| unsafe {
                ffi::createQureg(num_qubits, env.0)
            })?,
        })
    }

    ///  Creates a density matrix Qureg object.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = Qureg::try_new_density(2, env).unwrap();
    /// ```
    ///
    /// See [QuEST API][1] for more information.
    ///
    /// # Errors
    ///
    /// Returns [`QuestError::InvalidQuESTInputError`](crate::QuestError::InvalidQuESTInputError)
    /// on failure.  This is an exception thrown by `QuEST`.
    ///
    /// [1]: https://quest-kit.github.io/QuEST/modules.html
    pub fn try_new_density(
        num_qubits: i32,
        env: &'a QuestEnv,
    ) -> Result<Self, QuestError> {
        Ok(Self {
            env,
            reg: catch_quest_exception(|| unsafe {
                ffi::createDensityQureg(num_qubits, env.0)
            })?,
        })
    }

    #[must_use]
    pub fn is_density_matrix(&self) -> bool {
        self.reg.isDensityMatrix != 0
    }

    #[must_use]
    pub fn num_qubits_represented(&self) -> i32 {
        self.reg.numQubitsRepresented
    }
}

impl<'a> Drop for Qureg<'a> {
    fn drop(&mut self) {
        unsafe { ffi::destroyQureg(self.reg, self.env.0) };
    }
}

#[derive(Debug)]
pub struct QuestEnv(ffi::QuESTEnv);

impl QuestEnv {
    #[must_use]
    pub fn new() -> Self {
        Self(unsafe { ffi::createQuESTEnv() })
    }

    pub fn sync(&self) {
        unsafe {
            ffi::syncQuESTEnv(self.0);
        }
    }
}

impl Default for QuestEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for QuestEnv {
    fn drop(&mut self) {
        unsafe { ffi::destroyQuESTEnv(self.0) }
    }
}

/// Initialises a `ComplexMatrixN` instance to have the passed
/// `real` and `imag` values.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let mtr = &mut ComplexMatrixN::try_new(2).unwrap();
/// init_complex_matrix_n(
///     mtr,
///     &[&[1., 2.], &[3., 4.]],
///     &[&[5., 6.], &[7., 8.]],
/// )
/// .unwrap();
/// ```
///
/// See [QuEST API][1] for more information.
///
/// # Errors
///
/// Returns [`Error::ArrayLengthError`](crate::QuestError::ArrayLengthError), if
/// either `real` or `imag` is not a square array of dimension equal to the
/// number of qubits in `m`.  Otherwise, returns
/// [`QuestError::InvalidQuESTInputError`](crate::QuestError::InvalidQuESTInputError) on
/// failure. This is an exception thrown by `QuEST`.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::cast_sign_loss)]
pub fn init_complex_matrix_n(
    m: &mut ComplexMatrixN,
    real: &[&[Qreal]],
    imag: &[&[Qreal]],
) -> Result<(), QuestError> {
    let n = m.0.numQubits as usize;

    if real.len() < n || imag.len() < n {
        return Err(QuestError::ArrayLengthError);
    }
    for i in 0..n {
        if real[i].len() < n || imag[i].len() < n {
            return Err(QuestError::ArrayLengthError);
        }
    }

    let mut real_ptrs = Vec::with_capacity(n);
    let mut imag_ptrs = Vec::with_capacity(n);
    catch_quest_exception(|| unsafe {
        for i in 0..n {
            real_ptrs.push(real[i].as_ptr());
            imag_ptrs.push(imag[i].as_ptr());
        }

        ffi::initComplexMatrixN(m.0, real_ptrs.as_ptr(), imag_ptrs.as_ptr());
    })
}

/// Initialize [`PauliHamil`](crate::PauliHamil) instance with the given term
/// coefficients
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// use quest_bind::PauliOpType::*;
///
/// let hamil = &mut PauliHamil::try_new(2, 2).unwrap();
///
/// init_pauli_hamil(
///     hamil,
///     &[0.5, -0.5],
///     &[PAULI_X, PAULI_Y, PAULI_I, PAULI_I, PAULI_Z, PAULI_X],
/// )
/// .unwrap();
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn init_pauli_hamil(
    hamil: &mut PauliHamil,
    coeffs: &[Qreal],
    codes: &[PauliOpType],
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::initPauliHamil(hamil.0, coeffs.as_ptr(), codes.as_ptr());
    })
}

/// Update the GPU memory with the current values in `op`.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let op = &mut DiagonalOp::try_new(1, env).unwrap();
///
/// sync_diagonal_op(op).unwrap();
/// ```
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn sync_diagonal_op(op: &mut DiagonalOp) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::syncDiagonalOp(op.op);
    })
}

/// Overwrites the entire `DiagonalOp` with the given elements.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let mut op = &mut DiagonalOp::try_new(2, env).unwrap();
///
/// let real = &[1., 2., 3., 4.];
/// let imag = &[5., 6., 7., 8.];
/// init_diagonal_op(op, real, imag);
/// ```
/// See [QuEST API][1] for more information.
///
/// # Panics
///
/// This function will panic, if either `real` or `imag`
/// have length smaller than `2.pow(num_qubits)`.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::cast_sign_loss)]
pub fn init_diagonal_op(
    op: &mut DiagonalOp,
    real: &[Qreal],
    imag: &[Qreal],
) -> Result<(), QuestError> {
    let len_required = 2usize.pow(op.op.numQubits as u32);
    assert!(real.len() >= len_required);
    assert!(imag.len() >= len_required);
    catch_quest_exception(|| unsafe {
        ffi::initDiagonalOp(op.op, real.as_ptr(), imag.as_ptr());
    })
}

/// Populates the diagonal operator \p op to be equivalent to the given Pauli
/// Hamiltonian
///
/// Assuming `hamil` contains only `PAULI_I` or `PAULI_Z` operators.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// use quest_bind::PauliOpType::*;
///
/// let hamil = &mut PauliHamil::try_new(2, 2).unwrap();
/// init_pauli_hamil(
///     hamil,
///     &[0.5, -0.5],
///     &[PAULI_I, PAULI_Z, PAULI_Z, PAULI_Z],
/// )
/// .unwrap();
///
/// let env = &QuestEnv::new();
/// let mut op = &mut DiagonalOp::try_new(2, env).unwrap();
///
/// init_diagonal_op_from_pauli_hamil(op, hamil).unwrap();
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn init_diagonal_op_from_pauli_hamil(
    op: &mut DiagonalOp,
    hamil: &PauliHamil,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::initDiagonalOpFromPauliHamil(op.op, hamil.0);
    })
}

/// Modifies a subset of elements of `DiagonalOp`.
///
/// Starting at index `start_ind`, and ending at index
/// `start_ind +  num_elems`.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let op = &mut DiagonalOp::try_new(3, env).unwrap();
///
/// let num_elems = 4;
/// let re = &[1., 2., 3., 4.];
/// let im = &[1., 2., 3., 4.];
/// set_diagonal_op_elems(op, 0, re, im, num_elems).unwrap();
/// ```
///
/// # Panics
///
/// This function will panic if either
/// `real.len() >= num_elems as usize`, or
/// `imag.len() >= num_elems as usize`.
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_possible_truncation)]
pub fn set_diagonal_op_elems(
    op: &mut DiagonalOp,
    start_ind: i64,
    real: &[Qreal],
    imag: &[Qreal],
    num_elems: i64,
) -> Result<(), QuestError> {
    assert!(real.len() >= num_elems as usize);
    assert!(imag.len() >= num_elems as usize);

    catch_quest_exception(|| unsafe {
        ffi::setDiagonalOpElems(
            op.op,
            start_ind,
            real.as_ptr(),
            imag.as_ptr(),
            num_elems,
        );
    })
}

/// Apply a diagonal operator to the entire `qureg`.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// let op = &mut DiagonalOp::try_new(2, env).unwrap();
///
/// init_diagonal_op(op, &[1., 2., 3., 4.], &[5., 6., 7., 8.]).unwrap();
/// apply_diagonal_op(qureg, &op).unwrap();
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_diagonal_op(
    qureg: &mut Qureg,
    op: &DiagonalOp,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::applyDiagonalOp(qureg.reg, op.op);
    })
}

/// Computes the expected value of the diagonal operator `op`.
///
/// Since `op` is not necessarily Hermitian, the expected value may be a complex
/// number.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// let op = &mut DiagonalOp::try_new(2, env).unwrap();
///
/// init_zero_state(qureg);
/// init_diagonal_op(op, &[1., 2., 3., 4.], &[5., 6., 7., 8.]).unwrap();
///
/// let expec_val = calc_expec_diagonal_op(qureg, op).unwrap();
///
/// assert!((expec_val.re - 1.).abs() < EPSILON);
/// assert!((expec_val.im - 5.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn calc_expec_diagonal_op(
    qureg: &Qureg,
    op: &DiagonalOp,
) -> Result<Qcomplex, QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::calcExpecDiagonalOp(qureg.reg, op.op)
    })
    .map(Into::into)
}

pub fn report_state(qureg: &Qureg) {
    unsafe { ffi::reportState(qureg.reg) }
}

pub fn report_state_to_screen(
    qureg: &Qureg,
    env: &QuestEnv,
    report_rank: i32,
) {
    unsafe { ffi::reportStateToScreen(qureg.reg, env.0, report_rank) }
}

pub fn report_qureg_params(qureg: &Qureg) {
    unsafe {
        ffi::reportQuregParams(qureg.reg);
    }
}

pub fn report_pauli_hamil(hamil: &PauliHamil) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::reportPauliHamil(hamil.0);
    })
}

/// Returns the number of qubits represented.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &Qureg::try_new(3, env).unwrap();
///
/// assert_eq!(get_num_qubits(qureg), 3);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
#[must_use]
pub fn get_num_qubits(qureg: &Qureg) -> i32 {
    unsafe { ffi::getNumQubits(qureg.reg) }
}

/// Returns the number of complex amplitudes in a state-vector.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &Qureg::try_new(3, env).unwrap();
///
/// assert_eq!(get_num_amps(qureg).unwrap(), 8);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn get_num_amps(qureg: &Qureg) -> Result<i64, QuestError> {
    catch_quest_exception(|| unsafe { ffi::getNumAmps(qureg.reg) })
}

/// Initializes a `qureg` to have all-zero-amplitudes.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
///
/// init_blank_state(qureg);
///
/// assert!(get_prob_amp(qureg, 0).unwrap().abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn init_blank_state(qureg: &mut Qureg) {
    unsafe {
        ffi::initBlankState(qureg.reg);
    }
}

/// Initialize `qureg` into the zero state.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
///
/// init_zero_state(qureg);
///
/// assert!((get_prob_amp(qureg, 0).unwrap() - 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn init_zero_state(qureg: &mut Qureg) {
    unsafe {
        ffi::initZeroState(qureg.reg);
    }
}

/// Initialize `qureg` into the plus state.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
///
/// init_plus_state(qureg);
/// let prob = get_prob_amp(qureg, 0).unwrap();
///
/// assert!((prob - 0.125).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn init_plus_state(qureg: &mut Qureg) {
    unsafe {
        ffi::initPlusState(qureg.reg);
    }
}

/// Initialize `qureg` into a classical state.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
///
/// init_classical_state(qureg, 8);
/// let prob = get_prob_amp(qureg, 0).unwrap();
///
/// assert!(prob.abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn init_classical_state(
    qureg: &mut Qureg,
    state_ind: i64,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::initClassicalState(qureg.reg, state_ind);
    })
}

/// Initialize `qureg` into a pure state.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new_density(3, env).unwrap();
/// let pure_state = &mut Qureg::try_new(3, env).unwrap();
///
/// init_zero_state(pure_state);
/// init_pure_state(qureg, pure_state).unwrap();
///
/// assert!((calc_purity(qureg).unwrap() - 1.0).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn init_pure_state(
    qureg: &mut Qureg,
    pure_: &Qureg,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::initPureState(qureg.reg, pure_.reg);
    })
}

/// Initializes `qureg` to be in the debug state.
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn init_debug_state(qureg: &mut Qureg) {
    unsafe {
        ffi::initDebugState(qureg.reg);
    }
}

/// Initialize `qureg` by specifying all amplitudes.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
///
/// init_state_from_amps(qureg, &[1., 0., 0., 0.], &[0., 0., 0., 0.]);
/// let prob = get_prob_amp(qureg, 0).unwrap();
///
/// assert!((prob - 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn init_state_from_amps(
    qureg: &mut Qureg,
    reals: &[Qreal],
    imags: &[Qreal],
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::initStateFromAmps(qureg.reg, reals.as_ptr(), imags.as_ptr());
    })
}

/// Overwrites a contiguous subset of the amplitudes in state-vector `qureg`.
///
/// In distributed mode, this function assumes the subset `reals` and `imags`
/// exist (at least) on the node containing the ultimately updated elements.
///
/// # Examples
///
/// Below is the correct way to modify the full 8 elements of `qureg`when split
/// between 2 nodes.
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
///
/// let re = &mut [1., 2., 3., 4.];
/// let im = &mut [1., 2., 3., 4.];
/// let num_amps = 4;
///
/// set_amps(qureg, 0, re, im, num_amps);
///
/// // modify re and im to the next set of elements
/// for i in 0..4 {
///     re[i] += 4.;
///     im[i] += 4.;
/// }
/// set_amps(qureg, 4, re, im, num_amps);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn set_amps(
    qureg: &mut Qureg,
    start_ind: i64,
    reals: &[Qreal],
    imags: &[Qreal],
    num_amps: i64,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::setAmps(
            qureg.reg,
            start_ind,
            reals.as_ptr(),
            imags.as_ptr(),
            num_amps,
        );
    })
}

/// Overwrites a contiguous subset of the amplitudes in density-matrix `qureg`.
///
/// In distributed mode, this function assumes the subset `reals` and `imags`
/// exist (at least) on the node containing the ultimately updated elements.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new_density(3, env).unwrap();
///
/// let mut re = &[1., 2., 3., 4.];
/// let mut im = &[1., 2., 3., 4.];
/// let num_amps = 4;
///
/// set_density_amps(qureg, 0, 0, re, im, num_amps);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn set_density_amps(
    qureg: &mut Qureg,
    start_row: i64,
    start_col: i64,
    reals: &[Qreal],
    imags: &[Qreal],
    num_amps: i64,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::setDensityAmps(
            qureg.reg,
            start_row,
            start_col,
            reals.as_ptr(),
            imags.as_ptr(),
            num_amps,
        );
    })
}

/// Overwrite the amplitudes of `target_qureg` with those from `copy_qureg`.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let target_qureg = &mut Qureg::try_new(3, env).unwrap();
/// let copy_qureg = &Qureg::try_new(3, env).unwrap();
///
/// clone_qureg(target_qureg, copy_qureg);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn clone_qureg(
    target_qureg: &mut Qureg,
    copy_qureg: &Qureg,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::cloneQureg(target_qureg.reg, copy_qureg.reg);
    })
}

/// Shift the phase between `|0>` and `|1>` of a single qubit by a given angle.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
///
/// let target_qubit = 1;
/// let angle = 0.5;
///
/// phase_shift(qureg, target_qubit, angle).unwrap();
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn phase_shift(
    qureg: &mut Qureg,
    target_quibit: i32,
    angle: Qreal,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::phaseShift(qureg.reg, target_quibit, angle);
    })
}

/// Introduce a phase factor on state of qubits.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
///
/// let id_qubit1 = 0;
/// let id_qubit2 = 2;
/// let angle = 0.5;
/// controlled_phase_shift(qureg, id_qubit1, id_qubit2, angle).unwrap();
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn controlled_phase_shift(
    qureg: &mut Qureg,
    id_qubit1: i32,
    id_qubit2: i32,
    angle: Qreal,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::controlledPhaseShift(qureg.reg, id_qubit1, id_qubit2, angle);
    })
}

/// Introduce a phase factor of the passed qubits.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(4, env).unwrap();
///
/// let control_qubits = &[0, 1, 3];
/// let num_control_qubits = control_qubits.len() as i32;
/// let angle = 0.5;
///
/// multi_controlled_phase_shift(
///     qureg,
///     control_qubits,
///     num_control_qubits,
///     angle,
/// )
/// .unwrap();
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_controlled_phase_shift(
    qureg: &mut Qureg,
    control_qubits: &[i32],
    num_control_qubits: i32,
    angle: Qreal,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::multiControlledPhaseShift(
            qureg.reg,
            control_qubits.as_ptr(),
            num_control_qubits,
            angle,
        );
    })
}

/// Apply the (two-qubit) controlled phase flip gate
///
/// Also known as the controlled pauliZ gate.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(qureg);
///
/// controlled_phase_flip(qureg, 0, 1);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn controlled_phase_flip(
    qureg: &mut Qureg,
    id_qubit1: i32,
    id_qubit2: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::controlledPhaseFlip(qureg.reg, id_qubit1, id_qubit2);
    })
}

/// Apply the (multiple-qubit) controlled phase flip gate.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(4, env).unwrap();
/// init_zero_state(qureg);
///
/// let control_qubits = &[0, 1, 3];
/// multi_controlled_phase_flip(qureg, control_qubits);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_controlled_phase_flip(
    qureg: &mut Qureg,
    control_qubits: &[i32],
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::multiControlledPhaseFlip(
            qureg.reg,
            control_qubits.as_ptr(),
            control_qubits.len() as i32,
        );
    })
}

/// Apply the single-qubit S gate.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(qureg);
/// pauli_x(qureg, 0).unwrap();
///
/// s_gate(qureg, 0).unwrap();
///
/// let amp = get_imag_amp(qureg, 1).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn s_gate(
    qureg: &mut Qureg,
    target_qubit: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::sGate(qureg.reg, target_qubit);
    })
}

/// Apply the single-qubit T gate.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(qureg);
/// pauli_x(qureg, 0).unwrap();
///
/// t_gate(qureg, 0).unwrap();
///
/// let amp = get_imag_amp(qureg, 1).unwrap();
/// assert!((amp - SQRT_2 / 2.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn t_gate(
    qureg: &mut Qureg,
    target_qubit: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::tGate(qureg.reg, target_qubit);
    })
}

/// Performs a logical AND on all successCodes held by all processes.
///
/// If any one process has a zero `success_code`, all processes will return a
/// zero success code.
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
#[must_use]
pub fn sync_quest_success(success_code: i32) -> i32 {
    unsafe { ffi::syncQuESTSuccess(success_code) }
}

/// Report information about the `QuEST` environment
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn report_quest_env(env: &QuestEnv) {
    unsafe {
        ffi::reportQuESTEnv(env.0);
    }
}

/// Get a string containing information about the runtime environment,
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let env_str = get_environment_string(env).unwrap();
///
/// assert!(env_str.contains("OpenMP="));
/// assert!(env_str.contains("threads="));
/// assert!(env_str.contains("MPI="));
/// assert!(env_str.contains("ranks="));
/// assert!(env_str.contains("CUDA="));
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn get_environment_string(env: &QuestEnv) -> Result<String, QuestError> {
    let mut cstr =
        CString::new("CUDA=x OpenMP=x MPI=x threads=xxxxxxx ranks=xxxxxxx")
            .map_err(QuestError::NulError)?;
    unsafe {
        let cstr_ptr = cstr.into_raw();
        ffi::getEnvironmentString(env.0, cstr_ptr);
        cstr = CString::from_raw(cstr_ptr);
    }
    cstr.into_string().map_err(QuestError::IntoStringError)
}

/// >>>Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn copy_state_to_gpu(qureg: &mut Qureg) {
    unsafe {
        ffi::copyStateToGPU(qureg.reg);
    }
}

/// In GPU mode, this copies the state-vector (or density matrix) from RAM.
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn copy_state_from_gpu(qureg: &mut Qureg) {
    unsafe {
        ffi::copyStateFromGPU(qureg.reg);
    }
}

/// In GPU mode, this copies the state-vector (or density matrix) from GPU
/// memory.
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn copy_substate_to_gpu(
    qureg: &mut Qureg,
    start_ind: i64,
    num_amps: i64,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::copySubstateToGPU(qureg.reg, start_ind, num_amps);
    })
}

/// In GPU mode, this copies a substate of the state-vector (or density matrix)
/// from RAM.
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn copy_substate_from_gpu(
    qureg: &mut Qureg,
    start_ind: i64,
    num_amps: i64,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::copySubstateToGPU(qureg.reg, start_ind, num_amps);
    })
}

/// Get the complex amplitude at a given index in the state vector.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_plus_state(qureg);
///
/// let amp = get_amp(qureg, 0).unwrap().re;
/// assert!((amp - 0.5).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn get_amp(
    qureg: &Qureg,
    index: i64,
) -> Result<Qcomplex, QuestError> {
    catch_quest_exception(|| unsafe { ffi::getAmp(qureg.reg, index) })
        .map(Into::into)
}

/// Get the real component of the complex probability amplitude at an index in
/// the state vector.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_plus_state(qureg);
///
/// let amp = get_real_amp(qureg, 0).unwrap();
/// assert!((amp - 0.5).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn get_real_amp(
    qureg: &Qureg,
    index: i64,
) -> Result<Qreal, QuestError> {
    catch_quest_exception(|| unsafe { ffi::getRealAmp(qureg.reg, index) })
}

/// Get the imaginary component of the complex probability amplitude at an index
/// in the state vector.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_plus_state(qureg);
///
/// let amp = get_imag_amp(qureg, 0).unwrap();
/// assert!(amp.abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn get_imag_amp(
    qureg: &Qureg,
    index: i64,
) -> Result<Qreal, QuestError> {
    catch_quest_exception(|| unsafe { ffi::getImagAmp(qureg.reg, index) })
}

/// Get the probability of a state-vector at an index in the full state vector.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_plus_state(qureg);
///
/// let amp = get_prob_amp(qureg, 0).unwrap();
/// assert!((amp - 0.25).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn get_prob_amp(
    qureg: &Qureg,
    index: i64,
) -> Result<Qreal, QuestError> {
    catch_quest_exception(|| unsafe { ffi::getProbAmp(qureg.reg, index) })
}

/// Get an amplitude from a density matrix at a given row and column.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new_density(2, env).unwrap();
/// init_plus_state(qureg);
///
/// let amp = get_density_amp(qureg, 0, 0).unwrap().re;
/// assert!((amp - 0.25).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn get_density_amp(
    qureg: &Qureg,
    row: i64,
    col: i64,
) -> Result<Qcomplex, QuestError> {
    catch_quest_exception(|| unsafe { ffi::getDensityAmp(qureg.reg, row, col) })
        .map(Into::into)
}

/// A debugging function which calculates the probability of the qubits in
/// `qureg`
///
/// This function should always be 1 for correctly normalised states
/// (hence returning a real number).
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_plus_state(qureg);
///
/// let amp = calc_total_prob(qureg);
/// assert!((amp - 1.).abs() < EPSILON)
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
#[must_use]
pub fn calc_total_prob(qureg: &Qureg) -> Qreal {
    unsafe { ffi::calcTotalProb(qureg.reg) }
}

/// Apply a single-qubit unitary parameterized by two given complex scalars.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(qureg);
///
/// let norm = SQRT_2.recip();
/// let alpha = Qcomplex::new(0., norm);
/// let beta = Qcomplex::new(0., norm);
/// compact_unitary(qureg, 0, alpha, beta).unwrap();
///
/// let other_qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(other_qureg);
/// hadamard(other_qureg, 0).unwrap();
///
/// let fidelity = calc_fidelity(qureg, other_qureg).unwrap();
/// assert!((fidelity - 1.).abs() < 10. * EPSILON,);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn compact_unitary(
    qureg: &mut Qureg,
    target_qubit: i32,
    alpha: Qcomplex,
    beta: Qcomplex,
) -> Result<(), QuestError> {
    // Check if target_qubit is within bounds.  QuEST doesn't and seg-faults
    // sometimes
    if target_qubit >= qureg.num_qubits_represented() {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::compactUnitary(qureg.reg, target_qubit, alpha.into(), beta.into());
    })
}

/// Apply a general single-qubit unitary (including a global phase factor).
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(qureg);
///
/// let norm = SQRT_2.recip();
/// let mtr = ComplexMatrix2::new(
///     [[norm, norm], [norm, -norm]],
///     [[0., 0.], [0., 0.]],
/// );
/// unitary(qureg, 0, &mtr).unwrap();
///
/// let other_qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(other_qureg);
/// hadamard(other_qureg, 0).unwrap();
///
/// let fidelity = calc_fidelity(qureg, other_qureg).unwrap();
/// assert!((fidelity - 1.).abs() < 10. * EPSILON,);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn unitary(
    qureg: &mut Qureg,
    target_qubit: i32,
    u: &ComplexMatrix2,
) -> Result<(), QuestError> {
    // Check if target_qubit is within bounds.  QuEST doesn't and seg-faults
    // sometimes
    if target_qubit >= qureg.num_qubits_represented() {
        return Err(QuestError::QubitIndexError);
    }

    catch_quest_exception(|| unsafe {
        ffi::unitary(qureg.reg, target_qubit, u.0);
    })
}

/// Rotate a single qubit by a given angle around the X-axis of the
/// Bloch-sphere.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
/// let theta = PI;
///
/// rotate_x(qureg, 0, theta).unwrap();
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn rotate_x(
    qureg: &mut Qureg,
    rot_qubit: i32,
    angle: Qreal,
) -> Result<(), QuestError> {
    // Check if target_qubit is within bounds.  QuEST doesn't and seg-faults
    // sometimes
    if rot_qubit >= qureg.num_qubits_represented() {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::rotateX(qureg.reg, rot_qubit, angle);
    })
}

/// Rotate a single qubit by a given angle around the Y-axis of the
/// Bloch-sphere.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
/// let theta = PI;
///
/// rotate_y(qureg, 0, theta).unwrap();
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn rotate_y(
    qureg: &mut Qureg,
    rot_qubit: i32,
    angle: Qreal,
) -> Result<(), QuestError> {
    // Check if target_qubit is within bounds.  QuEST doesn't and seg-faults
    // sometimes
    if rot_qubit >= qureg.num_qubits_represented() {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::rotateY(qureg.reg, rot_qubit, angle);
    })
}

/// Rotate a single qubit by a given angle around the Z-axis of the
/// Bloch-sphere.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
/// let theta = PI;
///
/// rotate_z(qureg, 0, theta).unwrap();
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn rotate_z(
    qureg: &mut Qureg,
    rot_qubit: i32,
    angle: Qreal,
) -> Result<(), QuestError> {
    // Check if target_qubit is within bounds.  QuEST doesn't and seg-faults
    // sometimes
    if rot_qubit >= qureg.num_qubits_represented() {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::rotateZ(qureg.reg, rot_qubit, angle);
    })
}

/// Rotate a single qubit by a given angle around a given axis.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
/// init_zero_state(qureg);
///
/// let angle = 2.0 * PI;
/// let axis = &Vector::new(0., 0., 1.);
/// rotate_around_axis(qureg, 0, angle, axis).unwrap();
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn rotate_around_axis(
    qureg: &mut Qureg,
    rot_qubit: i32,
    angle: Qreal,
    axis: &Vector,
) -> Result<(), QuestError> {
    // Check if target_qubit is within bounds.  QuEST doesn't and seg-faults
    // sometimes
    if rot_qubit >= qureg.num_qubits_represented() {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::rotateAroundAxis(qureg.reg, rot_qubit, angle, axis.0);
    })
}

/// Applies a controlled rotation by a given angle around the X-axis of the
/// Bloch-sphere.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
///
/// let control_qubit = 1;
/// let target_qubit = 0;
/// let angle = PI;
/// controlled_rotate_x(qureg, control_qubit, target_qubit, angle).unwrap();
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn controlled_rotate_x(
    qureg: &mut Qureg,
    control_qubit: i32,
    target_qubit: i32,
    angle: Qreal,
) -> Result<(), QuestError> {
    // Check if target_qubit is within bounds.  QuEST doesn't and seg-faults
    // sometimes
    if control_qubit >= qureg.num_qubits_represented()
        || target_qubit >= qureg.num_qubits_represented()
    {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::controlledRotateX(qureg.reg, control_qubit, target_qubit, angle);
    })
}

/// Applies a controlled rotation by a given angle around the Y-axis of the
/// Bloch-sphere.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
///
/// let control_qubit = 1;
/// let target_qubit = 0;
/// let angle = PI;
/// controlled_rotate_y(qureg, control_qubit, target_qubit, angle).unwrap();
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn controlled_rotate_y(
    qureg: &mut Qureg,
    control_qubit: i32,
    target_qubit: i32,
    angle: Qreal,
) -> Result<(), QuestError> {
    // Check if target_qubit is within bounds.  QuEST doesn't and seg-faults
    // sometimes
    if control_qubit >= qureg.num_qubits_represented()
        || target_qubit >= qureg.num_qubits_represented()
    {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::controlledRotateY(qureg.reg, control_qubit, target_qubit, angle);
    })
}

/// Applies a controlled rotation by a given angle around the Z-axis of the
/// Bloch-sphere.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
///
/// let control_qubit = 1;
/// let target_qubit = 0;
/// let angle = PI;
/// controlled_rotate_z(qureg, control_qubit, target_qubit, angle).unwrap();
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn controlled_rotate_z(
    qureg: &mut Qureg,
    control_qubit: i32,
    target_qubit: i32,
    angle: Qreal,
) -> Result<(), QuestError> {
    // Check if target_qubit is within bounds.  QuEST doesn't and seg-faults
    // sometimes
    if control_qubit >= qureg.num_qubits_represented()
        || target_qubit >= qureg.num_qubits_represented()
    {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::controlledRotateZ(qureg.reg, control_qubit, target_qubit, angle);
    })
}

/// Applies a controlled rotation by a given angle around a given vector of the
/// Bloch-sphere.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
///
/// let control_qubit = 1;
/// let target_qubit = 0;
/// let angle = PI;
/// let vector = &Vector::new(0., 0., 1.);
/// controlled_rotate_around_axis(
///     qureg,
///     control_qubit,
///     target_qubit,
///     angle,
///     vector,
/// )
/// .unwrap();
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn controlled_rotate_around_axis(
    qureg: &mut Qureg,
    control_qubit: i32,
    target_qubit: i32,
    angle: Qreal,
    axis: &Vector,
) -> Result<(), QuestError> {
    // Check if target_qubit is within bounds.  QuEST doesn't and seg-faults
    // sometimes
    if control_qubit >= qureg.num_qubits_represented()
        || target_qubit >= qureg.num_qubits_represented()
    {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::controlledRotateAroundAxis(
            qureg.reg,
            control_qubit,
            target_qubit,
            angle,
            axis.0,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn controlled_compact_unitary(
    qureg: &mut Qureg,
    control_qubit: i32,
    target_qubit: i32,
    alpha: Qcomplex,
    beta: Qcomplex,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::controlledCompactUnitary(
            qureg.reg,
            control_qubit,
            target_qubit,
            alpha.into(),
            beta.into(),
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn controlled_unitary(
    qureg: &mut Qureg,
    control_qubit: i32,
    target_qubit: i32,
    u: &ComplexMatrix2,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::controlledUnitary(qureg.reg, control_qubit, target_qubit, u.0);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_controlled_unitary(
    qureg: &mut Qureg,
    control_qubits: &[i32],
    num_control_qubits: i32,
    target_qubit: i32,
    u: &ComplexMatrix2,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::multiControlledUnitary(
            qureg.reg,
            control_qubits.as_ptr(),
            num_control_qubits,
            target_qubit,
            u.0,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn pauli_x(
    qureg: &mut Qureg,
    target_qubit: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::pauliX(qureg.reg, target_qubit);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn pauli_y(
    qureg: &mut Qureg,
    target_qubit: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::pauliY(qureg.reg, target_qubit);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn pauli_z(
    qureg: &mut Qureg,
    target_qubit: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::pauliZ(qureg.reg, target_qubit);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn hadamard(
    qureg: &mut Qureg,
    target_qubit: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::hadamard(qureg.reg, target_qubit);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn controlled_not(
    qureg: &mut Qureg,
    control_qubit: i32,
    target_qubit: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::controlledNot(qureg.reg, control_qubit, target_qubit);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_controlled_multi_qubit_not(
    qureg: &mut Qureg,
    ctrls: &[i32],
    num_ctrls: i32,
    targs: &[i32],
    num_targs: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::multiControlledMultiQubitNot(
            qureg.reg,
            ctrls.as_ptr(),
            num_ctrls,
            targs.as_ptr(),
            num_targs,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_qubit_not(
    qureg: &mut Qureg,
    targs: &[i32],
    num_targs: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        let targs_ptr = targs.as_ptr();
        ffi::multiQubitNot(qureg.reg, targs_ptr, num_targs);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn controlled_pauli_y(
    qureg: &mut Qureg,
    control_qubit: i32,
    target_qubit: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::controlledPauliY(qureg.reg, control_qubit, target_qubit);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn calc_prob_of_outcome(
    qureg: &Qureg,
    measure_qubit: i32,
    outcome: i32,
) -> Result<Qreal, QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::calcProbOfOutcome(qureg.reg, measure_qubit, outcome)
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// # Panics
///
/// This function will panic if
/// `outcome_probs.len() < num_qubits as usize`
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::cast_sign_loss)]
pub fn calc_prob_of_all_outcomes(
    outcome_probs: &mut [Qreal],
    qureg: &Qureg,
    qubits: &[i32],
    num_qubits: i32,
) -> Result<(), QuestError> {
    assert!(outcome_probs.len() >= num_qubits as usize);
    catch_quest_exception(|| unsafe {
        ffi::calcProbOfAllOutcomes(
            outcome_probs.as_mut_ptr(),
            qureg.reg,
            qubits.as_ptr(),
            num_qubits,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn collapse_to_outcome(
    qureg: &mut Qureg,
    measure_qubit: i32,
    outcome: i32,
) -> Result<Qreal, QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::collapseToOutcome(qureg.reg, measure_qubit, outcome)
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn measure(
    qureg: &mut Qureg,
    measure_qubit: i32,
) -> Result<i32, QuestError> {
    catch_quest_exception(|| unsafe { ffi::measure(qureg.reg, measure_qubit) })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn measure_with_stats(
    qureg: &mut Qureg,
    measure_qubit: i32,
    outcome_prob: &mut Qreal,
) -> Result<i32, QuestError> {
    catch_quest_exception(|| unsafe {
        let outcome_prob_ptr = outcome_prob as *mut _;
        ffi::measureWithStats(qureg.reg, measure_qubit, outcome_prob_ptr)
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn calc_inner_product(
    bra: &Qureg,
    ket: &Qureg,
) -> Result<Qcomplex, QuestError> {
    catch_quest_exception(|| unsafe { ffi::calcInnerProduct(bra.reg, ket.reg) })
        .map(Into::into)
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn calc_density_inner_product(
    rho1: &Qureg,
    rho2: &Qureg,
) -> Result<Qreal, QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::calcDensityInnerProduct(rho1.reg, rho2.reg)
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn seed_quest_default(env: &mut QuestEnv) {
    unsafe {
        let env_ptr = std::ptr::addr_of_mut!(env.0);
        ffi::seedQuESTDefault(env_ptr);
    }
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn seed_quest(
    env: &mut QuestEnv,
    seed_array: &[u64],
    num_seeds: i32,
) {
    // QuEST's function signature is `c_ulong`. Let's use u64 for now...
    unsafe {
        let env_ptr = std::ptr::addr_of_mut!(env.0);
        let seed_array_ptr = seed_array.as_ptr();
        ffi::seedQuEST(env_ptr, seed_array_ptr, num_seeds);
    }
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::cast_sign_loss)]
#[must_use]
pub fn get_quest_seeds<'a: 'b, 'b>(env: &'a QuestEnv) -> (&'b mut [u64], i32) {
    unsafe {
        let seeds_ptr = &mut std::ptr::null_mut();
        let mut num_seeds = 0_i32;
        ffi::getQuESTSeeds(env.0, seeds_ptr, &mut num_seeds);

        let seeds =
            std::slice::from_raw_parts_mut(*seeds_ptr, num_seeds as usize);
        (seeds, num_seeds)
    }
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn start_recording_qasm(qureg: &mut Qureg) {
    unsafe {
        ffi::startRecordingQASM(qureg.reg);
    }
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn stop_recording_qasm(qureg: &mut Qureg) {
    unsafe {
        ffi::stopRecordingQASM(qureg.reg);
    }
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn clear_recorded_qasm(qureg: &mut Qureg) {
    unsafe {
        ffi::clearRecordedQASM(qureg.reg);
    }
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn print_recorded_qasm(qureg: &mut Qureg) {
    unsafe {
        ffi::printRecordedQASM(qureg.reg);
    }
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn write_recorded_qasm_to_file(
    qureg: &mut Qureg,
    filename: &str,
) -> Result<(), QuestError> {
    unsafe {
        let filename_cstr =
            CString::new(filename).map_err(QuestError::NulError)?;
        catch_quest_exception(|| {
            ffi::writeRecordedQASMToFile(qureg.reg, (*filename_cstr).as_ptr());
        })
    }
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn mix_dephasing(
    qureg: &mut Qureg,
    target_qubit: i32,
    prob: Qreal,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::mixDephasing(qureg.reg, target_qubit, prob);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn mix_two_qubit_dephasing(
    qureg: &mut Qureg,
    qubit1: i32,
    qubit2: i32,
    prob: Qreal,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::mixTwoQubitDephasing(qureg.reg, qubit1, qubit2, prob);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn mix_depolarising(
    qureg: &mut Qureg,
    target_qubit: i32,
    prob: Qreal,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::mixDepolarising(qureg.reg, target_qubit, prob);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn mix_damping(
    qureg: &mut Qureg,
    target_qubit: i32,
    prob: Qreal,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::mixDamping(qureg.reg, target_qubit, prob);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn mix_two_qubit_depolarising(
    qureg: &mut Qureg,
    qubit1: i32,
    qubit2: i32,
    prob: Qreal,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::mixTwoQubitDepolarising(qureg.reg, qubit1, qubit2, prob);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn mix_pauli(
    qureg: &mut Qureg,
    target_qubit: i32,
    prob_x: Qreal,
    prob_y: Qreal,
    prob_z: Qreal,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::mixPauli(qureg.reg, target_qubit, prob_x, prob_y, prob_z);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn mix_density_matrix(
    combine_qureg: &mut Qureg,
    prob: Qreal,
    other_qureg: &Qureg,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::mixDensityMatrix(combine_qureg.reg, prob, other_qureg.reg);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn calc_purity(qureg: &Qureg) -> Result<Qreal, QuestError> {
    catch_quest_exception(|| unsafe { ffi::calcPurity(qureg.reg) })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn calc_fidelity(
    qureg: &Qureg,
    pure_state: &Qureg,
) -> Result<Qreal, QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::calcFidelity(qureg.reg, pure_state.reg)
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn swap_gate(
    qureg: &mut Qureg,
    qubit1: i32,
    qubit2: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::swapGate(qureg.reg, qubit1, qubit2);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn sqrt_swap_gate(
    qureg: &mut Qureg,
    qb1: i32,
    qb2: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::swapGate(qureg.reg, qb1, qb2);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_state_controlled_unitary(
    qureg: &mut Qureg,
    control_qubits: &[i32],
    control_state: &[i32],
    num_control_qubits: i32,
    target_qubit: i32,
    u: &ComplexMatrix2,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::multiStateControlledUnitary(
            qureg.reg,
            control_qubits.as_ptr(),
            control_state.as_ptr(),
            num_control_qubits,
            target_qubit,
            u.0,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_rotate_z(
    qureg: &mut Qureg,
    qubits: &[i32],
    num_qubits: i32,
    angle: Qreal,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::multiRotateZ(qureg.reg, qubits.as_ptr(), num_qubits, angle);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_rotate_pauli(
    qureg: &mut Qureg,
    target_qubits: &[i32],
    target_paulis: &[PauliOpType],
    num_targets: i32,
    angle: Qreal,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::multiRotatePauli(
            qureg.reg,
            target_qubits.as_ptr(),
            target_paulis.as_ptr(),
            num_targets,
            angle,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_controlled_multi_rotate_z(
    qureg: &mut Qureg,
    control_qubits: &[i32],
    num_controls: i32,
    target_qubits: &[i32],
    num_targets: i32,
    angle: Qreal,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::multiControlledMultiRotateZ(
            qureg.reg,
            control_qubits.as_ptr(),
            num_controls,
            target_qubits.as_ptr(),
            num_targets,
            angle,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_controlled_multi_rotate_pauli(
    qureg: &mut Qureg,
    control_qubits: &[i32],
    num_controls: i32,
    target_qubits: &[i32],
    target_paulis: &[PauliOpType],
    num_targets: i32,
    angle: Qreal,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::multiControlledMultiRotatePauli(
            qureg.reg,
            control_qubits.as_ptr(),
            num_controls,
            target_qubits.as_ptr(),
            target_paulis.as_ptr(),
            num_targets,
            angle,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn calc_expec_pauli_prod(
    qureg: &Qureg,
    target_qubits: &[i32],
    pauli_codes: &[PauliOpType],
    num_targets: i32,
    workspace: &mut Qureg,
) -> Result<Qreal, QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::calcExpecPauliProd(
            qureg.reg,
            target_qubits.as_ptr(),
            pauli_codes.as_ptr(),
            num_targets,
            workspace.reg,
        )
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn calc_expec_pauli_sum(
    qureg: &Qureg,
    all_pauli_codes: &[PauliOpType],
    term_coeffs: &[Qreal],
    num_sum_terms: i32,
    workspace: &mut Qureg,
) -> Result<Qreal, QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::calcExpecPauliSum(
            qureg.reg,
            all_pauli_codes.as_ptr(),
            term_coeffs.as_ptr(),
            num_sum_terms,
            workspace.reg,
        )
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn calc_expec_pauli_hamil(
    qureg: &Qureg,
    hamil: &PauliHamil,
    workspace: &mut Qureg,
) -> Result<Qreal, QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::calcExpecPauliHamil(qureg.reg, hamil.0, workspace.reg)
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn two_qubit_unitary(
    qureg: &mut Qureg,
    target_qubit1: i32,
    target_qubit2: i32,
    u: &ComplexMatrix4,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::twoQubitUnitary(qureg.reg, target_qubit1, target_qubit2, u.0);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn controlled_two_qubit_unitary(
    qureg: &mut Qureg,
    control_qubit: i32,
    target_qubit1: i32,
    target_qubit2: i32,
    u: &ComplexMatrix4,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::controlledTwoQubitUnitary(
            qureg.reg,
            control_qubit,
            target_qubit1,
            target_qubit2,
            u.0,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_controlled_two_qubit_unitary(
    qureg: &mut Qureg,
    control_qubits: &[i32],
    num_control_qubits: i32,
    target_qubit1: i32,
    target_qubit2: i32,
    u: &ComplexMatrix4,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::multiControlledTwoQubitUnitary(
            qureg.reg,
            control_qubits.as_ptr(),
            num_control_qubits,
            target_qubit1,
            target_qubit2,
            u.0,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_qubit_unitary(
    qureg: &mut Qureg,
    targs: &[i32],
    num_targs: i32,
    u: &ComplexMatrixN,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::multiQubitUnitary(qureg.reg, targs.as_ptr(), num_targs, u.0);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn controlled_multi_qubit_unitary(
    qureg: &mut Qureg,
    ctrl: i32,
    targs: &[i32],
    num_targs: i32,
    u: &ComplexMatrixN,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::controlledMultiQubitUnitary(
            qureg.reg,
            ctrl,
            targs.as_ptr(),
            num_targs,
            u.0,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_controlled_multi_qubit_unitary(
    qureg: &mut Qureg,
    ctrls: &[i32],
    num_ctrls: i32,
    targs: &[i32],
    num_targs: i32,
    u: &ComplexMatrixN,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::multiControlledMultiQubitUnitary(
            qureg.reg,
            ctrls.as_ptr(),
            num_ctrls,
            targs.as_ptr(),
            num_targs,
            u.0,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn mix_kraus_map(
    qureg: &mut Qureg,
    target: i32,
    ops: &[ComplexMatrix2],
    num_ops: i32,
) {
    let ops_inner = ops.iter().map(|x| x.0).collect::<Vec<_>>();
    unsafe {
        ffi::mixKrausMap(qureg.reg, target, ops_inner.as_ptr(), num_ops);
    }
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn mix_two_qubit_kraus_map(
    qureg: &mut Qureg,
    target1: i32,
    target2: i32,
    ops: &[ComplexMatrix4],
    num_ops: i32,
) -> Result<(), QuestError> {
    let ops_inner = ops.iter().map(|x| x.0).collect::<Vec<_>>();
    catch_quest_exception(|| unsafe {
        ffi::mixTwoQubitKrausMap(
            qureg.reg,
            target1,
            target2,
            ops_inner.as_ptr(),
            num_ops,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn mix_multi_qubit_kraus_map(
    qureg: &mut Qureg,
    targets: &[i32],
    num_targets: i32,
    ops: &[ComplexMatrixN],
    num_ops: i32,
) -> Result<(), QuestError> {
    let ops_inner = ops.iter().map(|x| x.0).collect::<Vec<_>>();
    catch_quest_exception(|| unsafe {
        ffi::mixMultiQubitKrausMap(
            qureg.reg,
            targets.as_ptr(),
            num_targets,
            ops_inner.as_ptr(),
            num_ops,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn mix_nontp_kraus_map(
    qureg: &mut Qureg,
    target: i32,
    ops: &[ComplexMatrix2],
    num_ops: i32,
) {
    let ops_inner = ops.iter().map(|x| x.0).collect::<Vec<_>>();
    unsafe {
        ffi::mixNonTPKrausMap(qureg.reg, target, ops_inner.as_ptr(), num_ops);
    }
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn mix_nontp_two_qubit_kraus_map(
    qureg: &mut Qureg,
    target1: i32,
    target2: i32,
    ops: &[ComplexMatrix4],
    num_ops: i32,
) -> Result<(), QuestError> {
    let ops_inner = ops.iter().map(|x| x.0).collect::<Vec<_>>();
    catch_quest_exception(|| unsafe {
        ffi::mixNonTPTwoQubitKrausMap(
            qureg.reg,
            target1,
            target2,
            ops_inner.as_ptr(),
            num_ops,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn mix_nontp_multi_qubit_kraus_map(
    qureg: &mut Qureg,
    targets: &[i32],
    num_targets: i32,
    ops: &[ComplexMatrixN],
    num_ops: i32,
) -> Result<(), QuestError> {
    let ops_inner = ops.iter().map(|x| x.0).collect::<Vec<_>>();
    catch_quest_exception(|| unsafe {
        ffi::mixNonTPMultiQubitKrausMap(
            qureg.reg,
            targets.as_ptr(),
            num_targets,
            ops_inner.as_ptr(),
            num_ops,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn calc_hilbert_schmidt_distance(
    a: &Qureg,
    b: &Qureg,
) -> Result<Qreal, QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::calcHilbertSchmidtDistance(a.reg, b.reg)
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn set_weighted_qureg(
    fac1: Qcomplex,
    qureg1: &Qureg,
    fac2: Qcomplex,
    qureg2: &Qureg,
    fac_out: Qcomplex,
    out: &mut Qureg,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::setWeightedQureg(
            fac1.into(),
            qureg1.reg,
            fac2.into(),
            qureg2.reg,
            fac_out.into(),
            out.reg,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_pauli_sum(
    in_qureg: &Qureg,
    all_pauli_codes: &[PauliOpType],
    term_coeffs: &[Qreal],
    num_sum_terms: i32,
    out_qureg: &mut Qureg,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::applyPauliSum(
            in_qureg.reg,
            all_pauli_codes.as_ptr(),
            term_coeffs.as_ptr(),
            num_sum_terms,
            out_qureg.reg,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_pauli_hamil(
    in_qureg: &Qureg,
    hamil: &PauliHamil,
    out_qureg: &mut Qureg,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::applyPauliHamil(in_qureg.reg, hamil.0, out_qureg.reg);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_trotter_circuitit(
    qureg: &mut Qureg,
    hamil: &PauliHamil,
    time: Qreal,
    order: i32,
    reps: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::applyTrotterCircuit(qureg.reg, hamil.0, time, order, reps);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_matrix2(
    qureg: &mut Qureg,
    target_qubit: i32,
    u: &ComplexMatrix2,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::applyMatrix2(qureg.reg, target_qubit, u.0);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_matrix4(
    qureg: &mut Qureg,
    target_qubit1: i32,
    target_qubit2: i32,
    u: &ComplexMatrix4,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::applyMatrix4(qureg.reg, target_qubit1, target_qubit2, u.0);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_matrix_n(
    qureg: &mut Qureg,
    targs: &[i32],
    num_targs: i32,
    u: &ComplexMatrixN,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::applyMatrixN(qureg.reg, targs.as_ptr(), num_targs, u.0);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_multi_controlled_matrix_n(
    qureg: &mut Qureg,
    ctrls: &[i32],
    num_ctrls: i32,
    targs: &[i32],
    num_targs: i32,
    u: &ComplexMatrixN,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::applyMultiControlledMatrixN(
            qureg.reg,
            ctrls.as_ptr(),
            num_ctrls,
            targs.as_ptr(),
            num_targs,
            u.0,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_phase_func(
    qureg: &mut Qureg,
    qubits: &[i32],
    num_qubits: i32,
    encoding: BitEncoding,
    coeffs: &[Qreal],
    exponents: &[Qreal],
    num_terms: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::applyPhaseFunc(
            qureg.reg,
            qubits.as_ptr(),
            num_qubits,
            encoding,
            coeffs.as_ptr(),
            exponents.as_ptr(),
            num_terms,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::too_many_arguments)]
pub fn apply_phase_func_overrides(
    qureg: &mut Qureg,
    qubits: &[i32],
    num_qubits: i32,
    encoding: BitEncoding,
    coeffs: &[Qreal],
    exponents: &[Qreal],
    num_terms: i32,
    override_inds: &[i64],
    override_phases: &[Qreal],
    num_overrides: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::applyPhaseFuncOverrides(
            qureg.reg,
            qubits.as_ptr(),
            num_qubits,
            encoding,
            coeffs.as_ptr(),
            exponents.as_ptr(),
            num_terms,
            override_inds.as_ptr(),
            override_phases.as_ptr(),
            num_overrides,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::too_many_arguments)]
pub fn apply_multi_var_phase_func(
    qureg: &mut Qureg,
    qubits: &[i32],
    num_qubits_per_reg: &[i32],
    num_regs: i32,
    encoding: BitEncoding,
    coeffs: &[Qreal],
    exponents: &[Qreal],
    num_terms_per_reg: &[i32],
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::applyMultiVarPhaseFunc(
            qureg.reg,
            qubits.as_ptr(),
            num_qubits_per_reg.as_ptr(),
            num_regs,
            encoding,
            coeffs.as_ptr(),
            exponents.as_ptr(),
            num_terms_per_reg.as_ptr(),
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::too_many_arguments)]
pub fn apply_multi_var_phase_func_overrides(
    qureg: &mut Qureg,
    qubits: &[i32],
    num_qubits_per_reg: &[i32],
    num_regs: i32,
    encoding: BitEncoding,
    coeffs: &[Qreal],
    exponents: &[Qreal],
    num_terms_per_reg: &[i32],
    override_inds: &[i64],
    override_phases: &[Qreal],
    num_overrides: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::applyMultiVarPhaseFuncOverrides(
            qureg.reg,
            qubits.as_ptr(),
            num_qubits_per_reg.as_ptr(),
            num_regs,
            encoding,
            coeffs.as_ptr(),
            exponents.as_ptr(),
            num_terms_per_reg.as_ptr(),
            override_inds.as_ptr(),
            override_phases.as_ptr(),
            num_overrides,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_named_phase_func(
    qureg: &mut Qureg,
    qubits: &[i32],
    num_qubits_per_reg: &[i32],
    num_regs: i32,
    encoding: BitEncoding,
    function_name_code: PhaseFunc,
) {
    unsafe {
        ffi::applyNamedPhaseFunc(
            qureg.reg,
            qubits.as_ptr(),
            num_qubits_per_reg.as_ptr(),
            num_regs,
            encoding,
            function_name_code,
        );
    }
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::too_many_arguments)]
pub fn apply_named_phase_func_overrides(
    qureg: &mut Qureg,
    qubits: &[i32],
    num_qubits_per_reg: &[i32],
    num_regs: i32,
    encoding: BitEncoding,
    function_name_code: PhaseFunc,
    override_inds: &[i64],
    override_phases: &[Qreal],
    num_overrides: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::applyNamedPhaseFuncOverrides(
            qureg.reg,
            qubits.as_ptr(),
            num_qubits_per_reg.as_ptr(),
            num_regs,
            encoding,
            function_name_code,
            override_inds.as_ptr(),
            override_phases.as_ptr(),
            num_overrides,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::too_many_arguments)]
pub fn apply_param_named_phase_func(
    qureg: &mut Qureg,
    qubits: &[i32],
    num_qubits_per_reg: &[i32],
    num_regs: i32,
    encoding: BitEncoding,
    function_name_code: PhaseFunc,
    params: &[Qreal],
    num_params: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::applyParamNamedPhaseFunc(
            qureg.reg,
            qubits.as_ptr(),
            num_qubits_per_reg.as_ptr(),
            num_regs,
            encoding,
            function_name_code,
            params.as_ptr(),
            num_params,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::too_many_arguments)]
pub fn apply_param_named_phase_func_overrides(
    qureg: &mut Qureg,
    qubits: &[i32],
    num_qubits_per_reg: &[i32],
    num_regs: i32,
    encoding: BitEncoding,
    function_name_code: PhaseFunc,
    params: &[Qreal],
    num_params: i32,
    override_inds: &[i64],
    override_phases: &[Qreal],
    num_overrides: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::applyParamNamedPhaseFuncOverrides(
            qureg.reg,
            qubits.as_ptr(),
            num_qubits_per_reg.as_ptr(),
            num_regs,
            encoding,
            function_name_code,
            params.as_ptr(),
            num_params,
            override_inds.as_ptr(),
            override_phases.as_ptr(),
            num_overrides,
        );
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_full_qft(qureg: &mut Qureg) {
    unsafe {
        ffi::applyFullQFT(qureg.reg);
    }
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_qft(
    qureg: &mut Qureg,
    qubits: &[i32],
    num_qubits: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::applyQFT(qureg.reg, qubits.as_ptr(), num_qubits);
    })
}

/// Desc.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_projector(
    qureg: &mut Qureg,
    qubit: i32,
    outcome: i32,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::applyProjector(qureg.reg, qubit, outcome);
    })
}

#[cfg(test)]
mod tests;
