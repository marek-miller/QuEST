#![allow(clippy::missing_errors_doc)]

use std::ffi::CString;

mod exceptions;
use exceptions::catch_quest_exception;

mod ffi;
pub use ffi::{
    bitEncoding as BitEncoding,
    pauliOpType as PauliOpType,
    phaseFunc as PhaseFunc,
    phaseGateType as PhaseGateType,
};

mod precision;
pub use precision::{
    Qreal,
    EPSILON,
    LN_10,
    LN_2,
    PI,
    SQRT_2,
    TAU,
};

mod qureg;
pub use qureg::Qureg;

mod questenv;
pub use questenv::QuestEnv;

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
    NotDensityMatrix,
    NegativeProbability,
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

    /// Get the real part of the `i`th row of the matrix as shared slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use quest_bind::*;
    /// let num_qubits = 2;
    /// let mtr = &mut ComplexMatrixN::try_new(num_qubits).unwrap();
    /// init_complex_matrix_n(
    ///     mtr,
    ///     &[
    ///         &[111., 112., 113., 114.],
    ///         &[115., 116., 117., 118.],
    ///         &[119., 120., 121., 122.],
    ///         &[123., 124., 125., 126.],
    ///     ],
    ///     &[
    ///         &[211., 212., 213., 214.],
    ///         &[215., 216., 217., 218.],
    ///         &[219., 220., 221., 222.],
    ///         &[223., 224., 225., 226.],
    ///     ],
    /// )
    /// .unwrap();
    ///
    /// let i = 3;
    /// assert!(i < 1 << num_qubits);
    ///
    /// let row = mtr.row_real_as_slice(i);
    /// assert_eq!(row, &[123., 124., 125., 126.]);
    /// ```
    /// # Panics
    ///
    /// This function will panic if `i>= 2.pow(1<< num_qubits),
    /// where `num_qubits` is the number of qubits the matrix was initialized
    /// with.
    #[must_use]
    pub fn row_real_as_slice(
        &self,
        i: usize,
    ) -> &[Qreal] {
        assert!(i < 1 << self.0.numQubits);

        unsafe {
            std::slice::from_raw_parts(
                *(self.0.real).add(i),
                (1 << self.0.numQubits) as usize,
            )
        }
    }

    /// Get the real part of the `i`th row of the matrix as mutable slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use quest_bind::*;
    /// let num_qubits = 2;
    /// let mtr = &mut ComplexMatrixN::try_new(num_qubits).unwrap();
    /// init_complex_matrix_n(
    ///     mtr,
    ///     &[
    ///         &[111., 112., 113., 114.],
    ///         &[115., 116., 117., 118.],
    ///         &[119., 120., 121., 122.],
    ///         &[123., 124., 125., 126.],
    ///     ],
    ///     &[
    ///         &[211., 212., 213., 214.],
    ///         &[215., 216., 217., 218.],
    ///         &[219., 220., 221., 222.],
    ///         &[223., 224., 225., 226.],
    ///     ],
    /// )
    /// .unwrap();
    ///
    /// let i = 3;
    /// assert!(i < 1 << num_qubits);
    ///
    /// let row = mtr.row_real_as_mut_slice(i);
    /// assert_eq!(row, &[123., 124., 125., 126.]);
    /// ```
    /// # Panics
    ///
    /// This function will panic if `i>= 2.pow(1<< num_qubits),
    /// where `num_qubits` is the number of qubits the matrix was initialized
    /// with.
    pub fn row_real_as_mut_slice(
        &mut self,
        i: usize,
    ) -> &mut [Qreal] {
        unsafe {
            std::slice::from_raw_parts_mut(
                *(self.0.real).add(i),
                (1 << self.0.numQubits) as usize,
            )
        }
    }

    /// Get the imaginary part of the `i`th row of the matrix as shared slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use quest_bind::*;
    /// let num_qubits = 2;
    /// let mtr = &mut ComplexMatrixN::try_new(num_qubits).unwrap();
    /// init_complex_matrix_n(
    ///     mtr,
    ///     &[
    ///         &[111., 112., 113., 114.],
    ///         &[115., 116., 117., 118.],
    ///         &[119., 120., 121., 122.],
    ///         &[123., 124., 125., 126.],
    ///     ],
    ///     &[
    ///         &[211., 212., 213., 214.],
    ///         &[215., 216., 217., 218.],
    ///         &[219., 220., 221., 222.],
    ///         &[223., 224., 225., 226.],
    ///     ],
    /// )
    /// .unwrap();
    ///
    /// let i = 3;
    /// assert!(i < 1 << num_qubits);
    ///
    /// let row = mtr.row_imag_as_slice(i);
    /// assert_eq!(row, &[223., 224., 225., 226.]);
    /// ```
    /// # Panics
    ///
    /// This function will panic if `i>= 2.pow(1<< num_qubits),
    /// where `num_qubits` is the number of qubits the matrix was initialized
    /// with.
    #[must_use]
    pub fn row_imag_as_slice(
        &self,
        i: usize,
    ) -> &[Qreal] {
        unsafe {
            std::slice::from_raw_parts(
                *(self.0.imag).add(i),
                (1 << self.0.numQubits) as usize,
            )
        }
    }

    /// Get the imaginary part of the `i`th row of the matrix as mutable slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use quest_bind::*;
    /// let num_qubits = 2;
    /// let mtr = &mut ComplexMatrixN::try_new(num_qubits).unwrap();
    /// init_complex_matrix_n(
    ///     mtr,
    ///     &[
    ///         &[111., 112., 113., 114.],
    ///         &[115., 116., 117., 118.],
    ///         &[119., 120., 121., 122.],
    ///         &[123., 124., 125., 126.],
    ///     ],
    ///     &[
    ///         &[211., 212., 213., 214.],
    ///         &[215., 216., 217., 218.],
    ///         &[219., 220., 221., 222.],
    ///         &[223., 224., 225., 226.],
    ///     ],
    /// )
    /// .unwrap();
    ///
    /// let i = 3;
    /// assert!(i < 1 << num_qubits);
    ///
    /// let row = mtr.row_imag_as_mut_slice(i);
    /// assert_eq!(row, &[223., 224., 225., 226.]);
    /// ```
    /// # Panics
    ///
    /// This function will panic if `i>= 2.pow(1<< num_qubits),
    /// where `num_qubits` is the number of qubits the matrix was initialized
    /// with.
    pub fn row_imag_as_mut_slice(
        &mut self,
        i: usize,
    ) -> &mut [Qreal] {
        unsafe {
            std::slice::from_raw_parts_mut(
                *(self.0.imag).add(i),
                (1 << self.0.numQubits) as usize,
            )
        }
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
        catch_quest_exception(|| unsafe { ffi::destroyPauliHamil(self.0) })
            .expect("dropping PauliHamil should always succeed");
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
        catch_quest_exception(|| unsafe {
            ffi::destroyDiagonalOp(self.op, self.env.0);
        })
        .expect("dropping DiagonalOp should always succeed");
    }
}

/// Initialises a `ComplexMatrixN` instance to have the passed
/// `real` and `imag` values.
///
/// This function reimplements the functionality of `QuEST`'s
/// `initComplexMatrix()`, instead of calling that function directly.  This way,
/// we avoid transmuting the slice of slices passed as argument into a C array
/// and simply copy the matrix elements onto the `QuEST` matrix type.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let mtr = &mut ComplexMatrixN::try_new(1).unwrap();
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
    let num_elems = 1 << m.0.numQubits;

    if real.len() < num_elems || imag.len() < num_elems {
        return Err(QuestError::ArrayLengthError);
    }
    for i in 0..num_elems {
        if real[i].len() < num_elems || imag[i].len() < num_elems {
            return Err(QuestError::ArrayLengthError);
        }
    }

    for i in 0..num_elems {
        for j in 0..num_elems {
            unsafe {
                *(*m.0.real.add(i)).add(j) = real[i][j];
                *(*m.0.imag.add(i)).add(j) = imag[i][j];
            }
        }
    }
    Ok(())
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

/// Print the current state vector of probability amplitudes to file.
pub fn report_state(qureg: &Qureg) {
    unsafe { ffi::reportState(qureg.reg) }
}

/// Print the current state vector of probability amplitudes.
pub fn report_state_to_screen(
    qureg: &Qureg,
    env: &QuestEnv,
    report_rank: i32,
) {
    unsafe { ffi::reportStateToScreen(qureg.reg, env.0, report_rank) }
}

/// Report information about a set of qubits.
///
/// This function reports: number of qubits, number of probability amplitudes.
pub fn report_qureg_params(qureg: &Qureg) {
    unsafe {
        ffi::reportQuregParams(qureg.reg);
    }
}

/// Print the Hamiltonian `hamil` to screen.
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
    catch_quest_exception(|| unsafe {
        ffi::initBlankState(qureg.reg);
    })
    .expect("init_blank_state should always succeed");
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
    catch_quest_exception(|| unsafe {
        ffi::initZeroState(qureg.reg);
    })
    .expect("init_zero_state should always succeed");
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
    catch_quest_exception(|| unsafe {
        ffi::initPlusState(qureg.reg);
    })
    .expect("init_plus_state should always succeed");
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
    catch_quest_exception(|| unsafe {
        ffi::initDebugState(qureg.reg);
    })
    .expect("init_debug_state should always succeed");
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

/// Shift the phase of a single qubit by a given angle.
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
/// let angle = 0.5;
/// multi_controlled_phase_shift(qureg, control_qubits, angle).unwrap();
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_controlled_phase_shift(
    qureg: &mut Qureg,
    control_qubits: &[i32],
    angle: Qreal,
) -> Result<(), QuestError> {
    let num_control_qubits = control_qubits.len() as i32;
    if num_control_qubits > qureg.num_qubits_represented() {
        return Err(QuestError::ArrayLengthError);
    }
    for &idx in control_qubits {
        if idx >= qureg.num_qubits_represented() || idx < 0 {
            return Err(QuestError::QubitIndexError);
        }
    }
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
    catch_quest_exception(|| unsafe { ffi::syncQuESTSuccess(success_code) })
        .expect("sync_quest_success should always succeed")
}

/// Report information about the `QuEST` environment
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn report_quest_env(env: &QuestEnv) {
    catch_quest_exception(|| unsafe {
        ffi::reportQuESTEnv(env.0);
    })
    .expect("report_quest_env should always succeed");
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
    catch_quest_exception(|| {
        unsafe {
            let cstr_ptr = cstr.into_raw();
            ffi::getEnvironmentString(env.0, cstr_ptr);
            cstr = CString::from_raw(cstr_ptr);
        }

        cstr.into_string().map_err(QuestError::IntoStringError)
    })
    .expect("get_environment_string should always succeed")
}

/// Copy the state-vector (or density matrix) into GPU memory.
///
/// In GPU mode, this copies the state-vector (or density matrix) from RAM
/// to VRAM / GPU-memory, which is the version operated upon by other calls to
/// the API.
///
/// In CPU mode, this function has no effect.
///
/// In conjunction with [`copy_state_from_gpu()`][api-copy-state-from-gpu]
/// (which should be called first), this allows a user to directly modify the
/// state-vector in a hardware agnostic way. Note though that users should
/// instead use [`set_amps()`][api-set-amps] if possible.
///
/// # Parameters
///
/// - `qureg`: the qureg to copy
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
///
/// copy_state_to_gpu(qureg);
/// ```
///
/// See [QuEST API][quest-api] for more information.
///
/// [api-copy-state-from-gpu]: crate::copy_state_from_gpu()
/// [api-set-amps]: crate::set_amps()
/// [quest-api]: https://quest-kit.github.io/QuEST/modules.html
pub fn copy_state_to_gpu(qureg: &mut Qureg) {
    catch_quest_exception(|| unsafe {
        ffi::copyStateToGPU(qureg.reg);
    })
    .expect("copy_state_to_gpu should always succeed");
}

/// Copy the state-vector (or density matrix) from GPU memory.
///
/// In GPU mode, this copies the state-vector (or density matrix) from GPU
/// memory  to RAM , where it can be accessed/modified  by the user.
///
/// In CPU mode, this function has no effect.
///
/// In conjunction with [`copy_state_to_gpu()`][api-copy-state-to-gpu] , this
/// allows a user to directly modify the state-vector in a hardware agnostic
/// way. Note though that users should instead use [`set_amps()`][api-set-amps]
/// if possible.
///
/// # Parameters
///
/// - `qureg`: the qureg to copy
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
///
/// copy_state_from_gpu(qureg);
/// ```
///
/// See [QuEST API][quest-api] for more information.
///
/// [api-copy-state-to-gpu]: crate::copy_state_to_gpu()
/// [api-set-amps]: crate::set_amps()
/// [quest-api]: https://quest-kit.github.io/QuEST/modules.html
pub fn copy_state_from_gpu(qureg: &mut Qureg) {
    catch_quest_exception(|| unsafe { ffi::copyStateFromGPU(qureg.reg) })
        .expect("copy_state_from_gpu should always succeed");
}

/// Copy a part the state-vector (or density matrix) into GPU memory.
///
/// In GPU mode, this copies a substate of the state-vector (or density matrix)
/// from RAM to VRAM / GPU-memory.
///
/// In CPU mode, this function has no effect.
///
/// In conjunction with
/// [`copy_substate_from_gpu()`][api-copy-substate-from-gpu], this allows a user
/// to directly modify a subset of the amplitudes the state-vector in a hardware
/// agnostic way, without having to load the entire state via
/// [`copy_state_to_gpu()`][api-copy-state-to-gpu].
///
/// Note though that users should instead use [`set_amps()`][api-set-amps] if
/// possible.
///
/// # Parameters
///
/// - `qureg`: the qureg to copy
/// - `start_ind` the index of the first amplitude to copy
/// - `num_amps` the number of contiguous amplitudes to copy (starting with
///   `start_ind`)
///
/// # Errors
///
/// - [`InvalidQuESTInputError`][quest-error-except],
///   - if `start_ind` is an invalid amplitude index
///   - if `num_amps` is greater than the remaining amplitudes in the state,
///     from `start_ind`
///
///
/// See [QuEST API][quest-api] for more information.
///
/// [api-copy-substate-from-gpu]: crate::copy_substate_from_gpu()
/// [api-copy-state-to-gpu]: crate::copy_state_to_gpu()
/// [api-set-amps]: crate::set_amps()
/// [quest-error-except]: crate::QuestError::InvalidQuESTInputError
/// [quest-api]: https://quest-kit.github.io/QuEST/modules.html
pub fn copy_substate_to_gpu(
    qureg: &mut Qureg,
    start_ind: i64,
    num_amps: i64,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::copySubstateToGPU(qureg.reg, start_ind, num_amps);
    })
}

/// Copy a part the state-vector (or density matrix) from GPU memory.
///
/// In GPU mode, this copies a substate of the state-vector (or density matrix)
/// from  to VRAM / GPU-memory to RAM, which is the version operated upon by
/// other calls to the API.
///
/// In CPU mode, this function has no effect.
///
/// In conjunction with
/// [`copy_substate_to_gpu()`][api-copy-substate-to-gpu], this allows a user
/// to directly modify a subset of the amplitudes the state-vector in a hardware
/// agnostic way, without having to load the entire state via
/// [`copy_state_from_gpu()`][api-copy-state-from-gpu].
///
/// Note though that users should instead use [`set_amps()`][api-set-amps] if
/// possible.
///
/// # Parameters
///
/// - `qureg`: the qureg to copy
/// - `start_ind` the index of the first amplitude to copy
/// - `num_amps` the number of contiguous amplitudes to copy (starting with
///   `start_ind`)
///
/// # Errors
///
/// - [`InvalidQuESTInputError`][quest-error-except],
///   - if `start_ind` is an invalid amplitude index
///   - if `num_amps` is greater than the remaining amplitudes in the state,
///     from `start_ind`
///
///
/// See [QuEST API][quest-api] for more information.
///
/// [api-copy-substate-to-gpu]: crate::copy_substate_to_gpu()
/// [api-copy-state-from-gpu]: crate::copy_state_from_gpu()
/// [api-set-amps]: crate::set_amps()
/// [quest-error-except]: crate::QuestError::InvalidQuESTInputError
/// [quest-api]: https://quest-kit.github.io/QuEST/modules.html
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

/// Get the real part of the probability amplitude at an index in
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

/// Get the imaginary part of the probability amplitude at an index
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

/// A debugging function which calculates the total probability of the qubits.
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
    catch_quest_exception(|| unsafe { ffi::calcTotalProb(qureg.reg) })
        .expect("calc_total_prop should always succeed")
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
    if control_qubit >= qureg.num_qubits_represented()
        || target_qubit >= qureg.num_qubits_represented()
    {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::controlledRotateZ(qureg.reg, control_qubit, target_qubit, angle);
    })
}

/// Applies a controlled rotation by  around a given vector of the Bloch-sphere.
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

/// Apply a controlled unitary parameterized by
/// two given complex scalars.
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
/// controlled_compact_unitary(qureg, 0, 1, alpha, beta).unwrap();
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
    if control_qubit >= qureg.num_qubits_represented()
        || target_qubit >= qureg.num_qubits_represented()
    {
        return Err(QuestError::QubitIndexError);
    }
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

/// Apply a general controlled unitary.
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
/// let mtr = &ComplexMatrix2::new(
///     [[norm, norm], [norm, -norm]],
///     [[0., 0.], [0., 0.]],
/// );
/// controlled_unitary(qureg, 0, 1, mtr).unwrap();
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
    if control_qubit >= qureg.num_qubits_represented()
        || target_qubit >= qureg.num_qubits_represented()
    {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::controlledUnitary(qureg.reg, control_qubit, target_qubit, u.0);
    })
}

/// Apply a general multiple-control single-target unitary.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
/// init_zero_state(qureg);
///
/// let norm = SQRT_2.recip();
/// let mtr = &ComplexMatrix2::new(
///     [[norm, norm], [norm, -norm]],
///     [[0., 0.], [0., 0.]],
/// );
/// multi_controlled_unitary(qureg, &[1, 2], 0, mtr).unwrap();
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_controlled_unitary(
    qureg: &mut Qureg,
    control_qubits: &[i32],
    target_qubit: i32,
    u: &ComplexMatrix2,
) -> Result<(), QuestError> {
    let num_control_qubits = control_qubits.len() as i32;
    if num_control_qubits >= qureg.num_qubits_represented()
        || target_qubit >= qureg.num_qubits_represented()
    {
        return Err(QuestError::QubitIndexError);
    }
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

/// Apply the single-qubit Pauli-X gate.
///
/// # Parameters
///
///  - `qureg`: object representing the set of all qubits
///  - `target_qubit`: qubit to operate on
///
/// # Errors
///
/// - [`QubitIndexError`][quest-error-index], if `qubit` is outside [0,
///   [`qureg.num_qubits_represented()`][qureg-num-qubits]).
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(qureg);
///
/// pauli_x(qureg, 0).unwrap();
///
/// let amp = get_real_amp(qureg, 1).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][quest-api] for more information.
///
/// [quest-error-index]: crate::QuestError::QubitIndexError
/// [qureg-num-qubits]: crate::Qureg::num_qubits_represented()
/// [quest-api]: https://quest-kit.github.io/QuEST/modules.html
pub fn pauli_x(
    qureg: &mut Qureg,
    target_qubit: i32,
) -> Result<(), QuestError> {
    if target_qubit >= qureg.num_qubits_represented() || target_qubit < 0 {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::pauliX(qureg.reg, target_qubit);
    })
}

/// Apply the single-qubit Pauli-Y gate.
///
/// # Parameters
///
///  - `qureg`: object representing the set of all qubits
///  - `target_qubit`: qubit to operate on
///
/// # Errors
///
/// - [`QubitIndexError`][quest-error-index], if `qubit` is outside [0,
///   [`qureg.num_qubits_represented()`][qureg-num-qubits]).
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(qureg);
///
/// pauli_y(qureg, 0).unwrap();
///
/// let amp = get_imag_amp(qureg, 1).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][quest-api] for more information.
///
/// [quest-error-index]: crate::QuestError::QubitIndexError
/// [qureg-num-qubits]: crate::Qureg::num_qubits_represented()
/// [quest-api]: https://quest-kit.github.io/QuEST/modules.html
pub fn pauli_y(
    qureg: &mut Qureg,
    target_qubit: i32,
) -> Result<(), QuestError> {
    if target_qubit >= qureg.num_qubits_represented() || target_qubit < 0 {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::pauliY(qureg.reg, target_qubit);
    })
}

/// Apply the single-qubit Pauli-Z gate.
///
/// # Parameters
///
///  - `qureg`: object representing the set of all qubits
///  - `target_qubit`: qubit to operate on
///
/// # Errors
///
/// - [`QubitIndexError`][quest-error-index], if `qubit` is outside [0,
///   [`qureg.num_qubits_represented()`][qureg-num-qubits]).
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(qureg);
///
/// pauli_z(qureg, 0).unwrap();
///
/// let amp = get_real_amp(qureg, 0).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][quest-api] for more information.
///
/// [quest-error-index]: crate::QuestError::QubitIndexError
/// [qureg-num-qubits]: crate::Qureg::num_qubits_represented()
/// [quest-api]: https://quest-kit.github.io/QuEST/modules.html
pub fn pauli_z(
    qureg: &mut Qureg,
    target_qubit: i32,
) -> Result<(), QuestError> {
    if target_qubit >= qureg.num_qubits_represented() || target_qubit < 0 {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::pauliZ(qureg.reg, target_qubit);
    })
}

/// Apply the single-qubit Hadamard gate.
///
/// This function applies the following unitary on `qubit`:
///
/// ```text
/// SQRT_2.recip() *
///     [ 1  1 ]
///     [ 1 -1 ]
/// ```
///
/// # Parameters
///
///  - `qureg`: object representing the set of all qubits
///  - `target_qubit`: qubit to operate on
///
/// # Errors
///
/// Returns [`QubitIndexError`][error-qubit-index] if `target_qubit` is
/// outside [0, [`qureg.num_qubits_represented()`][qureg-num-qubits]).
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(qureg);
///
/// hadamard(qureg, 0).unwrap();
///
/// let amp = get_real_amp(qureg, 0).unwrap();
/// assert!((amp - SQRT_2.recip()).abs() < EPSILON);
/// ```
///
/// See [QuEST API][quest-api] for more information.
///
/// [error-qubit-index]: crate::QuestError::QubitIndexError
/// [qureg-num-qubits]: crate::Qureg::num_qubits_represented()
/// [quest-api]: https://quest-kit.github.io/QuEST/modules.html
pub fn hadamard(
    qureg: &mut Qureg,
    target_qubit: i32,
) -> Result<(), QuestError> {
    if target_qubit < 0 || target_qubit >= qureg.num_qubits_represented() {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::hadamard(qureg.reg, target_qubit);
    })
}

/// Apply the controlled not (single control, single target) gate.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(qureg);
/// pauli_x(qureg, 1).unwrap();
///
/// controlled_not(qureg, 1, 0).unwrap();
///
/// let amp = get_real_amp(qureg, 3).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
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
    if target_qubit >= qureg.num_qubits_represented()
        || control_qubit >= qureg.num_qubits_represented()
    {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::controlledNot(qureg.reg, control_qubit, target_qubit);
    })
}

/// Apply a NOT (or Pauli X) gate with multiple control and target qubits.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(4, env).unwrap();
/// init_zero_state(qureg);
/// pauli_x(qureg, 0).unwrap();
/// pauli_x(qureg, 1).unwrap();
///
/// let ctrls = &[0, 1];
/// let targs = &[2, 3];
/// multi_controlled_multi_qubit_not(qureg, ctrls, targs).unwrap();
///
/// let amp = get_real_amp(qureg, 15).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_controlled_multi_qubit_not(
    qureg: &mut Qureg,
    ctrls: &[i32],
    targs: &[i32],
) -> Result<(), QuestError> {
    let num_ctrls = ctrls.len() as i32;
    let num_targs = targs.len() as i32;
    if num_ctrls > qureg.num_qubits_represented()
        || num_targs > qureg.num_qubits_represented()
    {
        return Err(QuestError::ArrayLengthError);
    }
    for idx in ctrls.iter().chain(targs) {
        if *idx >= qureg.num_qubits_represented() {
            return Err(QuestError::QubitIndexError);
        }
    }

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

/// Apply a NOT (or Pauli X) gate with multiple target qubits,
///
/// which has the same  effect as (but is much faster than) applying each
/// single-qubit NOT gate in turn.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(qureg);
///
/// let targs = &[0, 1];
/// multi_qubit_not(qureg, targs).unwrap();
///
/// let amp = get_real_amp(qureg, 3).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_qubit_not(
    qureg: &mut Qureg,
    targs: &[i32],
) -> Result<(), QuestError> {
    let num_targs = targs.len() as i32;
    if num_targs > qureg.num_qubits_represented() {
        return Err(QuestError::ArrayLengthError);
    }
    for idx in targs {
        if *idx >= qureg.num_qubits_represented() {
            return Err(QuestError::QubitIndexError);
        }
    }
    catch_quest_exception(|| unsafe {
        let targs_ptr = targs.as_ptr();
        ffi::multiQubitNot(qureg.reg, targs_ptr, num_targs);
    })
}

/// Apply the controlled pauliY (single control, single target) gate.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(qureg);
/// pauli_x(qureg, 1).unwrap();
///
/// controlled_pauli_y(qureg, 1, 0).unwrap();
///
/// let amp = get_imag_amp(qureg, 3).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
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

/// Gives the probability of a qubit being measured in the given outcome.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
/// init_zero_state(qureg);
///
/// let prob = calc_prob_of_outcome(qureg, 0, 0).unwrap();
/// assert!((prob - 1.).abs() < EPSILON);
/// let prob = calc_prob_of_outcome(qureg, 0, 1).unwrap();
/// assert!(prob.abs() < EPSILON);
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

/// Calculate probabilities of every outcome of the sub-register.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
/// init_zero_state(qureg);
///
/// let qubits = &[1, 2];
/// let outcome_probs = &mut vec![0.; 4];
/// calc_prob_of_all_outcomes(outcome_probs, qureg, qubits).unwrap();
/// assert_eq!(outcome_probs, &vec![1., 0., 0., 0.]);
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
) -> Result<(), QuestError> {
    let num_qubits = qubits.len() as i32;
    if num_qubits > qureg.num_qubits_represented()
        || outcome_probs.len() < (1 << num_qubits)
    {
        return Err(QuestError::ArrayLengthError);
    }

    catch_quest_exception(|| unsafe {
        ffi::calcProbOfAllOutcomes(
            outcome_probs.as_mut_ptr(),
            qureg.reg,
            qubits.as_ptr(),
            num_qubits,
        );
    })
}

/// Updates `qureg` to be consistent with measuring qubit in the given outcome.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_plus_state(qureg);
///
/// collapse_to_outcome(qureg, 0, 0).unwrap();
///
/// // QuEST throws an exception if probability of outcome is 0.
/// init_zero_state(qureg);
/// collapse_to_outcome(qureg, 0, 1).unwrap_err();
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
    if measure_qubit >= qureg.num_qubits_represented() {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::collapseToOutcome(qureg.reg, measure_qubit, outcome)
    })
}

/// Measures a single qubit, collapsing it randomly to 0 or 1.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
///
/// // Prepare an entangled state `|00> + |11>`
/// init_zero_state(qureg);
/// hadamard(qureg, 0).and(controlled_not(qureg, 0, 1)).unwrap();
///
/// // Qubits are entangled now
/// let outcome1 = measure(qureg, 0).unwrap();
/// let outcome2 = measure(qureg, 1).unwrap();
///
/// assert_eq!(outcome1, outcome2);
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

/// Measures a single qubit, collapsing it randomly to 0 or 1
///
/// Additionally, the function gives the probability of that outcome.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
///
/// // Prepare an entangled state `|00> + |11>`
/// init_zero_state(qureg);
/// hadamard(qureg, 0).and(controlled_not(qureg, 0, 1)).unwrap();
///
/// // Qubits are entangled now
/// let prob = &mut -1.;
/// let outcome1 = measure_with_stats(qureg, 0, prob).unwrap();
/// assert!((*prob - 0.5).abs() < EPSILON);
///
/// let outcome2 = measure_with_stats(qureg, 1, prob).unwrap();
/// assert!((*prob - 1.).abs() < EPSILON);
///
/// assert_eq!(outcome1, outcome2);
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

/// Computes the inner product of two equal-size state vectors.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(qureg);
/// let other_qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_plus_state(other_qureg);
///
/// let prod = calc_inner_product(qureg, other_qureg).unwrap();
/// assert!((prod.re - 0.5).abs() < EPSILON);
/// assert!((prod.im).abs() < EPSILON);
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

/// Computes the Hilbert-Schmidt scalar product.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new_density(2, env).unwrap();
/// init_zero_state(qureg);
/// let other_qureg = &mut Qureg::try_new_density(2, env).unwrap();
/// init_plus_state(other_qureg);
///
/// let prod = calc_density_inner_product(qureg, other_qureg).unwrap();
/// assert!((prod - 0.25).abs() < EPSILON);
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

/// Seed the random number generator.
///
/// Seeds the random number generator with the (master node) current time and
/// process ID.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &mut QuestEnv::new();
///
/// seed_quest_default(env);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn seed_quest_default(env: &mut QuestEnv) {
    catch_quest_exception(|| unsafe {
        let env_ptr = std::ptr::addr_of_mut!(env.0);
        ffi::seedQuESTDefault(env_ptr);
    })
    .expect("seed_quest_default should always succeed");
}

/// Seeds the random number generator with a custom array of key(s).
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &mut QuestEnv::new();
///
/// seed_quest(env, &[1, 2, 3]);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn seed_quest(
    env: &mut QuestEnv,
    seed_array: &[u64],
) {
    let num_seeds = seed_array.len() as i32;
    // QuEST's function signature is `c_ulong`. Let's use u64 for now...
    catch_quest_exception(|| unsafe {
        let env_ptr = std::ptr::addr_of_mut!(env.0);
        let seed_array_ptr = seed_array.as_ptr();
        ffi::seedQuEST(env_ptr, seed_array_ptr, num_seeds);
    })
    .expect("seed_quest should always succeed");
}

/// Obtain the seeds presently used in random number generation.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let seeds = get_quest_seeds(env);
///
/// assert!(seeds.len() > 0);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::cast_sign_loss)]
#[must_use]
pub fn get_quest_seeds<'a: 'b, 'b>(env: &'a QuestEnv) -> &'b [u64] {
    catch_quest_exception(|| unsafe {
        let seeds_ptr = &mut std::ptr::null_mut();
        let num_seeds = &mut 0_i32;
        ffi::getQuESTSeeds(env.0, seeds_ptr, num_seeds);
        std::slice::from_raw_parts(*seeds_ptr, *num_seeds as usize)
    })
    .expect("get_quest_seeds should always succeed")
}

/// Enable QASM recording.
///
/// Gates applied to qureg will here-after be added to a growing log of QASM
/// instructions, progressively consuming more memory until disabled with
/// `stop_recording_qasm()`. The QASM log is bound to this qureg instance.
///
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
///
/// start_recording_qasm(qureg);
/// hadamard(qureg, 0).and(controlled_not(qureg, 0, 1)).unwrap();
/// stop_recording_qasm(qureg);
///
/// print_recorded_qasm(qureg);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn start_recording_qasm(qureg: &mut Qureg) {
    catch_quest_exception(|| unsafe {
        ffi::startRecordingQASM(qureg.reg);
    })
    .expect("start_recording_qasm should always succeed");
}

/// Disable QASM recording.
///
/// The recorded QASM will be maintained in qureg and continue to be appended to
/// if `startRecordingQASM` is recalled.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
///
/// start_recording_qasm(qureg);
/// hadamard(qureg, 0).and(controlled_not(qureg, 0, 1)).unwrap();
/// stop_recording_qasm(qureg);
///
/// print_recorded_qasm(qureg);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn stop_recording_qasm(qureg: &mut Qureg) {
    catch_quest_exception(|| unsafe {
        ffi::stopRecordingQASM(qureg.reg);
    })
    .expect("stop_recording_qasm should always succeed");
}

/// Clear all QASM so far recorded.
///
/// This does not start or stop recording.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// start_recording_qasm(qureg);
/// hadamard(qureg, 0).unwrap();
///
/// clear_recorded_qasm(qureg);
///
/// controlled_not(qureg, 0, 1).unwrap();
/// stop_recording_qasm(qureg);
/// print_recorded_qasm(qureg);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn clear_recorded_qasm(qureg: &mut Qureg) {
    catch_quest_exception(|| unsafe {
        ffi::clearRecordedQASM(qureg.reg);
    })
    .expect("clear_recorded_qasm should always succeed");
}

/// Print recorded QASM to stdout.
///
/// This does not clear the QASM log, nor does it start or stop QASM recording.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
///
/// start_recording_qasm(qureg);
/// hadamard(qureg, 0).and(controlled_not(qureg, 0, 1)).unwrap();
/// stop_recording_qasm(qureg);
///
/// print_recorded_qasm(qureg);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn print_recorded_qasm(qureg: &mut Qureg) {
    catch_quest_exception(|| unsafe {
        ffi::printRecordedQASM(qureg.reg);
    })
    .expect("print_recorded_qasm should always succeed");
}

/// Writes recorded QASM to a file, throwing an error if inaccessible.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
///
/// start_recording_qasm(qureg);
/// hadamard(qureg, 0).and(controlled_not(qureg, 0, 1)).unwrap();
/// stop_recording_qasm(qureg);
///
/// write_recorded_qasm_to_file(qureg, "/dev/null").unwrap();
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

///  Mixes a density matrix `qureg` to induce single-qubit dephasing noise.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new_density(2, env).unwrap();
/// init_plus_state(qureg);
///
/// mix_dephasing(qureg, 0, 0.5).unwrap();
///
/// let amp = get_density_amp(qureg, 0, 0).unwrap();
/// assert!((amp.re - 0.25).abs() < EPSILON);
/// let amp = get_density_amp(qureg, 0, 1).unwrap();
/// assert!(amp.re.abs() < EPSILON);
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
    if !qureg.is_density_matrix() {
        return Err(QuestError::NotDensityMatrix);
    }
    if target_qubit < 0 || target_qubit > qureg.num_qubits_represented() {
        return Err(QuestError::QubitIndexError);
    }
    if prob < 0. {
        return Err(QuestError::NegativeProbability);
    }
    catch_quest_exception(|| unsafe {
        ffi::mixDephasing(qureg.reg, target_qubit, prob);
    })
}

///  Mixes a density matrix `qureg` to induce two-qubit dephasing noise.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new_density(3, env).unwrap();
/// init_plus_state(qureg);
///
/// mix_two_qubit_dephasing(qureg, 0, 1, 0.75).unwrap();
///
/// let amp = get_density_amp(qureg, 0, 0).unwrap();
/// assert!((amp.re - 0.125).abs() < EPSILON);
/// let amp = get_density_amp(qureg, 0, 1).unwrap();
/// assert!(amp.re.abs() < EPSILON);
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
    if !qureg.is_density_matrix() {
        return Err(QuestError::NotDensityMatrix);
    }
    if qubit1 < 0 || qubit1 > qureg.num_qubits_represented() {
        return Err(QuestError::QubitIndexError);
    }
    if qubit2 < 0 || qubit2 > qureg.num_qubits_represented() {
        return Err(QuestError::QubitIndexError);
    }
    if prob < 0. {
        return Err(QuestError::NegativeProbability);
    }
    catch_quest_exception(|| unsafe {
        ffi::mixTwoQubitDephasing(qureg.reg, qubit1, qubit2, prob);
    })
}

/// Mixes a density matrix to induce single-qubit homogeneous
/// depolarising noise.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*; let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new_density(2, env).unwrap();
/// init_zero_state(qureg);
///
/// mix_depolarising(qureg, 0, 0.75).unwrap();
/// let amp = get_density_amp(qureg, 0, 0).unwrap();
///
/// assert!((amp.re - 0.5) < EPSILON);
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
    if !qureg.is_density_matrix() {
        return Err(QuestError::NotDensityMatrix);
    }
    if target_qubit < 0 || target_qubit > qureg.num_qubits_represented() {
        return Err(QuestError::QubitIndexError);
    }
    if prob < 0. {
        return Err(QuestError::NegativeProbability);
    }
    catch_quest_exception(|| unsafe {
        ffi::mixDepolarising(qureg.reg, target_qubit, prob);
    })
}

///  Mixes a density matrix to induce single-qubit amplitude damping.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new_density(2, env).unwrap();
/// init_plus_state(qureg);
///
/// mix_damping(qureg, 0, 1.).unwrap();
///
/// let amp = get_density_amp(qureg, 0, 0).unwrap();
/// assert!((amp.re - 1.) < EPSILON);
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
    if !qureg.is_density_matrix() {
        return Err(QuestError::NotDensityMatrix);
    }
    if target_qubit < 0 || target_qubit > qureg.num_qubits_represented() {
        return Err(QuestError::QubitIndexError);
    }
    if prob < 0. {
        return Err(QuestError::NegativeProbability);
    }
    catch_quest_exception(|| unsafe {
        ffi::mixDamping(qureg.reg, target_qubit, prob);
    })
}

/// Mixes a density matrix to induce two-qubit homogeneous depolarising
/// noise.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new_density(3, env).unwrap();
/// init_plus_state(qureg);
///
/// mix_two_qubit_depolarising(qureg, 0, 1, 15. / 16.).unwrap();
///
/// let amp = get_density_amp(qureg, 0, 0).unwrap();
/// assert!((amp.re - 0.125).abs() < EPSILON);
/// let amp = get_density_amp(qureg, 0, 1).unwrap();
/// assert!(amp.re.abs() < EPSILON);
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
    if !qureg.is_density_matrix() {
        return Err(QuestError::NotDensityMatrix);
    }
    if qubit1 < 0 || qubit1 > qureg.num_qubits_represented() {
        return Err(QuestError::QubitIndexError);
    }
    if qubit2 < 0 || qubit2 > qureg.num_qubits_represented() {
        return Err(QuestError::QubitIndexError);
    }
    if prob < 0. {
        return Err(QuestError::NegativeProbability);
    }
    catch_quest_exception(|| unsafe {
        ffi::mixTwoQubitDepolarising(qureg.reg, qubit1, qubit2, prob);
    })
}

/// Mixes a density matrix to induce general single-qubit Pauli noise.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new_density(2, env).unwrap();
/// init_zero_state(qureg);
///
/// let (prob_x, prob_y, prob_z) = (0.25, 0.25, 0.25);
/// mix_pauli(qureg, 0, prob_x, prob_y, prob_z).unwrap();
///
/// let mut outcome_prob = -1.;
/// let _ = measure_with_stats(qureg, 0, &mut outcome_prob).unwrap();
///
/// assert!((outcome_prob - 0.5).abs() < EPSILON);
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
    if target_qubit >= qureg.num_qubits_represented() || target_qubit < 0 {
        return Err(QuestError::QubitIndexError);
    }
    if !qureg.is_density_matrix() {
        return Err(QuestError::NotDensityMatrix);
    }
    catch_quest_exception(|| unsafe {
        ffi::mixPauli(qureg.reg, target_qubit, prob_x, prob_y, prob_z);
    })
}

/// Modifies `combine_qureg` with `other_qureg`.
///
/// The state becomes `(1-prob) combine_qureg +  prob other_qureg`.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let combine_qureg = &mut Qureg::try_new_density(2, env).unwrap();
/// let other_qureg = &mut Qureg::try_new_density(2, env).unwrap();
///
/// init_zero_state(combine_qureg);
/// init_classical_state(other_qureg, 3).unwrap();
///
/// mix_density_matrix(combine_qureg, 0.5, other_qureg).unwrap();
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

/// Calculate the purity of a density matrix.
///
/// The purity of a density matrix is calculated by taking the trace of the
/// density matrix squared. Returns `Tr (\rho^2)`.
/// For a pure state, this =1.
/// For a mixed state, the purity is less than 1 and is lower bounded by
/// `1/2^n`, where n is the number of qubits. The minimum purity is achieved for
/// the maximally mixed state `identity/2^n`.
///
/// This function does not accept state-vectors, which clearly have purity 1.
///
/// Note this function will give incorrect results for non-Hermitian Quregs
/// (i.e. invalid density matrices), which will disagree with
/// `Tr(\rho^2)`. Instead, this function returns `\sum_{ij}
/// |\rho_{ij}|^2`.
///
/// # Parameters
///
/// - `qureg`: a density matrix of which to measure the purity
///
/// # Errors
///
/// Returns [`InvalidQuESTInputError`][quest-error-except],
///
/// - if the argument `qureg` is not a density matrix
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new_density(2, env).unwrap();
/// init_zero_state(qureg);
///
/// let purity = calc_purity(qureg).unwrap();
/// assert!((purity - 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][quest-api] for more information.
///
/// [quest-error-except]: crate::QuestError::InvalidQuESTInputError
/// [quest-api]: https://quest-kit.github.io/QuEST/modules.html
pub fn calc_purity(qureg: &Qureg) -> Result<Qreal, QuestError> {
    catch_quest_exception(|| unsafe { ffi::calcPurity(qureg.reg) })
}

/// Calculates the fidelity of `qureg` (a state-vector or density matrix).
///
/// Calculates the fidelity against a reference pure state (necessarily a
/// state-vector).
///
/// - If `qureg` is a state-vector, this function computes
///
/// ```latex
///  |\langle \text{qureg} | \text{pure_state} \rangle|^2
/// ```
///
/// - If `qureg` is a density matrix, this function computes
///
/// ```latex
///  \langle \text{pure_state} | \text{qureg} | \text{pure_state} \rangle
/// ```
///
/// In either case, the returned fidelity lies in `[0, 1]` (assuming both input
/// states have valid normalisation). If any of the input `Qureg`s are not
/// normalised, this function will return the real component of the correct
/// linear algebra calculation.
///
/// The number of qubits represented in `qureg` and `pure_state` must match.
///
/// # Parameters
///
/// - `qureg`: a density matrix or state vector
/// - `pure_state`: a state vector
///
/// Returns the fidelity between the input registers
///
/// # Errors
///
/// Returns [`InvalidQuESTInputError`][quest-error-except],
///
/// - if the second argument `pure_state` is not a state-vector
/// - if the number of qubits `qureg` and `pure_state` do not match
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new_density(2, env).unwrap();
/// let pure_state = &mut Qureg::try_new(2, env).unwrap();
///
/// init_zero_state(qureg);
/// init_plus_state(pure_state);
///
/// let fidelity = calc_fidelity(qureg, pure_state).unwrap();
/// assert!((fidelity - 0.25).abs() < EPSILON);
/// ```
///
/// See [QuEST API][quest-api] for more information.
///
/// [quest-error-except]: crate::QuestError::InvalidQuESTInputError
/// [quest-api]: https://quest-kit.github.io/QuEST/modules.html
pub fn calc_fidelity(
    qureg: &Qureg,
    pure_state: &Qureg,
) -> Result<Qreal, QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::calcFidelity(qureg.reg, pure_state.reg)
    })
}

/// Performs a SWAP gate between `qubit1` and `qubit2`.

/// This effects
///
/// ```text
/// [1 0 0 0]
/// [0 0 1 0]
/// [0 1 0 0]
/// [0 0 0 1]
/// ```
///
/// on the designated qubits, though is performed internally by three CNOT
/// gates.
///
///
/// # Parameters
///
/// - `qureg`: object representing the set of all qubits
/// - `qubit1`: qubit to swap
/// - `qubit2`: other qubit to swap
///
/// # Errors
///
/// - [`QubitIndexError`][quest-error-index], if either `qubit1` or `qubit2` is
///   outside [0, [`qureg.num_qubits_represented()`][qureg-num-qubits]).
/// - [`InvalidQuESTInputError`][quest-error-except], if `qubit1` and `qubit2`
///   are equal
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
///
/// // init state |10>
/// init_classical_state(qureg, 1).unwrap();
/// // swap to |01>
/// swap_gate(qureg, 0, 1).unwrap();
///
/// let outcome = measure(qureg, 0).unwrap();
/// assert_eq!(outcome, 0);
/// ```
///
/// See [QuEST API][quest-api] for more information.
///
/// [quest-error-index]: crate::QuestError::QubitIndexError
/// [quest-error-except]: crate::QuestError::InvalidQuESTInputError
/// [qureg-num-qubits]: crate::Qureg::num_qubits_represented()
/// [quest-api]: https://quest-kit.github.io/QuEST/modules.html
pub fn swap_gate(
    qureg: &mut Qureg,
    qubit1: i32,
    qubit2: i32,
) -> Result<(), QuestError> {
    if qubit1 >= qureg.num_qubits_represented() || qubit1 < 0 {
        return Err(QuestError::QubitIndexError);
    }
    if qubit2 >= qureg.num_qubits_represented() || qubit2 < 0 {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::swapGate(qureg.reg, qubit1, qubit2);
    })
}

/// Performs a sqrt SWAP gate between `qubit1` and `qubit2`.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// // init state |10>
/// init_classical_state(qureg, 1).unwrap();
/// sqrt_swap_gate(qureg, 0, 1).unwrap();
/// sqrt_swap_gate(qureg, 0, 1).unwrap();
/// let outcome = measure(qureg, 0).unwrap();
/// assert_eq!(outcome, 0);
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
    if qb1 >= qureg.num_qubits_represented() || qb1 < 0 {
        return Err(QuestError::QubitIndexError);
    }
    if qb2 >= qureg.num_qubits_represented() || qb2 < 0 {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::sqrtSwapGate(qureg.reg, qb1, qb2);
    })
}

/// Apply a general single-qubit unitary with multiple control qubits.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
/// init_zero_state(qureg);
///
/// let control_qubits = &[1, 2];
/// let control_state = &[0, 0];
/// let target_qubit = 0;
/// let u = &ComplexMatrix2::new([[0., 1.], [1., 0.]], [[0., 0.], [0., 0.]]);
/// multi_state_controlled_unitary(
///     qureg,
///     control_qubits,
///     control_state,
///     target_qubit,
///     u,
/// )
/// .unwrap();
///
/// let amp = get_real_amp(qureg, 1).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_state_controlled_unitary(
    qureg: &mut Qureg,
    control_qubits: &[i32],
    control_state: &[i32],
    target_qubit: i32,
    u: &ComplexMatrix2,
) -> Result<(), QuestError> {
    let num_control_qubits = control_qubits.len() as i32;
    let num_qubits_rep = qureg.num_qubits_represented();
    for &idx in control_qubits {
        if idx >= num_qubits_rep {
            return Err(QuestError::QubitIndexError);
        }
    }
    if target_qubit >= qureg.num_qubits_represented() || target_qubit < 0 {
        return Err(QuestError::QubitIndexError);
    }
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

/// Apply a multi-qubit Z rotation on selected qubits.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_plus_state(qureg);
///
/// let qubits = &[0, 1];
/// let angle = PI;
/// multi_rotate_z(qureg, qubits, angle).unwrap();
///
/// let amp = get_imag_amp(qureg, 0).unwrap();
/// assert!((amp + 0.5).abs() < EPSILON);
/// let amp = get_imag_amp(qureg, 1).unwrap();
/// assert!((amp - 0.5).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_rotate_z(
    qureg: &mut Qureg,
    qubits: &[i32],
    angle: Qreal,
) -> Result<(), QuestError> {
    let num_qubits = qubits.len() as i32;
    if num_qubits > qureg.num_qubits_represented() {
        return Err(QuestError::ArrayLengthError);
    }

    catch_quest_exception(|| unsafe {
        ffi::multiRotateZ(qureg.reg, qubits.as_ptr(), num_qubits, angle);
    })
}

/// Apply a multi-qubit multi-Pauli rotation.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// use PauliOpType::PAULI_X;
///
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
/// init_zero_state(qureg);
///
/// let target_qubits = &[1, 2];
/// let target_paulis = &[PAULI_X, PAULI_X];
/// let angle = PI;
///
/// multi_rotate_pauli(qureg, target_qubits, target_paulis, angle).unwrap();
///
/// let amp = get_imag_amp(qureg, 6).unwrap();
/// assert!((amp + 1.).abs() < 2. * EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_rotate_pauli(
    qureg: &mut Qureg,
    target_qubits: &[i32],
    target_paulis: &[PauliOpType],
    angle: Qreal,
) -> Result<(), QuestError> {
    let num_targets = target_qubits.len() as i32;
    let num_qubits_rep = qureg.num_qubits_represented();
    if num_targets > num_qubits_rep {
        return Err(QuestError::ArrayLengthError);
    }
    for &idx in target_qubits {
        if idx >= num_qubits_rep || idx < 0 {
            return Err(QuestError::QubitIndexError);
        }
    }
    if target_paulis.len() < target_qubits.len() {
        return Err(QuestError::ArrayLengthError);
    }
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

/// Apply a multi-controlled multi-target Z rotation.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(4, env).unwrap();
///
/// // Initialize `|1111>`
/// init_zero_state(qureg);
/// (0..4).try_for_each(|i| pauli_x(qureg, i)).unwrap();
///
/// let control_qubits = &[0, 1];
/// let target_qubits = &[2, 3];
/// let angle = 2. * PI;
/// multi_controlled_multi_rotate_z(
///     qureg,
///     control_qubits,
///     target_qubits,
///     angle,
/// )
/// .unwrap();
///
/// // the state is now `-1. * |1111>`
/// let amp = get_real_amp(qureg, 15).unwrap();
/// assert!((amp + 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_controlled_multi_rotate_z(
    qureg: &mut Qureg,
    control_qubits: &[i32],
    target_qubits: &[i32],
    angle: Qreal,
) -> Result<(), QuestError> {
    let num_controls = control_qubits.len() as i32;
    let num_targets = target_qubits.len() as i32;
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

/// Apply a multi-controlled multi-target multi-Pauli rotation.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// use PauliOpType::PAULI_Z;
///
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(4, env).unwrap();
///
/// // Initialize `|1111>`
/// init_zero_state(qureg);
/// (0..4).try_for_each(|i| pauli_x(qureg, i)).unwrap();
///
/// let control_qubits = &[0, 1];
/// let target_qubits = &[2, 3];
/// let target_paulis = &[PAULI_Z, PAULI_Z];
/// let angle = 2. * PI;
/// multi_controlled_multi_rotate_pauli(
///     qureg,
///     control_qubits,
///     target_qubits,
///     target_paulis,
///     angle,
/// )
/// .unwrap();
///
/// // the state is now `-1. * |1111>`
/// let amp = get_real_amp(qureg, 15).unwrap();
/// assert!((amp + 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_controlled_multi_rotate_pauli(
    qureg: &mut Qureg,
    control_qubits: &[i32],
    target_qubits: &[i32],
    target_paulis: &[PauliOpType],
    angle: Qreal,
) -> Result<(), QuestError> {
    let num_controls = control_qubits.len() as i32;
    let num_targets = target_qubits.len() as i32;
    if target_paulis.len() != target_qubits.len() {
        return Err(QuestError::ArrayLengthError);
    }
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

/// Computes the expected value of a product of Pauli operators.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// use PauliOpType::PAULI_X;
///
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(qureg);
/// let workspace = &mut Qureg::try_new(2, env).unwrap();
///
/// let target_qubits = &[0, 1];
/// let pauli_codes = &[PAULI_X, PAULI_X];
///
/// calc_expec_pauli_prod(qureg, target_qubits, pauli_codes, workspace)
///     .unwrap();
/// let amp = get_real_amp(workspace, 3).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn calc_expec_pauli_prod(
    qureg: &Qureg,
    target_qubits: &[i32],
    pauli_codes: &[PauliOpType],
    workspace: &mut Qureg,
) -> Result<Qreal, QuestError> {
    let num_targets = target_qubits.len() as i32;
    if pauli_codes.len() as i32 != num_targets {
        return Err(QuestError::ArrayLengthError);
    }
    for &idx in target_qubits {
        if idx < 0 || idx > qureg.num_qubits_represented() {
            return Err(QuestError::QubitIndexError);
        }
    }
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

/// Computes the expected value of a sum of products of Pauli operators.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// use PauliOpType::{
///     PAULI_X,
///     PAULI_Z,
/// };
///
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(qureg);
/// let workspace = &mut Qureg::try_new(2, env).unwrap();
///
/// let all_pauli_codes = &[PAULI_X, PAULI_Z, PAULI_Z, PAULI_X];
/// let term_coeffs = &[0.5, 0.5];
///
/// calc_expec_pauli_sum(qureg, all_pauli_codes, term_coeffs, workspace)
///     .unwrap();
///
/// let amp = get_real_amp(workspace, 2).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn calc_expec_pauli_sum(
    qureg: &Qureg,
    all_pauli_codes: &[PauliOpType],
    term_coeffs: &[Qreal],
    workspace: &mut Qureg,
) -> Result<Qreal, QuestError> {
    let num_sum_terms = term_coeffs.len() as i32;
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

/// Computes the expected value of `qureg` under Hermitian operator `hamil`.
///
/// This function is merely an encapsulation of `calc_expec_pauli_sum()` - refer
/// to the doc there for an elaboration.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// use PauliOpType::{
///     PAULI_X,
///     PAULI_Z,
/// };
///
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(qureg);
/// let workspace = &mut Qureg::try_new(2, env).unwrap();
///
/// let hamil = &mut PauliHamil::try_new(2, 2).unwrap();
/// init_pauli_hamil(hamil, &[0.5, 0.5], &[PAULI_X, PAULI_X, PAULI_X, PAULI_Z])
///     .unwrap();
///
/// calc_expec_pauli_hamil(qureg, hamil, workspace).unwrap();
///
/// let amp = get_real_amp(workspace, 1).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
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

///  Apply a general two-qubit unitary (including a global phase factor).
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
/// init_zero_state(qureg);
/// pauli_x(qureg, 0).unwrap();
///
/// let target_qubit1 = 1;
/// let target_qubit2 = 2;
/// let u = &ComplexMatrix4::new(
///     [
///         [0., 0., 0., 1.],
///         [0., 1., 0., 0.],
///         [0., 0., 1., 0.],
///         [1., 0., 0., 0.],
///     ],
///     [
///         [0., 0., 0., 0.],
///         [0., 0., 0., 0.],
///         [0., 0., 0., 0.],
///         [0., 0., 0., 0.],
///     ],
/// );
///
/// two_qubit_unitary(qureg, target_qubit1, target_qubit2, u).unwrap();
///
/// let amp = get_real_amp(qureg, 7).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
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
    let num_qubits_rep = qureg.num_qubits_represented();
    if !((0..num_qubits_rep).contains(&target_qubit1)
        && (0..num_qubits_rep).contains(&target_qubit2))
    {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::twoQubitUnitary(qureg.reg, target_qubit1, target_qubit2, u.0);
    })
}

/// Apply a general controlled two-qubit unitary.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
/// init_zero_state(qureg);
/// pauli_x(qureg, 0).unwrap();
///
/// let control_qubit = 0;
/// let target_qubit1 = 1;
/// let target_qubit2 = 2;
/// let u = &ComplexMatrix4::new(
///     [
///         [0., 0., 0., 1.],
///         [0., 1., 0., 0.],
///         [0., 0., 1., 0.],
///         [1., 0., 0., 0.],
///     ],
///     [
///         [0., 0., 0., 0.],
///         [0., 0., 0., 0.],
///         [0., 0., 0., 0.],
///         [0., 0., 0., 0.],
///     ],
/// );
///
/// controlled_two_qubit_unitary(
///     qureg,
///     control_qubit,
///     target_qubit1,
///     target_qubit2,
///     u,
/// )
/// .unwrap();
///
/// let amp = get_real_amp(qureg, 7).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
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
    let num_qubits_rep = qureg.num_qubits_represented();
    if !((0..num_qubits_rep).contains(&target_qubit1)
        && (0..num_qubits_rep).contains(&target_qubit2)
        && (0..num_qubits_rep).contains(&control_qubit))
    {
        return Err(QuestError::QubitIndexError);
    }
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

/// Apply a general multi-qubit unitary with any number of target qubits.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(4, env).unwrap();
/// init_zero_state(qureg);
/// pauli_x(qureg, 0).unwrap();
/// pauli_x(qureg, 1).unwrap();
///
/// let control_qubits = &[0, 1];
/// let target_qubit1 = 2;
/// let target_qubit2 = 3;
/// let u = &ComplexMatrix4::new(
///     [
///         [0., 0., 0., 1.],
///         [0., 1., 0., 0.],
///         [0., 0., 1., 0.],
///         [1., 0., 0., 0.],
///     ],
///     [
///         [0., 0., 0., 0.],
///         [0., 0., 0., 0.],
///         [0., 0., 0., 0.],
///         [0., 0., 0., 0.],
///     ],
/// );
///
/// multi_controlled_two_qubit_unitary(
///     qureg,
///     control_qubits,
///     target_qubit1,
///     target_qubit2,
///     u,
/// )
/// .unwrap();
///
/// let amp = get_real_amp(qureg, 15).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_controlled_two_qubit_unitary(
    qureg: &mut Qureg,
    control_qubits: &[i32],
    target_qubit1: i32,
    target_qubit2: i32,
    u: &ComplexMatrix4,
) -> Result<(), QuestError> {
    let num_qubits_rep = qureg.num_qubits_represented();
    let num_control_qubits = control_qubits.len() as i32;
    for idx in control_qubits {
        if !(0..num_qubits_rep).contains(idx) {
            return Err(QuestError::QubitIndexError);
        }
    }
    if !((0..num_qubits_rep).contains(&target_qubit1)
        && (0..num_qubits_rep).contains(&target_qubit2))
    {
        return Err(QuestError::QubitIndexError);
    }
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

/// Apply a general multi-qubit unitary with any number of target qubits.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(qureg);
///
/// let u = &mut ComplexMatrixN::try_new(2).unwrap();
/// let zero_row = &[0., 0., 0., 0.];
/// init_complex_matrix_n(
///     u,
///     &[
///         &[0., 0., 0., 1.],
///         &[0., 1., 0., 0.],
///         &[0., 0., 1., 0.],
///         &[1., 0., 0., 0.],
///     ],
///     &[zero_row, zero_row, zero_row, zero_row],
/// )
/// .unwrap();
///
/// multi_qubit_unitary(qureg, &[0, 1], u).unwrap();
///
/// // Check if the register is now in the state `|11>`
/// let amp = get_real_amp(qureg, 3).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_qubit_unitary(
    qureg: &mut Qureg,
    targs: &[i32],
    u: &ComplexMatrixN,
) -> Result<(), QuestError> {
    let num_targs = targs.len() as i32;
    for &idx in targs {
        if idx < 0 || idx >= qureg.num_qubits_represented() {
            return Err(QuestError::QubitIndexError);
        }
    }
    catch_quest_exception(|| unsafe {
        ffi::multiQubitUnitary(qureg.reg, targs.as_ptr(), num_targs, u.0);
    })
}

/// Apply a general controlled multi-qubit unitary (including a global phase
/// factor).
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
/// init_zero_state(qureg);
/// pauli_x(qureg, 0).unwrap();
///
/// let u = &mut ComplexMatrixN::try_new(2).unwrap();
/// let zero_row = &[0., 0., 0., 0.];
/// init_complex_matrix_n(
///     u,
///     &[
///         &[0., 0., 0., 1.],
///         &[0., 1., 0., 0.],
///         &[0., 0., 1., 0.],
///         &[1., 0., 0., 0.],
///     ],
///     &[zero_row, zero_row, zero_row, zero_row],
/// )
/// .unwrap();
///
/// let ctrl = 0;
/// let targs = &[1, 2];
/// controlled_multi_qubit_unitary(qureg, ctrl, targs, u).unwrap();
///
/// // Check if the register is now in the state `|111>`
/// let amp = get_real_amp(qureg, 7).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn controlled_multi_qubit_unitary(
    qureg: &mut Qureg,
    ctrl: i32,
    targs: &[i32],
    u: &ComplexMatrixN,
) -> Result<(), QuestError> {
    if ctrl < 0 || ctrl >= qureg.num_qubits_represented() {
        return Err(QuestError::QubitIndexError);
    }
    let num_targs = targs.len() as i32;
    for &idx in targs {
        if idx < 0 || idx >= qureg.num_qubits_represented() {
            return Err(QuestError::QubitIndexError);
        }
    }
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

/// Apply a general multi-controlled multi-qubit unitary (including a global
/// phase factor).
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(4, env).unwrap();
/// init_zero_state(qureg);
/// pauli_x(qureg, 0).unwrap();
/// pauli_x(qureg, 1).unwrap();
///
/// let u = &mut ComplexMatrixN::try_new(2).unwrap();
/// let zero_row = &[0., 0., 0., 0.];
/// init_complex_matrix_n(
///     u,
///     &[
///         &[0., 0., 0., 1.],
///         &[0., 1., 0., 0.],
///         &[0., 0., 1., 0.],
///         &[1., 0., 0., 0.],
///     ],
///     &[zero_row, zero_row, zero_row, zero_row],
/// )
/// .unwrap();
///
/// let ctrls = &[0, 1];
/// let targs = &[2, 3];
/// multi_controlled_multi_qubit_unitary(qureg, ctrls, targs, u).unwrap();
///
/// // Check if the register is now in the state `|1111>`
/// let amp = get_real_amp(qureg, 15).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn multi_controlled_multi_qubit_unitary(
    qureg: &mut Qureg,
    ctrls: &[i32],
    targs: &[i32],
    u: &ComplexMatrixN,
) -> Result<(), QuestError> {
    let num_qubits_rep = qureg.num_qubits_represented();
    let num_ctrls = ctrls.len() as i32;
    for &idx in ctrls {
        if idx < 0 || idx >= num_qubits_rep {
            return Err(QuestError::QubitIndexError);
        }
    }
    let num_targs = targs.len() as i32;
    for &idx in targs {
        if idx < 0 || idx >= num_qubits_rep {
            return Err(QuestError::QubitIndexError);
        }
    }
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

/// Apply a general single-qubit Kraus map to a density matrix.
///
/// The map is specified by at most four Kraus operators.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new_density(2, env).unwrap();
/// init_zero_state(qureg);
///
/// let m = &ComplexMatrix2::new([[0., 1.], [1., 0.]], [[0., 0.], [0., 0.]]);
/// let target = 1;
/// mix_kraus_map(qureg, target, &[m]).unwrap();
///
/// // Check is the register is now in the state |01>
/// let amp = get_density_amp(qureg, 2, 2).unwrap();
/// assert!((amp.re - 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn mix_kraus_map(
    qureg: &mut Qureg,
    target: i32,
    ops: &[&ComplexMatrix2],
) -> Result<(), QuestError> {
    let num_qubits_rep = qureg.num_qubits_represented();
    if target < 0 || target >= num_qubits_rep {
        return Err(QuestError::QubitIndexError);
    }

    let num_ops = ops.len() as i32;
    if !(1..=4).contains(&num_ops) {
        return Err(QuestError::ArrayLengthError);
    }
    let ops_inner = ops.iter().map(|x| x.0).collect::<Vec<_>>();
    catch_quest_exception(|| unsafe {
        ffi::mixKrausMap(qureg.reg, target, ops_inner.as_ptr(), num_ops);
    })
}

/// Apply a general two-qubit Kraus map to a density matrix.
///
/// The map is specified by at most sixteen Kraus operators.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new_density(3, env).unwrap();
/// init_zero_state(qureg);
///
/// let m = &ComplexMatrix4::new(
///     [
///         [0., 0., 0., 1.],
///         [0., 1., 0., 0.],
///         [0., 0., 1., 0.],
///         [1., 0., 0., 0.],
///     ],
///     [
///         [0., 0., 0., 0.],
///         [0., 0., 0., 0.],
///         [0., 0., 0., 0.],
///         [0., 0., 0., 0.],
///     ],
/// );
/// let target1 = 1;
/// let target2 = 2;
/// mix_two_qubit_kraus_map(qureg, target1, target2, &[m]).unwrap();
///
/// // Check is the register is now in the state |011>
/// let amp = get_density_amp(qureg, 6, 6).unwrap();
/// assert!((amp.re - 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn mix_two_qubit_kraus_map(
    qureg: &mut Qureg,
    target1: i32,
    target2: i32,
    ops: &[&ComplexMatrix4],
) -> Result<(), QuestError> {
    let num_qubits_rep = qureg.num_qubits_represented();
    if target1 < 0 || target1 >= num_qubits_rep {
        return Err(QuestError::QubitIndexError);
    }
    if target2 < 0 || target2 >= num_qubits_rep {
        return Err(QuestError::QubitIndexError);
    }
    let num_ops = ops.len() as i32;
    if !(1..=16).contains(&num_ops) {
        return Err(QuestError::ArrayLengthError);
    }
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

/// Apply a general N-qubit Kraus map to a density matrix.
///
/// The map is specified by at most `(2N)^2` Kraus operators.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new_density(3, env).unwrap();
/// init_zero_state(qureg);
/// let m = &mut ComplexMatrixN::try_new(2).unwrap();
/// init_complex_matrix_n(
///     m,
///     &[
///         &[0., 0., 0., 1.],
///         &[0., 1., 0., 0.],
///         &[0., 0., 1., 0.],
///         &[1., 0., 0., 0.],
///     ],
///     &[
///         &[0., 0., 0., 0.],
///         &[0., 0., 0., 0.],
///         &[0., 0., 0., 0.],
///         &[0., 0., 0., 0.],
///     ],
/// )
/// .unwrap();
/// let targets = &[1, 2];
/// mix_multi_qubit_kraus_map(qureg, targets, &[m]).unwrap();
///
/// // Check if the register is now in the state |011>
/// let amp = get_density_amp(qureg, 6, 6).unwrap();
/// assert!((amp.re - 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn mix_multi_qubit_kraus_map(
    qureg: &mut Qureg,
    targets: &[i32],
    ops: &[&ComplexMatrixN],
) -> Result<(), QuestError> {
    let num_qubits_rep = qureg.num_qubits_represented();
    let num_targets = targets.len() as i32;
    for &target in targets {
        if target < 0 || target >= num_qubits_rep {
            return Err(QuestError::QubitIndexError);
        }
    }
    let num_ops = ops.len() as i32;
    if !(1..=1 << (2 * num_qubits_rep)).contains(&num_ops) {
        return Err(QuestError::ArrayLengthError);
    }
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

/// Apply a general non-trace-preserving single-qubit Kraus map.
///
/// The state must be a density matrix, and the map is specified by at most four
/// operators.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new_density(2, env).unwrap();
/// init_zero_state(qureg);
///
/// let m = &ComplexMatrix2::new([[0., 1.], [0., 0.]], [[0., 0.], [0., 0.]]);
/// let target = 1;
/// mix_nontp_kraus_map(qureg, target, &[m]).unwrap();
///
/// // The register is in an unphysical null state
/// let amp = get_density_amp(qureg, 2, 2).unwrap();
/// assert!(amp.re.abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn mix_nontp_kraus_map(
    qureg: &mut Qureg,
    target: i32,
    ops: &[&ComplexMatrix2],
) -> Result<(), QuestError> {
    let num_ops = ops.len() as i32;
    if !(0..qureg.num_qubits_represented()).contains(&target) {
        return Err(QuestError::QubitIndexError);
    }
    let ops_inner = ops.iter().map(|x| x.0).collect::<Vec<_>>();
    catch_quest_exception(|| unsafe {
        ffi::mixNonTPKrausMap(qureg.reg, target, ops_inner.as_ptr(), num_ops);
    })
}

/// Apply a general non-trace-preserving two-qubit Kraus map.
///
/// The state must be a density matrix, and the map is specified
/// by at most 16 operators.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new_density(3, env).unwrap();
/// init_zero_state(qureg);
///
/// let m = &ComplexMatrix4::new(
///     [
///         [0., 0., 0., 1.],
///         [0., 1., 0., 0.],
///         [0., 0., 1., 0.],
///         [0., 0., 0., 0.],
///     ],
///     [
///         [0., 0., 0., 0.],
///         [0., 0., 0., 0.],
///         [0., 0., 0., 0.],
///         [0., 0., 0., 0.],
///     ],
/// );
/// let target1 = 1;
/// let target2 = 2;
/// mix_nontp_two_qubit_kraus_map(qureg, target1, target2, &[m]).unwrap();
///
/// // The register is in an unphysical null state
/// let amp = get_density_amp(qureg, 6, 6).unwrap();
/// assert!(amp.re.abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn mix_nontp_two_qubit_kraus_map(
    qureg: &mut Qureg,
    target1: i32,
    target2: i32,
    ops: &[&ComplexMatrix4],
) -> Result<(), QuestError> {
    let num_qubits_rep = qureg.num_qubits_represented();
    if target1 < 0 || target1 >= num_qubits_rep {
        return Err(QuestError::QubitIndexError);
    }
    if target2 < 0 || target2 >= num_qubits_rep {
        return Err(QuestError::QubitIndexError);
    }
    let num_ops = ops.len() as i32;
    if !(1..=16).contains(&num_ops) {
        return Err(QuestError::ArrayLengthError);
    }
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

/// Apply a general N-qubit non-trace-preserving Kraus map.
///
/// The state must be a density matrix, and the map is specified
/// by at most `2^(2N)` operators.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new_density(3, env).unwrap();
/// init_zero_state(qureg);
/// let m = &mut ComplexMatrixN::try_new(2).unwrap();
/// init_complex_matrix_n(
///     m,
///     &[
///         &[0., 0., 0., 1.],
///         &[0., 1., 0., 0.],
///         &[0., 0., 1., 0.],
///         &[0., 0., 0., 0.],
///     ],
///     &[
///         &[0., 0., 0., 0.],
///         &[0., 0., 0., 0.],
///         &[0., 0., 0., 0.],
///         &[0., 0., 0., 0.],
///     ],
/// )
/// .unwrap();
/// let targets = &[1, 2];
/// mix_nontp_multi_qubit_kraus_map(qureg, targets, &[m]).unwrap();
///
/// // The register is in an unphysical null state
/// let amp = get_density_amp(qureg, 6, 6).unwrap();
/// assert!(amp.re.abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn mix_nontp_multi_qubit_kraus_map(
    qureg: &mut Qureg,
    targets: &[i32],
    ops: &[&ComplexMatrixN],
) -> Result<(), QuestError> {
    let num_qubits_rep = qureg.num_qubits_represented();
    let num_targets = targets.len() as i32;
    for &target in targets {
        if target < 0 || target >= num_qubits_rep {
            return Err(QuestError::QubitIndexError);
        }
    }
    let num_ops = ops.len() as i32;
    if !(1..=1 << (2 * num_qubits_rep)).contains(&num_ops) {
        return Err(QuestError::ArrayLengthError);
    }
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

/// Computes the Hilbert Schmidt distance between two density matrices.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let a = &mut Qureg::try_new_density(2, env).unwrap();
/// init_zero_state(a);
/// let b = &mut Qureg::try_new_density(2, env).unwrap();
/// init_classical_state(b, 1).unwrap();
///
/// let dist = calc_hilbert_schmidt_distance(a, b).unwrap();
/// assert!((dist - SQRT_2).abs() < EPSILON, "{:?}", dist);
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

/// Set `qureg` to a weighted sum of states.
///
/// Modifies qureg `out` to the result of `$(\p facOut \p out + \p fac1 \p
/// qureg1 + \p fac2 \p qureg2)$`, imposing no constraints on normalisation.
///
/// Works for both state-vectors and density matrices. Note that afterward, \p
/// out may not longer be normalised and ergo no longer a valid  state-vector or
/// density matrix. Users must therefore be careful passing \p out to other
/// QuEST functions which assume normalisation in order to function correctly.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// # use num::Zero;
/// let env = &QuestEnv::new();
/// let qureg1 = &mut Qureg::try_new(1, env).unwrap();
/// init_zero_state(qureg1);
/// let qureg2 = &mut Qureg::try_new(1, env).unwrap();
/// init_zero_state(qureg2);
/// pauli_x(qureg2, 0).unwrap();
///
/// let out = &mut Qureg::try_new(1, env).unwrap();
/// init_zero_state(out);
///
/// let fac1 = Qcomplex::new(SQRT_2.recip(), 0.);
/// let fac2 = Qcomplex::new(SQRT_2.recip(), 0.);
/// let fac_out = Qcomplex::zero();
///
/// set_weighted_qureg(fac1, qureg1, fac2, qureg2, fac_out, out).unwrap();
///
/// hadamard(out, 0).unwrap();
/// let amp = get_real_amp(out, 0).unwrap();
/// assert!((amp - 1.).abs() < 10. * EPSILON);
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

/// Apply the weighted sum of Pauli products.
///
/// In theory, `in_qureg` is unchanged though its state is temporarily modified
/// and is reverted by re-applying Paulis (XX=YY=ZZ=I), so may see a change by
/// small numerical errors. The initial state in `out_qureg` is not used.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// use PauliOpType::{
///     PAULI_I,
///     PAULI_X,
/// };
///
/// let env = &QuestEnv::new();
/// let in_qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(in_qureg);
/// let out_qureg = &mut Qureg::try_new(2, env).unwrap();
/// let all_pauli_codes = &[PAULI_I, PAULI_X, PAULI_X, PAULI_I];
/// let term_coeffs = &[SQRT_2.recip(), SQRT_2.recip()];
///
/// apply_pauli_sum(in_qureg, all_pauli_codes, term_coeffs, out_qureg).unwrap();
///
/// // out_qureg is now in `|01> + |10>` state:
/// let qb1 = measure(out_qureg, 0).unwrap();
/// let qb2 = measure(out_qureg, 1).unwrap();
/// assert!(qb1 != qb2);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_pauli_sum(
    in_qureg: &mut Qureg,
    all_pauli_codes: &[PauliOpType],
    term_coeffs: &[Qreal],
    out_qureg: &mut Qureg,
) -> Result<(), QuestError> {
    let num_sum_terms = term_coeffs.len() as i32;
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

/// Apply Hamiltonian `PauliHamil`.
///
/// Modifies `out_qureg` to be the result of applying `PauliHamil` (a Hermitian
/// but not necessarily unitary operator) to `in_qureg`.
///
/// In theory, `in_qureg` is unchanged though its state is temporarily modified
/// and is reverted by re-applying Paulis (XX=YY=ZZ=I), so may see a change by
/// small numerical errors. The initial state in `out_qureg` is not used.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// use PauliOpType::{
///     PAULI_I,
///     PAULI_X,
/// };
///
/// let env = &QuestEnv::new();
/// let in_qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(in_qureg);
/// let out_qureg = &mut Qureg::try_new(2, env).unwrap();
///
/// let hamil = &mut PauliHamil::try_new(2, 2).unwrap();
/// let coeffs = &[SQRT_2.recip(), SQRT_2.recip()];
/// let codes = &[PAULI_I, PAULI_X, PAULI_X, PAULI_I];
/// init_pauli_hamil(hamil, coeffs, codes).unwrap();
///
/// apply_pauli_hamil(in_qureg, hamil, out_qureg).unwrap();
///
/// // out_qureg is now in `|01> + |10>` state:
/// let qb1 = measure(out_qureg, 0).unwrap();
/// let qb2 = measure(out_qureg, 1).unwrap();
/// assert!(qb1 != qb2);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_pauli_hamil(
    in_qureg: &mut Qureg,
    hamil: &PauliHamil,
    out_qureg: &mut Qureg,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::applyPauliHamil(in_qureg.reg, hamil.0, out_qureg.reg);
    })
}

/// Applies a trotterisation of unitary evolution.
///
/// The unitary evelution `$\exp(-i \, \text{hamil} \, \text{time})$` is applied
/// to `qureg`. # Examples
///
/// ```rust
/// # use quest_bind::*;
/// use PauliOpType::PAULI_X;
///
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(1, env).unwrap();
/// init_zero_state(qureg);
///
/// let hamil = &mut PauliHamil::try_new(1, 1).unwrap();
/// let coeffs = &[1.];
/// let codes = &[PAULI_X];
/// init_pauli_hamil(hamil, coeffs, codes).unwrap();
///
/// let time = PI / 2.;
/// let order = 1;
/// let reps = 1;
/// apply_trotter_circuit(qureg, hamil, time, order, reps).unwrap();
///
/// // qureg is now in `|1>` state:
/// let qb1 = measure(qureg, 0).unwrap();
/// assert_eq!(qb1, 1);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_trotter_circuit(
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

/// Apply a general 2-by-2 matrix, which may be non-unitary.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(qureg);
///
/// let target_qubit = 0;
/// let u = &ComplexMatrix2::new([[0., 1.], [1., 0.]], [[0., 0.], [0., 0.]]);
///
/// apply_matrix2(qureg, target_qubit, u).unwrap();
///
/// let amp = get_real_amp(qureg, 1).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
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
    if target_qubit >= qureg.num_qubits_represented() || target_qubit < 0 {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::applyMatrix2(qureg.reg, target_qubit, u.0);
    })
}

/// Apply a general 4-by-4 matrix, which may be non-unitary.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(qureg);
///
/// let target_qubit1 = 0;
/// let target_qubit2 = 1;
/// let u = &ComplexMatrix4::new(
///     [
///         [0., 1., 0., 0.],
///         [1., 0., 0., 0.],
///         [0., 0., 1., 0.],
///         [0., 0., 0., 1.],
///     ],
///     [
///         [0., 0., 0., 0.],
///         [0., 0., 0., 0.],
///         [0., 0., 0., 0.],
///         [0., 0., 0., 0.],
///     ],
/// );
///
/// apply_matrix4(qureg, target_qubit1, target_qubit2, u).unwrap();
///
/// let amp = get_real_amp(qureg, 1).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
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
    if target_qubit1 >= qureg.num_qubits_represented() || target_qubit1 < 0 {
        return Err(QuestError::QubitIndexError);
    }
    if target_qubit2 >= qureg.num_qubits_represented() || target_qubit2 < 0 {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::applyMatrix4(qureg.reg, target_qubit1, target_qubit2, u.0);
    })
}

/// Apply a general N-by-N matrix on any number of target qubits.
///
/// The matrix need not be unitary.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
/// init_zero_state(qureg);
///
/// let mtr = &mut ComplexMatrixN::try_new(3).unwrap();
/// let empty = &[0., 0., 0., 0., 0., 0., 0., 0.];
/// init_complex_matrix_n(
///     mtr,
///     &[
///         &[0., 0., 0., 0., 0., 0., 0., 1.],
///         &[0., 1., 0., 0., 0., 0., 0., 0.],
///         &[0., 0., 1., 0., 0., 0., 0., 0.],
///         &[0., 0., 0., 1., 0., 0., 0., 0.],
///         &[0., 0., 0., 0., 1., 0., 0., 0.],
///         &[0., 0., 0., 0., 0., 1., 0., 0.],
///         &[0., 0., 0., 0., 0., 0., 1., 0.],
///         &[1., 0., 0., 0., 0., 0., 0., 0.],
///     ],
///     &[empty, empty, empty, empty, empty, empty, empty, empty],
/// )
/// .unwrap();
///
/// let targets = &[0, 1, 2];
/// apply_matrix_n(qureg, targets, mtr).unwrap();
///
/// // Check if the state is now `|111>`
/// let amp = get_real_amp(qureg, 7).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_matrix_n(
    qureg: &mut Qureg,
    targs: &[i32],
    u: &ComplexMatrixN,
) -> Result<(), QuestError> {
    let num_targs = targs.len() as i32;
    let num_qubits_rep = qureg.num_qubits_represented();
    if num_targs > num_qubits_rep {
        return Err(QuestError::ArrayLengthError);
    }
    for &idx in targs {
        if idx < 0 || idx >= num_qubits_rep {
            return Err(QuestError::QubitIndexError);
        }
    }
    catch_quest_exception(|| unsafe {
        ffi::applyMatrixN(qureg.reg, targs.as_ptr(), num_targs, u.0);
    })
}

/// Apply a general N-by-N matrix with additional controlled qubits.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(4, env).unwrap();
/// init_zero_state(qureg);
/// pauli_x(qureg, 0).unwrap();
/// pauli_x(qureg, 1).unwrap();
///
/// let ctrls = &[0, 1];
/// let targs = &[2, 3];
/// let u = &mut ComplexMatrixN::try_new(2).unwrap();
/// let zero_row = &[0., 0., 0., 0.];
/// init_complex_matrix_n(
///     u,
///     &[
///         &[0., 0., 0., 1.],
///         &[0., 1., 0., 0.],
///         &[0., 0., 1., 0.],
///         &[1., 0., 0., 0.],
///     ],
///     &[zero_row, zero_row, zero_row, zero_row],
/// )
/// .unwrap();
/// apply_multi_controlled_matrix_n(qureg, ctrls, targs, u).unwrap();
///
/// // Assert `qureg` is now in the state `|1111>`
/// let amp = get_real_amp(qureg, 15).unwrap();
/// assert!((amp - 1.).abs() < EPSILON);
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_multi_controlled_matrix_n(
    qureg: &mut Qureg,
    ctrls: &[i32],
    targs: &[i32],
    u: &ComplexMatrixN,
) -> Result<(), QuestError> {
    let num_ctrls = ctrls.len() as i32;
    let num_targs = targs.len() as i32;
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

/// Apply a phase function.
///
/// Induces a phase change upon each amplitude of `qureg`, determined by the
/// passed exponential polynomial *phase function*.  This effects a diagonal
/// unitary of unit complex scalars, targeting the nominated `qubits`.
///
/// - Arguments `coeffs` and `exponents` together specify a real exponential
///   polynomial `f(r)` with `num_terms` terms, of the form
///  
///   ```latex
///   f(r) =
///     \sum\limits_{i}^{\text{num_terms}} \text{coeffs}[i] \;
///     r^{\, \text{exponents}[i]}\,, \f],
///   ```
///   where both `coeffs` and `exponents` can be negative, positive and
///   fractional. For example,
///  
///   ```rust,no_run
///   let coeffs = [1., -3.14];
///   let exponents = [2., -5.5];
///   ```
///  
///   constitutes the function: `f(r) =  1 * r^2 - 3.14 * r^(-5.5)`.  Note
///   that you cannot use fractional exponents with `encoding` being
///   [`BitEncoding::TWOS_COMPLEMENT`][api-bit-encoding-twos-cplm],  since the
///   negative   indices would generate (illegal) complex phases, and  must be
///   overriden with
///   [`apply_phase_func_overrides()`][api-apply-phase-func-overrides].  
///  
///   If your function `f(r)` diverges at one or more `r` values, you
///   must instead use `apply_phase_func_overrides()` and specify explicit phase
///   changes for these values. Otherwise, the corresponding amplitudes of the
///   state-vector will become indeterminate (like `NaN`). Note that use of any
///   negative exponent will result in divergences at `r=0`.
///
/// - The function `f(r)` specifies the phase change to induce upon amplitude
///   `alpha` of computational basis state with index `r`, such that
///
///   ```latex
///   \alpha |r\rangle \rightarrow \, \exp(i f(r))  \alpha \,  |r\rangle.
///   ```
///
///   The index `r` associated with each computational basis
///   state is determined by the binary value of the specified `qubits`
///   (ordered least to most significant), interpreted under the given
///   [`BitEncoding`][api-bit-encoding] encoding.
///
/// - If `qureg` is a density matrix `rho`, this function modifies `qureg` to:
///
/// ```latex
/// \rho \rightarrow \hat{D} \, \rho \, \hat{D}^\dagger,
/// ```
///
///  where   `\hat{D}` is the diagonal unitary operator:
///
/// ```latex
///  \hat{D} = \text{diag}
///   \, \{ \; e^{i f(r_0)}, \; e^{i f(r_1)}, \;  \dots \; \}.
/// ```
///
/// - The interpreted phase function can be previewed in the QASM log, as a
///   comment.
///
/// - This function may become numerically imprecise for quickly growing phase
///   functions which admit very large phases, for example of `10^10`.
///
/// # Parameters
///
/// - `qureg`: the state-vector or density matrix to be modified
/// - `qubits`: a list of the indices of the qubits which will inform `r` for
///   each amplitude in `qureg`
/// - `encoding`: the [`BitEncoding`][api-bit-encoding] under which to infer the
///   binary value `r` from the bits of `qubits` in each basis state of `qureg`
/// - `coeffs`: the coefficients of the exponential polynomial phase function
///   `f(r)`
/// - `exponents`: the exponents of the exponential polynomial phase function
///   `f(r)`
///
/// The length of list `coeffs` must be the same as that of `exponents`
///
/// # Errors
///
/// - [`ArrayLengthError`][quest-error-array-len], if the length of `coeffs` is
///   different than that of `exponents`
/// - [`InvalidQuESTInputError`][quest-error-except]
///   - if any qubit in `qubits` has an invalid index (i.e. does not satisfy `0
///     <= qubit < qureg.num_qubits_represented()`
///   - if the elements of `qubits` are not unique
///   - if `qubits.len() >= qureg.num_qubits_represented()`
///   - if `encoding` is not compatible with `qubits.len()` (e.g.
///     `TWOS_COMPLEMENT` with only 1 qubit)
///   - if `exponents` contains a fractional number despite `encoding` being
///     `TWOS_COMPLEMENT` (you must instead use `apply_phase_func_overrides()`
///     and override all negative indices)
///   - if `exponents` contains a negative power (you must instead use
///     apply_phase_func_overrides()` and override the zero index)
///
/// # Examples
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
/// init_zero_state(qureg);
/// pauli_x(qureg, 1).unwrap();
///
/// let qubits = &[0, 1];
/// let encoding = BitEncoding::UNSIGNED;
/// let coeffs = &[0.5, 0.5];
/// let exponents = &[0., 2.];
///
/// apply_phase_func(qureg, qubits, encoding, coeffs, exponents).unwrap();
/// ```
///
/// See [QuEST API][quest-api] for more information.
///
/// [api-bit-encoding-twos-cplm]: crate::BitEncoding::TWOS_COMPLEMENT
/// [api-bit-encoding]: crate::BitEncoding
/// [api-apply-phase-func-overrides]: crate::apply_phase_func_overrides()
/// [qureg-num-qubits]: crate::Qureg::num_qubits_represented()
/// [quest-error-array-len]: crate::QuestError::ArrayLengthError
/// [quest-error-except]: crate::QuestError::InvalidQuESTInputError
/// [quest-api]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_phase_func(
    qureg: &mut Qureg,
    qubits: &[i32],
    encoding: BitEncoding,
    coeffs: &[Qreal],
    exponents: &[Qreal],
) -> Result<(), QuestError> {
    let num_qubits = qubits.len() as i32;
    let num_terms = coeffs.len();
    if exponents.len() != num_terms {
        return Err(QuestError::ArrayLengthError);
    }
    let num_terms = num_terms as i32;
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

/// Apply a phase function with overrides.
///
/// Induces a phase change upon each amplitude of `qureg`, determined by the
/// passed  exponential polynomial "phase function", and an explicit set of
/// 'overriding' values at specific state indices.
///
/// See [`apply_phase_func()`][api-apply-phase-func] for a full desctiption.
///
/// - As in `apply_phase_func()`, the arguments `coeffs` and `exponents` specify
///   a phase function `f(r)`, where `r` is determined by `qubits` and
///   `encoding` for each basis state of `qureg`.
/// - Additionally, `override_inds` is a list specifying the values of `r` for
///   which to explicitly set the induced phase change. The overriding phase
///   changes are specified in the corresponding elements of `override_phases`.
/// - Note that if `encoding` is `TWOS_COMPLEMENT`, and `f(r)` features a
///   fractional exponent, then every negative phase index must be overriden.
///   This is checked and enforced by QuEST's validation, unless there are more
///   than 16 targeted qubits, in which case valid input is assumed (due to an
///   otherwise prohibitive performance overhead).
/// - Overriding phases are checked at each computational basis state of `qureg`
///   *before* evaluating the phase function `f(r)`, and hence are useful for
///   avoiding singularities or errors at diverging values of `r`.
/// - The interpreted phase function and list of overrides can be previewed in
///   the QASM log, as a comment.
///
/// # Parameters
///
/// - `qureg`:  the state-vector or density matrix to be modified
/// - `qubits`: a list of the indices of the qubits which will inform `r` for
///   each amplitude in `qureg`
/// - `encoding`: [`BitEncoding`][api-bit-encoding] under which to infer the
///   binary value `r` from the bits of `qubits` in each basis state of `qureg`
/// - `coeffs`: the coefficients of the exponential polynomial phase function
///   `f(r)`
/// - `exponents`: the exponents of the exponential polynomial phase function
///   `f(r)`
/// - `override_inds`: a list of sub-state indices (values of `r` of which to
///   explicit set the phase change
/// - `override_phases`: a list of replacement phase changes, for the
///   corresponding `r` values in `override_inds` (one to one)
///
/// # Errors
///
/// - [`ArrayLengthError`][quest-error-array-len],
///   - if the length of `coeffs` is different than that of `exponents`
///   - if the length of `override_inds` is different than that of
///     `override_phases`
/// - [`InvalidQuESTInputError`][quest-error-except],
///   - if any qubit in `qubits` has an invalid index (i.e. does not satisfy `0
///     <= qubit < qureg.num_qubits_represented()`
///   - if the elements of `qubits` are not unique
///   - if `qubits.len() >= qureg.num_qubits_represented()`
///   - if `encoding` is not compatible with `qubits.len()` (e.g.
///     `TWOS_COMPLEMENT` with only 1 qubit)
///   - if `exponents` contains a fractional number despite `encoding` being
///     `TWOS_COMPLEMENT` (you must instead use `apply_phase_func_overrides()`
///     and override all negative indices)
///   - if `exponents` contains a negative power and the (consequently
///     diverging) zero index is not contained in `override_inds`
///   - if any value in `override_inds` is not producible by `qubits` under the
///     given `encoding` (e.g. 2 unsigned qubits cannot represent index 9)
///   - if `encoding` is `TWOS_COMPLEMENT`, and `exponents` contains a
///     fractional number, but `override_inds` does not contain every possible
///     negative index (checked only up to 16 targeted qubits)
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
/// init_zero_state(qureg);
/// pauli_x(qureg, 1).unwrap();
///
/// let qubits = &[0, 1];
/// let encoding = BitEncoding::UNSIGNED;
/// let coeffs = &[0.5, 0.5];
/// let exponents = &[-2., 2.];
/// let override_inds = &[0];
/// let override_phases = &[0.];
///
/// apply_phase_func_overrides(
///     qureg,
///     qubits,
///     encoding,
///     coeffs,
///     exponents,
///     override_inds,
///     override_phases,
/// )
/// .unwrap();
/// ```
///
/// See [QuEST API][quest-api] for more information.
///
/// [api-apply-phase-func]: crate::apply_phase_func()
/// [api-bit-encoding]: crate::BitEncoding
/// [quest-error-array-len]: crate::QuestError::ArrayLengthError
/// [quest-error-except]: crate::QuestError::InvalidQuESTInputError
/// [quest-api]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::too_many_arguments)]
pub fn apply_phase_func_overrides(
    qureg: &mut Qureg,
    qubits: &[i32],
    encoding: BitEncoding,
    coeffs: &[Qreal],
    exponents: &[Qreal],
    override_inds: &[i64],
    override_phases: &[Qreal],
) -> Result<(), QuestError> {
    let num_qubits = qubits.len() as i32;
    let num_terms = coeffs.len();
    if exponents.len() != num_terms {
        return Err(QuestError::ArrayLengthError);
    }
    let num_terms = num_terms as i32;
    let num_overrides = override_inds.len();
    if override_phases.len() != num_overrides {
        return Err(QuestError::ArrayLengthError);
    }
    let num_overrides = num_overrides as i32;
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

/// Apply a multi-variable exponential polynomial.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
/// init_zero_state(qureg);
/// pauli_x(qureg, 1).unwrap();
///
/// let qubits = &[0, 1];
/// let num_qubits_per_reg = &[1, 1];
/// let encoding = BitEncoding::UNSIGNED;
/// let coeffs = &[0.5, 0.5];
/// let exponents = &[2., 2.];
/// let num_terms_per_reg = &[1, 1];
///
/// apply_multi_var_phase_func(
///     qureg,
///     qubits,
///     num_qubits_per_reg,
///     encoding,
///     coeffs,
///     exponents,
///     num_terms_per_reg,
/// )
/// .unwrap();
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
    encoding: BitEncoding,
    coeffs: &[Qreal],
    exponents: &[Qreal],
    num_terms_per_reg: &[i32],
) -> Result<(), QuestError> {
    let num_regs = num_qubits_per_reg.len() as i32;
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

/// Apply a multi-variable exponential polynomial with overrides.
///
/// Induces a phase change upon each amplitude of \p qureg, determined by a
/// phase function, and an explicit set of 'overriding' values at specific
/// state indices.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
/// init_zero_state(qureg);
/// pauli_x(qureg, 1).unwrap();
///
/// let qubits = &[0, 1];
/// let num_qubits_per_reg = &[1, 1];
/// let encoding = BitEncoding::UNSIGNED;
/// let coeffs = &[0.5, 0.5];
/// let exponents = &[2., 2.];
/// let num_terms_per_reg = &[1, 1];
/// let override_inds = &[0, 1, 0, 1];
/// let override_phases = &[0., 0.];
///
/// apply_multi_var_phase_func_overrides(
///     qureg,
///     qubits,
///     num_qubits_per_reg,
///     encoding,
///     coeffs,
///     exponents,
///     num_terms_per_reg,
///     override_inds,
///     override_phases,
/// )
/// .unwrap();
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
    encoding: BitEncoding,
    coeffs: &[Qreal],
    exponents: &[Qreal],
    num_terms_per_reg: &[i32],
    override_inds: &[i64],
    override_phases: &[Qreal],
) -> Result<(), QuestError> {
    let num_regs = num_qubits_per_reg.len() as i32;
    let num_overrides = override_phases.len() as i32;
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

/// Apply a named phase function.
///
/// Induces a phase change upon each amplitude of `qureg`, determined by a named
/// (and potentially multi-variable) phase function.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(qureg);
///
/// let qubits = &[0, 1];
/// let num_qubits_per_reg = &[1, 1];
/// let encoding = BitEncoding::UNSIGNED;
/// let function_name_code = PhaseFunc::DISTANCE;
///
/// apply_named_phase_func(
///     qureg,
///     qubits,
///     num_qubits_per_reg,
///     encoding,
///     function_name_code,
/// )
/// .unwrap();
/// ```
///
/// See [QuEST API][1] for more information.
///
/// [1]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_named_phase_func(
    qureg: &mut Qureg,
    qubits: &[i32],
    num_qubits_per_reg: &[i32],
    encoding: BitEncoding,
    function_name_code: PhaseFunc,
) -> Result<(), QuestError> {
    let num_regs = num_qubits_per_reg.len() as i32;
    catch_quest_exception(|| unsafe {
        ffi::applyNamedPhaseFunc(
            qureg.reg,
            qubits.as_ptr(),
            num_qubits_per_reg.as_ptr(),
            num_regs,
            encoding,
            function_name_code,
        );
    })
}

/// Apply a named phase function with overrides.
///
/// Induces a phase change upon each amplitude of \p qureg, determined by a
/// named (and potentially multi-variable) phase function, and an explicit set
/// of 'overriding' values at specific state indices.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
/// init_zero_state(qureg);
///
/// let qubits = &[0, 1];
/// let num_qubits_per_reg = &[1, 1];
/// let encoding = BitEncoding::UNSIGNED;
/// let function_name_code = PhaseFunc::DISTANCE;
/// let override_inds = &[0, 1, 0, 1];
/// let override_phases = &[0., 0.];
///
/// apply_named_phase_func_overrides(
///     qureg,
///     qubits,
///     num_qubits_per_reg,
///     encoding,
///     function_name_code,
///     override_inds,
///     override_phases,
/// )
/// .unwrap();
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
    encoding: BitEncoding,
    function_name_code: PhaseFunc,
    override_inds: &[i64],
    override_phases: &[Qreal],
) -> Result<(), QuestError> {
    let num_regs = num_qubits_per_reg.len() as i32;
    let num_overrides = override_phases.len() as i32;
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

/// Apply a parametrized phase function.
///
/// Induces a phase change upon each amplitude of \p qureg, determined by a
/// named, paramaterized (and potentially multi-variable) phase function.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_zero_state(qureg);
///
/// let qubits = &[0, 1];
/// let num_qubits_per_reg = &[1, 1];
/// let encoding = BitEncoding::UNSIGNED;
/// let function_name_code = PhaseFunc::SCALED_INVERSE_SHIFTED_NORM;
/// let params = &[0., 0., 0., 0.];
///
/// apply_param_named_phase_func(
///     qureg,
///     qubits,
///     num_qubits_per_reg,
///     encoding,
///     function_name_code,
///     params,
/// )
/// .unwrap();
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
    encoding: BitEncoding,
    function_name_code: PhaseFunc,
    params: &[Qreal],
) -> Result<(), QuestError> {
    let num_regs = num_qubits_per_reg.len() as i32;
    let num_params = params.len() as i32;
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

/// Apply a parametrized phase function with overrides.
///
/// Induces a phase change upon each amplitude of \p qureg, determined by a
/// named, parameterised (and potentially multi-variable) phase function, and an
/// explicit set of "overriding" values at specific state indices.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
/// init_zero_state(qureg);
///
/// let qubits = &[0, 1];
/// let num_qubits_per_reg = &[1, 1];
/// let encoding = BitEncoding::UNSIGNED;
/// let function_name_code = PhaseFunc::SCALED_INVERSE_SHIFTED_NORM;
/// let params = &[0., 0., 0., 0.];
/// let override_inds = &[0, 1, 0, 1];
/// let override_phases = &[0., 0.];
///
/// apply_param_named_phase_func_overrides(
///     qureg,
///     qubits,
///     num_qubits_per_reg,
///     encoding,
///     function_name_code,
///     params,
///     override_inds,
///     override_phases,
/// )
/// .unwrap();
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
    encoding: BitEncoding,
    function_name_code: PhaseFunc,
    params: &[Qreal],
    override_inds: &[i64],
    override_phases: &[Qreal],
) -> Result<(), QuestError> {
    let num_regs = num_qubits_per_reg.len() as i32;
    let num_params = params.len() as i32;
    let num_overrides = override_phases.len() as i32;
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

/// Apply the full quantum Fourier transform (QFT).
///
/// - If `qureg` is a state-vector, the output amplitudes are the discrete
///   Fourier transform (DFT) of the input amplitudes, in the exact ordering.
///   This is true even if `qureg` is unnormalised.
///
/// - If `qureg` is a density matrix, it will be changed under the unitary
///   action of the QFT. This can be imagined as each mixed state-vector
///   undergoing the DFT on its amplitudes. This is true even if `qureg` is
///   unnormalised.
///
/// This function merges contiguous controlled-phase gates into single
/// invocations of [apply_named_phase_func()][api-apply-named-phase-func], and
/// hence is significantly faster than performing
/// the QFT circuit directly.
///
/// Furthermore, in distributed mode, this function requires only `log2(#nodes)`
/// rounds of pair-wise communication, and hence is exponentially faster than
/// directly performing the DFT on the amplitudes of `qureg`.
///
/// See [`apply_qft()`][api-apply-qft] to apply the QFT to a sub-register of
/// `qureg`.
///
/// # Parameters
///
/// - `qureg`: a state-vector or density matrix to modify
///
/// See [QuEST API][quest-api] for more information.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
/// init_zero_state(qureg);
///
/// apply_full_qft(qureg);
/// ```
///
/// [api-apply-named-phase-func]: crate::apply_named_phase_func()
/// [api-apply-qft]: crate::apply_qft()
/// [quest-api]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_full_qft(qureg: &mut Qureg) {
    catch_quest_exception(|| unsafe {
        ffi::applyFullQFT(qureg.reg);
    })
    .expect("apply_full_qft should always succeed");
}

/// Applies the quantum Fourier transform (QFT) to a specific subset of qubits.
///
/// The order of qubits affects the ultimate unitary.
/// The canonical full-state QFT ([`apply_full_qft()`][api-apply-full-qft]) is
/// achieved by targeting every qubit in increasing order.
///
/// - If `qureg` is a state-vector, the output amplitudes are a kronecker
///   product of the discrete Fourier transform (DFT) acting upon the targeted
///   amplitudes.
/// - If `qureg` is a density matrix, it will be changed under the unitary
///   action of the QFT. This can be imagined as each mixed state-vector
///   undergoing the DFT on its amplitudes. This is true even if `qureg` is
///   unnormalised.
///
/// This function merges contiguous controlled-phase gates into single
/// invocations of [apply_named_phase_func()][api-apply-named-phase-func], and
/// hence is significantly faster than performing
/// the QFT circuit directly.
///
///
/// Furthermore, in distributed mode, this function requires only `log2(#nodes)`
/// rounds of pair-wise communication, and hence is exponentially faster than
/// directly performing the DFT on the amplitudes of `qureg`.
///
/// See [`apply_full_qft()`][api-apply-full-qft] to apply the QFT to he entirety
/// of `qureg`.
///
/// # Parameters
///
/// `qureg`: a state-vector or density matrix to modify
/// `qubits` a list of the qubits to operate the QFT upon
///
/// # Errors
///
/// - [`ArrayLengthError`][quest-error-array-len], if the length of `qubits` is
///   less than [`qureg.num_qubits_represented()`][qureg-num-qubits]
/// - [`QubitIndexError`][quest-error-index], if any of `qubits` is outside [0,
///   [`qureg.num_qubits_represented()`][qureg-num-qubits]).
/// - [`InvalidQuESTInputError`][quest-error-except], if `qubits` contains any
///   repetitions
///
/// See [QuEST API][quest-api] for more information.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(3, env).unwrap();
/// init_zero_state(qureg);
///
/// apply_qft(qureg, &[0, 1]).unwrap();
/// ```
///
/// [api-apply-full-qft]: crate::apply_full_qft()
/// [api-apply-named-phase-func]: crate::apply_named_phase_func()
/// [quest-error-array-len]: crate::QuestError::ArrayLengthError
/// [quest-error-except]: crate::QuestError::InvalidQuESTInputError
/// [quest-error-index]: crate::QuestError::QubitIndexError
/// [qureg-num-qubits]: crate::Qureg::num_qubits_represented()
/// [quest-api]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_qft(
    qureg: &mut Qureg,
    qubits: &[i32],
) -> Result<(), QuestError> {
    let num_qubits = qubits.len() as i32;
    let num_qubits_rep = qureg.num_qubits_represented();
    if num_qubits > num_qubits_rep {
        return Err(QuestError::ArrayLengthError);
    }
    for &idx in qubits {
        if idx >= num_qubits_rep {
            return Err(QuestError::QubitIndexError);
        }
    }
    catch_quest_exception(|| unsafe {
        ffi::applyQFT(qureg.reg, qubits.as_ptr(), num_qubits);
    })
}

/// Apply a projector.
///
/// Force the target `qubit` of `qureg` into the given classical `outcome`,
/// via a non-renormalising projection.
///
/// This function zeroes all amplitudes in the state-vector or density-matrix
/// which correspond to the opposite `outcome` given. Unlike
/// [`collapse_to_outcome()`][api-collapse-to-outcome], it does not
/// thereafter normalise `qureg`, and hence may leave it in a non-physical
/// state.
///
/// Note there is no requirement that the `outcome` state has a non-zero
/// proability, and hence this function may leave `qureg` in a blank state,
/// like that produced by [`init_blank_state()`][api-init-blank-state].
///
/// See [`collapse_to_outcome()`][api-collapse-to-outcome] for a norm-preserving
/// equivalent, like a forced   measurement
///
/// # Parameters
///
/// - `qureg`:  a state-vector or density matrix to modify
/// - `qubit`: the qubit to which to apply the projector
/// - `outcome`: the single-qubit outcome (`0` or `1`) to project `qubit`
///
/// # Errors
///
/// - [`QubitIndexError`][quest-error-index], if `qubit` is outside [0,
///   [`qureg.num_qubits_represented()`][qureg-num-qubits]).
/// - [`InvalidQuESTInputError`][quest-error-except], if `outcome` is not in
///   {0,1}
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut Qureg::try_new(2, env).unwrap();
/// init_plus_state(qureg);
///
/// apply_projector(qureg, 0, 0).unwrap();
///
/// let amp = get_real_amp(qureg, 3).unwrap();
/// assert!(amp.abs() < EPSILON);
/// ```
///
/// See [QuEST API][quest-api] for more information.
///
/// [api-collapse-to-outcome]: crate::collapse_to_outcome()
/// [api-init-blank-state]: crate::init_blank_state()
/// [qureg-num-qubits]: crate::Qureg::num_qubits_represented()
/// [quest-error-except]: crate::QuestError::InvalidQuESTInputError
/// [quest-error-index]: crate::QuestError::QubitIndexError
/// [quest-api]: https://quest-kit.github.io/QuEST/modules.html
pub fn apply_projector(
    qureg: &mut Qureg,
    qubit: i32,
    outcome: i32,
) -> Result<(), QuestError> {
    if qubit >= qureg.num_qubits_represented() || qubit < 0 {
        return Err(QuestError::QubitIndexError);
    }
    catch_quest_exception(|| unsafe {
        ffi::applyProjector(qureg.reg, qubit, outcome);
    })
}

#[cfg(test)]
mod tests;
