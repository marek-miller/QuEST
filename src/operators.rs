use std::ffi::CString;

use crate::{
    exceptions::catch_quest_exception,
    ffi::{
        self,
        pauliOpType as PauliOpType,
    },
    Qcomplex,
    Qreal,
    QuestEnv,
    QuestError,
    Qureg,
};

#[derive(Debug)]
pub struct PauliHamil(pub(crate) ffi::PauliHamil);

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
    /// See [QuEST API] for more information.
    ///
    /// # Errors
    ///
    /// Returns [`QuestError::InvalidQuESTInputError`](crate::QuestError::InvalidQuESTInputError) on
    /// failure. This is an exception thrown by `QuEST`.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
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
    pub(crate) env: &'a QuestEnv,
    pub(crate) op:  ffi::DiagonalOp,
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
/// See [QuEST API] for more information.
///
/// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::needless_pass_by_ref_mut)]
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
/// See [QuEST API] for more information.
///
/// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::needless_pass_by_ref_mut)]
pub fn sync_diagonal_op(op: &mut DiagonalOp<'_>) -> Result<(), QuestError> {
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
/// See [QuEST API] for more information.
///
/// # Panics
///
/// This function will panic, if either `real` or `imag`
/// have length smaller than `2.pow(num_qubits)`.
///
/// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::needless_pass_by_ref_mut)]
pub fn init_diagonal_op(
    op: &mut DiagonalOp<'_>,
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
/// See [QuEST API] for more information.
///
/// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::needless_pass_by_ref_mut)]
pub fn init_diagonal_op_from_pauli_hamil(
    op: &mut DiagonalOp<'_>,
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
/// See [QuEST API] for more information.
///
/// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::cast_sign_loss)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::needless_pass_by_ref_mut)]
pub fn set_diagonal_op_elems(
    op: &mut DiagonalOp<'_>,
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
/// See [QuEST API] for more information.
///
/// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::needless_pass_by_ref_mut)]
pub fn apply_diagonal_op<const N: usize>(
    qureg: &mut Qureg<'_, N>,
    op: &DiagonalOp<'_>,
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
/// See [QuEST API] for more information.
///
/// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
pub fn calc_expec_diagonal_op<const N: usize>(
    qureg: &Qureg<'_, N>,
    op: &DiagonalOp<'_>,
) -> Result<Qcomplex, QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::calcExpecDiagonalOp(qureg.reg, op.op)
    })
    .map(Into::into)
}
