use crate::{
    exceptions::catch_quest_exception,
    ffi,
    Qreal,
    QuestError,
};

#[derive(Debug, Clone, Copy)]
pub struct ComplexMatrix2(pub(crate) ffi::ComplexMatrix2);

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
pub struct ComplexMatrix4(pub(crate) ffi::ComplexMatrix4);

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
pub struct ComplexMatrixN(pub(crate) ffi::ComplexMatrixN);

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
    /// See [QuEST API] for more information.
    ///
    /// # Errors
    ///
    /// Returns [`QuestError::InvalidQuESTInputError`](crate::QuestError::InvalidQuESTInputError)
    /// on failure.  This is an exception thrown by `QuEST`.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
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
pub struct Vector(pub(crate) ffi::Vector);

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
/// See [QuEST API] for more information.
///
/// # Errors
///
/// Returns [`Error::ArrayLengthError`](crate::QuestError::ArrayLengthError), if
/// either `real` or `imag` is not a square array of dimension equal to the
/// number of qubits in `m`.  Otherwise, returns
/// [`QuestError::InvalidQuESTInputError`](crate::QuestError::InvalidQuESTInputError) on
/// failure. This is an exception thrown by `QuEST`.
///
/// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
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
