use std::ffi::CString;

use super::{
    catch_quest_exception,
    ffi,
    BitEncoding,
    ComplexMatrix2,
    ComplexMatrix4,
    ComplexMatrixN,
    PauliHamil,
    PauliOpType,
    PhaseFunc,
    Qcomplex,
    Qreal,
    QuestEnv,
    QuestError,
    Vector,
};

#[must_use]
pub fn create_qureg<const N: u16>(env: &QuestEnv) -> Qureg<'_, N> {
    Qureg::try_new(env).expect("cannot allocate new state-vector Qureg")
}

#[must_use]
pub fn create_density_qureg<const N: u16>(env: &QuestEnv) -> Qureg<'_, N> {
    Qureg::try_new_density(env).expect("cannot allocate new density Qureg")
}

#[derive(Debug)]
pub struct Qureg<'a, const N: u16> {
    pub(crate) env: &'a QuestEnv,
    pub(crate) reg: ffi::Qureg,
}

impl<'a, const N: u16> Qureg<'a, N> {
    /// Creates a state-vector Qureg object.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = create_qureg::<2>(env);
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
    pub fn try_new(env: &'a QuestEnv) -> Result<Self, QuestError> {
        let num_qubits = i32::from(N);
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
    pub fn try_new_density(env: &'a QuestEnv) -> Result<Self, QuestError> {
        let num_qubits = i32::from(N);
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

    /// Print the current state vector of probability amplitudes to file.
    ///
    /// ## File format:
    ///
    /// ```text
    /// real, imag
    /// realComponent1, imagComponent1
    /// realComponent2, imagComponent2
    /// ...
    /// realComponentN, imagComponentN
    /// ```
    ///
    ///  ## File naming convention:
    ///
    /// For each node that the program runs on, a file
    /// `state_rank_[node_rank].csv` is generated. If there is  more than
    /// one node, ranks after the first do not include the header:
    ///
    /// ```text
    /// real, imag
    /// ```
    ///
    /// so that files are easier to combine.
    ///
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    pub fn report_state(&self) {
        catch_quest_exception(|| unsafe { ffi::reportState(self.reg) })
            .expect("report_state should never fail");
    }

    /// Print the current state vector of probability amplitudes.
    ///
    /// Print the current state vector of probability amplitudes for a set of
    /// qubits to standard out. For debugging purposes. Each rank should
    /// print output serially.  Only print output for systems <= 5 qubits.
    ///
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    pub fn report_state_to_screen(
        &self,
        report_rank: i32,
    ) {
        catch_quest_exception(|| unsafe {
            ffi::reportStateToScreen(self.reg, self.env.0, report_rank);
        })
        .expect("report_state_to screen should never fail");
    }

    /// Returns the number of qubits represented.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = create_qureg::<3>(env);
    ///
    /// assert_eq!(qureg.get_num_qubits(), 3);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[must_use]
    pub fn get_num_qubits(&self) -> i32 {
        catch_quest_exception(|| unsafe { ffi::getNumQubits(self.reg) })
            .expect("get_num_qubits should never fail")
    }

    /// Return the number of complex amplitudes in a state-vector.
    ///
    /// In distributed mode, this returns the total number of amplitudes in the
    /// full representation of `qureg`, and so may be larger than the number
    /// stored on each node.
    ///
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`], if `Qureg` is a density matrix
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &create_qureg::<3>(env);
    ///
    /// assert_eq!(qureg.get_num_amps().unwrap(), 8);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    pub fn get_num_amps(&self) -> Result<i64, QuestError> {
        catch_quest_exception(|| unsafe { ffi::getNumAmps(self.reg) })
    }

    /// Return the total number of amplitudes in the register.
    ///
    /// - If `Qureg` is a state-vector, this is equal to: `2^N`, where `N` is
    ///   the number of qubits in the register [`get_num_qubits()`]
    /// - If `Qureg` is a density matrix, this is equal to `2^(2N)`
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &Qureg::try_new_density(3, env).unwrap();
    ///
    /// assert_eq!(qureg.get_num_amps_total(), 64);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`get_num_qubits()`]: crate::Qureg::get_num_qubits()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[must_use]
    pub fn get_num_amps_total(&self) -> i64 {
        self.reg.numAmpsTotal
    }

    /// Report information about a set of qubits.
    ///
    /// This function prints to stdout: number of qubits, number of probability
    /// amplitudes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &create_qureg::<2>(env);
    ///
    /// qureg.report_qureg_params();
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    pub fn report_qureg_params(&self) {
        catch_quest_exception(|| unsafe {
            ffi::reportQuregParams(self.reg);
        })
        .expect("report_qureg_params should never fail");
    }

    /// Initializes a `Qureg` to have all-zero-amplitudes.
    ///
    /// This is an unphysical state, useful for iteratively building a state
    /// with functions like
    /// [`set_weighted_qureg()`][api-set-weighted-qureg], and should
    /// not be confused with [`init_zero_state()`][api-init-zero-state].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    ///
    /// init_blank_state(qureg);
    ///
    /// assert!(get_prob_amp(qureg, 0).unwrap().abs() < EPSILON);
    /// assert!(get_prob_amp(qureg, 1).unwrap().abs() < EPSILON);
    /// assert!(get_prob_amp(qureg, 2).unwrap().abs() < EPSILON);
    /// assert!(get_prob_amp(qureg, 3).unwrap().abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [api-set-weighted-qureg]: crate::set_weighted_qureg()
    /// [api-init-zero-state]: crate::Qureg::init_zero_state()
    /// [api-qureg]: crate::Qureg
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn init_blank_state(&mut self) {
        catch_quest_exception(|| unsafe {
            ffi::initBlankState(self.reg);
        })
        .expect("init_blank_state should always succeed");
    }

    /// Initialize `qureg` into the zero state.
    ///
    /// If `qureg` is a state-vector of `N` qubits, it is modified to state
    /// `|0>^{\otimes N}`.  If `qureg` is a density matrix of `N` qubits, it is
    /// modified to state `|0><0|^{\otimes N}`.
    ///
    /// # Parameters
    ///
    /// - `qureg`: a [`Qureg`][api-qureg] of which to clear all amplitudes
    ///
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    ///
    /// init_zero_state(qureg);
    ///
    /// assert!((get_prob_amp(qureg, 0).unwrap() - 1.).abs() < EPSILON);
    /// assert!(get_prob_amp(qureg, 1).unwrap().abs() < EPSILON);
    /// assert!(get_prob_amp(qureg, 2).unwrap().abs() < EPSILON);
    /// assert!(get_prob_amp(qureg, 3).unwrap().abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [api-qureg]: crate::Qureg
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn init_zero_state(&mut self) {
        catch_quest_exception(|| unsafe {
            ffi::initZeroState(self.reg);
        })
        .expect("init_zero_state should always succeed");
    }

    /// Initialize `qureg` into the plus state.
    ///
    /// If `qureg` is a state-vector of `N` qubits, it is modified to state:
    ///
    /// ```latex
    ///   {| + \rangle}^{\otimes N} = \frac{1}{\sqrt{2^N}} (| 0 \rangle + | 1 \rangle)^{\otimes N}.
    /// ```
    ///
    /// If `qureg` is a density matrix of `N`, it is modified to state:
    ///
    /// ```latex
    ///   {| + \rangle\langle+|}^{\otimes N} = \frac{1}{{2^N}} \sum_i\sum_j |i\rangle\langle j|.
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    ///
    /// init_plus_state(qureg);
    ///
    /// assert!((get_prob_amp(qureg, 0).unwrap() - 0.25).abs() < EPSILON);
    /// assert!((get_prob_amp(qureg, 1).unwrap() - 0.25).abs() < EPSILON);
    /// assert!((get_prob_amp(qureg, 2).unwrap() - 0.25).abs() < EPSILON);
    /// assert!((get_prob_amp(qureg, 3).unwrap() - 0.25).abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [api-qureg]: crate::Qureg
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn init_plus_state(&mut self) {
        catch_quest_exception(|| unsafe {
            ffi::initPlusState(self.reg);
        })
        .expect("init_plus_state should always succeed");
    }

    /// Initialize `qureg` into a classical state.
    ///
    /// This state is also known as a "computational basis state" with index
    /// `state_ind`.
    ///
    /// If `qureg` is a state-vector, it will become: `|state_ind>`. If
    /// `qureg`is a density matrix, it will become:
    ///
    /// ```text
    ///   |state_ind> <state_ind|
    /// ````
    ///
    /// Classical states are indexed from zero, so that `state_ind=0` produces
    /// `|0..00>`,  and  `state_ind=1` produces `|00..01>`, and `state_ind=2^N -
    /// 1` produces `|11..11>`. Subsequent calls to
    /// [`get_prob_amp()`][api-get-prob-amp] will yield `0` for all indices
    /// except `state_ind`,  and the phase of `state_ind`'s amplitude will
    /// be `1` (real).
    ///
    /// This function can be used to initialise `qureg` into a specific binary
    /// state  (e.g. `11001`) using a binary literal.
    ///
    /// # Parameters
    ///
    ///  - `qureg`: the register to modify
    ///  - `state_ind` the index of the basis state to modify `qureg` into
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if `state_ind` is outside [0, qureg.[`num_qubits_represented()`]).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
    ///
    /// init_classical_state(qureg, 8);
    /// let prob = get_prob_amp(qureg, 0).unwrap();
    ///
    /// assert!((prob.abs() - 1.) < EPSILON);
    /// ```
    ///
    ///
    /// See [QuEST API] for more information.
    ///
    /// [api-get-prob-amp]: crate::get_prob_amp()
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn init_classical_state(
        &mut self,
        state_ind: i64,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::initClassicalState(self.reg, state_ind);
        })
    }

    /// Initialize `qureg` into a pure state.
    ///
    /// - If `qureg` is a state-vector, this merely clones `pure` into `qureg`.
    /// - If `qureg` is a density matrix, this makes `qureg` 100% likely to be
    ///   in the `pure` state.
    ///
    /// # Parameters
    ///
    /// - `qureg`: the register to modify
    /// - `pure`: a state-vector containing the pure state into which to
    ///   initialize `qureg`
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if `qureg` and `pure` have mismatching dimensions
    ///   - if `pure` is a density matrix
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut Qureg::try_new_density(3, env).unwrap();
    /// let pure_state = &mut create_qureg::<3>(env);
    ///
    /// init_zero_state(pure_state);
    /// init_pure_state(qureg, pure_state).unwrap();
    ///
    /// assert!((calc_purity(qureg).unwrap() - 1.0).abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn init_pure_state(
        &mut self,
        pure_: &Qureg<'_, N>,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::initPureState(self.reg, pure_.reg);
        })
    }

    /// Initialize `qureg` to be in a debug state.
    ///
    /// Set `qureg` to be in the un-normalized, non-physical state with
    /// with `n`th complex amplitude given by:
    ///
    /// ```text
    ///   2n/10 + i*(2n+1)/10.
    /// ```
    ///
    /// This is used internally for debugging and testing.
    ///
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn init_debug_state(&mut self) {
        catch_quest_exception(|| unsafe {
            ffi::initDebugState(self.reg);
        })
        .expect("init_debug_state() should always succeed");
    }

    /// Initialize `qureg` by specifying all amplitudes.
    ///
    /// For density matrices, it is assumed the amplitudes have been flattened
    /// column-wise into the given arrays.
    ///
    /// The real and imaginary components of the amplitudes are passed in
    /// separate arrays, `reals` and `imags`, each of which must have length
    /// [`qureg.get_num_amps_total()`]. There is no automatic checking that the
    /// passed arrays are L2 normalized, so this can be used to prepare `qureg`
    /// in a non-physical state.
    ///
    /// In distributed mode, this would require the complete state to fit in
    /// every node. To manually prepare a state for which all amplitudes cannot
    /// fit into a single node, use [`set_amps()`]
    ///
    /// # Parameters
    ///
    /// - `qureg`: the register to overwrite
    /// - `reals`: array of the real components of the new amplitudes
    /// - `imags`: array of the imaginary components of the new amplitudes
    ///
    /// # Errors
    ///
    /// - [`ArrayLengthError`],
    ///   - if either `reals` or `imags` have fewer than
    ///     [`qureg.get_num_amps_total()`] elements
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    ///
    /// init_state_from_amps(qureg, &[1., 0., 0., 0.], &[0., 0., 0., 0.]);
    /// let prob = get_prob_amp(qureg, 0).unwrap();
    ///
    /// assert!((prob - 1.).abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`qureg.get_num_amps_total()`]: crate::Qureg::get_num_amps_total()
    /// [`set_amps()`]: crate::set_amps()
    /// [`ArrayLengthError`]: crate::QuestError::ArrayLengthError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn init_state_from_amps(
        &mut self,
        reals: &[Qreal],
        imags: &[Qreal],
    ) -> Result<(), QuestError> {
        let num_amps_total = self.get_num_amps_total() as usize;
        if reals.len() < num_amps_total || imags.len() < num_amps_total {
            return Err(QuestError::ArrayLengthError);
        }
        catch_quest_exception(|| unsafe {
            ffi::initStateFromAmps(self.reg, reals.as_ptr(), imags.as_ptr());
        })
    }

    /// Overwrites a contiguous subset of the amplitudes in a state-vector.
    ///
    /// Only amplitudes with indices in `[start_ind,  start_ind + reals.len()]`
    /// will be changed. The resulting `qureg` may not necessarily be in an
    /// L2 normalized state.
    ///
    /// In distributed mode, this function assumes the subset `reals` and
    /// `imags` exist (at least) on the node containing the ultimately
    /// updated elements. For example, below is the correct way to modify
    /// the full 8 elements of `qureg` when split between 2 nodes:
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
    ///
    /// let re = &mut [1., 2., 3., 4.];
    /// let im = &mut [1., 2., 3., 4.];
    /// set_amps(qureg, 0, re, im);
    ///
    /// // modify re and im to the next set of elements
    /// for i in 0..4 {
    ///     re[i] += 4.;
    ///     im[i] += 4.;
    /// }
    /// set_amps(qureg, 4, re, im);
    /// ```
    ///
    /// # Parameters
    ///
    /// - `qureg`: the state-vector to modify
    /// - `start_ind`: the index of the first amplitude in `qureg` to modify
    /// - `reals`: array of the real components of the new amplitudes
    /// - `imags`: array of the imaginary components of the new amplitudes
    ///
    /// # Errors
    ///
    /// - [`ArrayLengthError`]
    ///   - if `reals.len()` and `imags.len()` are different
    ///
    /// - [`InvalidQuESTInputError`]
    ///   - if `qureg` is not a state-vector (i.e. is a density matrix)
    ///   - if `start_ind` is outside [0, [`qureg.get_num_amps_total()`]]
    ///   - if `reals.len()` is outside [0, `qureg.get_num_amps_total()`]
    ///   - if `reals.len()` + `start_ind` >= `qureg.get_num_amps_total()`
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    ///
    /// let re = &[1., 2., 3.];
    /// let im = &[4., 5., 6.];
    /// let start_ind = 1;
    /// set_amps(qureg, start_ind, re, im);
    ///
    /// let amp = get_real_amp(qureg, 3).unwrap();
    /// assert!((amp - 3.).abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`qureg.get_num_amps_total()`]: crate::Qureg::get_num_amps_total()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`ArrayLengthError`]: crate::QuestError::ArrayLengthError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn set_amps(
        &mut self,
        start_ind: i64,
        reals: &[Qreal],
        imags: &[Qreal],
    ) -> Result<(), QuestError> {
        if reals.len() != imags.len() {
            return Err(QuestError::ArrayLengthError);
        }
        let num_amps = reals.len() as i64;
        catch_quest_exception(|| unsafe {
            ffi::setAmps(
                self.reg,
                start_ind,
                reals.as_ptr(),
                imags.as_ptr(),
                num_amps,
            );
        })
    }

    /// Overwrites a contiguous subset of the amplitudes in a density-matrix.
    ///
    /// Only the first `reals.len()` amplitudes starting from row-column index
    /// `(start_row, start_col)`, and proceeding down the column (wrapping
    /// around between rows) will be modified. The resulting `qureg` may not
    /// necessarily be in an L2 normalized state.
    ///
    /// In distributed mode, this function assumes the subset `reals` and
    /// `imags` exist (at least) on the node containing the ultimately
    /// updated elements. See also [`set_amps()`] for more details.
    ///
    /// # Parameters
    ///
    /// - `qureg`: the state-vector to modify
    /// - `start_row`: the row-index of the first amplitude in `qureg` to modify
    /// - `start_col`: the column-index of the first amplitude in `qureg` to
    ///   modify
    /// - `reals`: array of the real components of the new amplitudes
    /// - `imags`: array of the imaginary components of the new amplitudes
    ///
    /// # Errors
    ///
    /// - [`ArrayLengthError`]
    ///   - if `reals.len()` and `imags.len()` are different
    ///
    /// - [`InvalidQuESTInputError`]
    ///   - if `qureg` is not a density-matrix (i.e. is a state vector)
    ///   - if `start_row` is outside [0, 1 <<
    ///     [`qureg.num_qubits_represented()`]]
    ///   - if `start_col` is outside [0, 1 <<
    ///     [`qureg.num_qubits_represented()`]]
    ///   - if `reals.len()` is outside [0, `qureg.get_num_amps_total()`]
    ///   - if `reals.len()` is larger than the remaining number of amplitudes
    ///     from (`start_row`, `start_col`), column-wise
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut Qureg::try_new_density(2, env).unwrap();
    ///
    /// let re = &[1., 2., 3.];
    /// let im = &[4., 5., 6.];
    /// let start_row = 1;
    /// let start_col = 1;
    /// set_density_amps(qureg, start_row, start_col, re, im);
    ///
    /// let amp = get_density_amp(qureg, 2, 1).unwrap();
    ///
    /// assert!((amp.re - 2.).abs() < EPSILON);
    /// assert!((amp.im - 5.).abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`set_amps()`]: crate::set_amps()
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`ArrayLengthError`]: crate::QuestError::ArrayLengthError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn set_density_amps(
        &mut self,
        start_row: i64,
        start_col: i64,
        reals: &[Qreal],
        imags: &[Qreal],
    ) -> Result<(), QuestError> {
        if reals.len() != imags.len() {
            return Err(QuestError::ArrayLengthError);
        }
        let num_amps = reals.len() as i64;
        catch_quest_exception(|| unsafe {
            ffi::setDensityAmps(
                self.reg,
                start_row,
                start_col,
                reals.as_ptr(),
                imags.as_ptr(),
                num_amps,
            );
        })
    }

    /// Shift the phase of a single qubit by a given angle.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
    ///
    /// let target_qubit = 1;
    /// let angle = 0.5;
    ///
    /// phase_shift(qureg, target_qubit, angle).unwrap();
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn phase_shift(
        &mut self,
        target_quibit: i32,
        angle: Qreal,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::phaseShift(self.reg, target_quibit, angle);
        })
    }

    /// Introduce a phase factor on state of qubits.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
    ///
    /// let id_qubit1 = 0;
    /// let id_qubit2 = 2;
    /// let angle = 0.5;
    /// controlled_phase_shift(qureg, id_qubit1, id_qubit2, angle).unwrap();
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn controlled_phase_shift(
        &mut self,
        id_qubit1: i32,
        id_qubit2: i32,
        angle: Qreal,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::controlledPhaseShift(self.reg, id_qubit1, id_qubit2, angle);
        })
    }

    /// Introduce a phase factor of the passed qubits.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<4>(env);
    ///
    /// let control_qubits = &[0, 1, 3];
    /// let angle = 0.5;
    /// multi_controlled_phase_shift(qureg, control_qubits, angle).unwrap();
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn multi_controlled_phase_shift(
        &mut self,
        control_qubits: &[i32],
        angle: Qreal,
    ) -> Result<(), QuestError> {
        let num_control_qubits = control_qubits.len() as i32;
        catch_quest_exception(|| unsafe {
            ffi::multiControlledPhaseShift(
                self.reg,
                control_qubits.as_ptr(),
                num_control_qubits,
                angle,
            );
        })
    }

    /// Apply the (two-qubit) controlled phase flip gate.
    ///
    /// Also known as the controlled pauliZ gate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_zero_state(qureg);
    ///
    /// controlled_phase_flip(qureg, 0, 1);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn controlled_phase_flip(
        &mut self,
        id_qubit1: i32,
        id_qubit2: i32,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::controlledPhaseFlip(self.reg, id_qubit1, id_qubit2);
        })
    }

    /// Apply the (multiple-qubit) controlled phase flip gate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<4>(env);
    /// init_zero_state(qureg);
    ///
    /// let control_qubits = &[0, 1, 3];
    /// multi_controlled_phase_flip(qureg, control_qubits);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn multi_controlled_phase_flip(
        &mut self,
        control_qubits: &[i32],
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::multiControlledPhaseFlip(
                self.reg,
                control_qubits.as_ptr(),
                control_qubits.len() as i32,
            );
        })
    }

    /// Apply the single-qubit S gate.
    ///
    /// This is a rotation of `PI/2` around the Z-axis on the Bloch sphere, or
    /// the unitary:
    ///
    /// ```text
    ///   [ 1  0 ]
    ///   [ 0  i ]
    /// ```
    ///
    /// # Parameters
    ///
    /// - `qureg`: object representing the set of all qubits
    /// - `target_qubit`: qubit to operate upon
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if `target_qubit` is outside [0,
    ///     qureg.[`num_qubits_represented()`]).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_zero_state(qureg);
    /// pauli_x(qureg, 0).unwrap();
    ///
    /// s_gate(qureg, 0).unwrap();
    ///
    /// let amp = get_imag_amp(qureg, 1).unwrap();
    /// assert!((amp - 1.).abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn s_gate(
        &mut self,
        target_qubit: i32,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::sGate(self.reg, target_qubit);
        })
    }

    /// Apply the single-qubit T gate.
    ///
    /// This is a rotation of `PI/4` around the Z-axis on the Bloch sphere, or
    /// the unitary:
    ///
    /// ```text
    ///   [ 1       0       ]
    ///   [ 0  e^(i PI / 4) ]
    /// ```
    ///
    /// # Parameters
    ///
    /// - `qureg`: object representing the set of all qubits
    /// - `target_qubit`: qubit to operate upon
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if `target_qubit` is outside [0,
    ///     qureg.[`num_qubits_represented()`]).
    ///
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_zero_state(qureg);
    /// pauli_x(qureg, 0).unwrap();
    ///
    /// t_gate(qureg, 0).unwrap();
    ///
    /// let amp = get_imag_amp(qureg, 1).unwrap();
    /// assert!((amp - SQRT_2 / 2.).abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn t_gate(
        &mut self,
        target_qubit: i32,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::tGate(self.reg, target_qubit);
        })
    }

    /// Overwrite the amplitudes of `target_qureg` with those from `copy_qureg`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let target_qureg = &mut create_qureg::<3>(env);
    /// let copy_qureg = &create_qureg::<3>(env);
    ///
    /// clone_qureg(target_qureg, copy_qureg);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn clone_qureg(
        &mut self,
        copy_qureg: &Qureg<'_, N>,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::cloneQureg(self.reg, copy_qureg.reg);
        })
    }

    /// Performs a logical AND on all successCodes held by all processes.
    ///
    /// If any one process has a zero `success_code`, all processes will return
    /// a zero success code.
    ///
    /// # Parameters
    ///
    /// - `success_code`: `1` if process task succeeded, `0` if process task
    ///   failed
    ///
    /// # Returns
    ///
    /// `1` if all processes succeeded, `0` if any one process failed
    ///
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[must_use]
    pub fn sync_quest_success(success_code: i32) -> i32 {
        catch_quest_exception(|| unsafe { ffi::syncQuESTSuccess(success_code) })
            .expect("sync_quest_success should always succeed")
    }

    /// Copy the state-vector (or density matrix) into GPU memory.
    ///
    /// In GPU mode, this copies the state-vector (or density matrix) from RAM
    /// to VRAM / GPU-memory, which is the version operated upon by other calls
    /// to the API.
    ///
    /// In CPU mode, this function has no effect.
    ///
    /// In conjunction with [`copy_state_from_gpu()`][api-copy-state-from-gpu]
    /// (which should be called first), this allows a user to directly modify
    /// the state-vector in a hardware agnostic way. Note though that users
    /// should instead use [`set_amps()`][api-set-amps] if possible.
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
    /// let qureg = &mut create_qureg::<2>(env);
    ///
    /// copy_state_to_gpu(qureg);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [api-copy-state-from-gpu]: crate::copy_state_from_gpu()
    /// [api-set-amps]: crate::set_amps()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn copy_state_to_gpu(&mut self) {
        catch_quest_exception(|| unsafe {
            ffi::copyStateToGPU(self.reg);
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
    /// In conjunction with [`copy_state_to_gpu()`][api-copy-state-to-gpu] ,
    /// this allows a user to directly modify the state-vector in a hardware
    /// agnostic way. Note though that users should instead use
    /// [`set_amps()`][api-set-amps] if possible.
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
    /// let qureg = &mut create_qureg::<2>(env);
    ///
    /// copy_state_from_gpu(qureg);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [api-copy-state-to-gpu]: crate::copy_state_to_gpu()
    /// [api-set-amps]: crate::set_amps()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn copy_state_from_gpu(&mut self) {
        catch_quest_exception(|| unsafe { ffi::copyStateFromGPU(self.reg) })
            .expect("copy_state_from_gpu should always succeed");
    }

    /// Copy a part the state-vector (or density matrix) into GPU memory.
    ///
    /// In GPU mode, this copies a substate of the state-vector (or density
    /// matrix) from RAM to VRAM / GPU-memory.
    ///
    /// In CPU mode, this function has no effect.
    ///
    /// In conjunction with
    /// [`copy_substate_from_gpu()`][api-copy-substate-from-gpu], this allows a
    /// user to directly modify a subset of the amplitudes the state-vector
    /// in a hardware agnostic way, without having to load the entire state
    /// via [`copy_state_to_gpu()`][api-copy-state-to-gpu].
    ///
    /// Note though that users should instead use [`set_amps()`][api-set-amps]
    /// if possible.
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
    /// - [`InvalidQuESTInputError`],
    ///   - if `start_ind` is an invalid amplitude index
    ///   - if `num_amps` is greater than the remaining amplitudes in the state,
    ///     from `start_ind`
    ///
    ///
    /// See [QuEST API] for more information.
    ///
    /// [api-copy-substate-from-gpu]: crate::copy_substate_from_gpu()
    /// [api-copy-state-to-gpu]: crate::copy_state_to_gpu()
    /// [api-set-amps]: crate::set_amps()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn copy_substate_to_gpu(
        &mut self,
        start_ind: i64,
        num_amps: i64,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::copySubstateToGPU(self.reg, start_ind, num_amps);
        })
    }

    /// Copy a part the state-vector (or density matrix) from GPU memory.
    ///
    /// In GPU mode, this copies a substate of the state-vector (or density
    /// matrix) from  to VRAM / GPU-memory to RAM, which is the version
    /// operated upon by other calls to the API.
    ///
    /// In CPU mode, this function has no effect.
    ///
    /// In conjunction with
    /// [`copy_substate_to_gpu()`][api-copy-substate-to-gpu], this allows a user
    /// to directly modify a subset of the amplitudes the state-vector in a
    /// hardware agnostic way, without having to load the entire state via
    /// [`copy_state_from_gpu()`][api-copy-state-from-gpu].
    ///
    /// Note though that users should instead use [`set_amps()`][api-set-amps]
    /// if possible.
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
    /// - [`InvalidQuESTInputError`],
    ///   - if `start_ind` is an invalid amplitude index
    ///   - if `num_amps` is greater than the remaining amplitudes in the state,
    ///     from `start_ind`
    ///
    ///
    /// See [QuEST API] for more information.
    ///
    /// [api-copy-substate-to-gpu]: crate::copy_substate_to_gpu()
    /// [api-copy-state-from-gpu]: crate::copy_state_from_gpu()
    /// [api-set-amps]: crate::set_amps()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn copy_substate_from_gpu(
        &mut self,
        start_ind: i64,
        num_amps: i64,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::copySubstateToGPU(self.reg, start_ind, num_amps);
        })
    }

    /// Get the complex amplitude at a given index in the state vector.
    ///
    /// # Parameters
    ///
    /// - `qureg`: object representing a set of qubits
    /// - `index`: index in state vector of probability amplitudes
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if `qureg` is a density matrix
    ///   - if `index` is outside [0, qureg.[`num_qubits_represented()`]).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_plus_state(qureg);
    ///
    /// let amp = get_amp(qureg, 0).unwrap().re;
    /// assert!((amp - 0.5).abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    pub fn get_amp(
        &self,
        index: i64,
    ) -> Result<Qcomplex, QuestError> {
        catch_quest_exception(|| unsafe { ffi::getAmp(self.reg, index) })
            .map(Into::into)
    }

    /// Get the real part of the probability amplitude at an index in
    /// the state vector.
    ///
    /// # Parameters
    ///
    /// - `qureg`: object representing a set of qubits
    /// - `index`: index in state vector of probability amplitudes
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if `qureg` is a density matrix
    ///   - if `index` is outside [0, qureg.[`num_qubits_represented()`]).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_plus_state(qureg);
    ///
    /// let amp = get_real_amp(qureg, 0).unwrap();
    /// assert!((amp - 0.5).abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    pub fn get_real_amp(
        &self,
        index: i64,
    ) -> Result<Qreal, QuestError> {
        catch_quest_exception(|| unsafe { ffi::getRealAmp(self.reg, index) })
    }

    /// Get the imaginary part of the probability amplitude at an index
    /// in the state vector.
    ///
    /// # Parameters
    ///
    /// - `qureg`: object representing a set of qubits
    /// - `index`: index in state vector of probability amplitudes
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if `qureg` is a density matrix
    ///   - if `index` is outside [0, qureg.[`num_qubits_represented()`]).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_plus_state(qureg);
    ///
    /// let amp = get_imag_amp(qureg, 0).unwrap();
    /// assert!(amp.abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    pub fn get_imag_amp(
        &self,
        index: i64,
    ) -> Result<Qreal, QuestError> {
        catch_quest_exception(|| unsafe { ffi::getImagAmp(self.reg, index) })
    }

    /// Get the probability of a state-vector at an index in the full state
    /// vector.
    ///
    /// # Parameters
    ///
    /// - `qureg`: object representing a set of qubits
    /// - `index`: index in state vector of probability amplitudes
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if `qureg` is a density matrix
    ///   - if `index` is outside [0, qureg.[`num_qubits_represented()`]).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_plus_state(qureg);
    ///
    /// let amp = get_prob_amp(qureg, 0).unwrap();
    /// assert!((amp - 0.25).abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    pub fn get_prob_amp(
        &self,
        index: i64,
    ) -> Result<Qreal, QuestError> {
        catch_quest_exception(|| unsafe { ffi::getProbAmp(self.reg, index) })
    }

    /// Get an amplitude from a density matrix at a given row and column.
    ///
    /// # Parameters
    ///
    /// - `qureg`: object representing a set of qubits
    /// - `row`: row of the desired amplitude in the density matrix
    /// - `col`: column of the desired amplitude in the density matrix
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if `qureg` is a state vector
    ///   - if `row` or `col` are outside [0,
    ///     qureg.[`num_qubits_represented()`]).
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
    /// See [QuEST API] for more information.
    ///
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    pub fn get_density_amp(
        &self,
        row: i64,
        col: i64,
    ) -> Result<Qcomplex, QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::getDensityAmp(self.reg, row, col)
        })
        .map(Into::into)
    }

    /// A debugging function which calculates the total probability of the
    /// qubits.
    ///
    /// This function should always be 1 for correctly normalised states
    /// (hence returning a real number).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_plus_state(qureg);
    ///
    /// let amp = calc_total_prob(qureg);
    /// assert!((amp - 1.).abs() < EPSILON)
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[must_use]
    pub fn calc_total_prob(&self) -> Qreal {
        catch_quest_exception(|| unsafe { ffi::calcTotalProb(self.reg) })
            .expect("calc_total_prop should always succeed")
    }

    /// Apply a single-qubit unitary parameterized by two given complex scalars.
    ///
    /// Given valid complex numbers `alpha` and `beta`, applies the unitary:
    ///
    /// ```text
    /// [ alpha -beta.conj() ]
    /// [ beta  alpha.conj() ]
    /// ```
    ///
    /// Valid `alpha`, `beta` satisfy `|alpha|^2 + |beta|^2 = 1`.
    /// The target unitary is general up to a global phase factor.
    ///
    /// # Parameters
    ///
    /// - `qureg`: object representing the set of all qubits
    /// - `target_qubit`: qubit to operate on
    /// - `alpha`: complex unitary parameter (row 1, column 1)
    /// - `beta`: complex unitary parameter (row 2, column 1)
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if `target_qubit` is outside [0,
    ///     qureg.[`num_qubits_represented()`]).
    ///   - if  `alpha`, `beta` don't satisfy: `|alpha|^2 + |beta|^2 = 1`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_zero_state(qureg);
    ///
    /// let norm = SQRT_2.recip();
    /// let alpha = Qcomplex::new(0., norm);
    /// let beta = Qcomplex::new(0., norm);
    /// compact_unitary(qureg, 0, alpha, beta).unwrap();
    ///
    /// let other_qureg = &mut create_qureg::<2>(env);
    /// init_zero_state(other_qureg);
    /// hadamard(other_qureg, 0).unwrap();
    ///
    /// let fidelity = calc_fidelity(qureg, other_qureg).unwrap();
    /// assert!((fidelity - 1.).abs() < 10. * EPSILON,);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn compact_unitary(
        &mut self,
        target_qubit: i32,
        alpha: Qcomplex,
        beta: Qcomplex,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::compactUnitary(
                self.reg,
                target_qubit,
                alpha.into(),
                beta.into(),
            );
        })
    }

    /// Apply a general single-qubit unitary (including a global phase factor).
    ///
    /// The passed 2x2 `ComplexMatrix` must be unitary, otherwise an error is
    /// thrown.
    ///
    /// # Parameters
    ///
    /// - `qureg`: object representing the set of all qubits
    /// - `target_qubit`: qubit to operate on
    /// - `u`: single-qubit unitary matrix to apply
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if `target_qubit` is outside [0,
    ///     qureg.[`num_qubits_represented()`]).
    ///   - if `u` is not unitary
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_zero_state(qureg);
    ///
    /// let norm = SQRT_2.recip();
    /// let mtr = ComplexMatrix2::new(
    ///     [[norm, norm], [norm, -norm]],
    ///     [[0., 0.], [0., 0.]],
    /// );
    /// unitary(qureg, 0, &mtr).unwrap();
    ///
    /// let other_qureg = &mut create_qureg::<2>(env);
    /// init_zero_state(other_qureg);
    /// hadamard(other_qureg, 0).unwrap();
    ///
    /// let fidelity = calc_fidelity(qureg, other_qureg).unwrap();
    /// assert!((fidelity - 1.).abs() < 10. * EPSILON,);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn unitary(
        &mut self,
        target_qubit: i32,
        u: &ComplexMatrix2,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::unitary(self.reg, target_qubit, u.0);
        })
    }

    /// Rotate a single qubit by a given angle around the X-axis of the
    /// Bloch-sphere.
    ///
    /// For angle `theta`, this applies
    /// ```text
    /// [    cos(theta/2)   -i sin(theta/2) ]
    /// [ -i sin(theta/2)      cos(theta/2) ]
    /// ```
    ///
    /// # Parameters
    ///
    /// - `qureg`: object representing the set of all qubits
    /// - `rot_qubit`: qubit to rotate
    /// - `angle`: angle by which to rotate in radians
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if `rot_qubit` is outside [0, qureg.[`num_qubits_represented()`]).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
    /// let theta = PI;
    ///
    /// rotate_x(qureg, 0, theta).unwrap();
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn rotate_x(
        &mut self,
        rot_qubit: i32,
        angle: Qreal,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::rotateX(self.reg, rot_qubit, angle);
        })
    }

    /// Rotate a single qubit by a given angle around the Y-axis of the
    /// Bloch-sphere.
    ///
    /// For angle `theta`, this applies
    /// ```text
    /// [  cos(theta/2)   -sin(theta/2) ]
    /// [ -sin(theta/2)    cos(theta/2) ]
    /// ```
    ///
    /// # Parameters
    ///
    /// - `qureg`: object representing the set of all qubits
    /// - `rot_qubit`: qubit to rotate
    /// - `angle`: angle by which to rotate in radians
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if `rot_qubit` is outside [0, qureg.[`num_qubits_represented()`]).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
    /// let theta = PI;
    ///
    /// rotate_y(qureg, 0, theta).unwrap();
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn rotate_y(
        &mut self,
        rot_qubit: i32,
        angle: Qreal,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::rotateY(self.reg, rot_qubit, angle);
        })
    }

    /// Rotate a single qubit by a given angle around the Z-axis of the
    /// Bloch-sphere.
    ///
    /// For angle `theta`, this applies
    /// ```text
    /// [ exp(-i theta/2)         0     ]
    /// [       0          exp(theta/2) ]
    /// ```
    ///
    /// # Parameters
    ///
    /// - `qureg`: object representing the set of all qubits
    /// - `rot_qubit`: qubit to rotate
    /// - `angle`: angle by which to rotate in radians
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if `rot_qubit` is outside [0, qureg.[`num_qubits_represented()`]).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
    /// let theta = PI;
    ///
    /// rotate_z(qureg, 0, theta).unwrap();
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn rotate_z(
        &mut self,
        rot_qubit: i32,
        angle: Qreal,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::rotateZ(self.reg, rot_qubit, angle);
        })
    }

    /// Rotate a single qubit by a given angle around a given axis.
    ///
    /// The axis of rotation is given by a [`Vector`] on the Bloch-sphere.      
    /// The vector must not be zero (or else an error is thrown), but needn't be
    /// unit magnitude, since the normalization will be computed by by `QuEST`.
    ///
    /// # Parameters
    ///
    /// - `qureg`: object representing the set of all qubits
    /// - `rot_qubit`: qubit to rotate
    /// - `angle`: angle by which to rotate in radians
    /// - `axis`: vector around which to rotate (can be non-unit; will be
    ///   normalized)
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`]
    ///   - if `rot_qubit` is outside [0, qureg.[`num_qubits_represented()`])
    ///   - if `axis` is the zero vector
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
    /// init_zero_state(qureg);
    ///
    /// let angle = 2.0 * PI;
    /// let axis = &Vector::new(0., 0., 1.);
    /// rotate_around_axis(qureg, 0, angle, axis).unwrap();
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`Vector`]: crate::Vector
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn rotate_around_axis(
        &mut self,
        rot_qubit: i32,
        angle: Qreal,
        axis: &Vector,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::rotateAroundAxis(self.reg, rot_qubit, angle, axis.0);
        })
    }

    /// Applies a controlled rotation by a given angle around the X-axis of the
    /// Bloch-sphere.
    ///
    /// The target qubit is rotated in states where the control qubit has value
    /// `1`.
    ///
    /// # Parameters
    ///
    /// - `qureg`: object representing the set of all qubits
    /// - `control_qubit`: qubit which has value `1` in the rotated states
    /// - `target_qubit`: qubit to rotate
    /// - `angle`: angle by which to rotate the target qubit in radians
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`]
    ///   - if either `control_qubit` or `target_qubit` are outside [0,
    ///     qureg.[`num_qubits_represented()`])
    ///   - if `control_qubit` and `target_qubit` are equal
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
    ///
    /// let control_qubit = 1;
    /// let target_qubit = 0;
    /// let angle = PI;
    /// controlled_rotate_x(qureg, control_qubit, target_qubit, angle).unwrap();
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn controlled_rotate_x(
        &mut self,
        control_qubit: i32,
        target_qubit: i32,
        angle: Qreal,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::controlledRotateX(
                self.reg,
                control_qubit,
                target_qubit,
                angle,
            );
        })
    }

    /// Applies a controlled rotation by a given angle around the Y-axis of the
    /// Bloch-sphere.
    ///
    /// The target qubit is rotated in states where the control qubit has value
    /// `1`.
    ///
    /// # Parameters
    ///
    /// - `qureg`: object representing the set of all qubits
    /// - `control_qubit`: qubit which has value `1` in the rotated states
    /// - `target_qubit`: qubit to rotate
    /// - `angle`: angle by which to rotate the target qubit in radians
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`]
    ///   - if either `control_qubit` or `target_qubit` are outside [0,
    ///     qureg.[`num_qubits_represented()`])
    ///   - if `control_qubit` and `target_qubit` are equal
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
    ///
    /// let control_qubit = 1;
    /// let target_qubit = 0;
    /// let angle = PI;
    /// controlled_rotate_y(qureg, control_qubit, target_qubit, angle).unwrap();
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn controlled_rotate_y(
        &mut self,
        control_qubit: i32,
        target_qubit: i32,
        angle: Qreal,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::controlledRotateY(
                self.reg,
                control_qubit,
                target_qubit,
                angle,
            );
        })
    }

    /// Applies a controlled rotation by a given angle around the Z-axis of the
    /// Bloch-sphere.
    ///
    /// The target qubit is rotated in states where the control qubit has value
    /// `1`.
    ///
    /// # Parameters
    ///
    /// - `qureg`: object representing the set of all qubits
    /// - `control_qubit`: qubit which has value `1` in the rotated states
    /// - `target_qubit`: qubit to rotate
    /// - `angle`: angle by which to rotate the target qubit in radians
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`]
    ///   - if either `control_qubit` or `target_qubit` are outside [0,
    ///     qureg.[`num_qubits_represented()`])
    ///   - if `control_qubit` and `target_qubit` are equal
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
    ///
    /// let control_qubit = 1;
    /// let target_qubit = 0;
    /// let angle = PI;
    /// controlled_rotate_z(qureg, control_qubit, target_qubit, angle).unwrap();
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn controlled_rotate_z(
        &mut self,
        control_qubit: i32,
        target_qubit: i32,
        angle: Qreal,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::controlledRotateZ(
                self.reg,
                control_qubit,
                target_qubit,
                angle,
            );
        })
    }

    /// Applies a controlled rotation by  around a given vector of the
    /// Bloch-sphere.
    ///
    /// The vector must not be zero (else an error is thrown), but needn't be
    /// unit magnitude.
    ///
    /// # Parameters
    ///
    /// - `qureg`: object representing the set of all qubits
    /// - `control_qubit`: qubit which has value `1` in the rotated states
    /// - `target_qubit`: qubit to rotate
    /// - `angle`: angle by which to rotate in radians
    /// - `axis`: vector around which to rotate (can be non-unit; will be
    ///   normalized)
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`]
    ///   - if either `control_qubit` or `target_qubit` are outside [0,
    ///     qureg.[`num_qubits_represented()`])
    ///   - if `control_qubit` and `target_qubit` are equal
    ///   - if `axis` is the zero vector
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [`Vector`]: crate::Vector
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn controlled_rotate_around_axis(
        &mut self,
        control_qubit: i32,
        target_qubit: i32,
        angle: Qreal,
        axis: &Vector,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::controlledRotateAroundAxis(
                self.reg,
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
    ///  Given valid complex numbers `alpha` and `beta`, applies the two-qubit
    /// unitary:
    ///
    /// ```text
    /// [ alpha -beta.conj() ]
    /// [ beta  alpha.conj() ]
    /// ```
    ///
    /// Valid `alpha`, `beta` satisfy `|alpha|^2 + |beta|^2 = 1`.
    /// The target unitary is general up to a global phase factor.  
    ///
    /// # Parameters
    ///
    /// - `qureg`: object representing the set of all qubits
    /// - `control_qubit`: applies unitary if this qubit is `1`
    /// - `target_qubit`: qubit to operate on
    /// - `alpha`: complex unitary parameter (row 1, column 1)
    /// - `beta`: complex unitary parameter (row 2, column 1)
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if either `target_qubit` or `control_qubit` is outside [0,
    ///     [`qureg.num_qubits_represented()`]).
    ///   - if `control_qubits` and `target_qubit` are equal
    ///   - if  `alpha`, `beta` don't satisfy: `|alpha|^2 + |beta|^2 = 1`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_zero_state(qureg);
    ///
    /// let norm = SQRT_2.recip();
    /// let alpha = Qcomplex::new(0., norm);
    /// let beta = Qcomplex::new(0., norm);
    /// controlled_compact_unitary(qureg, 0, 1, alpha, beta).unwrap();
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn controlled_compact_unitary(
        &mut self,
        control_qubit: i32,
        target_qubit: i32,
        alpha: Qcomplex,
        beta: Qcomplex,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::controlledCompactUnitary(
                self.reg,
                control_qubit,
                target_qubit,
                alpha.into(),
                beta.into(),
            );
        })
    }

    /// Apply a general controlled unitary.
    ///
    /// The unitary can include a global phase factor and is applied
    /// to the target qubit if the control qubit has value `1`.
    ///
    /// # Parameters
    ///
    /// - `qureg`: object representing the set of all qubits
    /// - `control_qubit`: applies unitary if this qubit is `1`
    /// - `target_qubit`: qubit to operate on
    /// - `u`: single-qubit unitary matrix to apply
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if either `target_qubit` or `control_qubit` is outside [0,
    ///    [`qureg.num_qubits_represented()`]).
    ///   - if `control_qubits` and `target_qubit` are equal
    ///   - if `u` is not unitary
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn controlled_unitary(
        &mut self,
        control_qubit: i32,
        target_qubit: i32,
        u: &ComplexMatrix2,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::controlledUnitary(self.reg, control_qubit, target_qubit, u.0);
        })
    }

    /// Apply a general multiple-control single-target unitary.
    ///
    /// The unitary can include a global phase factor. Any number of control
    /// qubits can be specified, and if all have value `1`, the given
    /// unitary is applied to the target qubit.
    ///
    /// # Parameters
    ///
    /// - `qureg`: object representing the set of all qubits
    /// - `control_qubits`: applies unitary if all qubits in this slice are
    ///   equal to `1`
    /// - `target_qubit`: qubit to operate on
    /// - `u`: single-qubit unitary matrix to apply
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if `target_qubit` or any of `control_qubits` is outside [0,
    ///     [`qureg.num_qubits_represented()`]).
    ///   - if any qubit in `control_qubits` is repeated
    ///   - if `control_qubits` contains `target_qubit`
    ///   - if `u` is not unitary
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn multi_controlled_unitary(
        &mut self,
        control_qubits: &[i32],
        target_qubit: i32,
        u: &ComplexMatrix2,
    ) -> Result<(), QuestError> {
        let num_control_qubits = control_qubits.len() as i32;
        catch_quest_exception(|| unsafe {
            ffi::multiControlledUnitary(
                self.reg,
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
    /// - [`InvalidQuESTInputError`],
    ///   - if either `control_qubit` or `target_qubit` is outside [0,
    ///     [`qureg.num_qubits_represented()`])
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_zero_state(qureg);
    ///
    /// pauli_x(qureg, 0).unwrap();
    ///
    /// let amp = get_real_amp(qureg, 1).unwrap();
    /// assert!((amp - 1.).abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn pauli_x(
        &mut self,
        target_qubit: i32,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::pauliX(self.reg, target_qubit);
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
    /// - [`InvalidQuESTInputError`],
    ///   - if either `control_qubit` or `target_qubit` is outside [0,
    ///     [`qureg.num_qubits_represented()`])
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_zero_state(qureg);
    ///
    /// pauli_y(qureg, 0).unwrap();
    ///
    /// let amp = get_imag_amp(qureg, 1).unwrap();
    /// assert!((amp - 1.).abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn pauli_y(
        &mut self,
        target_qubit: i32,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::pauliY(self.reg, target_qubit);
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
    /// - [`InvalidQuESTInputError`],
    ///   - if either `control_qubit` or `target_qubit` is outside [0,
    ///     [`qureg.num_qubits_represented()`])
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_zero_state(qureg);
    ///
    /// pauli_z(qureg, 0).unwrap();
    ///
    /// let amp = get_real_amp(qureg, 0).unwrap();
    /// assert!((amp - 1.).abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn pauli_z(
        &mut self,
        target_qubit: i32,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::pauliZ(self.reg, target_qubit);
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
    /// - [`InvalidQuESTInputError`],
    ///   - if either `control_qubit` or `target_qubit` is outside [0,
    ///     [`qureg.num_qubits_represented()`])
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_zero_state(qureg);
    ///
    /// hadamard(qureg, 0).unwrap();
    ///
    /// let amp = get_real_amp(qureg, 0).unwrap();
    /// assert!((amp - SQRT_2.recip()).abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn hadamard(
        &mut self,
        target_qubit: i32,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::hadamard(self.reg, target_qubit);
        })
    }

    /// Apply the controlled not (single control, single target) gate.
    ///
    /// The gate is also known as the c-X, c-sigma-X, c-Pauli-X and c-bit-flip
    /// gate. This applies pauliX to the target qubit if the control qubit
    /// has value 1. This effects the two-qubit unitary:
    ///
    /// ```text
    ///  [ 1  0  0  0 ]
    ///  [ 0  1  0  0 ]
    ///  [ 0  0  0  1 ]
    ///  [ 0  0  1  0 ]
    /// ```
    ///
    /// on the control and target qubits.
    ///
    /// # Parameters
    ///
    /// - `qureg`: the state-vector or density matrix to modify
    /// - `control_qubit`: "nots" the target if this qubit is `1`
    /// - `target_qubit`: qubit to "not"
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if either `control_qubit` or `target_qubit` is outside [0,
    ///     qureg.[`num_qubits_represented()`])
    ///   - if `control_qubit` and `target_qubit` are equal
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_zero_state(qureg);
    /// pauli_x(qureg, 1).unwrap();
    ///
    /// controlled_not(qureg, 1, 0).unwrap();
    ///
    /// let amp = get_real_amp(qureg, 3).unwrap();
    /// assert!((amp - 1.).abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn controlled_not(
        &mut self,
        control_qubit: i32,
        target_qubit: i32,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::controlledNot(self.reg, control_qubit, target_qubit);
        })
    }

    /// Apply a NOT (or Pauli X) gate with multiple control and target qubits.
    ///
    /// This applies pauliX to qubits `targs` on every basis state for which the
    /// control qubits `ctrls` are all in the `|1>` state. The ordering within
    /// each of `ctrls` and `targs` has no effect on the operation.
    ///
    /// This function is equivalent, but significantly faster (approximately
    /// `targs.len()` times) than applying controlled NOTs on each qubit in
    /// `targs` in turn.
    ///
    /// In distributed mode, this operation requires at most a single round of)
    /// pair-wise communication between nodes, and hence is as efficient as
    /// [`pauli_x()`][api-pauli-x].
    ///
    /// # Parameters
    ///
    ///  - `qureg`: a state-vector or density matrix to modify
    ///  - `ctrls`: a list of the control qubit indices
    ///  - `targs`: a list of the qubits to be targeted by the X gates
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if any qubit in `ctrls` and `targs` is invalid, i.e. outside [0,
    ///     [`qureg.num_qubits_represented()`]).
    ///   - if the length of `targs` or `ctrls` is larger than
    ///     [`qureg.num_qubits_represented()`]
    ///   - if `ctrls` or `targs` contain any repetitions
    ///   - if any qubit in `ctrls` is also in `targs` (and vice versa)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<4>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [api-pauli-x]: crate::pauli_x()
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn multi_controlled_multi_qubit_not(
        &mut self,
        ctrls: &[i32],
        targs: &[i32],
    ) -> Result<(), QuestError> {
        let num_ctrls = ctrls.len() as i32;
        let num_targs = targs.len() as i32;
        catch_quest_exception(|| unsafe {
            ffi::multiControlledMultiQubitNot(
                self.reg,
                ctrls.as_ptr(),
                num_ctrls,
                targs.as_ptr(),
                num_targs,
            );
        })
    }

    /// Apply a NOT (or Pauli X) gate with multiple target qubits.
    ///
    /// This has the same  effect as (but is much faster than) applying each
    /// single-qubit NOT gate in turn.
    ///
    /// The ordering within `targs` has no effect on the operation.
    ///
    /// This function is equivalent, but significantly faster (approximately
    /// `targs.len()` times) than applying NOT on each qubit in `targs` in turn.
    ///
    /// In distributed mode, this operation requires at most a single round of)
    /// pair-wise communication between nodes, and hence is as efficient as
    /// [`pauli_x()`][api-pauli-x].
    ///
    /// # Parameters
    ///
    ///  - `qureg`: a state-vector or density matrix to modify
    ///  - `targs`: a list of the qubits to be targeted by the X gates
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if any qubit in `targs` is invalid, i.e. outside [0,
    ///     [`qureg.num_qubits_represented()`]).
    ///   - if the length of `targs` is larger than
    ///     [`qureg.num_qubits_represented()`]
    ///   - if `targs` contains any repetitions
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_zero_state(qureg);
    ///
    /// let targs = &[0, 1];
    /// multi_qubit_not(qureg, targs).unwrap();
    ///
    /// let amp = get_real_amp(qureg, 3).unwrap();
    /// assert!((amp - 1.).abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [api-pauli-x]: crate::pauli_x()
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn multi_qubit_not(
        &mut self,
        targs: &[i32],
    ) -> Result<(), QuestError> {
        let num_targs = targs.len() as i32;
        catch_quest_exception(|| unsafe {
            let targs_ptr = targs.as_ptr();
            ffi::multiQubitNot(self.reg, targs_ptr, num_targs);
        })
    }

    /// Apply the controlled pauli Y (single control, single target) gate.
    ///
    /// The gate is also known as the c-Y and c-sigma-Y gate.
    /// This applies pauli Y to the target qubit, if the control qubit has value
    /// 1. This effects the two-qubit unitary:
    ///
    /// ```text
    ///  [ 1  0  0   0 ]
    ///  [ 0  1  0   0 ]
    ///  [ 0  0  0  -i ]
    ///  [ 0  0  i   0 ]
    /// ```
    ///
    /// on the control and target qubits.
    ///
    /// # Parameters
    ///
    /// - `qureg`: the state-vector or density matrix to modify
    /// - `control_qubit`: applies pauli Y the target if this qubit is `1`
    /// - `target_qubit`: qubit to modify
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if either `control_qubit` or `target_qubit` is outside [0,
    ///     qureg.[`num_qubits_represented()`])
    ///   - if `control_qubit` and `target_qubit` are equal
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_zero_state(qureg);
    /// pauli_x(qureg, 1).unwrap();
    ///
    /// controlled_pauli_y(qureg, 1, 0).unwrap();
    ///
    /// let amp = get_imag_amp(qureg, 3).unwrap();
    /// assert!((amp - 1.).abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn controlled_pauli_y(
        &mut self,
        control_qubit: i32,
        target_qubit: i32,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::controlledPauliY(self.reg, control_qubit, target_qubit);
        })
    }

    /// Gives the probability of a qubit being measured in the given outcome.
    ///
    /// This performs no actual measurement and does not change the state of the
    /// qubits.
    ///
    /// - For state-vectors, this function works by summing the
    ///   absolute-value-squared of every amplitude in the state-vector for
    ///   which `measure_qubit = 0`. If `outcome = 1`, it returns `1` minus this
    ///   value. Hence for unnormalized state-vectors, this result will differ
    ///   from the absolute-value-squared of every amplitude where
    ///   `measure_qubit = outcome`.
    ///
    /// - For density matrices, this function sums the diagonal values (should
    ///   be real) corresponding to `measure_qubit = 0` (returning 1 minus this
    ///   if `outcome = 1`).
    ///
    /// # Parameters
    ///
    /// - `qureg`: object representing the set of all qubits
    /// - `measure_qubit`: qubit to study
    /// - `outcome`: for which to find the probability of the qubit being
    ///   measured in
    ///
    /// # Return
    ///
    /// Returns probability of qubit `measure_qubit` being measured in the given
    /// outcome.
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if `measure_qubit` is outside [0,
    ///     qureg.[`num_qubits_represented()`])
    ///   - if `outcome` is not in {0, 1}
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
    /// init_zero_state(qureg);
    ///
    /// let prob = calc_prob_of_outcome(qureg, 0, 0).unwrap();
    /// assert!((prob - 1.).abs() < EPSILON);
    /// let prob = calc_prob_of_outcome(qureg, 0, 1).unwrap();
    /// assert!(prob.abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    pub fn calc_prob_of_outcome(
        &self,
        measure_qubit: i32,
        outcome: i32,
    ) -> Result<Qreal, QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::calcProbOfOutcome(self.reg, measure_qubit, outcome)
        })
    }

    /// Calculate probabilities of every outcome of the sub-register.
    ///
    /// This function populates `outcome_probs` with the probabilities of every
    /// outcome of the sub-register contained in `qubits`.
    ///
    /// This performs no actual measurement and does not modify `qureg`.
    ///
    /// - `outcome_probs` must be a pre-allocated array of length
    ///   `2^qubits.len()`. In distributed mode, every node receives the full
    ///   list of outcome probabilities.
    ///
    /// - Note that the indices in `qubits` need not be adjacent nor ordered.
    ///   The order of `qubits` determines the order of `outcome_probs`, whereby
    ///   `qubits` are treated as *increasing* significance.
    ///
    /// - Since all probability amplitudes of a state-vector are ultimately
    ///   involved in the output probabilities, this function works as expected
    ///   for unnormalized states. This is similarly true for density matrices,
    ///   where all  diagonal elements are involved, although only the real
    ///   values of the diagonal elements will be consulted.
    ///
    /// # Parameters
    ///
    /// - `outcome_probs`: a pre-allocated array of length `1 << n`, where `n =
    ///   qubits.len()`  which will be modified to contain all outcome
    ///   probabilities
    /// - `qureg`: a state-vector or density matrix to study
    /// - `qubits`: a list of qubits to study
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if any index in `qubits` is invalid, i.e. outside [0,
    ///     qureg.[`num_qubits_represented()`])
    ///   - if `qubits` contains any repetitions
    /// - [`ArrayLengthError`],
    ///   - if `outcome_probs.len() < 1 << qubits.len()`
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
    /// init_zero_state(qureg);
    ///
    /// let qubits = &[1, 2];
    /// let outcome_probs = &mut vec![0.; 4];
    /// calc_prob_of_all_outcomes(outcome_probs, qureg, qubits).unwrap();
    /// assert_eq!(outcome_probs, &vec![1., 0., 0., 0.]);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`ArrayLengthError`]: crate::QuestError::ArrayLengthError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn calc_prob_of_all_outcomes(
        &self,
        outcome_probs: &mut [Qreal],
        qubits: &[i32],
    ) -> Result<(), QuestError> {
        if outcome_probs.len() < 1 << qubits.len() {
            return Err(QuestError::ArrayLengthError);
        }
        let num_qubits = qubits.len() as i32;
        let outcome_probs_ptr = outcome_probs.as_mut_ptr();
        catch_quest_exception(|| unsafe {
            ffi::calcProbOfAllOutcomes(
                outcome_probs_ptr,
                self.reg,
                qubits.as_ptr(),
                num_qubits,
            );
        })
    }

    /// Updates `qureg` to be consistent with measuring qubit in the given
    /// outcome.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_plus_state(qureg);
    ///
    /// collapse_to_outcome(qureg, 0, 0).unwrap();
    ///
    /// // QuEST throws an exception if probability of outcome is 0.
    /// init_zero_state(qureg);
    /// collapse_to_outcome(qureg, 0, 1).unwrap_err();
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn collapse_to_outcome(
        &mut self,
        measure_qubit: i32,
        outcome: i32,
    ) -> Result<Qreal, QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::collapseToOutcome(self.reg, measure_qubit, outcome)
        })
    }

    /// Measures a single qubit, collapsing it randomly to 0 or 1.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn measure(
        &mut self,
        measure_qubit: i32,
    ) -> Result<i32, QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::measure(self.reg, measure_qubit)
        })
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
    /// let qureg = &mut create_qureg::<2>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn measure_with_stats(
        &mut self,
        measure_qubit: i32,
        outcome_prob: &mut Qreal,
    ) -> Result<i32, QuestError> {
        let outcome_prob_ptr = outcome_prob as *mut _;
        catch_quest_exception(|| unsafe {
            ffi::measureWithStats(self.reg, measure_qubit, outcome_prob_ptr)
        })
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
    /// let qureg = &mut create_qureg::<2>(env);
    ///
    /// start_recording_qasm(qureg);
    /// hadamard(qureg, 0).and(controlled_not(qureg, 0, 1)).unwrap();
    /// stop_recording_qasm(qureg);
    ///
    /// print_recorded_qasm(qureg);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn start_recording_qasm(&mut self) {
        catch_quest_exception(|| unsafe {
            ffi::startRecordingQASM(self.reg);
        })
        .expect("start_recording_qasm should always succeed");
    }

    /// Disable QASM recording.
    ///
    /// The recorded QASM will be maintained in qureg and continue to be
    /// appended to if `startRecordingQASM` is recalled.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    ///
    /// start_recording_qasm(qureg);
    /// hadamard(qureg, 0).and(controlled_not(qureg, 0, 1)).unwrap();
    /// stop_recording_qasm(qureg);
    ///
    /// print_recorded_qasm(qureg);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn stop_recording_qasm(&mut self) {
        catch_quest_exception(|| unsafe {
            ffi::stopRecordingQASM(self.reg);
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
    /// let qureg = &mut create_qureg::<2>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn clear_recorded_qasm(&mut self) {
        catch_quest_exception(|| unsafe {
            ffi::clearRecordedQASM(self.reg);
        })
        .expect("clear_recorded_qasm should always succeed");
    }

    /// Print recorded QASM to stdout.
    ///
    /// This does not clear the QASM log, nor does it start or stop QASM
    /// recording.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    ///
    /// start_recording_qasm(qureg);
    /// hadamard(qureg, 0).and(controlled_not(qureg, 0, 1)).unwrap();
    /// stop_recording_qasm(qureg);
    ///
    /// print_recorded_qasm(qureg);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn print_recorded_qasm(&mut self) {
        catch_quest_exception(|| unsafe {
            ffi::printRecordedQASM(self.reg);
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
    /// let qureg = &mut create_qureg::<2>(env);
    ///
    /// start_recording_qasm(qureg);
    /// hadamard(qureg, 0).and(controlled_not(qureg, 0, 1)).unwrap();
    /// stop_recording_qasm(qureg);
    ///
    /// write_recorded_qasm_to_file(qureg, "/dev/null").unwrap();
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn write_recorded_qasm_to_file(
        &mut self,
        filename: &str,
    ) -> Result<(), QuestError> {
        unsafe {
            let filename_cstr =
                CString::new(filename).map_err(QuestError::NulError)?;
            catch_quest_exception(|| {
                ffi::writeRecordedQASMToFile(
                    self.reg,
                    (*filename_cstr).as_ptr(),
                );
            })
        }
    }

    /// Mixes a density matrix to induce single-qubit dephasing noise.
    ///
    /// With probability `prob`, applies Pauli Z to `target_qubit` in `qureg`.
    ///
    /// This transforms `qureg = rho` into the mixed state:
    ///
    /// ```text
    /// (1 - prob) * rho  +  prob * Z_q rho Z_q,
    /// ```
    ///
    /// where `q = target_qubit`. The coefficient `prob` cannot exceed `1/2`,
    /// which maximally mixes `target_qubit`.
    ///
    /// # Parameters
    ///
    /// - `qureg`: a density matrix
    /// - `target_qubit`: qubit upon which to induce dephasing noise
    /// - `prob`: the probability of the phase error occurring
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if `qureg` is not a density matrix
    ///   - if `target_qubit` is outside [0,
    ///     qureg.[`num_qubits_represented()`]).
    ///   - if `prob` is not in `[0, 1/2]`
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
    /// See [QuEST API] for more information.
    ///
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn mix_dephasing(
        &mut self,
        target_qubit: i32,
        prob: Qreal,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::mixDephasing(self.reg, target_qubit, prob);
        })
    }

    /// Mixes a density matrix `qureg` to induce two-qubit dephasing noise.
    ///
    /// With probability `prob`, applies Pauli Z to either or both in `qureg`.
    ///
    /// This transforms `qureg = rho` into the mixed state:
    ///
    /// ```text
    /// (1 - prob) * rho  +  prob * 1/3 * (
    ///         Z_a  rho  Z_a +
    ///         Z_b  rho  Z_b +
    ///         Z_a Z_b rho Z_a Z_b
    ///    )
    /// ```
    ///
    /// where `a = qubit1`, `b=qubit2`. The coefficient `prob` cannot exceed
    /// `3/4`, at which maximal mixing occurs.
    ///
    /// # Parameters
    ///
    /// - `qureg`: a density matrix
    /// - `qubit1`: qubit upon which to induce dephasing noise
    /// - `qubit2`: qubit upon which to induce dephasing noise
    /// - `prob`: the probability of the phase error occurring
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if `qureg` is not a density matrix
    ///   - if `qubit1` or `qubit2` are outside [0,
    ///     qureg.[`num_qubits_represented()`]).
    ///   - if `qubit1 = qubit2`
    ///   - if `prob` is not in `[0, 3/4]`
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
    /// See [QuEST API] for more information.
    ///
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn mix_two_qubit_dephasing(
        &mut self,
        qubit1: i32,
        qubit2: i32,
        prob: Qreal,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::mixTwoQubitDephasing(self.reg, qubit1, qubit2, prob);
        })
    }

    /// Mixes a density matrix to induce single-qubit homogeneous
    /// depolarising noise.
    /// This is equivalent to, with probability `prob`, uniformly randomly
    /// applying either Pauli X, Y, or Z to `target_qubit`.
    ///
    /// This transforms `qureg = rho` into the mixed state:
    ///
    /// ```text
    /// (1 - prob) * rho  +  prob * 1/3 * (
    ///      X_q rho X_q +
    ///      Y_q rho Y_q +
    ///      Z_q rho Z_q
    /// )
    /// ```
    ///
    /// where `q = target_qubit`. The coefficient `prob` cannot exceed `3/4`, at
    /// which maximal mixing occurs.
    ///
    /// # Parameters
    ///
    /// - `qureg`: a density matrix
    /// - `target_qubit`: qubit upon which to induce depolarizing noise
    /// - `prob`: the probability of the depolarizing error occurring
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if `qureg` is not a density matrix
    ///   - if `target_qubit` is outside [0,
    ///     qureg.[`num_qubits_represented()`]).
    ///   - if `prob` is not in `[0, 3/4]`
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
    /// See [QuEST API] for more information.
    ///
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn mix_depolarising(
        &mut self,
        target_qubit: i32,
        prob: Qreal,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::mixDepolarising(self.reg, target_qubit, prob);
        })
    }

    ///  Mixes a density matrix to induce single-qubit amplitude damping.
    ///
    /// With probability `prob`, applies damping (transition from `1` to `0`
    /// state). This transforms `qureg = rho` into the mixed state:
    ///
    /// ```text
    ///  K_0 rho K_0^\dagger + K_1 rho K_1^\dagger
    /// ```
    ///
    /// where `q = target_qubit` and `K_0` and `$K_1` are Kraus operators:
    ///
    /// ```text
    ///      K_0 =  [ 1       0       ]   K_1 = [ 0  sqrt(prob) ]
    ///             [ 0  sqrt(1-prob) ]         [ 0      0      ]
    /// ```
    ///
    /// The coefficient `prob` cannot exceed 1, at which total damping/decay
    /// occurs.
    ///
    /// Note that unlike [`mix_dephasing()`] and [`mix_depolarising()`], this
    /// function can increase the purity of a mixed state (by, as `prob` becomes
    /// `1`, gaining certainty that the qubit is in the 0 state).
    ///
    /// # Parameters
    ///
    /// - `qureg`: a density matrix
    /// - `target_qubit`: qubit upon which to induce amplitude damping
    /// - `prob`: the probability of the damping
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if `qureg` is not a density matrix
    ///   - if `target_qubit` is outside [0,
    ///     qureg.[`num_qubits_represented()`]).
    ///   - if `prob` is not in `[0, 1]`
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
    /// See [QuEST API] for more information.
    ///
    /// [`mix_dephasing()`]: crate::mix_dephasing()
    /// [`mix_depolarising()`]: crate::mix_depolarising()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn mix_damping(
        &mut self,
        target_qubit: i32,
        prob: Qreal,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::mixDamping(self.reg, target_qubit, prob);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn mix_two_qubit_depolarising(
        &mut self,
        qubit1: i32,
        qubit2: i32,
        prob: Qreal,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::mixTwoQubitDepolarising(self.reg, qubit1, qubit2, prob);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn mix_pauli(
        &mut self,
        target_qubit: i32,
        prob_x: Qreal,
        prob_y: Qreal,
        prob_z: Qreal,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::mixPauli(self.reg, target_qubit, prob_x, prob_y, prob_z);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn mix_density_matrix(
        &mut self,
        prob: Qreal,
        other_qureg: &Qureg<'_, N>,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::mixDensityMatrix(self.reg, prob, other_qureg.reg);
        })
    }

    /// Calculate the purity of a density matrix.
    ///
    /// The purity of a density matrix is calculated by taking the trace of the
    /// density matrix squared. Returns `Tr (\rho^2)`.
    /// For a pure state, this =1.
    /// For a mixed state, the purity is less than 1 and is lower bounded by
    /// `1/2^n`, where n is the number of qubits. The minimum purity is achieved
    /// for the maximally mixed state `identity/2^n`.
    ///
    /// This function does not accept state-vectors, which clearly have purity
    /// 1.
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
    /// Returns [`InvalidQuESTInputError`],
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
    /// See [QuEST API] for more information.
    ///
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    pub fn calc_purity(&self) -> Result<Qreal, QuestError> {
        catch_quest_exception(|| unsafe { ffi::calcPurity(self.reg) })
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
    /// In either case, the returned fidelity lies in `[0, 1]` (assuming both
    /// input states have valid normalisation). If any of the input `Qureg`s
    /// are not normalised, this function will return the real component of
    /// the correct linear algebra calculation.
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
    /// Returns [`InvalidQuESTInputError`],
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
    /// let pure_state = &mut create_qureg::<2>(env);
    ///
    /// init_zero_state(qureg);
    /// init_plus_state(pure_state);
    ///
    /// let fidelity = calc_fidelity(qureg, pure_state).unwrap();
    /// assert!((fidelity - 0.25).abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    pub fn calc_fidelity(
        &self,
        pure_state: &Qureg<'_, N>,
    ) -> Result<Qreal, QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::calcFidelity(self.reg, pure_state.reg)
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
    /// - [`InvalidQuESTInputError`],
    ///   - if either `qubit1` or `qubit2` is outside [0,
    ///     [`qureg.num_qubits_represented()`]).
    ///   - if `qubit1` and `qubit2` are equal
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn swap_gate(
        &mut self,
        qubit1: i32,
        qubit2: i32,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::swapGate(self.reg, qubit1, qubit2);
        })
    }

    /// Performs a sqrt SWAP gate between `qubit1` and `qubit2`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    /// // init state |10>
    /// init_classical_state(qureg, 1).unwrap();
    /// sqrt_swap_gate(qureg, 0, 1).unwrap();
    /// sqrt_swap_gate(qureg, 0, 1).unwrap();
    /// let outcome = measure(qureg, 0).unwrap();
    /// assert_eq!(outcome, 0);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn sqrt_swap_gate(
        &mut self,
        qb1: i32,
        qb2: i32,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::sqrtSwapGate(self.reg, qb1, qb2);
        })
    }

    /// Apply a general single-qubit unitary with multiple control qubits.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn multi_state_controlled_unitary(
        &mut self,
        control_qubits: &[i32],
        control_state: &[i32],
        target_qubit: i32,
        u: &ComplexMatrix2,
    ) -> Result<(), QuestError> {
        let num_control_qubits = control_qubits.len() as i32;
        catch_quest_exception(|| unsafe {
            ffi::multiStateControlledUnitary(
                self.reg,
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
    /// let qureg = &mut create_qureg::<2>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn multi_rotate_z(
        &mut self,
        qubits: &[i32],
        angle: Qreal,
    ) -> Result<(), QuestError> {
        let num_qubits = qubits.len() as i32;
        catch_quest_exception(|| unsafe {
            ffi::multiRotateZ(self.reg, qubits.as_ptr(), num_qubits, angle);
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
    /// let qureg = &mut create_qureg::<3>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn multi_rotate_pauli(
        &mut self,
        target_qubits: &[i32],
        target_paulis: &[PauliOpType],
        angle: Qreal,
    ) -> Result<(), QuestError> {
        let num_targets = target_qubits.len() as i32;
        catch_quest_exception(|| unsafe {
            ffi::multiRotatePauli(
                self.reg,
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
    /// let qureg = &mut create_qureg::<4>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn multi_controlled_multi_rotate_z(
        &mut self,
        control_qubits: &[i32],
        target_qubits: &[i32],
        angle: Qreal,
    ) -> Result<(), QuestError> {
        let num_controls = control_qubits.len() as i32;
        let num_targets = target_qubits.len() as i32;
        catch_quest_exception(|| unsafe {
            ffi::multiControlledMultiRotateZ(
                self.reg,
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
    /// let qureg = &mut create_qureg::<4>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn multi_controlled_multi_rotate_pauli(
        &mut self,
        control_qubits: &[i32],
        target_qubits: &[i32],
        target_paulis: &[PauliOpType],
        angle: Qreal,
    ) -> Result<(), QuestError> {
        let num_controls = control_qubits.len() as i32;
        let num_targets = target_qubits.len() as i32;
        catch_quest_exception(|| unsafe {
            ffi::multiControlledMultiRotatePauli(
                self.reg,
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
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_zero_state(qureg);
    /// let workspace = &mut create_qureg::<2>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn calc_expec_pauli_prod(
        &self,
        target_qubits: &[i32],
        pauli_codes: &[PauliOpType],
        workspace: &mut Qureg<'_, N>,
    ) -> Result<Qreal, QuestError> {
        let num_targets = target_qubits.len() as i32;
        catch_quest_exception(|| unsafe {
            ffi::calcExpecPauliProd(
                self.reg,
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
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_zero_state(qureg);
    /// let workspace = &mut create_qureg::<2>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn calc_expec_pauli_sum(
        &self,
        all_pauli_codes: &[PauliOpType],
        term_coeffs: &[Qreal],
        workspace: &mut Qureg<'_, N>,
    ) -> Result<Qreal, QuestError> {
        let num_sum_terms = term_coeffs.len() as i32;
        catch_quest_exception(|| unsafe {
            ffi::calcExpecPauliSum(
                self.reg,
                all_pauli_codes.as_ptr(),
                term_coeffs.as_ptr(),
                num_sum_terms,
                workspace.reg,
            )
        })
    }

    /// Computes the expected value of `qureg` under Hermitian operator `hamil`.
    ///
    /// This function is merely an encapsulation of `calc_expec_pauli_sum()` -
    /// refer to the doc there for an elaboration.
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
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_zero_state(qureg);
    /// let workspace = &mut create_qureg::<2>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn calc_expec_pauli_hamil(
        &self,
        hamil: &PauliHamil,
        workspace: &mut Qureg<'_, N>,
    ) -> Result<Qreal, QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::calcExpecPauliHamil(self.reg, hamil.0, workspace.reg)
        })
    }

    ///  Apply a general two-qubit unitary (including a global phase factor).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn two_qubit_unitary(
        &mut self,
        target_qubit1: i32,
        target_qubit2: i32,
        u: &ComplexMatrix4,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::twoQubitUnitary(self.reg, target_qubit1, target_qubit2, u.0);
        })
    }

    /// Apply a general controlled two-qubit unitary.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn controlled_two_qubit_unitary(
        &mut self,
        control_qubit: i32,
        target_qubit1: i32,
        target_qubit2: i32,
        u: &ComplexMatrix4,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::controlledTwoQubitUnitary(
                self.reg,
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
    /// let qureg = &mut create_qureg::<4>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn multi_controlled_two_qubit_unitary(
        &mut self,
        control_qubits: &[i32],
        target_qubit1: i32,
        target_qubit2: i32,
        u: &ComplexMatrix4,
    ) -> Result<(), QuestError> {
        let num_control_qubits = control_qubits.len() as i32;
        catch_quest_exception(|| unsafe {
            ffi::multiControlledTwoQubitUnitary(
                self.reg,
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
    /// let qureg = &mut create_qureg::<2>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn multi_qubit_unitary(
        &mut self,
        targs: &[i32],
        u: &ComplexMatrixN,
    ) -> Result<(), QuestError> {
        let num_targs = targs.len() as i32;
        catch_quest_exception(|| unsafe {
            ffi::multiQubitUnitary(self.reg, targs.as_ptr(), num_targs, u.0);
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
    /// let qureg = &mut create_qureg::<3>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn controlled_multi_qubit_unitary(
        &mut self,
        ctrl: i32,
        targs: &[i32],
        u: &ComplexMatrixN,
    ) -> Result<(), QuestError> {
        let num_targs = targs.len() as i32;
        catch_quest_exception(|| unsafe {
            ffi::controlledMultiQubitUnitary(
                self.reg,
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
    /// let qureg = &mut create_qureg::<4>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn multi_controlled_multi_qubit_unitary(
        &mut self,
        ctrls: &[i32],
        targs: &[i32],
        u: &ComplexMatrixN,
    ) -> Result<(), QuestError> {
        let num_ctrls = ctrls.len() as i32;
        let num_targs = targs.len() as i32;
        catch_quest_exception(|| unsafe {
            ffi::multiControlledMultiQubitUnitary(
                self.reg,
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn mix_kraus_map(
        &mut self,
        target: i32,
        ops: &[&ComplexMatrix2],
    ) -> Result<(), QuestError> {
        let num_ops = ops.len() as i32;
        let ops_inner = ops.iter().map(|x| x.0).collect::<Vec<_>>();
        catch_quest_exception(|| unsafe {
            ffi::mixKrausMap(self.reg, target, ops_inner.as_ptr(), num_ops);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn mix_two_qubit_kraus_map(
        &mut self,
        target1: i32,
        target2: i32,
        ops: &[&ComplexMatrix4],
    ) -> Result<(), QuestError> {
        let num_ops = ops.len() as i32;
        let ops_inner = ops.iter().map(|x| x.0).collect::<Vec<_>>();
        catch_quest_exception(|| unsafe {
            ffi::mixTwoQubitKrausMap(
                self.reg,
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn mix_multi_qubit_kraus_map(
        &mut self,
        targets: &[i32],
        ops: &[&ComplexMatrixN],
    ) -> Result<(), QuestError> {
        let num_targets = targets.len() as i32;
        let num_ops = ops.len() as i32;
        let ops_inner = ops.iter().map(|x| x.0).collect::<Vec<_>>();
        catch_quest_exception(|| unsafe {
            ffi::mixMultiQubitKrausMap(
                self.reg,
                targets.as_ptr(),
                num_targets,
                ops_inner.as_ptr(),
                num_ops,
            );
        })
    }

    /// Apply a general non-trace-preserving single-qubit Kraus map.
    ///
    /// The state must be a density matrix, and the map is specified by at most
    /// four operators.
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn mix_nontp_kraus_map(
        &mut self,
        target: i32,
        ops: &[&ComplexMatrix2],
    ) -> Result<(), QuestError> {
        let num_ops = ops.len() as i32;
        let ops_inner = ops.iter().map(|x| x.0).collect::<Vec<_>>();
        catch_quest_exception(|| unsafe {
            ffi::mixNonTPKrausMap(
                self.reg,
                target,
                ops_inner.as_ptr(),
                num_ops,
            );
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn mix_nontp_two_qubit_kraus_map(
        &mut self,
        target1: i32,
        target2: i32,
        ops: &[&ComplexMatrix4],
    ) -> Result<(), QuestError> {
        let num_ops = ops.len() as i32;
        let ops_inner = ops.iter().map(|x| x.0).collect::<Vec<_>>();
        catch_quest_exception(|| unsafe {
            ffi::mixNonTPTwoQubitKrausMap(
                self.reg,
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn mix_nontp_multi_qubit_kraus_map(
        &mut self,
        targets: &[i32],
        ops: &[&ComplexMatrixN],
    ) -> Result<(), QuestError> {
        let num_targets = targets.len() as i32;
        let num_ops = ops.len() as i32;
        let ops_inner = ops.iter().map(|x| x.0).collect::<Vec<_>>();
        catch_quest_exception(|| unsafe {
            ffi::mixNonTPMultiQubitKrausMap(
                self.reg,
                targets.as_ptr(),
                num_targets,
                ops_inner.as_ptr(),
                num_ops,
            );
        })
    }

    /// Applies a trotterisation of unitary evolution.
    ///
    /// The unitary evelution `$\exp(-i \, \text{hamil} \, \text{time})$` is
    /// applied to `qureg`. # Examples
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn apply_trotter_circuit(
        &mut self,
        hamil: &PauliHamil,
        time: Qreal,
        order: i32,
        reps: i32,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::applyTrotterCircuit(self.reg, hamil.0, time, order, reps);
        })
    }

    /// Apply a general 2-by-2 matrix, which may be non-unitary.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn apply_matrix2(
        &mut self,
        target_qubit: i32,
        u: &ComplexMatrix2,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::applyMatrix2(self.reg, target_qubit, u.0);
        })
    }

    /// Apply a general 4-by-4 matrix, which may be non-unitary.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn apply_matrix4(
        &mut self,
        target_qubit1: i32,
        target_qubit2: i32,
        u: &ComplexMatrix4,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::applyMatrix4(self.reg, target_qubit1, target_qubit2, u.0);
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
    /// let qureg = &mut create_qureg::<3>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn apply_matrix_n(
        &mut self,
        targs: &[i32],
        u: &ComplexMatrixN,
    ) -> Result<(), QuestError> {
        let num_targs = targs.len() as i32;
        catch_quest_exception(|| unsafe {
            ffi::applyMatrixN(self.reg, targs.as_ptr(), num_targs, u.0);
        })
    }

    /// Apply a general N-by-N matrix with additional controlled qubits.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<4>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn apply_multi_controlled_matrix_n(
        &mut self,
        ctrls: &[i32],
        targs: &[i32],
        u: &ComplexMatrixN,
    ) -> Result<(), QuestError> {
        let num_ctrls = ctrls.len() as i32;
        let num_targs = targs.len() as i32;
        catch_quest_exception(|| unsafe {
            ffi::applyMultiControlledMatrixN(
                self.reg,
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
    ///
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
    ///   [`BitEncoding::TWOS_COMPLEMENT`],  since the
    ///   negative   indices would generate (illegal) complex phases, and  must
    /// be   overriden with
    ///   [`apply_phase_func_overrides()`].  
    ///  
    ///   If your function `f(r)` diverges at one or more `r` values, you
    ///   must instead use `apply_phase_func_overrides()` and specify explicit
    /// phase   changes for these values. Otherwise, the corresponding
    /// amplitudes of the   state-vector will become indeterminate (like
    /// `NaN`). Note that use of any   negative exponent will result in
    /// divergences at `r=0`.
    ///
    /// - The function `f(r)` specifies the phase change to induce upon
    ///   amplitude `alpha` of computational basis state with index `r`, such
    ///   that
    ///
    ///   ```latex
    ///   \alpha |r\rangle \rightarrow \, \exp(i f(r))  \alpha \,  |r\rangle.
    ///   ```
    ///
    ///   The index `r` associated with each computational basis
    ///   state is determined by the binary value of the specified `qubits`
    ///   (ordered least to most significant), interpreted under the given
    ///   [`BitEncoding`] encoding.
    ///
    /// - If `qureg` is a density matrix `rho`, this function modifies `qureg`
    ///   to:
    ///
    ///   ```latex
    ///   \rho \rightarrow \hat{D} \, \rho \, \hat{D}^\dagger,
    ///   ```
    ///
    ///   where   `\hat{D}` is the diagonal unitary operator:
    ///
    ///   ```latex
    ///    \hat{D} = \text{diag}
    ///     \, \{ \; e^{i f(r_0)}, \; e^{i f(r_1)}, \;  \dots \; \}.
    ///   ```
    ///
    /// - The interpreted phase function can be previewed in the QASM log, as a
    ///   comment.
    ///
    /// - This function may become numerically imprecise for quickly growing
    ///   phase functions which admit very large phases, for example of `10^10`.
    ///
    /// # Parameters
    ///
    /// - `qureg`: the state-vector or density matrix to be modified
    /// - `qubits`: a list of the indices of the qubits which will inform `r`
    ///   for each amplitude in `qureg`
    /// - `encoding`: the [`BitEncoding`] under which to infer the binary value
    ///   `r` from the bits of `qubits` in each basis state of `qureg`
    /// - `coeffs`: the coefficients of the exponential polynomial phase
    ///   function `f(r)`
    /// - `exponents`: the exponents of the exponential polynomial phase
    ///   function `f(r)`
    ///
    /// The length of list `coeffs` must be the same as that of `exponents`
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`]
    ///   - if the length of `coeffs` is different than that of  `exponents`
    ///   - if any qubit in `qubits` has an invalid index (i.e. does not satisfy
    ///     `0 <= qubit < qureg.num_qubits_represented()`
    ///   - if the elements of `qubits` are not unique
    ///   - if `qubits.len() >= qureg.num_qubits_represented()`
    ///   - if `encoding` is not compatible with `qubits.len()` (e.g.
    ///     `TWOS_COMPLEMENT` with only 1 qubit)
    ///   - if `exponents` contains a fractional number despite `encoding` being
    ///     `TWOS_COMPLEMENT` (you must instead use
    ///     `apply_phase_func_overrides()` and override all negative indices)
    ///   - if `exponents` contains a negative power (you must instead use
    ///     apply_phase_func_overrides()` and override the zero index)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [`BitEncoding::TWOS_COMPLEMENT`]: crate::BitEncoding::TWOS_COMPLEMENT
    /// [`BitEncoding`]: crate::BitEncoding
    /// [`apply_phase_func_overrides()`]: crate::apply_phase_func_overrides()
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn apply_phase_func(
        &mut self,
        qubits: &[i32],
        encoding: BitEncoding,
        coeffs: &[Qreal],
        exponents: &[Qreal],
    ) -> Result<(), QuestError> {
        let num_qubits = qubits.len() as i32;
        let num_terms = coeffs.len() as i32;
        catch_quest_exception(|| unsafe {
            ffi::applyPhaseFunc(
                self.reg,
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
    /// See [`apply_phase_func()`] for a full desctiption.
    ///
    /// - As in `apply_phase_func()`, the arguments `coeffs` and `exponents`
    ///   specify a phase function `f(r)`, where `r` is determined by `qubits`
    ///   and `encoding` for each basis state of `qureg`.
    /// - Additionally, `override_inds` is a list specifying the values of `r`
    ///   for which to explicitly set the induced phase change. The overriding
    ///   phase changes are specified in the corresponding elements of
    ///   `override_phases`.
    /// - Note that if `encoding` is `TWOS_COMPLEMENT`, and `f(r)` features a
    ///   fractional exponent, then every negative phase index must be
    ///   overriden. This is checked and enforced by `QuEST`'s validation,
    ///   unless there are more than 16 targeted qubits, in which case valid
    ///   input is assumed (due to an otherwise prohibitive performance
    ///   overhead).
    /// - Overriding phases are checked at each computational basis state of
    ///   `qureg` *before* evaluating the phase function `f(r)`, and hence are
    ///   useful for avoiding singularities or errors at diverging values of
    ///   `r`.
    /// - The interpreted phase function and list of overrides can be previewed
    ///   in the QASM log, as a comment.
    ///
    /// # Parameters
    ///
    /// - `qureg`:  the state-vector or density matrix to be modified
    /// - `qubits`: a list of the indices of the qubits which will inform `r`
    ///   for each amplitude in `qureg`
    /// - `encoding`: [`BitEncoding`] under which to infer the binary value `r`
    ///   from the bits of `qubits` in each basis state of `qureg`
    /// - `coeffs`: the coefficients of the exponential polynomial phase
    ///   function `f(r)`
    /// - `exponents`: the exponents of the exponential polynomial phase
    ///   function `f(r)`
    /// - `override_inds`: a list of sub-state indices (values of `r` of which
    ///   to explicit set the phase change
    /// - `override_phases`: a list of replacement phase changes, for the
    ///   corresponding `r` values in `override_inds` (one to one)
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if the length of `override_inds` is different than that of
    ///     `override_phases`
    ///   - if the length of `coeffs` is different than that of `exponents`
    ///   - if any qubit in `qubits` has an invalid index (i.e. does not satisfy
    ///     `0 <= qubit < qureg.num_qubits_represented()`
    ///   - if the elements of `qubits` are not unique
    ///   - if `qubits.len() >= qureg.num_qubits_represented()`
    ///   - if `encoding` is not compatible with `qubits.len()` (e.g.
    ///     `TWOS_COMPLEMENT` with only 1 qubit)
    ///   - if `exponents` contains a fractional number despite `encoding` being
    ///     `TWOS_COMPLEMENT` (you must instead use
    ///     `apply_phase_func_overrides()` and override all negative indices)
    ///   - if `exponents` contains a negative power and the (consequently
    ///     diverging) zero index is not contained in `override_inds`
    ///   - if any value in `override_inds` is not producible by `qubits` under
    ///     the given `encoding` (e.g. 2 unsigned qubits cannot represent index
    ///     9)
    ///   - if `encoding` is `TWOS_COMPLEMENT`, and `exponents` contains a
    ///     fractional number, but `override_inds` does not contain every
    ///     possible negative index (checked only up to 16 targeted qubits)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [`apply_phase_func()`]: crate::apply_phase_func()
    /// [`BitEncoding`]: crate::BitEncoding
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn apply_phase_func_overrides(
        &mut self,
        qubits: &[i32],
        encoding: BitEncoding,
        coeffs: &[Qreal],
        exponents: &[Qreal],
        override_inds: &[i64],
        override_phases: &[Qreal],
    ) -> Result<(), QuestError> {
        let num_qubits = qubits.len() as i32;
        let num_terms = coeffs.len() as i32;
        let num_overrides = override_inds.len() as i32;
        catch_quest_exception(|| unsafe {
            ffi::applyPhaseFuncOverrides(
                self.reg,
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
    /// Induces a phase change upon each amplitude of `qureg`, determined by the
    /// multi-variable exponential polynomial "phase function".
    ///
    /// This is a multi-variable extension of [`apply_phase_func()`], whereby
    /// multiple sub-registers inform separate variables in the exponential
    /// polynomial function, and effects a diagonal unitary operator.
    ///
    /// - Arguments `coeffs`, `exponents` and `num_terms_per_reg` together
    ///   specify a real exponential polynomial `f(r)` of the form
    ///
    ///   ```latex
    ///   f(r_1,\dots, \; r_{\text{numRegs}}) =
    ///   \sum\limits_j^{\text{numRegs}}
    /// \sum\limits_{i}^{\text{numTermsPerReg}[j]}     c_{i,j} \;
    /// {r_j}^{p_{i,j}},   ```
    ///
    ///   where both coefficients `c_{i,j}` and exponents `p_{i,j}` can be any
    /// real   number, subject to constraints described below.
    ///  
    ///   While `coeffs` and `exponents` are flat lists, they should be
    /// considered   grouped into `num_qubits_per_reg.len()` sublists with
    /// lengths given by   `num_qubits_per_reg`.
    ///
    ///   For example,
    ///
    ///   ```rust,no_run
    ///   let coeffs =            [1., 2., 4., -3.14];
    ///   let exponents =         [2., 1., 5., 0.5];
    ///   let num_terms_per_reg = [1., 2.,     1.];
    ///   ```
    ///
    ///   constitutes the function: `f(\vec{r}) =  1 * {r_1}^2 + 2 * {r_2} + 4
    /// \,   {r_2}^{5} - 3.14 \, {r_3}^{0.5}`.   This means lists `coeffs`
    /// and   `exponents` should both be of length   equal to the sum of
    ///   `num_terms_per_reg`.
    ///
    ///
    /// - Unlike [`apply_phase_func()`], this function places additional
    ///   constraints on the   exponents in `f(\vec{r})`, due to the
    ///   exponentially growing costs of overriding diverging indices. Namely:
    ///
    ///   - `exponents` must not contain a negative number, since this would
    ///     result in a divergence when that register is zero, which would need
    ///     to be overriden for every other register basis state.  If
    ///     `f(\vec{r})` must contain a negative exponent, you should instead
    ///     call [`apply_phase_func_overrides()`] once for each
    ///     register/variable, and override the zero index for the relevant
    ///     variable. This works, because `\exp( i \sum_j f_j(r_j) ) = \prod_j
    ///     \exp(i f_j(r_j) )`.
    ///   - `exponents` must not contain a fractional number if `endoding =
    ///     TWOS_COMPLEMENT`, because such a term would produce illegal complex
    ///     values at negative register indices. Similar to the problem above,
    ///     each negative register index would require overriding at every index
    ///     of the other registers, and hence require an exponential number of
    ///     overrides. Therefore, if `f(\vec{r})` must contain a negative
    ///     exponent, you should instead call `apply_phase_func_overrides()`
    ///     once for each register/variable, and override every negative index
    ///     of each register in turn.
    ///
    /// - Lists `qubits` and `num_qubits_per_reg` together describe
    ///   sub-registers of `qureg`, which can each contain a different number of
    ///   qubits. Although `qubits` is a flat list of unique qubit indices, it
    ///   should be imagined grouped into sub-lists, of lengths given by
    ///   `num_qubits_per_reg`.
    ///
    ///   Note that the qubits need not be ordered increasing, and
    ///   qubits within each sub-register are assumed ordered least to most
    ///   significant in that sub-register. List `qubits` should have length
    /// equal   to the sum of elements in `num_qubits_per_reg`.
    ///
    /// - Each sub-register is associated with a variable `r_j` in phase
    ///   function `f(\vec{r})`. For a given computational basis state of
    ///   `qureg`, the value of each variable is determined by the binary value
    ///   in the corresponding sub-register, when intepreted with
    ///   [`BitEncoding`] `encoding`.
    ///
    /// - The function `f(\vec{r})` specifies the phase change to induce upon
    ///   amplitude `alpha` of computational basis state with the nominated
    ///   sub-registers encoding values.
    ///
    /// - The interpreted phase function can be previewed in the QASM log, as a
    ///   comment.
    ///
    /// # Parameters
    ///
    /// - `qureg`: the state-vector or density matrix to be modified
    /// - `qubits`: a list of all the qubit indices contained in each
    ///   sub-register
    /// - `num_qubits_per_reg`: a list of the lengths of each sub-list in
    ///   `qubits`
    /// - `encoding`: [`BitEncoding`] under which to infer the binary value
    ///   `r_j` from the bits of a sub-register
    /// - `coeffs`: the coefficients of all terms of the exponential polynomial
    ///   phase function `f(\vec{r})`
    /// - `exponents`: the exponents of all terms of the exponential polynomial
    ///   phase function `f(\vec{r})`
    /// - `num_terms_per_reg` a list of the number of `coeff` and `exponent`
    ///   terms supplied for each variable/sub-register
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if any qubit in `qubits` has an invalid index (i.e. does not satisfy
    ///     0 <= qubit < `qureg.num_qubits_represented()`)
    ///   - if the elements of `qubits` are not unique (including if
    ///     sub-registers overlap)
    ///   - if `num_qubits_per_reg.len() = 0 or > 100` (constrained by
    ///     `MAX_NUM_REGS_APPLY_ARBITRARY_PHASE` in `QuEST_precision.h`)
    ///   - if the size of any sub-register is incompatible with `encoding`
    ///     (e.g. contains fewer than two qubits if `encoding =
    ///     TWOS_COMPLEMENT`)
    ///   - if any element of `num_terms_per_reg < 1`
    ///   - if `exponents` contains a negative number
    ///   - if `exponents` contains a fractional number despite `encoding =
    ///     TWOS_COMPLEMENT`
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [`apply_phase_func()`]: crate::apply_phase_func()
    /// [`apply_phase_func_overrides()`]: crate::apply_phase_func_overrides()
    /// [`BitEncoding`]: crate::BitEncoding
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn apply_multi_var_phase_func(
        &mut self,
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
                self.reg,
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
    /// Induces a phase change upon each amplitude of `qureg`, determined by a
    /// phase function, and an explicit set of 'overriding' values at specific
    /// state indices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn apply_multi_var_phase_func_overrides(
        &mut self,
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
                self.reg,
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
    /// Induces a phase change upon each amplitude of `qureg`, determined by a
    /// named (and potentially multi-variable) phase function.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn apply_named_phase_func(
        &mut self,
        qubits: &[i32],
        num_qubits_per_reg: &[i32],
        encoding: BitEncoding,
        function_name_code: PhaseFunc,
    ) -> Result<(), QuestError> {
        let num_regs = num_qubits_per_reg.len() as i32;
        catch_quest_exception(|| unsafe {
            ffi::applyNamedPhaseFunc(
                self.reg,
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
    /// named (and potentially multi-variable) phase function, and an explicit
    /// set of 'overriding' values at specific state indices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn apply_named_phase_func_overrides(
        &mut self,
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
                self.reg,
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
    /// let qureg = &mut create_qureg::<2>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn apply_param_named_phase_func(
        &mut self,
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
                self.reg,
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
    /// named, parameterised (and potentially multi-variable) phase function,
    /// and an explicit set of "overriding" values at specific state
    /// indices.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
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
    /// See [QuEST API] for more information.
    ///
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn apply_param_named_phase_func_overrides(
        &mut self,
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
                self.reg,
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
    ///   Fourier transform (DFT) of the input amplitudes, in the exact
    ///   ordering. This is true even if `qureg` is unnormalised.
    ///
    /// - If `qureg` is a density matrix, it will be changed under the unitary
    ///   action of the QFT. This can be imagined as each mixed state-vector
    ///   undergoing the DFT on its amplitudes. This is true even if `qureg` is
    ///   unnormalised.
    ///
    /// This function merges contiguous controlled-phase gates into single
    /// invocations of [`apply_named_phase_func`()][api-apply-named-phase-func],
    /// and hence is significantly faster than performing
    /// the QFT circuit directly.
    ///
    /// Furthermore, in distributed mode, this function requires only
    /// `log2(#nodes)` rounds of pair-wise communication, and hence is
    /// exponentially faster than directly performing the DFT on the
    /// amplitudes of `qureg`.
    ///
    /// See [`apply_qft()`][api-apply-qft] to apply the QFT to a sub-register of
    /// `qureg`.
    ///
    /// # Parameters
    ///
    /// - `qureg`: a state-vector or density matrix to modify
    ///
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
    /// init_zero_state(qureg);
    ///
    /// apply_full_qft(qureg);
    /// ```
    /// See [QuEST API] for more information.
    ///
    /// [api-apply-named-phase-func]: crate::apply_named_phase_func()
    /// [api-apply-qft]: crate::apply_qft()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn apply_full_qft(&mut self) {
        catch_quest_exception(|| unsafe {
            ffi::applyFullQFT(self.reg);
        })
        .expect("apply_full_qft should always succeed");
    }

    /// Applies the quantum Fourier transform (QFT) to a specific subset of
    /// qubits.
    ///
    /// The order of qubits affects the ultimate unitary.
    /// The canonical full-state QFT ([`apply_full_qft()`]) is
    /// achieved by targeting every qubit in increasing order.
    ///
    /// - If `qureg` is a state-vector, the output amplitudes are a kronecker
    ///   product of the discrete Fourier transform (DFT) acting upon the
    ///   targeted amplitudes.
    /// - If `qureg` is a density matrix, it will be changed under the unitary
    ///   action of the QFT. This can be imagined as each mixed state-vector
    ///   undergoing the DFT on its amplitudes. This is true even if `qureg` is
    ///   unnormalised.
    ///
    /// This function merges contiguous controlled-phase gates into single
    /// invocations of [`apply_named_phase_func()`], and
    /// hence is significantly faster than performing
    /// the QFT circuit directly.
    ///
    ///
    /// Furthermore, in distributed mode, this function requires only
    /// `log2(#nodes)` rounds of pair-wise communication, and hence is
    /// exponentially faster than directly performing the DFT on the
    /// amplitudes of `qureg`.
    ///
    /// See [`apply_full_qft()`] to apply the QFT to he entirety
    /// of `qureg`.
    ///
    /// # Parameters
    ///
    /// `qureg`: a state-vector or density matrix to modify
    /// `qubits` a list of the qubits to operate the QFT upon
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if the length of `qubits` is less than
    ///     [`qureg.num_qubits_represented()`]
    ///   - if any of `qubits` is outside [0,
    ///     qureg.[`num_qubits_represented()`]).
    ///   - if `qubits` contains any repetitions
    ///
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<3>(env);
    /// init_zero_state(qureg);
    ///
    /// apply_qft(qureg, &[0, 1]).unwrap();
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`apply_full_qft()`]: crate::apply_full_qft()
    /// [`apply_named_phase_func()`]: crate::apply_named_phase_func()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn apply_qft(
        &mut self,
        qubits: &[i32],
    ) -> Result<(), QuestError> {
        let num_qubits = qubits.len() as i32;
        catch_quest_exception(|| unsafe {
            ffi::applyQFT(self.reg, qubits.as_ptr(), num_qubits);
        })
    }

    /// Apply a projector.
    ///
    /// Force the target `qubit` of `qureg` into the given classical `outcome`,
    /// via a non-renormalising projection.
    ///
    /// This function zeroes all amplitudes in the state-vector or
    /// density-matrix which correspond to the opposite `outcome` given.
    /// Unlike [`collapse_to_outcome()`], it does not thereafter normalise
    /// `qureg`, and hence may leave it in a non-physical state.
    ///
    /// Note there is no requirement that the `outcome` state has a non-zero
    /// proability, and hence this function may leave `qureg` in a blank state,
    /// like that produced by [`init_blank_state()`].
    ///
    /// See [`collapse_to_outcome()`] for a norm-preserving equivalent, like a
    /// forced   measurement
    ///
    /// # Parameters
    ///
    /// - `qureg`:  a state-vector or density matrix to modify
    /// - `qubit`: the qubit to which to apply the projector
    /// - `outcome`: the single-qubit outcome (`0` or `1`) to project `qubit`
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`],
    ///   - if `qubit` is outside [0, qureg.[`num_qubits_represented()`]).
    ///   - if `outcome` is not in {0,1}
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &mut create_qureg::<2>(env);
    /// init_plus_state(qureg);
    ///
    /// apply_projector(qureg, 0, 0).unwrap();
    ///
    /// let amp = get_real_amp(qureg, 3).unwrap();
    /// assert!(amp.abs() < EPSILON);
    /// ```
    ///
    /// See [QuEST API] for more information.
    ///
    /// [`collapse_to_outcome()`]: crate::collapse_to_outcome()
    /// [`init_blank_state()`]: crate::Qureg::init_blank_state()
    /// [`QubitIndexError`]: crate::QuestError::QubitIndexError
    /// [`num_qubits_represented()`]: crate::Qureg::num_qubits_represented()
    /// [`InvalidQuESTInputError`]: crate::QuestError::InvalidQuESTInputError
    /// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn apply_projector(
        &mut self,
        qubit: i32,
        outcome: i32,
    ) -> Result<(), QuestError> {
        catch_quest_exception(|| unsafe {
            ffi::applyProjector(self.reg, qubit, outcome);
        })
    }
} // Qureg

impl<'a, const N: u16> Drop for Qureg<'a, N> {
    fn drop(&mut self) {
        catch_quest_exception(|| {
            unsafe { ffi::destroyQureg(self.reg, self.env.0) };
        })
        .expect("dropping Qureg should always succeed");
    }
}

/// Apply Hamiltonian `PauliHamil`.
///
/// Modifies `out_qureg` to be the result of applying `PauliHamil` (a
/// Hermitian but not necessarily unitary operator) to `in_qureg`.
///
/// In theory, `in_qureg` is unchanged though its state is temporarily
/// modified and is reverted by re-applying Paulis (XX=YY=ZZ=I), so may
/// see a change by small numerical errors. The initial state in
/// `out_qureg` is not used.
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
/// let in_qureg = &mut create_qureg::<2>(env);
/// init_zero_state(in_qureg);
/// let out_qureg = &mut create_qureg::<2>(env);
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
/// See [QuEST API] for more information.
///
/// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::needless_pass_by_ref_mut)]
pub fn apply_pauli_hamil<const N: u16>(
    in_qureg: &mut Qureg<'_, N>,
    hamil: &PauliHamil,
    out_qureg: &mut Qureg<'_, N>,
) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::applyPauliHamil(in_qureg.reg, hamil.0, out_qureg.reg);
    })
}

/// Apply the weighted sum of Pauli products.
///
/// In theory, `in_qureg` is unchanged though its state is temporarily
/// modified and is reverted by re-applying Paulis (XX=YY=ZZ=I), so may
/// see a change by small numerical errors. The initial state in
/// `out_qureg` is not used.
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
/// let in_qureg = &mut create_qureg::<2>(env);
/// init_zero_state(in_qureg);
/// let out_qureg = &mut create_qureg::<2>(env);
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
/// See [QuEST API] for more information.
///
/// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::needless_pass_by_ref_mut)]
pub fn apply_pauli_sum<const N: u16>(
    in_qureg: &mut Qureg<'_, N>,
    all_pauli_codes: &[PauliOpType],
    term_coeffs: &[Qreal],
    out_qureg: &mut Qureg<'_, N>,
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
/// See [QuEST API] for more information.
///
/// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
pub fn calc_hilbert_schmidt_distance<const N: u16>(
    a: &Qureg<'_, N>,
    b: &Qureg<'_, N>,
) -> Result<Qreal, QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::calcHilbertSchmidtDistance(a.reg, b.reg)
    })
}

/// Computes the inner product of two equal-size state vectors.
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &QuestEnv::new();
/// let qureg = &mut create_qureg::<2>(env);
/// init_zero_state(qureg);
/// let other_qureg = &mut create_qureg::<2>(env);
/// init_plus_state(other_qureg);
///
/// let prod = calc_inner_product(qureg, other_qureg).unwrap();
/// assert!((prod.re - 0.5).abs() < EPSILON);
/// assert!((prod.im).abs() < EPSILON);
/// ```
///
/// See [QuEST API] for more information.
///
/// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
pub fn calc_inner_product<const N: u16>(
    bra: &Qureg<'_, N>,
    ket: &Qureg<'_, N>,
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
/// See [QuEST API] for more information.
///
/// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
pub fn calc_density_inner_product<const N: u16>(
    rho1: &Qureg<'_, N>,
    rho2: &Qureg<'_, N>,
) -> Result<Qreal, QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::calcDensityInnerProduct(rho1.reg, rho2.reg)
    })
}

/// Set `qureg` to a weighted sum of states.
///
/// Modifies qureg `out` to the result of `$(\p facOut \p out + \p fac1 \p
/// qureg1 + \p fac2 \p qureg2)$`, imposing no constraints on normalisation.
///
/// Works for both state-vectors and density matrices. Note that afterward,
/// \p out may not longer be normalised and ergo no longer a valid
/// state-vector or density matrix. Users must therefore be careful
/// passing \p out to other `QuEST` functions which assume normalisation
/// in order to function correctly.
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
/// See [QuEST API] for more information.
///
/// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::needless_pass_by_ref_mut)]
pub fn set_weighted_qureg<const N: u16>(
    fac1: Qcomplex,
    qureg1: &Qureg<'_, N>,
    fac2: Qcomplex,
    qureg2: &Qureg<'_, N>,
    fac_out: Qcomplex,
    out: &mut Qureg<'_, N>,
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
