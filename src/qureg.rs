use super::{
    catch_quest_exception,
    ffi,
    QuestEnv,
    QuestError,
};

#[derive(Debug)]
pub struct Qureg<'a> {
    pub(crate) env: &'a QuestEnv,
    pub(crate) reg: ffi::Qureg,
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
        if num_qubits < 0 {
            return Err(QuestError::QubitIndexError);
        }
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
        if num_qubits < 0 {
            return Err(QuestError::QubitIndexError);
        }
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
    /// # Parameters
    ///
    /// - `qureg` a state-vector or density matrix
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
            ffi::reportStateToScreen(self.reg, self.env.0, report_rank)
        })
        .expect("report_state_to screen should never fail");
    }

    /// Returns the number of qubits represented.
    ///
    /// # Parameters
    ///
    /// - `qureg` a state-vector or density matrix
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &Qureg::try_new(3, env).unwrap();
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
    /// # Parameters
    ///
    /// - `qureg` a state-vector or density matrix
    ///
    /// # Errors
    ///
    /// - [`InvalidQuESTInputError`], if `qureg` is a density matrix
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = &QuestEnv::new();
    /// let qureg = &Qureg::try_new(3, env).unwrap();
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
}

impl<'a> Drop for Qureg<'a> {
    fn drop(&mut self) {
        catch_quest_exception(|| {
            unsafe { ffi::destroyQureg(self.reg, self.env.0) };
        })
        .expect("dropping Qureg should always succeed");
    }
}
