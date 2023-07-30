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
}

impl<'a> Drop for Qureg<'a> {
    fn drop(&mut self) {
        catch_quest_exception(|| {
            unsafe { ffi::destroyQureg(self.reg, self.env.0) };
        })
        .expect("dropping Qureg should always succeed");
    }
}
