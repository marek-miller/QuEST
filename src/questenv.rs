use crate::{
    exceptions::catch_quest_exception,
    ffi,
};

/// Information about the QuEST environment.
///
/// In practice, this holds info about MPI ranks and helps to hide MPI
/// initialization code.
#[derive(Debug)]
pub struct QuestEnv(pub(crate) ffi::QuESTEnv);

impl QuestEnv {
    /// Create a new environment.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = QuestEnv::new();
    /// report_quest_env(&env);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self(unsafe { ffi::createQuESTEnv() })
    }

    /// Sync environment in distributed mode.
    ///
    /// Guarantees that all code up to the given point has been executed on all
    /// nodes (if running in distributed mode).
    ///
    ///  # Examples
    ///
    /// ```rust
    /// # use quest_bind::*;
    /// let env = QuestEnv::new();
    /// env.sync();
    /// ```
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
        catch_quest_exception(|| unsafe { ffi::destroyQuESTEnv(self.0) })
            .expect("dropping QuestEnv should always succeed")
    }
}
