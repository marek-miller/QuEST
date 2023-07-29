use std::ffi::CString;

use crate::{
    exceptions::catch_quest_exception,
    ffi,
    QuestError,
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
    /// env.report_quest_env();
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

    /// Report information about the `QuEST` environment.
    ///
    /// The information if printed to standard output.
    ///
    /// See [QuEST API][quest-api] for more information.
    ///
    /// [quest-api]: https://quest-kit.github.io/QuEST/modules.html
    pub fn report_quest_env(&self) {
        catch_quest_exception(|| unsafe {
            ffi::reportQuESTEnv(self.0);
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
    /// let env_str = env.get_environment_string().unwrap();
    ///
    /// assert!(env_str.contains("OpenMP="));
    /// assert!(env_str.contains("threads="));
    /// assert!(env_str.contains("MPI="));
    /// assert!(env_str.contains("ranks="));
    /// assert!(env_str.contains("CUDA="));
    /// ```
    ///
    /// See [QuEST API][quest-api] for more information.
    ///
    /// [quest-api]: https://quest-kit.github.io/QuEST/modules.html
    pub fn get_environment_string(&self) -> Result<String, QuestError> {
        let mut cstr =
            CString::new("CUDA=x OpenMP=x MPI=x threads=xxxxxxx ranks=xxxxxxx")
                .map_err(QuestError::NulError)?;
        catch_quest_exception(|| {
            unsafe {
                let cstr_ptr = cstr.into_raw();
                ffi::getEnvironmentString(self.0, cstr_ptr);
                cstr = CString::from_raw(cstr_ptr);
            }

            cstr.into_string().map_err(QuestError::IntoStringError)
        })
        .expect("get_environment_string should always succeed")
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

// SAFETY:  The way we handle API calls to QuEST by locking the exception
// handler makes each call atomic and prevents data races.
unsafe impl Send for QuestEnv {}
unsafe impl Sync for QuestEnv {}
