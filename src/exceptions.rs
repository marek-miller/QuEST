//! Catch exceptions thrown by `QuEST`.
//!
//! On failure, `QuEST` throws exceptions via user-configurable global
//! [`invalidQuESTInputError()`]. By default, this function prints an error
//! message and aborts, which is problematic in a large distributed setup. We
//! opt for catching all exceptions early.
//!
//! This is an internal module that doesn't contain any useful user interface.
//!
//! [`invalidQuESTInputError()`]: https://quest-kit.github.io/QuEST/group__debug.html#ga51a64b05d31ef9bcf6a63ce26c0092db

use std::{
    ffi::{
        c_char,
        CStr,
    },
    panic::{
        self,
        UnwindSafe,
    },
};

use super::QuestError;

/// Report error in a `QuEST` API call.
///
/// This function is called by `QuEST` whenever an error occurs.
/// We redefine it to put the error message and site reported into
/// `QuestError::InvalidQuESTInputError` and start unwinding the stack.  The
/// function `catch_quest_exception()` should be able to catch it.
///
/// # Panics
///
/// This function will panic if strings returned by `QuEST` are not properly
/// formatted (null terminated) C strings.
#[allow(non_snake_case)]
#[no_mangle]
unsafe extern "C" fn invalidQuESTInputError(
    errMsg: *const c_char,
    errFunc: *const c_char,
) {
    // SAFETY: errMsg and errFunc are always non-null as a result of
    // a call to QuEST's function: QuESTAssert()
    let err_msg = unsafe { CStr::from_ptr(errMsg) }.to_str().expect(
        "String (errMsg) returned by QuEST should be properly formatted",
    );
    let err_func = unsafe { CStr::from_ptr(errFunc) }.to_str().expect(
        "String (errFunc) returned by QuEST should be properly formatted",
    );
    log::error!("QueST Error in function {err_func}: {err_msg}");

    panic::resume_unwind(Box::new(QuestError::InvalidQuESTInputError {
        err_msg:  err_msg.to_owned(),
        err_func: err_func.to_owned(),
    }));
}

/// Execute a call to `QuEST` API and catch exceptions.
pub fn catch_quest_exception<T, F>(f: F) -> Result<T, QuestError>
where
    F: FnOnce() -> T + UnwindSafe,
{
    // Call QuEST API, unwrap the error sent by invalidInputQuestError()
    panic::catch_unwind(f).map_err(|e| match e.downcast::<QuestError>() {
        Ok(boxed_err) => *boxed_err,
        Err(e) => panic::resume_unwind(e),
    })
}

#[cfg(test)]
mod tests {
    use std::thread;

    use crate::{
        ComplexMatrixN,
        PauliHamil,
    };

    #[test]
    fn catch_exception_01() {
        let _ = ComplexMatrixN::try_new(1).unwrap();
        // Seems like supplying other invalid params here, like e.g. -3,
        // causes QuEST to hang.  Or is this a bug on our side?
        let _ = ComplexMatrixN::try_new(0).unwrap_err();
    }

    #[test]
    fn catch_exception_02() {
        let _ = PauliHamil::try_new(-11, -3).unwrap_err();
        let _ = PauliHamil::try_new(2, 2).unwrap();
    }

    #[test]
    fn catch_exception_parallel_01() {
        thread::scope(|s| {
            s.spawn(|| {
                catch_exception_01();
                catch_exception_01();
            });
            s.spawn(|| {
                catch_exception_01();
                catch_exception_01();
            });
        });
    }

    #[test]
    fn catch_exception_parallel_02() {
        thread::scope(|s| {
            s.spawn(|| {
                catch_exception_02();
                catch_exception_02();
            });
            s.spawn(|| {
                catch_exception_02();
                catch_exception_02();
            });
        });
    }

    #[test]
    fn catch_exception_parallel_03() {
        thread::scope(|s| {
            s.spawn(|| {
                catch_exception_parallel_01();
                catch_exception_parallel_02();
            });
            s.spawn(|| {
                catch_exception_parallel_02();
                catch_exception_parallel_01();
            });
        });
    }
}
