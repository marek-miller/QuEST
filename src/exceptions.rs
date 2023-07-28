//! Catch exceptions thrown by `QuEST`.
//!
//! On failure, `QuEST` throws exceptions via user-configurable global
//! [`invalidQuESTInputError()`][1]. By default, this function prints an error
//! message and aborts, which is problematic in a large distributed setup. We
//! opt for catching all exceptions early. The exception handler is locked
//! during an API call. This means that calling `QuEST` functions is synchronous
//! and should be thread-safe, but comes at the expense of being able to run
//! only one `QuEST` API call at the time.
//!
//! The present implementation works because `QuEST`'s
//! `invalidQuESTInputError()` is synchronous and *all* `QuEST` API calls from
//! us that can potentially throw an exception are wrapped with
//! `catch_quest_exception()`. This way we ensure the calls are synchronous and
//! all exceptions have been thrown already when it's time for us to scoop them.
//!
//! This is an internal module that doesn't contain any useful user interface.
//!
//! [1]: https://quest-kit.github.io/QuEST/group__debug.html#ga51a64b05d31ef9bcf6a63ce26c0092db

use std::{
    ffi::{
        c_char,
        CStr,
    },
    sync::{
        Mutex,
        OnceLock,
    },
};

use crossbeam::channel::{
    unbounded,
    Receiver,
    Sender,
};

use super::QuestError;

static QUEST_EXCEPT_GUARD: Mutex<()> = Mutex::new(());
static QUEST_EXCEPT_ERROR: OnceLock<(
    Sender<QuestError>,
    Receiver<QuestError>,
)> = OnceLock::new();

/// Report error in a `QuEST` API call.
///
/// This function is called by `QuEST` whenever an error occurs.  We redefine it
/// to put the error message and site reported into a thread-safe global
/// storage, instead of just aborting.
///
/// # Panics
///
/// This function will panic if strings returned by `QuEST` are not properly
/// formatted (null terminated) C strings, or if the channel used for passing
/// error messages gets disconnected.
#[allow(non_snake_case)]
#[no_mangle]
unsafe extern "C" fn invalidQuESTInputError(
    errMsg: *const c_char,
    errFunc: *const c_char,
) {
    let err_msg = unsafe { CStr::from_ptr(errMsg) }.to_str().unwrap();
    let err_func = unsafe { CStr::from_ptr(errFunc) }.to_str().unwrap();

    QUEST_EXCEPT_ERROR
        .get_or_init(unbounded)
        .0
        .send(QuestError::InvalidQuESTInputError {
            err_msg:  err_msg.to_owned(),
            err_func: err_func.to_owned(),
        })
        .expect("channel transmitting error messages disconnected");

    log::error!("QueST Error in function {err_func}: {err_msg}");
}

/// Execute a call to `QuEST` API and catch exceptions.
///
/// This function achieves synchronous execution between threads
/// by locking the global `QUEST_EXCEPTION_GUARD` each time,
/// then executing the closure supplied, and checking the global
/// lock-free storage `QUEST_EXCEPTION_ERROR` for any error messages reported.
///
/// This way, interacting with `QuEST` API should stay thread-safe at all times,
/// at the expense of being able to call only one function at the time.
/// This is not an undesired property and shouldn't matter much for the overall
/// performance of the simulation, since each function retains access to all
/// the parallelism available in the system.
pub fn catch_quest_exception<T, F>(f: F) -> Result<T, QuestError>
where
    F: FnOnce() -> T,
{
    // Lock QuEST to our call
    let guard = QUEST_EXCEPT_GUARD.lock().unwrap();

    // Call QuEST API
    let res = f();

    // At this point all exceptions have been thrown.
    // Get the first exception thrown.  Drain the nonblocking iterator to leave
    // it empty for the next call.
    let mut err_iter = QUEST_EXCEPT_ERROR.get_or_init(unbounded).1.try_iter();
    let err = err_iter.next();
    let _ = err_iter.last();

    // Drop the guard as soon as we don't need it anymore:
    drop(guard);

    // This might be a little confusing:
    // If there is Some error, report it;
    // if there is None, everything's Ok.
    match err {
        Some(e) => Err(e),
        None => Ok(res),
    }
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
