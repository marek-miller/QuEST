#[derive(Debug, PartialEq)]
#[non_exhaustive]
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
}
