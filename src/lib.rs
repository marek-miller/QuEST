// #![deny(missing_debug_implementations)]
// #![deny(missing_docs)]
// #![deny(unreachable_pub)]
#![warn(rust_2018_idioms)]

use exceptions::catch_quest_exception;

mod errors;
mod exceptions;
mod ffi;
mod matrices;
mod numbers;
mod operators;
mod questenv;
mod qureg;
#[cfg(test)]
mod tests;

pub use errors::QuestError;
pub use ffi::{
    bitEncoding as BitEncoding,
    pauliOpType as PauliOpType,
    phaseFunc as PhaseFunc,
    phaseGateType as PhaseGateType,
};
pub use matrices::{
    init_complex_matrix_n,
    ComplexMatrix2,
    ComplexMatrix4,
    ComplexMatrixN,
    Vector,
};
pub use numbers::{
    Qcomplex,
    Qreal,
    EPSILON,
    LN_10,
    LN_2,
    PI,
    SQRT_2,
    TAU,
};
pub use operators::{
    apply_diagonal_op,
    calc_expec_diagonal_op,
    init_diagonal_op,
    init_diagonal_op_from_pauli_hamil,
    init_pauli_hamil,
    set_diagonal_op_elems,
    sync_diagonal_op,
    DiagonalOp,
    PauliHamil,
};
pub use questenv::QuestEnv;
pub use qureg::{
    apply_pauli_hamil,
    apply_pauli_sum,
    calc_density_inner_product,
    calc_hilbert_schmidt_distance,
    calc_inner_product,
    create_density_qureg,
    create_qureg,
    set_weighted_qureg,
    Qureg,
};

/// Print the Hamiltonian `hamil` to screen.
pub fn report_pauli_hamil(hamil: &PauliHamil) -> Result<(), QuestError> {
    catch_quest_exception(|| unsafe {
        ffi::reportPauliHamil(hamil.0);
    })
}

/// Seed the random number generator.
///
/// The seed is based on (master node) current time and process ID.
///
/// This is the default seeding used internally by [`QuestEnv::new()`], and
/// determines the outcomes in functions like [`measure()`] and
/// [`measure_with_stats()`]. In distributed mode, every node agrees on the
/// seed (nominated by the master node) such that every node generates
/// the same sequence of pseudorandom numbers.
///
/// `QuEST` uses the [Mersenne Twister] for random number generation.
///
/// # Parameters
///
/// - `env`: a mutable reference to the [`QuestEnv`] runtime environment
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &mut QuestEnv::new();
///
/// seed_quest_default(env);
/// ```
///
/// See [QuEST API] for more information.
///
/// [`QuestEnv::new()`]: crate::QuestEnv::new()
/// [`measure()`]: crate::Qureg::measure()
/// [`measure_with_stats()`]: crate::Qureg::measure_with_stats()
/// [Mersenne Twister]: http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/emt19937ar.html
/// [`QuestEnv`]: crate::QuestEnv
/// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::needless_pass_by_ref_mut)]
pub fn seed_quest_default(env: &mut QuestEnv) {
    let env_ptr = std::ptr::addr_of_mut!(env.0);
    catch_quest_exception(|| unsafe {
        ffi::seedQuESTDefault(env_ptr);
    })
    .expect("seed_quest_default should always succeed");
}

/// Seeds the random number generator with a custom array of key(s).
///
/// This overrides the default keys, and determines the outcomes in
/// functions like [`measure()`] and [`measure_with_stats()`]. In
/// distributed mode, every node agrees on the seed (nominated by the
/// master node) such that every node generates the same sequence of
/// pseudorandom numbers.
///
/// `QuEST` uses the [Mersenne Twister] for random number generation.
///
/// The values of `seed_array` are copied and stored internally. This
/// function allows the PRNG to be initialized with more than a 32-bit
/// integer, if required.
///
/// # Parameters
///
/// - `env`: a mutable reference to the [`QuestEnv`] runtime environment
/// - `seed_array`: array of integers to use as seed
///
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let env = &mut QuestEnv::new();
///
/// seed_quest(env, &[1, 2, 3]);
/// ```
///
/// See [QuEST API] for more information.
///
/// [`QuestEnv::new()`]: crate::QuestEnv::new()
/// [`measure()`]: crate::Qureg::measure()
/// [`measure_with_stats()`]: crate::Qureg::measure_with_stats()
/// [Mersenne Twister]: http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/emt19937ar.html
/// [`QuestEnv`]: crate::QuestEnv
/// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
pub fn seed_quest(
    env: &mut QuestEnv,
    seed_array: &[u64],
) {
    let num_seeds = seed_array.len() as i32; // QuEST's function signature is`c_ulong`. Let's use u64 for now...
    let env_ptr = std::ptr::addr_of_mut!(env.0);
    let seed_array_ptr = seed_array.as_ptr();
    catch_quest_exception(|| unsafe {
        ffi::seedQuEST(env_ptr, seed_array_ptr, num_seeds);
    })
    .expect("seed_quest should always succeed");
}

/// Obtain the seeds presently used in random number generation.
///
/// This function returns a reference to the internal array of keys
/// which have seeded `QuEST`'s PRNG. These are the seeds which inform the
/// outcomes of random functions like [`measure()`] and
/// [`measure_with_stats()`], and are set using [`seed_quest()`] and
/// [`seed_quest_default()`].
///
/// Obtaining `QuEST`'s seeds is useful for seeding your own random number
/// generators, so that a simulation (with random `QuEST` measurements, and
/// your own random decisions) can be precisely repeated later, just by
/// calling [`seed_quest()`].
///
/// One should not rely, however, upon the reference returned to be
/// automatically updated after a subsequent call to [`seed_quest()`] or
/// [`seed_quest_default()`]. Instead, the present function should be
/// recalled.
///
/// # Parameters
///
/// - `env`: the [`QuestEnv`] runtime environment
///
/// # Examples
///
/// ```rust
/// # use quest_bind::*;
/// let mut env = QuestEnv::new();
/// let seeds = &[1, 2, 3];
/// seed_quest(&mut env, seeds);
///
/// let check_seeds = get_quest_seeds(&env);
/// assert_eq!(seeds, check_seeds);
/// ```
///
/// See [QuEST API] for more information.
///
/// [`QuestEnv::new()`]: crate::QuestEnv::new()
/// [`measure()`]: crate::Qureg::measure()
/// [`measure_with_stats()`]: crate::Qureg::measure_with_stats()
/// [`seed_quest_default()`]: crate::seed_quest()
/// [`seed_quest_default()`]: crate::seed_quest_default()
/// [`QuestEnv`]: crate::QuestEnv
/// [QuEST API]: https://quest-kit.github.io/QuEST/modules.html
#[allow(clippy::cast_sign_loss)]
#[must_use]
pub fn get_quest_seeds<'a: 'b, 'b>(env: &'a QuestEnv) -> &'b [u64] {
    catch_quest_exception(|| unsafe {
        let seeds_ptr = &mut std::ptr::null_mut();
        let num_seeds = &mut 0_i32;
        ffi::getQuESTSeeds(env.0, seeds_ptr, num_seeds);

        // SAFETY: The lifetime bound assures that seed_ptr points
        // to the correct address as long as env is in scope.
        std::slice::from_raw_parts(*seeds_ptr, *num_seeds as usize)
    })
    .expect("get_quest_seeds should always succeed")
}

/// The impl of `SendPtr` was taken from [`rayon`] crate.
/// Rayon is distributed under MIT License.
///
/// We need to transmit raw pointers across threads. It is possible to do this
/// without any unsafe code by converting pointers to usize or to `AtomicPtr`<T>
/// then back to a raw pointer for use. We prefer this approach because code
/// that uses this type is more explicit.
///
/// Unsafe code is still required to dereference the pointer, so this type is
/// not unsound on its own, although it does partly lift the unconditional
/// !Send and !Sync on raw pointers. As always, dereference with care.
///
/// [`rayon`]: https://crates.io/crates/rayon
#[derive(Debug)]
#[repr(transparent)]
pub(crate) struct SendPtr<T>(*mut T);

// SAFETY: !Send for raw pointers is not for safety, just as a lint
unsafe impl<T: Send> Send for SendPtr<T> {}

// SAFETY: !Sync for raw pointers is not for safety, just as a lint
unsafe impl<T: Send> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    // Helper to avoid disjoint captures of `send_ptr.0`
    #[allow(dead_code)]
    #[must_use]
    pub fn get(self) -> *mut T {
        self.0
    }
}

// Implement Copy without the T: Copy bound from the derive
impl<T> Copy for SendPtr<T> {}

// Implement Clone without the T: Clone bound from the derive
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
