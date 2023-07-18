# Releases

## v0.3.3 (??/??/2023)

- Simplify handling QuEST exceptions
- Implement `Sync` and `Send` traits for QuestEnv
- Expand and improve documentation

## v0.3.2 (09/07/2023)

- Expand and improve documentation
- Various bug fixes

## v0.3.1 (02/07/2023)

- Publish to [crates.io](https://crates.io/crates/quest_bind)

## v0.3.0 (02/07/2023)

New features/improvements:

- Expanded and improved documentation and test suite.

API breaking changes:

- Change signature of the following functions:

  - `mix_nontp_kraus_map()`
  - `mix_nontp_two_qubit_kraus_map()`
  - `mix_nontp_multi_qubit_kraus_map()`

  These functions now take the list of Kraus operators by reference.

  - `apply_named_phase_func()`

  This function returns now `Result<(), QuestError>`.

  - `apply_pauli_sum()`
  - `apply_pauli_hamil()`

  These functions take argument `in_qureg` as `&mut` now (instead of a shared
  reference).

- Fix typo in the function name: `apply_trotter_circuit()`

- Function: `multi_controlled_multi_rotate_pauli()` also changes signature.

## v0.2.1 (01/07/2023)

New features/improvements:

- Expand and improve documentation and test suite

## v0.2.0 (24/06/2023)

New features/improvements:

- Improve documentation
- Catch exceptions thrown by QuEST
- Add build script
- Constructors/destructors for QuEST structs
- Add example: `grovers_search.rs`
- Use `Complex<f64>` type from `num` crate (as `QComplex`)
- Use compile flag `"f32"` to set floating point precision
- Add Github workflows CT

### v0.1.0 (11/06/2023)

Initial release.
