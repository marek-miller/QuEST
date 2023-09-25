# quest_bind

[![Test](https://github.com/marek-miller/quest_bind/actions/workflows/test.yml/badge.svg)](https://github.com/marek-miller/quest_bind/actions/workflows/test.yml)
[![Docs](https://github.com/marek-miller/quest_bind/actions/workflows/docs.yml/badge.svg)](https://github.com/marek-miller/quest_bind/actions/workflows/docs.yml)

A wrapper around [QuEST](https://github.com/QuEST-Kit/QuEST/) v3.6.0.

Quantum Exact Simulation Toolkit (QuEST) is a no-fluff, bent-on-speed quantum
circuit simulator [[1]](https://doi.org/10.1038/s41598-019-47174-9). It is
distributed under MIT License.

## How to use it

Initialize a new binary crate:

```sh
cargo new tryme
cd tryme/
```

Add `quest_bind` to your project's dependencies:

```sh
cargo add quest_bind
```

Now write some code and put it in `./src/main.rs`:

```rust
use quest_bind::*;

fn main() -> Result<(), QuestError> {
    // Initialize QuEST environment and report to screen
    let env = &QuestEnv::new();
    env.report_quest_env();

    // Create a 2-qubit register and report its parameters
    let mut qureg = Qureg::try_new(2, env).expect("cannot allocate new Qureg");
    qureg.report_qureg_params();
    // Initialize |00> state and print out the state to screen
    qureg.init_zero_state();
    qureg.report_state_to_screen(0);

    // Prepare a Bell state `|00> + |11>`: apply Hadamard gate
    // on qubit 0, then NOT on qubit 1, controlled by qubit 0.
    println!("---\nPrepare Bell state: |00> + |11>");
    qureg.hadamard(0).and(qureg.controlled_not(0, 1))?;

    // Measure both qubits
    let outcome0 = qureg.measure(0)?;
    let outcome1 = qureg.measure(1)?;
    println!("Qubit \"0\" measured in state: |{outcome0}>");
    println!("Qubit \"1\" measured in state: |{outcome1}>");

    // Because the state was entangled, the outcomes
    // should always be the same
    if outcome0 == outcome1 {
        println!("They match!");
        Ok(())
    } else {
        panic!("qubits in Bell state should be perfectly correlated");
    }

    // At this point both `qureg` and `env` are dropped and
    // the allocated memory is freed.
}
```

The documentation is available
[online](https://docs.rs/quest_bind/latest/quest_bind/), as well as locally:

```sh
cargo doc --open
```

Lastly, compile and run the program:

```sh
cargo run
```

You should be able to see something like:

```text
EXECUTION ENVIRONMENT:
Running locally on one node
Number of ranks is 1
OpenMP enabled
Number of threads available is 8
Precision: size of qreal is 8 bytes
QUBITS:
Number of qubits is 2.
Number of amps is 4.
Number of amps per rank is 4.
---
Prepare Bell state: |00> + |11>
Qubit "0" measured in state: |0>
Qubit "1" measured in state: |0>
They match!
```

## Distributed and GPU-accelerated mode

QuEST support for MPI and GPU-accelerated computation ca be enabled in
`quest_bind` by setting appropriate feature flags. To enable QuEST's MPI mode,
set the `mpi` feature for `quest_bind`. Simply edit `Cargo.toml` of your binary
crate:

```toml
[package]
name = "tryme"
version = "0.1.0"
edition = "2021"

[dependencies]
quest_bind = { features = ["mpi"] }
```

Now if you compile and run the above program again, the output should be:

```text
EXECUTION ENVIRONMENT:
Running distributed (MPI) version
Number of ranks is 1
...
```

The feature `"gpu"` enables the GPU-accelerated mode. These two features are
mutually exclusive and in case both flags are set, the feature `"mpi"` takes
precedence.

## Testing

To run unit tests for this library, first clone the repository together with
QuEST source code as submodule:

```sh
git clone --recurse-submodules https://github.com/marek-miller/quest_bind.git
cd quest_bind
```

Then run:

```sh
cargo test
```

Note that `quest_bind` will not run `QuEST`'s test suite, nor will it check
`QuEST`'s correctness. The tests here are intended to check if the C API is
invoked correctly, and if Rust's types are passed safely back and forth across
the FFI boundary.

If you want to run the test suite in the single-precision floating point mode,
make sure the build script recompiles `libQuEST.so` with the right type
definitions:

```sh
cargo clean
cargo test --features=f32
```

By defualt, `quest_bind` uses Rust's double precision floating-point type:
`f64`. See [Numercal types](#numerical-types) section below.

You can also try the available examples by running, e.g.:

```sh
 cargo run --release --example grovers_search
```

To see the list of all available examples, try:

```sh
cargo run --example
```

## Note on performance

In the typical case when it's the numerical computation that dominates the CPU
usage, and not API calls, there should be no discernible difference in
performance between programs calling QuEST routines directly and analogous
applications using `quest_bind`. Remember, however, to enable optimizations for
both `quest_bind` and `QuEST` by compiling your code using the "release"
profile:

```sh
cargo run --release
```

## Handling exceptions

On failure, QuEST throws exceptions via user-configurable global
[`invalidQuESTInputError()`](https://quest-kit.github.io/QuEST/group__debug.html#ga51a64b05d31ef9bcf6a63ce26c0092db).
By default, this function prints an error message and aborts, which is
problematic in a large distributed setup.

We opt for catching all exceptions early by reimplementing
`invalidQuESTInputError()` to unwind the stack using Rust's
[`panic`](https://doc.rust-lang.org/std/panic/index.html) mechanism.

Additionally, all error messages reported by QuEST are logged as errors. To be
able to see them, add a logger as a dependency to your crate, e.g.:

```sh
cargo add env_logger
```

Then enable logging in your application:

```rust
fn main()  {
    env_logger::init();
    // (...)
}
```

and run:

```sh
RUST_LOG=info cargo run
```

See [`log` crate](https://docs.rs/log/latest/log/) for more on logging in Rust.

The type `QuestError` doesn't contain (possibly malformed) data returned by the
API call on failure. Only successful calls can reach the library user. This is
intentional, following guidelines from the QuEST documentation:

> [*Upon failure*] Users must ensure that the triggered API call does not
> continue (e.g. the user exits or throws an exception), else QuEST will
> continue with the valid [*sic!*] input and likely trigger a seg-fault.

See
[Quest API](https://quest-kit.github.io/QuEST/group__debug.html#ga51a64b05d31ef9bcf6a63ce26c0092db)
for more information.

## Numerical types

For now, numerical types used by `quest_bind` match exactly the C types that
QuEST uses on `x86_64`. This is a safe, but not very portable strategy. We pass
Rust types directly to QuEST without casting, assuming the following type
definitions:

```rust
pub type c_float = f32;
pub type c_double = f64;
pub type c_int = i32;
pub type c_longlong = i64;
pub type c_ulong = u64;
```

This should work for many different architectures. If your system uses slightly
different numerical types, `quest_bind` simply won't compile and there is not
much you can do besides manually altering the source code.

To check what C types are defined by your Rust installation, see the local
documentation for the module `std::ffi` in Rust's Standard Library:

```sh
rustup doc
```

## Contributing

Here's a few things to know, if you'd like to contribute to `quest_bind`.

- The Rust codebase is formatted according to the settings in `./rustfmt.toml`.
  We enable some unstable features of `rustfmt`. To format your patches
  correctly, you will need the nightly version of the Rust compiler. Before
  opening a pull request, remove lint from the code by running:

  ```sh
  just lint
  ```
