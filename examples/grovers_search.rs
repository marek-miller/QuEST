//! This is an adapted example from: `grovers_search.c` in
//! `QuEST` repository.  `QuEST` is distributed under MIT License
//!
//! Implements Grover's algorithm for unstructured search,
//! using only X, H and multi-controlled Z gates.

use quest_bind::{
    get_prob_amp,
    hadamard,
    init_plus_state,
    multi_controlled_phase_flip,
    pauli_x,
    QuestEnv,
    QuestError,
    Qureg,
    PI,
};
use rand::Rng;

const NUM_QUBITS: i32 = 0x10;
const NUM_ELEMS: i64 = 1 << NUM_QUBITS;

fn tensor_gate<F>(
    qureg: &mut Qureg<'_>,
    gate: F,
    qubits: &[i32],
) -> Result<(), QuestError>
where
    F: Fn(&mut Qureg, i32) -> Result<(), QuestError>,
{
    qubits.iter().try_for_each(|&q| gate(qureg, q))
}

fn apply_oracle(
    qureg: &mut Qureg,
    qubits: &[i32],
    sol_elem: i64,
) -> Result<(), QuestError> {
    let sol_ctrls = &qubits
        .iter()
        .filter_map(|&q| ((sol_elem >> q) & 1 == 0).then_some(q))
        .collect::<Vec<_>>();

    // apply X to transform |solElem> into |111>
    tensor_gate(qureg, pauli_x, sol_ctrls)
        // effect |111> -> -|111>
        .and(multi_controlled_phase_flip(qureg, qubits))
        // apply X to transform |111> into |solElem>
        .and(tensor_gate(qureg, pauli_x, sol_ctrls))
}

fn apply_diffuser(
    qureg: &mut Qureg,
    qubits: &[i32],
) -> Result<(), QuestError> {
    // apply H to transform |+> into |0>
    tensor_gate(qureg, hadamard, qubits)
        // apply X to transform |11..1> into |00..0>
        .and(tensor_gate(qureg, pauli_x, qubits))?;

    // effect |11..1> -> -|11..1>
    multi_controlled_phase_flip(qureg, qubits)?;

    tensor_gate(qureg, pauli_x, qubits)
        .and(tensor_gate(qureg, hadamard, qubits))
}

fn main() -> Result<(), QuestError> {
    let env = &QuestEnv::new();

    let num_reps = (PI / 4.0 * (NUM_ELEMS as f64).sqrt()).ceil() as usize;
    println!(
        "num_qubits: {NUM_QUBITS}, num_elems: {NUM_ELEMS}, num_reps: \
         {num_reps}"
    );
    // randomly choose the element for which to search
    let mut rng = rand::thread_rng();
    let sol_elem = rng.gen_range(0..NUM_ELEMS);

    // prepare |+>
    let qureg = &mut Qureg::try_new(NUM_QUBITS, env)?;
    init_plus_state(qureg);
    // use all qubits in the register
    let qubits = &(0..NUM_QUBITS).collect::<Vec<_>>();

    // apply Grover's algorithm
    (0..num_reps).try_for_each(|_| {
        apply_oracle(qureg, qubits, sol_elem)
            .and(apply_diffuser(qureg, qubits))
            .and(get_prob_amp(qureg, sol_elem))
            .map(|prob| {
                println!("prob of solution |{sol_elem}> = {}", prob);
            })
    })
}
