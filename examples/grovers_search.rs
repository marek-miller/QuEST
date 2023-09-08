//! This is an adapted example from: `grovers_search.c` in
//! `QuEST` repository.  `QuEST` is distributed under MIT License
//!
//! Implements Grover's algorithm for unstructured search,
//! using only X, H and multi-controlled Z gates.

use quest_bind::{
    Qreal,
    QuestEnv,
    QuestError,
    Qureg,
    PI,
};
use rand::Rng;

const NUM_QUBITS: i32 = 0x10;
const NUM_ELEMS: i64 = 1 << NUM_QUBITS;

fn tensor_gate<'a, F>(
    qureg: &mut Qureg<'a>,
    gate: F,
    qubits: &[i32],
) -> Result<(), QuestError>
where
    F: for<'b> Fn(&'b mut Qureg<'a>, i32) -> Result<(), QuestError>,
{
    qubits.iter().try_for_each(|&q| gate(qureg, q))
}

fn apply_oracle(
    qureg: &mut Qureg<'_>,
    qubits: &[i32],
    sol_elem: i64,
) -> Result<(), QuestError> {
    let sol_ctrls = &qubits
        .iter()
        .filter_map(|&q| ((sol_elem >> q) & 1 == 0).then_some(q))
        .collect::<Vec<_>>();

    // apply X to transform |solElem> into |111>
    qureg
        .multi_qubit_not(sol_ctrls)
        // effect |111> -> -|111>
        .and(qureg.multi_controlled_phase_flip(qubits))
        // apply X to transform |111> into |solElem>
        .and(qureg.multi_qubit_not(sol_ctrls))
}

fn apply_diffuser(
    qureg: &mut Qureg<'_>,
    qubits: &[i32],
) -> Result<(), QuestError> {
    // apply H to transform |+> into |0>
    tensor_gate(qureg, Qureg::hadamard, qubits)
        // apply X to transform |11..1>  into |00..0>
        .and(qureg.multi_qubit_not(qubits))?;

    // effect |11..1> -> -|11..1>
    qureg.multi_controlled_phase_flip(qubits)?;

    qureg.multi_qubit_not(qubits).and(tensor_gate(
        qureg,
        Qureg::hadamard,
        qubits,
    ))
}

fn main() -> Result<(), QuestError> {
    let env = &QuestEnv::new();

    let num_reps = (PI / 4.0 * (NUM_ELEMS as Qreal).sqrt()).ceil() as usize;
    println!(
        "num_qubits: {NUM_QUBITS}, num_elems: {NUM_ELEMS}, num_reps: \
         {num_reps}"
    );
    // randomly choose the element for which to search
    let mut rng = rand::thread_rng();
    let sol_elem = rng.gen_range(0..NUM_ELEMS);

    // FIXME
    const N: i32 = NUM_QUBITS;
    // prepare |+>
    let mut qureg = Qureg::<'_>::try_new(N, env)?;
    qureg.init_plus_state();
    // use all qubits in the register
    let qubits = &(0..NUM_QUBITS).collect::<Vec<_>>();

    // apply Grover's algorithm
    (0..num_reps).try_for_each(|_| {
        apply_oracle(&mut qureg, qubits, sol_elem)
            .and(apply_diffuser(&mut qureg, qubits))
            .and(qureg.get_prob_amp(sol_elem))
            .map(|prob| {
                println!("prob of solution |{sol_elem}> = {:.8}", prob);
            })
    })
}
