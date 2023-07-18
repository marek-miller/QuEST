use std::thread;

use quest_bind::*;

#[test]
fn sync_qureg_01() {
    // Qureg must be Sync for this to compile
    let env = &QuestEnv::new();
    let qureg = &Qureg::try_new(2, env).unwrap();

    std::thread::scope(|s| {
        s.spawn(|| {
            report_qureg_params(qureg);
            assert!(!qureg.is_density_matrix());
        });

        report_qureg_params(qureg);
        assert!(!qureg.is_density_matrix());
    });

    report_qureg_params(qureg);
    assert!(!qureg.is_density_matrix());
}

#[test]
fn sync_qureg_02() {
    let env = &QuestEnv::new();
    let qureg = &Qureg::try_new(2, env).unwrap();

    let mut num_qubits = -1;
    let mut num_amps = -1;
    thread::scope(|s| {
        s.spawn(|| num_qubits = get_num_qubits(qureg));
        s.spawn(|| num_amps = get_num_amps(qureg).unwrap());
    });

    assert_eq!(num_qubits, 2);
    assert_eq!(num_amps, 4);
}

#[test]
fn sync_qureg_03() {
    let env = &QuestEnv::new();
    let qureg = &Qureg::try_new(2, env).unwrap();

    let mut num_qubits = -1;
    let mut num_amps = -1;
    thread::scope(|s| {
        s.spawn(|| num_qubits = get_num_qubits(qureg));
        s.spawn(|| num_amps = get_num_amps(qureg).unwrap());
    });

    assert_eq!(num_qubits, 2);
    assert_eq!(num_amps, 4);
}

// #[test]
// fn calc_expec_diagonal_op_01() {
//     let env = &QuestEnv::new();
//     let qureg = &mut Qureg::try_new(2, env).unwrap();
//     init_zero_state(qureg);

//     let mut expec_val = Qcomplex::new(0.,0.);

//     thread::scope(|s| {
//         s.spawn(|| {
//             let op = &mut DiagonalOp::try_new(2, env).unwrap();
//             init_diagonal_op(op, &[1., 2., 3., 4.], &[5., 6., 7.,
// 8.]).unwrap();

//             expec_val = calc_expec_diagonal_op(qureg, op).unwrap();
//         });
//     });

//     assert!((expec_val.re - 1.).abs() < EPSILON);
//     assert!((expec_val.im - 5.).abs() < EPSILON);
// }
