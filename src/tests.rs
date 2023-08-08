#![allow(clippy::cast_sign_loss)]

use num::Zero;

use super::*;

#[test]
fn create_qureg_01() -> Result<(), QuestError> {
    let env = &QuestEnv::new();
    let _ = Qureg::<'_, 1>::try_new(env)?;
    let _ = Qureg::<'_, 5>::try_new(env)?;

    let _ = Qureg::<'_, 0>::try_new(env).unwrap_err();
    Ok(())
}

// #[test]
// fn create_qureg_negative_num_qubits() {
//     let env = &QuestEnv::new();
//     let _ = Qureg::try_new(-1, env).unwrap_err();
//     let _ = Qureg::try_new_density(-1, env).unwrap_err();
// }

#[test]
fn create_density_qureg_01() -> Result<(), QuestError> {
    let env = &QuestEnv::new();
    {
        let _ = Qureg::<'_, 1>::try_new_density(env)?;
        let _ = Qureg::<'_, 5>::try_new_density(env)?;

        let _ = Qureg::<'_, 0>::try_new_density(env).unwrap_err();
    }
    Ok(())
}

#[test]
fn get_matrix_n_elem_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);

    qureg.init_zero_state();
    let amp = get_imag_amp(qureg, 0).unwrap();
    assert_eq!(amp, 0.);

    let mtr = &mut ComplexMatrixN::try_new(1).unwrap();
    init_complex_matrix_n(
        mtr,
        &[&[1., 2.], &[3., 4.]],
        &[&[11., 12.], &[13., 14.]],
    )
    .unwrap();
    apply_matrix_n(qureg, &[0], mtr).unwrap();
    let amp = get_imag_amp(qureg, 0).unwrap();
    assert!((amp - 11.).abs() < EPSILON);
}

#[test]
fn init_complex_matrix_n_02() {
    let mut m = ComplexMatrixN::try_new(1).unwrap();
    init_complex_matrix_n(
        &mut m,
        &[&[1., 2.], &[3., 4.]],
        &[&[11., 12.], &[13., 14.]],
    )
    .unwrap();

    assert_eq!(m.row_real_as_slice(0), &[1., 2.]);
    assert_eq!(m.row_real_as_slice(1), &[3., 4.]);

    assert_eq!(m.row_imag_as_slice(0), &[11., 12.]);
    assert_eq!(m.row_imag_as_slice(1), &[13., 14.]);
}

#[test]
fn init_complex_matrix_n_03() {
    let mut m = ComplexMatrixN::try_new(2).unwrap();
    init_complex_matrix_n(
        &mut m,
        &[
            &[111., 112., 113., 114.],
            &[115., 116., 117., 118.],
            &[119., 120., 121., 122.],
            &[123., 124., 125., 126.],
        ],
        &[
            &[211., 212., 213., 214.],
            &[215., 216., 217., 218.],
            &[219., 220., 221., 222.],
            &[223., 224., 225., 226.],
        ],
    )
    .unwrap();

    assert_eq!(m.row_real_as_slice(0), &[111., 112., 113., 114.]);
    assert_eq!(m.row_real_as_slice(1), &[115., 116., 117., 118.]);
    assert_eq!(m.row_real_as_slice(2), &[119., 120., 121., 122.]);
    assert_eq!(m.row_real_as_slice(3), &[123., 124., 125., 126.]);

    assert_eq!(m.row_imag_as_slice(0), &[211., 212., 213., 214.]);
    assert_eq!(m.row_imag_as_slice(1), &[215., 216., 217., 218.]);
    assert_eq!(m.row_imag_as_slice(2), &[219., 220., 221., 222.]);
    assert_eq!(m.row_imag_as_slice(3), &[223., 224., 225., 226.]);
}

#[test]
fn complex_matrix_n_row_slice_02() {
    let mtr = &mut ComplexMatrixN::try_new(2).unwrap();

    let row = mtr.row_real_as_mut_slice(0);
    row[0] = 1.;
    row[1] = 2.;
    assert_eq!(row.len(), 4);

    let row = mtr.row_real_as_slice(0);
    assert_eq!(row.len(), 4);
    assert_eq!(row[0], 1.);
    assert_eq!(row[1], 2.);

    let row = mtr.row_imag_as_mut_slice(0);
    row[0] = 3.;
    row[1] = 4.;
    assert_eq!(row.len(), 4);

    let row = mtr.row_imag_as_slice(0);
    assert_eq!(row.len(), 4);
    assert_eq!(row[0], 3.);
    assert_eq!(row[1], 4.);
}

#[test]
fn create_diagonal_op_01() {
    let env = &QuestEnv::new();

    let _ = DiagonalOp::try_new(1, env).unwrap();
    let _ = DiagonalOp::try_new(0, env).unwrap_err();
    let _ = DiagonalOp::try_new(-1, env).unwrap_err();
}

#[test]
fn set_diagonal_op_elems_01() {
    let env = &QuestEnv::new();
    let mut op = DiagonalOp::try_new(3, env).unwrap();

    let num_elems = 3;
    let re = [1., 2., 3.];
    let im = [1., 2., 3.];
    set_diagonal_op_elems(&mut op, 0, &re, &im, num_elems).unwrap();
    set_diagonal_op_elems(&mut op, -1, &re, &im, num_elems).unwrap_err();
    set_diagonal_op_elems(&mut op, 9, &re, &im, 3).unwrap_err();
}

#[test]
fn apply_diagonal_op_01() {
    let env = &QuestEnv::new();
    let mut qureg = create_qureg::<2>(env);
    let mut op = DiagonalOp::try_new(2, env).unwrap();

    init_diagonal_op(&mut op, &[1., 2., 3., 4.], &[5., 6., 7., 8.]).unwrap();
    apply_diagonal_op(&mut qureg, &op).unwrap();

    let mut op = DiagonalOp::try_new(1, env).unwrap();
    init_diagonal_op(&mut op, &[1., 2.], &[5., 6.]).unwrap();
    apply_diagonal_op(&mut qureg, &op).unwrap_err();
}

#[test]
fn calc_expec_diagonal_op_() {
    let env = &QuestEnv::new();
    let mut qureg = create_qureg::<2>(env);
    let mut op = DiagonalOp::try_new(2, env).unwrap();

    qureg.init_plus_state();
    init_diagonal_op(&mut op, &[1., 2., 3., 4.], &[5., 6., 7., 8.]).unwrap();

    let _ = calc_expec_diagonal_op(&qureg, &op).unwrap();

    let mut op = DiagonalOp::try_new(1, env).unwrap();
    init_diagonal_op(&mut op, &[1., 2.], &[5., 6.]).unwrap();
    let _ = calc_expec_diagonal_op(&qureg, &op).unwrap_err();
}

#[test]
fn create_pauli_hamil_01() {
    let _ = PauliHamil::try_new(1, 1).unwrap();
    let _ = PauliHamil::try_new(2, 3).unwrap();
    let _ = PauliHamil::try_new(3, 2).unwrap();

    let _ = PauliHamil::try_new(0, 1).unwrap_err();
    let _ = PauliHamil::try_new(-1, 1).unwrap_err();
    let _ = PauliHamil::try_new(1, 0).unwrap_err();
    let _ = PauliHamil::try_new(1, -1).unwrap_err();
    let _ = PauliHamil::try_new(0, 0).unwrap_err();
}

#[test]
fn initialize_pauli_hamil_01() {
    use PauliOpType::*;
    let mut hamil = PauliHamil::try_new(2, 2).unwrap();

    init_pauli_hamil(
        &mut hamil,
        &[0.5, -0.5],
        &[PAULI_X, PAULI_Y, PAULI_I, PAULI_I, PAULI_Z, PAULI_X],
    )
    .unwrap();
}

#[test]
fn set_amps_01() {
    let env = &QuestEnv::new();
    let mut qureg = create_qureg::<3>(env);

    let re = [1., 2., 3., 4.];
    let im = [1., 2., 3., 4.];

    set_amps(&mut qureg, 0, &re, &im).unwrap();
    assert!((get_real_amp(&qureg, 0).unwrap() - 1.).abs() < EPSILON);

    set_amps(&mut qureg, 3, &re, &im).unwrap();

    set_amps(&mut qureg, 9, &re, &im).unwrap_err();
    set_amps(&mut qureg, 7, &re, &im).unwrap_err();
    set_amps(&mut qureg, -1, &re, &im).unwrap_err();
}

#[test]
fn set_amps_02() {
    let env = &QuestEnv::new();
    let mut qureg = create_qureg::<2>(env);

    let re = [1.];
    let im = [1., 2.];

    let res = set_amps(&mut qureg, 0, &re, &im).unwrap_err();
    assert_eq!(res, QuestError::ArrayLengthError);

    let re = [1., 2.];
    let im = [1.];

    let res = set_amps(&mut qureg, 0, &re, &im).unwrap_err();
    assert_eq!(res, QuestError::ArrayLengthError);
}

#[test]
fn set_density_amps_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<3>(env);

    let re = &[1., 2., 3., 4.];
    let im = &[1., 2., 3., 4.];

    set_density_amps(qureg, 0, 0, re, im).unwrap();
    assert!((get_density_amp(qureg, 0, 0).unwrap().re - 1.).abs() < EPSILON);

    set_density_amps(qureg, 1, 3, re, im).unwrap();

    set_amps(qureg, 0, re, im).unwrap_err();

    set_density_amps(qureg, 0, 9, re, im).unwrap_err();
    set_density_amps(qureg, 8, 7, re, im).unwrap_err();
    set_density_amps(qureg, 0, -1, re, im).unwrap_err();
}

#[test]
fn set_density_amps_02() {
    let env = &QuestEnv::new();
    let mut qureg = create_density_qureg::<2>(env);

    let re = [1.];
    let im = [1., 2.];

    let res = set_density_amps(&mut qureg, 0, 0, &re, &im).unwrap_err();
    assert_eq!(res, QuestError::ArrayLengthError);

    let re = [1., 2.];
    let im = [1.];

    let res = set_density_amps(&mut qureg, 0, 0, &re, &im).unwrap_err();
    assert_eq!(res, QuestError::ArrayLengthError);
}

#[test]
fn phase_shift_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);

    phase_shift(qureg, 0, 0.0).unwrap();
    phase_shift(qureg, 1, 0.5).unwrap();
    phase_shift(qureg, 2, 1.0).unwrap();

    phase_shift(qureg, 3, 0.0).unwrap_err();
    phase_shift(qureg, -11, 0.0).unwrap_err();
}

#[test]
fn controlled_phase_shift_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);

    controlled_phase_shift(qureg, 0, 1, 0.5).unwrap();
    controlled_phase_shift(qureg, 0, 2, 0.5).unwrap();

    controlled_phase_shift(qureg, 0, 3, 0.5).unwrap_err();
    controlled_phase_shift(qureg, -1, 1, 0.5).unwrap_err();
}

#[test]
fn multi_controlled_phase_shift_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    multi_controlled_phase_shift(qureg, &[0, 1, 2], 0.5).unwrap();
    multi_controlled_phase_shift(qureg, &[2, 1, 0], 0.5).unwrap();

    multi_controlled_phase_shift(qureg, &[0, 1, 0], 0.5).unwrap_err();
    multi_controlled_phase_shift(qureg, &[0, 1, 1], 0.5).unwrap_err();

    multi_controlled_phase_shift(qureg, &[0, 4, 3, 4], 0.5).unwrap_err();
}

#[test]
fn controlled_phase_flip_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);

    controlled_phase_flip(qureg, 0, 1).unwrap();
    controlled_phase_flip(qureg, 0, 2).unwrap();

    controlled_phase_flip(qureg, 0, 3).unwrap_err();
    controlled_phase_flip(qureg, -1, 1).unwrap_err();
}

#[test]
fn multi_controlled_phase_flip_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::<'_, 4>::try_new(env).unwrap();
    multi_controlled_phase_flip(qureg, &[0, 1, 3]).unwrap();
    multi_controlled_phase_flip(qureg, &[0, 1, 3]).unwrap();

    multi_controlled_phase_flip(qureg, &[0, 4, 3, 4]).unwrap_err();
    multi_controlled_phase_flip(qureg, &[0, 7, -1]).unwrap_err();
}

#[test]
fn s_gate_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    s_gate(qureg, 0).unwrap();
    assert!((get_imag_amp(qureg, 0).unwrap()).abs() < EPSILON);

    pauli_x(qureg, 0).unwrap();
    s_gate(qureg, 0).unwrap();

    let amp = get_imag_amp(qureg, 1).unwrap();
    assert!((amp - 1.).abs() < EPSILON);

    s_gate(qureg, -1).unwrap_err();
    s_gate(qureg, 3).unwrap_err();
}

#[test]
fn t_gate_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    t_gate(qureg, 0).unwrap();
    t_gate(qureg, -1).unwrap_err();
    t_gate(qureg, 3).unwrap_err();
}

#[test]
fn get_amp_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_plus_state();

    get_amp(qureg, 0).unwrap();
    get_amp(qureg, 1).unwrap();
    get_amp(qureg, 2).unwrap();
    get_amp(qureg, 3).unwrap();

    get_amp(qureg, 4).unwrap_err();
    get_amp(qureg, -1).unwrap_err();
}

#[test]
fn get_real_amp_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_plus_state();

    get_real_amp(qureg, 0).unwrap();
    get_real_amp(qureg, 1).unwrap();
    get_real_amp(qureg, 2).unwrap();
    get_real_amp(qureg, 3).unwrap();

    get_real_amp(qureg, 4).unwrap_err();
    get_real_amp(qureg, -1).unwrap_err();
}

#[test]
fn get_imag_amp_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_plus_state();

    get_imag_amp(qureg, 0).unwrap();
    get_imag_amp(qureg, 1).unwrap();
    get_imag_amp(qureg, 2).unwrap();
    get_imag_amp(qureg, 3).unwrap();

    get_imag_amp(qureg, 4).unwrap_err();
    get_imag_amp(qureg, -1).unwrap_err();
}

#[test]
fn get_prob_amp_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_plus_state();

    get_prob_amp(qureg, 0).unwrap();
    get_prob_amp(qureg, 1).unwrap();
    get_prob_amp(qureg, 2).unwrap();
    get_prob_amp(qureg, 3).unwrap();

    get_prob_amp(qureg, 4).unwrap_err();
    get_prob_amp(qureg, -1).unwrap_err();
}

#[test]
fn get_density_amp_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);

    get_density_amp(qureg, 0, 0).unwrap_err();
    get_density_amp(qureg, 1, 0).unwrap_err();
    get_density_amp(qureg, 2, 0).unwrap_err();
    get_density_amp(qureg, 3, 0).unwrap_err();
    get_density_amp(qureg, -1, 5).unwrap_err();
    get_density_amp(qureg, 4, 0).unwrap_err();
}

#[test]
fn get_density_amp_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<2>(env);

    get_density_amp(qureg, 0, 0).unwrap();
    get_density_amp(qureg, 1, 0).unwrap();
    get_density_amp(qureg, 2, 0).unwrap();
    get_density_amp(qureg, 3, 0).unwrap();
    get_density_amp(qureg, -1, 0).unwrap_err();
    get_density_amp(qureg, 4, 0).unwrap_err();
}

#[test]
fn compact_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    let norm = SQRT_2.recip();
    let alpha = Qcomplex::new(0., norm);
    let beta = Qcomplex::new(0., norm);

    compact_unitary(qureg, 0, alpha, beta).unwrap();
    compact_unitary(qureg, 1, alpha, beta).unwrap();

    compact_unitary(qureg, 4, alpha, beta).unwrap_err();
    compact_unitary(qureg, -1, alpha, beta).unwrap_err();
}

#[test]
fn compact_unitary_02() {
    // env_logger::init();
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    // this doesn't define a unitary matrix
    let alpha = Qcomplex::new(1., 2.);
    let beta = Qcomplex::new(2., 1.);

    compact_unitary(qureg, 0, alpha, beta).unwrap_err();
    compact_unitary(qureg, 1, alpha, beta).unwrap_err();

    compact_unitary(qureg, 4, alpha, beta).unwrap_err();
    compact_unitary(qureg, -1, alpha, beta).unwrap_err();
}

#[test]
fn unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    let norm = SQRT_2.recip();
    let mtr = ComplexMatrix2::new(
        [[norm, norm], [norm, -norm]],
        [[0., 0.], [0., 0.]],
    );
    unitary(qureg, 0, &mtr).unwrap();
    unitary(qureg, 1, &mtr).unwrap();
    unitary(qureg, 2, &mtr).unwrap_err();
    unitary(qureg, -1, &mtr).unwrap_err();
}

#[test]
fn unitary_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    // This isn't a unitary
    let mtr = ComplexMatrix2::new([[1., 2.], [0., 0.]], [[0., 0.], [1., 2.]]);
    unitary(qureg, 0, &mtr).unwrap_err();
    unitary(qureg, 1, &mtr).unwrap_err();
    unitary(qureg, 2, &mtr).unwrap_err();
    unitary(qureg, -1, &mtr).unwrap_err();
}

#[test]
fn rotate_x_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    let theta = PI;
    rotate_x(qureg, 0, theta).unwrap();
    rotate_x(qureg, 1, theta).unwrap();
    rotate_x(qureg, 2, theta).unwrap();

    rotate_x(qureg, 3, theta).unwrap_err();
    rotate_x(qureg, -1, theta).unwrap_err();
}

#[test]
fn rotate_y_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    let theta = PI;
    rotate_y(qureg, 0, theta).unwrap();
    rotate_y(qureg, 1, theta).unwrap();
    rotate_y(qureg, 2, theta).unwrap();

    rotate_y(qureg, 3, theta).unwrap_err();
    rotate_y(qureg, -1, theta).unwrap_err();
}

#[test]
fn rotate_z_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    let theta = PI;
    rotate_z(qureg, 0, theta).unwrap();
    rotate_z(qureg, 1, theta).unwrap();
    rotate_z(qureg, 2, theta).unwrap();

    rotate_z(qureg, 3, theta).unwrap_err();
    rotate_z(qureg, -1, theta).unwrap_err();
}

#[test]
fn rotate_around_axis_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();

    let angle = 0.;
    let axis = &Vector::new(0., 0., 1.);
    rotate_around_axis(qureg, 0, angle, axis).unwrap();
    rotate_around_axis(qureg, 1, angle, axis).unwrap();
    rotate_around_axis(qureg, 2, angle, axis).unwrap();

    rotate_around_axis(qureg, 3, angle, axis).unwrap_err();
    rotate_around_axis(qureg, -1, angle, axis).unwrap_err();
}

#[test]
fn rotate_around_axis_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();

    let angle = 0.;
    // zero vector should throw an exception
    let axis = &Vector::new(0., 0., 0.);
    rotate_around_axis(qureg, 0, angle, axis).unwrap_err();
    rotate_around_axis(qureg, 1, angle, axis).unwrap_err();
    rotate_around_axis(qureg, 2, angle, axis).unwrap_err();

    rotate_around_axis(qureg, 3, angle, axis).unwrap_err();
    rotate_around_axis(qureg, -1, angle, axis).unwrap_err();
}

#[test]
fn controlled_rotate_x_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);

    controlled_rotate_x(qureg, 1, 0, 0.5).unwrap();
    controlled_rotate_x(qureg, 1, 2, 0.5).unwrap();

    controlled_rotate_x(qureg, 1, 1, 0.5).unwrap_err();
    controlled_rotate_x(qureg, 2, 2, 0.5).unwrap_err();
    controlled_rotate_x(qureg, -1, 2, 0.5).unwrap_err();
    controlled_rotate_x(qureg, 2, -1, 0.5).unwrap_err();
    controlled_rotate_x(qureg, 0, 4, 0.5).unwrap_err();
    controlled_rotate_x(qureg, 4, 0, 0.5).unwrap_err();
}

#[test]
fn controlled_rotate_y_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);

    controlled_rotate_y(qureg, 1, 0, 0.5).unwrap();
    controlled_rotate_y(qureg, 1, 2, 0.5).unwrap();

    controlled_rotate_y(qureg, 1, 1, 0.5).unwrap_err();
    controlled_rotate_y(qureg, 2, 2, 0.5).unwrap_err();
    controlled_rotate_y(qureg, -1, 2, 0.5).unwrap_err();
    controlled_rotate_y(qureg, 2, -1, 0.5).unwrap_err();
    controlled_rotate_y(qureg, 0, 4, 0.5).unwrap_err();
    controlled_rotate_y(qureg, 4, 0, 0.5).unwrap_err();
}

#[test]
fn controlled_rotate_z_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);

    controlled_rotate_z(qureg, 1, 0, 0.5).unwrap();
    controlled_rotate_z(qureg, 1, 2, 0.5).unwrap();

    controlled_rotate_z(qureg, 1, 1, 0.5).unwrap_err();
    controlled_rotate_z(qureg, 2, 2, 0.5).unwrap_err();
    controlled_rotate_z(qureg, -1, 2, 0.5).unwrap_err();
    controlled_rotate_z(qureg, 2, -1, 0.5).unwrap_err();
    controlled_rotate_z(qureg, 0, 4, 0.5).unwrap_err();
    controlled_rotate_z(qureg, 4, 0, 0.5).unwrap_err();
}

#[test]
fn controlled_rotate_around_axis_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    let vector = &Vector::new(0., 0., 1.);

    controlled_rotate_around_axis(qureg, 1, 0, 0.5, vector).unwrap();
    controlled_rotate_around_axis(qureg, 1, 2, 0.5, vector).unwrap();

    controlled_rotate_around_axis(qureg, 1, 1, 0.5, vector).unwrap_err();
    controlled_rotate_around_axis(qureg, 2, 2, 0.5, vector).unwrap_err();
    controlled_rotate_around_axis(qureg, -1, 2, 0.5, vector).unwrap_err();
    controlled_rotate_around_axis(qureg, 2, -1, 0.5, vector).unwrap_err();
    controlled_rotate_around_axis(qureg, 0, 4, 0.5, vector).unwrap_err();
    controlled_rotate_around_axis(qureg, 4, 0, 0.5, vector).unwrap_err();
}

#[test]
fn controlled_rotate_around_axis_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    // vector cannot be zero
    let vector = &Vector::new(0., 0., 0.);

    controlled_rotate_around_axis(qureg, 1, 0, 0.5, vector).unwrap_err();
    controlled_rotate_around_axis(qureg, 1, 2, 0.5, vector).unwrap_err();
}

#[test]
fn controlled_compact_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    let norm = SQRT_2.recip();
    let alpha = Qcomplex::new(0., norm);
    let beta = Qcomplex::new(0., norm);

    controlled_compact_unitary(qureg, 0, 1, alpha, beta).unwrap();
    controlled_compact_unitary(qureg, 1, 0, alpha, beta).unwrap();

    controlled_compact_unitary(qureg, 1, 1, alpha, beta).unwrap_err();
    controlled_compact_unitary(qureg, 2, 2, alpha, beta).unwrap_err();
    controlled_compact_unitary(qureg, 4, 1, alpha, beta).unwrap_err();
    controlled_compact_unitary(qureg, -1, 1, alpha, beta).unwrap_err();
}

#[test]
fn controlled_compact_unitary_02() {
    // env_logger::init();
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    // this doesn't define a unitary matrix
    let alpha = Qcomplex::new(1., 2.);
    let beta = Qcomplex::new(2., 1.);

    controlled_compact_unitary(qureg, 0, 1, alpha, beta).unwrap_err();
    controlled_compact_unitary(qureg, 1, 0, alpha, beta).unwrap_err();

    controlled_compact_unitary(qureg, 1, 1, alpha, beta).unwrap_err();
    controlled_compact_unitary(qureg, 4, 1, alpha, beta).unwrap_err();
    controlled_compact_unitary(qureg, -1, 2, alpha, beta).unwrap_err();
}

#[test]
fn controlled_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    let norm = SQRT_2.recip();
    let mtr = &ComplexMatrix2::new(
        [[norm, norm], [norm, -norm]],
        [[0., 0.], [0., 0.]],
    );

    controlled_unitary(qureg, 0, 1, mtr).unwrap();
    controlled_unitary(qureg, 1, 0, mtr).unwrap();

    controlled_unitary(qureg, 1, 1, mtr).unwrap_err();
    controlled_unitary(qureg, 2, 2, mtr).unwrap_err();
    controlled_unitary(qureg, 4, 1, mtr).unwrap_err();
    controlled_unitary(qureg, -1, 1, mtr).unwrap_err();
}

#[test]
fn controlled_unitary_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    // this doesn't define a unitary matrix
    let mtr = &ComplexMatrix2::new([[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]);

    controlled_unitary(qureg, 0, 1, mtr).unwrap_err();
    controlled_unitary(qureg, 1, 0, mtr).unwrap_err();

    controlled_unitary(qureg, 1, 1, mtr).unwrap_err();
    controlled_unitary(qureg, 2, 2, mtr).unwrap_err();
    controlled_unitary(qureg, 4, 1, mtr).unwrap_err();
    controlled_unitary(qureg, -1, 1, mtr).unwrap_err();
}

#[test]
fn multi_controlled_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();

    let norm = SQRT_2.recip();
    let mtr = &ComplexMatrix2::new(
        [[norm, norm], [norm, -norm]],
        [[0., 0.], [0., 0.]],
    );

    multi_controlled_unitary(qureg, &[0], 2, mtr).unwrap();
    multi_controlled_unitary(qureg, &[0, 1], 2, mtr).unwrap();
    multi_controlled_unitary(qureg, &[1, 0], 2, mtr).unwrap();
    multi_controlled_unitary(qureg, &[1, 2], 0, mtr).unwrap();

    multi_controlled_unitary(qureg, &[1, 1], 1, mtr).unwrap_err();
    multi_controlled_unitary(qureg, &[1, 1], 4, mtr).unwrap_err();
    multi_controlled_unitary(qureg, &[-1, 1], 0, mtr).unwrap_err();
    multi_controlled_unitary(qureg, &[1, 1, 1], 0, mtr).unwrap_err();
    multi_controlled_unitary(qureg, &[0, 1, 2, 3], 0, mtr).unwrap_err();
}

#[test]
fn multi_controlled_unitary_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();

    // this doesn't define a unitary matrix
    let mtr = &ComplexMatrix2::new([[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]);

    multi_controlled_unitary(qureg, &[0, 1], 2, mtr).unwrap_err();
    multi_controlled_unitary(qureg, &[1, 2], 0, mtr).unwrap_err();
    multi_controlled_unitary(qureg, &[1, 1], 1, mtr).unwrap_err();
    multi_controlled_unitary(qureg, &[1, 1], 4, mtr).unwrap_err();
}

#[test]
fn pauli_x_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    pauli_x(qureg, 0).unwrap();
    pauli_x(qureg, 1).unwrap();

    pauli_x(qureg, 2).unwrap_err();
    pauli_x(qureg, -1).unwrap_err();
}

#[test]
fn pauli_y_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    pauli_y(qureg, 0).unwrap();
    pauli_y(qureg, 1).unwrap();

    pauli_y(qureg, 2).unwrap_err();
    pauli_y(qureg, -1).unwrap_err();
}

#[test]
fn pauli_z_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    pauli_z(qureg, 0).unwrap();
    pauli_z(qureg, 1).unwrap();

    pauli_z(qureg, 2).unwrap_err();
    pauli_z(qureg, -1).unwrap_err();
}

#[test]
fn hadamard_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    hadamard(qureg, 0).unwrap();
    hadamard(qureg, 1).unwrap();
    hadamard(qureg, 2).unwrap_err();
    hadamard(qureg, -1).unwrap_err();
}

#[test]
fn controlled_not_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();
    pauli_x(qureg, 1).unwrap();

    controlled_not(qureg, 1, 0).unwrap();
    controlled_not(qureg, 0, 1).unwrap();

    controlled_not(qureg, 0, 0).unwrap_err();
    controlled_not(qureg, 1, 1).unwrap_err();
    controlled_not(qureg, 1, 2).unwrap_err();
    controlled_not(qureg, 2, 4).unwrap_err();
    controlled_not(qureg, 2, -1).unwrap_err();
}

#[test]
fn multi_qubit_not_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    multi_qubit_not(qureg, &[0, 1]).unwrap();
    multi_qubit_not(qureg, &[1, 0]).unwrap();
    multi_qubit_not(qureg, &[0, 0]).unwrap_err();
    multi_qubit_not(qureg, &[1, 1]).unwrap_err();
    multi_qubit_not(qureg, &[4, 1]).unwrap_err();
    multi_qubit_not(qureg, &[0, -1]).unwrap_err();
}

#[test]
fn multi_controlled_multi_qubit_not_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::<'_, 4>::try_new(env).unwrap();
    qureg.init_zero_state();

    multi_controlled_multi_qubit_not(qureg, &[0, 1], &[2, 3]).unwrap();
    multi_controlled_multi_qubit_not(qureg, &[1, 0], &[3, 2]).unwrap();
    multi_controlled_multi_qubit_not(qureg, &[1, 0], &[3]).unwrap();
    multi_controlled_multi_qubit_not(qureg, &[1], &[3, 0]).unwrap();

    multi_controlled_multi_qubit_not(qureg, &[1, 0], &[0]).unwrap_err();
    multi_controlled_multi_qubit_not(qureg, &[0, 0], &[1]).unwrap_err();
    multi_controlled_multi_qubit_not(qureg, &[0, 0], &[-1]).unwrap_err();
    multi_controlled_multi_qubit_not(qureg, &[4, 1], &[0]).unwrap_err();
    multi_controlled_multi_qubit_not(qureg, &[0, 1], &[4]).unwrap_err();
}

#[test]
fn controlled_pauli_y_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    controlled_pauli_y(qureg, 1, 0).unwrap();
    controlled_pauli_y(qureg, 0, 1).unwrap();

    controlled_pauli_y(qureg, 0, 0).unwrap_err();
    controlled_pauli_y(qureg, 1, 1).unwrap_err();
    controlled_pauli_y(qureg, 1, 2).unwrap_err();
    controlled_pauli_y(qureg, 2, 4).unwrap_err();
    controlled_pauli_y(qureg, 2, -1).unwrap_err();
}

#[test]
fn calc_prob_of_outcome_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    let _ = calc_prob_of_outcome(qureg, 0, 0).unwrap();
    let _ = calc_prob_of_outcome(qureg, 0, 1).unwrap();
    let _ = calc_prob_of_outcome(qureg, 1, 0).unwrap();
    let _ = calc_prob_of_outcome(qureg, 1, 1).unwrap();

    let _ = calc_prob_of_outcome(qureg, 0, 2).unwrap_err();
    let _ = calc_prob_of_outcome(qureg, 0, -2).unwrap_err();
    let _ = calc_prob_of_outcome(qureg, 1, 3).unwrap_err();
    let _ = calc_prob_of_outcome(qureg, 4, 0).unwrap_err();
}

#[test]
fn calc_prob_of_all_outcomes_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();

    let outcome_probs = &mut vec![0.; 4];
    calc_prob_of_all_outcomes(outcome_probs, qureg, &[1, 2]).unwrap();
    calc_prob_of_all_outcomes(outcome_probs, qureg, &[0, 1]).unwrap();
    calc_prob_of_all_outcomes(outcome_probs, qureg, &[0, 2]).unwrap();

    calc_prob_of_all_outcomes(outcome_probs, qureg, &[1, 3]).unwrap_err();
    calc_prob_of_all_outcomes(outcome_probs, qureg, &[0, 0]).unwrap_err();
    calc_prob_of_all_outcomes(outcome_probs, qureg, &[4, 0]).unwrap_err();
    calc_prob_of_all_outcomes(outcome_probs, qureg, &[0, -1]).unwrap_err();
}

#[test]
fn calc_prob_of_all_outcomes_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();

    let outcome_probs = &mut vec![0.; 3];

    calc_prob_of_all_outcomes(outcome_probs, qureg, &[1, 2]).unwrap_err();
}

#[test]
fn collapse_to_outcome_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);

    qureg.init_zero_state();
    collapse_to_outcome(qureg, 0, 0).unwrap();

    qureg.init_zero_state();
    collapse_to_outcome(qureg, 0, 1).unwrap_err();

    qureg.init_zero_state();
    collapse_to_outcome(qureg, -1, 0).unwrap_err();
    collapse_to_outcome(qureg, 3, 0).unwrap_err();
    collapse_to_outcome(qureg, 1, 3).unwrap_err();
    collapse_to_outcome(qureg, 4, 3).unwrap_err();
}

#[test]
fn measure_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);

    qureg.init_zero_state();

    let _ = measure(qureg, 0).unwrap();
    let _ = measure(qureg, 1).unwrap();
    let _ = measure(qureg, -1).unwrap_err();
    let _ = measure(qureg, 3).unwrap_err();
}

#[test]
fn measure_with_stats_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);

    // Prepare a triplet state `|00> + |11>`
    qureg.init_zero_state();

    let prob = &mut -1.;
    let _ = measure_with_stats(qureg, 0, prob).unwrap();
    let _ = measure_with_stats(qureg, 1, prob).unwrap();
    let _ = measure_with_stats(qureg, -1, prob).unwrap_err();
    let _ = measure_with_stats(qureg, 3, prob).unwrap_err();
}

#[test]
fn calc_inner_product_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();
    let other_qureg = &mut create_qureg::<2>(env);
    other_qureg.init_zero_state();

    let _ = calc_inner_product(qureg, other_qureg).unwrap();
}

// #[test]
// fn calc_inner_product_02() {
//     let env = &QuestEnv::new();
//     let qureg = &mut create_qureg::<2>(env);
//     qureg.init_zero_state();
//     let other_qureg = &mut Qureg::<'_, 1>::try_new(env).unwrap();
//     other_qureg.init_zero_state();

//     let _ = calc_inner_product(qureg, other_qureg).unwrap_err();
// }

#[test]
fn calc_inner_product_03() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();
    let other_qureg = &mut create_density_qureg::<2>(env);
    other_qureg.init_zero_state();

    let _ = calc_inner_product(qureg, other_qureg).unwrap_err();
}

#[test]
fn calc_density_inner_product_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<2>(env);
    qureg.init_zero_state();
    let other_qureg = &mut create_density_qureg::<2>(env);
    other_qureg.init_zero_state();

    let _ = calc_density_inner_product(qureg, other_qureg).unwrap();
}

// #[test]
// fn calc_density_inner_product_02() {
//     let env = &QuestEnv::new();
//     let qureg = &mut create_density_qureg::<2>(env);
//     qureg.init_zero_state();
//     let other_qureg = &mut Qureg::try_new_density(1, env).unwrap();
//     other_qureg.init_zero_state();

//     let _ = calc_density_inner_product(qureg, other_qureg).unwrap_err();
// }

#[test]
fn calc_density_inner_product_03() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<2>(env);
    qureg.init_zero_state();
    let other_qureg = &mut create_qureg::<2>(env);
    other_qureg.init_zero_state();

    let _ = calc_density_inner_product(qureg, other_qureg).unwrap_err();
}

#[test]
fn get_quest_seeds_01() {
    let env = &QuestEnv::new();
    let seeds = get_quest_seeds(env);

    assert!(!seeds.is_empty());
}

#[test]
fn get_quest_seeds_02() {
    let env = &mut QuestEnv::new();
    let seed_array = &[0, 1, 2, 3];
    seed_quest(env, seed_array);
    let seeds = get_quest_seeds(env);

    assert!(!seeds.is_empty());
    assert_eq!(seed_array, seeds);
}

#[test]
fn start_recording_qasm_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);

    start_recording_qasm(qureg);
    hadamard(qureg, 0).and(controlled_not(qureg, 0, 1)).unwrap();
    stop_recording_qasm(qureg);

    print_recorded_qasm(qureg);
}

#[test]
fn mix_dephasing_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<2>(env);
    qureg.init_plus_state();

    mix_dephasing(qureg, 0, 0.5).unwrap();
    mix_dephasing(qureg, 1, 0.0).unwrap();

    mix_dephasing(qureg, 0, 0.75).unwrap_err();
    mix_dephasing(qureg, 2, 0.25).unwrap_err();
    mix_dephasing(qureg, 0, -0.25).unwrap_err();
    mix_dephasing(qureg, -10, 0.25).unwrap_err();
}

#[test]
fn mix_dephasing_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_plus_state();

    // qureg is not a density matrix
    mix_dephasing(qureg, 0, 0.5).unwrap_err();
    mix_dephasing(qureg, 1, 0.0).unwrap_err();
}

#[test]
fn mix_two_qubit_dephasing_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<3>(env);
    qureg.init_plus_state();

    mix_two_qubit_dephasing(qureg, 0, 1, 0.75).unwrap();
    mix_two_qubit_dephasing(qureg, 0, 2, 0.75).unwrap();
    mix_two_qubit_dephasing(qureg, 1, 2, 0.75).unwrap();
    mix_two_qubit_dephasing(qureg, 1, 0, 0.75).unwrap();
    mix_two_qubit_dephasing(qureg, 2, 1, 0.75).unwrap();

    mix_two_qubit_dephasing(qureg, 0, 1, 0.99).unwrap_err();
    mix_two_qubit_dephasing(qureg, 2, 1, 0.99).unwrap_err();

    mix_two_qubit_dephasing(qureg, 4, 0, 0.1).unwrap_err();
    mix_two_qubit_dephasing(qureg, 0, 4, 0.1).unwrap_err();

    mix_two_qubit_dephasing(qureg, -1, 0, 0.1).unwrap_err();
    mix_two_qubit_dephasing(qureg, 0, -1, 0.1).unwrap_err();
}

#[test]
fn mix_two_qubit_dephasing_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_plus_state();

    // qureg is not a density matrix
    mix_two_qubit_dephasing(qureg, 0, 1, 0.75).unwrap_err();
    mix_two_qubit_dephasing(qureg, 0, 2, 0.75).unwrap_err();
}

#[test]
fn mix_depolarising_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<2>(env);
    qureg.init_zero_state();

    mix_depolarising(qureg, 0, 0.00).unwrap();
    mix_depolarising(qureg, 0, 0.75).unwrap();
    mix_depolarising(qureg, 1, 0.75).unwrap();

    mix_depolarising(qureg, 0, 0.99).unwrap_err();
    mix_depolarising(qureg, 1, 0.99).unwrap_err();
    mix_depolarising(qureg, 0, -0.99).unwrap_err();
    mix_depolarising(qureg, -1, 0.99).unwrap_err();
    mix_depolarising(qureg, -1, -0.99).unwrap_err();
}

#[test]
fn mix_damping_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<2>(env);
    qureg.init_plus_state();

    mix_damping(qureg, 0, 1.).unwrap();
    mix_damping(qureg, 0, 0.).unwrap();
    mix_damping(qureg, 1, 1.).unwrap();
    mix_damping(qureg, 1, 0.).unwrap();

    mix_damping(qureg, 0, 10.).unwrap_err();
    mix_damping(qureg, 0, -10.).unwrap_err();
    mix_damping(qureg, 1, 10.).unwrap_err();
    mix_damping(qureg, 1, -10.).unwrap_err();
    mix_damping(qureg, 3, 0.5).unwrap_err();
    mix_damping(qureg, -3, 0.5).unwrap_err();
}

#[test]
fn mix_damping_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_plus_state();

    // qureg is not a density matrix
    mix_damping(qureg, 1, 0.).unwrap_err();
    mix_damping(qureg, 0, 0.).unwrap_err();
    // QuEST seg faults here:
    mix_damping(qureg, 0, 0.5).unwrap_err();
}

#[test]
fn mix_two_qubit_depolarising_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<3>(env);
    qureg.init_plus_state();

    mix_two_qubit_depolarising(qureg, 0, 1, 15. / 16.).unwrap();
    mix_two_qubit_depolarising(qureg, 0, 2, 15. / 16.).unwrap();
    mix_two_qubit_depolarising(qureg, 1, 2, 15. / 16.).unwrap();
    mix_two_qubit_depolarising(qureg, 1, 0, 15. / 16.).unwrap();
    mix_two_qubit_depolarising(qureg, 2, 1, 15. / 16.).unwrap();

    mix_two_qubit_depolarising(qureg, 0, 1, 0.99).unwrap_err();
    mix_two_qubit_depolarising(qureg, 2, 1, 0.99).unwrap_err();

    mix_two_qubit_depolarising(qureg, 4, 0, 0.1).unwrap_err();
    mix_two_qubit_depolarising(qureg, 0, 4, 0.1).unwrap_err();

    mix_two_qubit_depolarising(qureg, -1, 0, 0.1).unwrap_err();
    mix_two_qubit_depolarising(qureg, 0, -1, 0.1).unwrap_err();
}

#[test]
fn mix_two_qubit_depolarising_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_plus_state();

    // qureg is not a density matrix
    mix_two_qubit_depolarising(qureg, 0, 1, 0.75).unwrap_err();
    mix_two_qubit_depolarising(qureg, 0, 2, 0.75).unwrap_err();
}

#[test]
fn mix_pauli_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<2>(env);
    qureg.init_zero_state();

    let (prob_x, prob_y, prob_z) = (0.25, 0.25, 0.25);
    mix_pauli(qureg, 0, prob_x, prob_y, prob_z).unwrap();
    mix_pauli(qureg, 1, prob_x, prob_y, prob_z).unwrap();

    mix_pauli(qureg, 2, prob_x, prob_y, prob_z).unwrap_err();
    mix_pauli(qureg, -2, prob_x, prob_y, prob_z).unwrap_err();
}

#[test]
fn mix_pauli_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<2>(env);
    qureg.init_zero_state();

    // this is not a prob distribution
    let (prob_x, prob_y, prob_z) = (0.5, 0.5, 0.5);
    mix_pauli(qureg, 0, prob_x, prob_y, prob_z).unwrap_err();
    mix_pauli(qureg, 1, prob_x, prob_y, prob_z).unwrap_err();
}

#[test]
fn mix_pauli_03() {
    let env = &QuestEnv::new();
    // not a density matrix
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    let (prob_x, prob_y, prob_z) = (0.25, 0.25, 0.25);
    mix_pauli(qureg, 0, prob_x, prob_y, prob_z).unwrap_err();
    mix_pauli(qureg, 1, prob_x, prob_y, prob_z).unwrap_err();
}

#[test]
fn mix_density_matrix_01() {
    let env = &QuestEnv::new();
    let combine_qureg = &mut create_density_qureg::<2>(env);
    let other_qureg = &mut create_density_qureg::<2>(env);

    combine_qureg.init_zero_state();
    other_qureg.init_zero_state();

    mix_density_matrix(combine_qureg, 0.0, other_qureg).unwrap();
    mix_density_matrix(combine_qureg, 0.5, other_qureg).unwrap();
    mix_density_matrix(combine_qureg, 0.99, other_qureg).unwrap();

    mix_density_matrix(combine_qureg, 1.01, other_qureg).unwrap_err();
    mix_density_matrix(combine_qureg, -1.01, other_qureg).unwrap_err();
}

#[test]
fn mix_density_matrix_02() {
    let env = &QuestEnv::new();
    // this is not a density matrix
    let combine_qureg = &mut create_qureg::<2>(env);
    let other_qureg = &mut create_density_qureg::<2>(env);

    combine_qureg.init_zero_state();
    other_qureg.init_zero_state();

    mix_density_matrix(combine_qureg, 0.0, other_qureg).unwrap_err();
}

#[test]
fn mix_density_matrix_03() {
    let env = &QuestEnv::new();
    let combine_qureg = &mut create_density_qureg::<2>(env);
    // this is not a density matrix
    let other_qureg = &mut create_qureg::<2>(env);

    combine_qureg.init_zero_state();
    other_qureg.init_zero_state();

    mix_density_matrix(combine_qureg, 0.0, other_qureg).unwrap_err();
}

// #[test]
// fn mix_density_matrix_04() {
//     let env = &QuestEnv::new();
//     // dimensions don't match
//     let combine_qureg = &mut create_density_qureg::<2>(env);
//     let other_qureg = &mut create_density_qureg::<3>(env);

//     combine_qureg.init_zero_state();
//     other_qureg.init_zero_state();

//     mix_density_matrix(combine_qureg, 0.0, other_qureg).unwrap_err();
// }

#[test]
fn calc_purity_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<2>(env);
    qureg.init_zero_state();

    let _ = calc_purity(qureg).unwrap();

    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();
    let _ = calc_purity(qureg).unwrap_err();
}

#[test]
fn calc_fidelity_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<2>(env);
    let pure_state = &mut create_qureg::<2>(env);

    qureg.init_zero_state();
    pure_state.init_zero_state();

    let _ = calc_fidelity(qureg, pure_state).unwrap();
}

// #[test]
// fn calc_fidelity_02() {
//     let env = &QuestEnv::new();
//     let qureg = &mut create_density_qureg::<3>(env);
//     let pure_state = &mut create_qureg::<2>(env);

//     qureg.init_zero_state();
//     pure_state.init_zero_state();

//     let _ = calc_fidelity(qureg, pure_state).unwrap_err();
// }

#[test]
fn calc_fidelity_03() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    let pure_state = &mut create_density_qureg::<2>(env);

    qureg.init_zero_state();
    pure_state.init_zero_state();

    let _ = calc_fidelity(qureg, pure_state).unwrap_err();
}

#[test]
fn swap_gate_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    swap_gate(qureg, 0, 1).unwrap();
    swap_gate(qureg, 1, 0).unwrap();

    swap_gate(qureg, 0, 0).unwrap_err();
    swap_gate(qureg, 1, 1).unwrap_err();
}

#[test]
fn swap_gate_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    swap_gate(qureg, 0, 2).unwrap_err();
    swap_gate(qureg, 2, 0).unwrap_err();

    swap_gate(qureg, -1, 0).unwrap_err();
    swap_gate(qureg, 0, -1).unwrap_err();

    swap_gate(qureg, 4, 0).unwrap_err();
    swap_gate(qureg, 0, 4).unwrap_err();
    swap_gate(qureg, 4, 4).unwrap_err();
}

#[test]
fn swap_gate_03() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    // QuEST seg faults here
    swap_gate(qureg, 4, -4).unwrap_err();
    swap_gate(qureg, -4, 4).unwrap_err();
    swap_gate(qureg, -4, -4).unwrap_err();
}

#[test]
fn sqrt_swap_gate_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    sqrt_swap_gate(qureg, 0, 1).unwrap();
    sqrt_swap_gate(qureg, 1, 0).unwrap();

    sqrt_swap_gate(qureg, 0, 0).unwrap_err();
    sqrt_swap_gate(qureg, 1, 1).unwrap_err();
}

#[test]
fn sqrt_swap_gate_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    sqrt_swap_gate(qureg, 0, 2).unwrap_err();
    sqrt_swap_gate(qureg, 2, 0).unwrap_err();

    sqrt_swap_gate(qureg, -1, 0).unwrap_err();
    sqrt_swap_gate(qureg, 0, -1).unwrap_err();

    sqrt_swap_gate(qureg, 4, 0).unwrap_err();
    sqrt_swap_gate(qureg, 0, 4).unwrap_err();
    sqrt_swap_gate(qureg, 4, 4).unwrap_err();
}

#[test]
fn sqrt_swap_gate_03() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    // QuEST seg faults here
    sqrt_swap_gate(qureg, 4, -4).unwrap_err();
    sqrt_swap_gate(qureg, -4, 4).unwrap_err();
    sqrt_swap_gate(qureg, -4, -4).unwrap_err();
}

#[test]
fn multi_rotate_z_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    multi_rotate_z(qureg, &[0, 1], PI).unwrap();
    multi_rotate_z(qureg, &[0, 1, 2], PI).unwrap_err();
    multi_rotate_z(qureg, &[0, 2], PI).unwrap_err();
    multi_rotate_z(qureg, &[0, 0], PI).unwrap_err();
}

#[test]
fn multi_state_controlled_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();

    let u = &ComplexMatrix2::new([[0., 1.], [1., 0.]], [[0., 0.], [0., 0.]]);
    multi_state_controlled_unitary(qureg, &[1, 2], &[0, 0], 0, u).unwrap();
    multi_state_controlled_unitary(qureg, &[0, 2], &[0, 0], 1, u).unwrap();
    multi_state_controlled_unitary(qureg, &[0, 1], &[0, 0], 2, u).unwrap();

    multi_state_controlled_unitary(qureg, &[0, 1], &[0, 0], 0, u).unwrap_err();
    multi_state_controlled_unitary(qureg, &[0, 1], &[0, 0], 1, u).unwrap_err();
    multi_state_controlled_unitary(qureg, &[0, 0], &[0, 0], 1, u).unwrap_err();

    multi_state_controlled_unitary(qureg, &[0, 0], &[0, 0], 3, u).unwrap_err();
    multi_state_controlled_unitary(qureg, &[0, 0], &[0, 0], -1, u).unwrap_err();
    multi_state_controlled_unitary(qureg, &[4, 0], &[0, 0], 1, u).unwrap_err();
    multi_state_controlled_unitary(qureg, &[4, -1], &[0, 0], 1, u).unwrap_err();
}

#[test]
fn multi_qubit_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    let u = &mut ComplexMatrixN::try_new(2).unwrap();
    let zero_row = &[0., 0., 0., 0.];
    init_complex_matrix_n(
        u,
        &[
            &[0., 0., 0., 1.],
            &[0., 1., 0., 0.],
            &[0., 0., 1., 0.],
            &[1., 0., 0., 0.],
        ],
        &[zero_row, zero_row, zero_row, zero_row],
    )
    .unwrap();

    multi_qubit_unitary(qureg, &[0, 1], u).unwrap();
    multi_qubit_unitary(qureg, &[1, 0], u).unwrap();

    multi_qubit_unitary(qureg, &[0, 0], u).unwrap_err();
    multi_qubit_unitary(qureg, &[1, 1], u).unwrap_err();

    multi_qubit_unitary(qureg, &[0, 2], u).unwrap_err();
    multi_qubit_unitary(qureg, &[1, -1], u).unwrap_err();
}

#[test]
fn multi_qubit_unitary_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    let u = &mut ComplexMatrixN::try_new(2).unwrap();
    let zero_row = &[0., 0., 0., 0.];

    // This is not a unitary matrix
    init_complex_matrix_n(
        u,
        &[zero_row, zero_row, zero_row, zero_row],
        &[zero_row, zero_row, zero_row, zero_row],
    )
    .unwrap();

    multi_qubit_unitary(qureg, &[0, 1], u).unwrap_err();
    multi_qubit_unitary(qureg, &[1, 0], u).unwrap_err();

    multi_qubit_unitary(qureg, &[0, 0], u).unwrap_err();
    multi_qubit_unitary(qureg, &[1, 1], u).unwrap_err();

    multi_qubit_unitary(qureg, &[0, 2], u).unwrap_err();
    multi_qubit_unitary(qureg, &[1, -1], u).unwrap_err();
}

#[test]
fn controlled_multi_qubit_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();

    let u = &mut ComplexMatrixN::try_new(2).unwrap();
    let zero_row = &[0., 0., 0., 0.];
    init_complex_matrix_n(
        u,
        &[
            &[0., 0., 0., 1.],
            &[0., 1., 0., 0.],
            &[0., 0., 1., 0.],
            &[1., 0., 0., 0.],
        ],
        &[zero_row, zero_row, zero_row, zero_row],
    )
    .unwrap();

    controlled_multi_qubit_unitary(qureg, 2, &[0, 1], u).unwrap();
    controlled_multi_qubit_unitary(qureg, 2, &[1, 0], u).unwrap();

    controlled_multi_qubit_unitary(qureg, 2, &[0, 0], u).unwrap_err();
    controlled_multi_qubit_unitary(qureg, 2, &[1, 1], u).unwrap_err();

    controlled_multi_qubit_unitary(qureg, 2, &[0, 2], u).unwrap_err();
    controlled_multi_qubit_unitary(qureg, 2, &[1, -1], u).unwrap_err();

    controlled_multi_qubit_unitary(qureg, -1, &[0, 1], u).unwrap_err();
    controlled_multi_qubit_unitary(qureg, 4, &[0, 1], u).unwrap_err();
}

#[test]
fn controlled_multi_qubit_unitary_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();

    let u = &mut ComplexMatrixN::try_new(2).unwrap();
    let zero_row = &[0., 0., 0., 0.];
    // This matrix is not unitary
    init_complex_matrix_n(
        u,
        &[zero_row, zero_row, zero_row, zero_row],
        &[zero_row, zero_row, zero_row, zero_row],
    )
    .unwrap();

    controlled_multi_qubit_unitary(qureg, 2, &[0, 1], u).unwrap_err();
    controlled_multi_qubit_unitary(qureg, 2, &[1, 0], u).unwrap_err();

    controlled_multi_qubit_unitary(qureg, 2, &[0, 0], u).unwrap_err();
    controlled_multi_qubit_unitary(qureg, 2, &[1, 1], u).unwrap_err();

    controlled_multi_qubit_unitary(qureg, 2, &[0, 2], u).unwrap_err();
    controlled_multi_qubit_unitary(qureg, 2, &[1, -1], u).unwrap_err();

    controlled_multi_qubit_unitary(qureg, -1, &[0, 1], u).unwrap_err();
    controlled_multi_qubit_unitary(qureg, 4, &[0, 1], u).unwrap_err();
}

#[test]
fn muti_controlled_multi_qubit_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::<'_, 4>::try_new(env).unwrap();
    qureg.init_zero_state();

    let u = &mut ComplexMatrixN::try_new(2).unwrap();
    let zero_row = &[0., 0., 0., 0.];
    init_complex_matrix_n(
        u,
        &[
            &[0., 0., 0., 1.],
            &[0., 1., 0., 0.],
            &[0., 0., 1., 0.],
            &[1., 0., 0., 0.],
        ],
        &[zero_row, zero_row, zero_row, zero_row],
    )
    .unwrap();

    multi_controlled_multi_qubit_unitary(qureg, &[2, 3], &[0, 1], u).unwrap();
    multi_controlled_multi_qubit_unitary(qureg, &[2, 3], &[1, 0], u).unwrap();

    multi_controlled_multi_qubit_unitary(qureg, &[2, 3], &[0, 0], u)
        .unwrap_err();
    multi_controlled_multi_qubit_unitary(qureg, &[2, 3], &[1, 1], u)
        .unwrap_err();

    multi_controlled_multi_qubit_unitary(qureg, &[2, 3], &[0, 2], u)
        .unwrap_err();
    multi_controlled_multi_qubit_unitary(qureg, &[2, 3], &[1, -1], u)
        .unwrap_err();

    multi_controlled_multi_qubit_unitary(qureg, &[2, 3], &[0, 2, 1], u)
        .unwrap_err();
    multi_controlled_multi_qubit_unitary(qureg, &[2, 3], &[1, -1, 5], u)
        .unwrap_err();
}

#[test]
fn muti_controlled_multi_qubit_unitary_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::<'_, 4>::try_new(env).unwrap();
    qureg.init_zero_state();

    let u = &mut ComplexMatrixN::try_new(2).unwrap();
    let zero_row = &[0., 0., 0., 0.];
    init_complex_matrix_n(
        u,
        &[
            &[0., 0., 0., 1.],
            &[0., 1., 0., 0.],
            &[0., 0., 1., 0.],
            &[1., 0., 0., 0.],
        ],
        &[zero_row, zero_row, zero_row, zero_row],
    )
    .unwrap();

    multi_controlled_multi_qubit_unitary(qureg, &[0, 1], &[2, 3], u).unwrap();
    multi_controlled_multi_qubit_unitary(qureg, &[1, 0], &[2, 3], u).unwrap();

    multi_controlled_multi_qubit_unitary(qureg, &[0, 0], &[2, 3], u)
        .unwrap_err();
    multi_controlled_multi_qubit_unitary(qureg, &[1, 1], &[2, 3], u)
        .unwrap_err();

    multi_controlled_multi_qubit_unitary(qureg, &[0, 2], &[2, 3], u)
        .unwrap_err();
    multi_controlled_multi_qubit_unitary(qureg, &[1, -1], &[2, 3], u)
        .unwrap_err();

    multi_controlled_multi_qubit_unitary(qureg, &[0, 2, 1], &[2, 3], u)
        .unwrap_err();
    multi_controlled_multi_qubit_unitary(qureg, &[1, -1, 5], &[2, 3], u)
        .unwrap_err();
}

#[test]
fn muti_controlled_multi_qubit_unitary_03() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::<'_, 4>::try_new(env).unwrap();
    qureg.init_zero_state();

    let u = &mut ComplexMatrixN::try_new(2).unwrap();
    let zero_row = &[0., 0., 0., 0.];
    // This matrix is not unitary
    init_complex_matrix_n(
        u,
        &[zero_row, zero_row, zero_row, zero_row],
        &[zero_row, zero_row, zero_row, zero_row],
    )
    .unwrap();

    multi_controlled_multi_qubit_unitary(qureg, &[0, 1], &[2, 3], u)
        .unwrap_err();
    multi_controlled_multi_qubit_unitary(qureg, &[1, 0], &[2, 3], u)
        .unwrap_err();

    multi_controlled_multi_qubit_unitary(qureg, &[0, 0], &[2, 3], u)
        .unwrap_err();
    multi_controlled_multi_qubit_unitary(qureg, &[1, 1], &[2, 3], u)
        .unwrap_err();

    multi_controlled_multi_qubit_unitary(qureg, &[0, 2], &[2, 3], u)
        .unwrap_err();
    multi_controlled_multi_qubit_unitary(qureg, &[1, -1], &[2, 3], u)
        .unwrap_err();

    multi_controlled_multi_qubit_unitary(qureg, &[0, 2, 1], &[2, 3], u)
        .unwrap_err();
    multi_controlled_multi_qubit_unitary(qureg, &[1, -1, 5], &[2, 3], u)
        .unwrap_err();
}

#[test]
fn apply_matrix2_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    let m = &ComplexMatrix2::new([[0., 1.], [1., 0.]], [[0., 0.], [0., 0.]]);

    apply_matrix2(qureg, 0, m).unwrap();
    apply_matrix2(qureg, 1, m).unwrap();

    apply_matrix2(qureg, -1, m).unwrap_err();
    apply_matrix2(qureg, 2, m).unwrap_err();
}

#[test]
fn mix_kraus_map_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<2>(env);
    qureg.init_zero_state();

    let m = &ComplexMatrix2::new([[0., 1.], [1., 0.]], [[0., 0.], [0., 0.]]);

    mix_kraus_map(qureg, 0, &[m]).unwrap();
    mix_kraus_map(qureg, 1, &[m]).unwrap();

    mix_kraus_map(qureg, -1, &[m]).unwrap_err();
    mix_kraus_map(qureg, 2, &[m]).unwrap_err();
}

#[test]
fn mix_kraus_map_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<2>(env);
    qureg.init_zero_state();

    // This is not a CPTP map
    let m1 = &ComplexMatrix2::new([[0., 1.], [1., 0.]], [[0., 0.], [0., 0.]]);
    let m2 = &ComplexMatrix2::new([[0., 1.], [1., 1.]], [[0., 0.], [0., 0.]]);

    mix_kraus_map(qureg, 0, &[m1, m2]).unwrap_err();
    mix_kraus_map(qureg, 1, &[m1, m2]).unwrap_err();
}

#[test]
fn mix_kraus_map_03() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<2>(env);
    qureg.init_zero_state();

    let m1 = &ComplexMatrix2::new([[0., 1.], [1., 0.]], [[0., 0.], [0., 0.]]);

    mix_kraus_map(qureg, 0, &[]).unwrap_err();
    mix_kraus_map(qureg, 0, &[m1, m1]).unwrap_err();
    mix_kraus_map(qureg, 0, &[m1, m1, m1]).unwrap_err();
    mix_kraus_map(qureg, 0, &[m1, m1, m1, m1]).unwrap_err();
}

#[test]
fn mix_two_qubit_kraus_map_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<3>(env);
    qureg.init_zero_state();

    let m = &ComplexMatrix4::new(
        [
            [0., 0., 0., 1.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
        ],
        [
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ],
    );
    mix_two_qubit_kraus_map(qureg, 0, 1, &[m]).unwrap();
    mix_two_qubit_kraus_map(qureg, 1, 2, &[m]).unwrap();
    mix_two_qubit_kraus_map(qureg, 0, 2, &[m]).unwrap();

    mix_two_qubit_kraus_map(qureg, 0, 0, &[m]).unwrap_err();
    mix_two_qubit_kraus_map(qureg, 1, 1, &[m]).unwrap_err();
    mix_two_qubit_kraus_map(qureg, 2, 2, &[m]).unwrap_err();

    mix_two_qubit_kraus_map(qureg, -1, 0, &[m]).unwrap_err();
    mix_two_qubit_kraus_map(qureg, 0, 4, &[m]).unwrap_err();
}

#[test]
fn mix_two_qubit_kraus_map_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<3>(env);
    qureg.init_zero_state();

    // This is not a TP map
    let m = &ComplexMatrix4::new(
        [
            [99., 0., 0., 1.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
        ],
        [
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ],
    );
    mix_two_qubit_kraus_map(qureg, 0, 1, &[m]).unwrap_err();
    mix_two_qubit_kraus_map(qureg, 1, 2, &[m]).unwrap_err();
    mix_two_qubit_kraus_map(qureg, 0, 2, &[m]).unwrap_err();
}

#[test]
fn mix_two_qubit_kraus_map_03() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<3>(env);
    qureg.init_zero_state();

    let m = &ComplexMatrix4::new(
        [
            [0., 0., 0., 1.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
        ],
        [
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ],
    );
    mix_two_qubit_kraus_map(qureg, 0, 1, &[]).unwrap_err();
    mix_two_qubit_kraus_map(qureg, 1, 2, &[m, m]).unwrap_err();
    mix_two_qubit_kraus_map(qureg, 0, 2, &[m, m, m]).unwrap_err();
}

#[test]
fn mix_multi_qubit_kraus_map_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<3>(env);
    qureg.init_zero_state();
    let m = &mut ComplexMatrixN::try_new(2).unwrap();
    init_complex_matrix_n(
        m,
        &[
            &[0., 0., 0., 1.],
            &[0., 1., 0., 0.],
            &[0., 0., 1., 0.],
            &[1., 0., 0., 0.],
        ],
        &[
            &[0., 0., 0., 0.],
            &[0., 0., 0., 0.],
            &[0., 0., 0., 0.],
            &[0., 0., 0., 0.],
        ],
    )
    .unwrap();

    mix_multi_qubit_kraus_map(qureg, &[1, 2], &[m]).unwrap();
    mix_multi_qubit_kraus_map(qureg, &[0, 1], &[m]).unwrap();
    mix_multi_qubit_kraus_map(qureg, &[0, 2], &[m]).unwrap();
    mix_multi_qubit_kraus_map(qureg, &[2, 0], &[m]).unwrap();

    mix_multi_qubit_kraus_map(qureg, &[0, 0], &[m]).unwrap_err();
    mix_multi_qubit_kraus_map(qureg, &[1, 1], &[m]).unwrap_err();
    mix_multi_qubit_kraus_map(qureg, &[2, 2], &[m]).unwrap_err();

    mix_multi_qubit_kraus_map(qureg, &[-1, 0], &[m]).unwrap_err();
    mix_multi_qubit_kraus_map(qureg, &[0, 4], &[m]).unwrap_err();
}

#[test]
fn mix_multi_qubit_kraus_map_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<3>(env);
    qureg.init_zero_state();
    let m = &mut ComplexMatrixN::try_new(2).unwrap();

    // This is not at TP map
    init_complex_matrix_n(
        m,
        &[
            &[99., 0., 0., 1.],
            &[0., 1., 0., 0.],
            &[0., 0., 1., 0.],
            &[1., 0., 0., 0.],
        ],
        &[
            &[0., 0., 0., 0.],
            &[0., 0., 0., 0.],
            &[0., 0., 0., 0.],
            &[0., 0., 0., 0.],
        ],
    )
    .unwrap();

    mix_multi_qubit_kraus_map(qureg, &[1, 2], &[m]).unwrap_err();
    mix_multi_qubit_kraus_map(qureg, &[0, 1], &[m]).unwrap_err();
    mix_multi_qubit_kraus_map(qureg, &[0, 2], &[m]).unwrap_err();
    mix_multi_qubit_kraus_map(qureg, &[2, 0], &[m]).unwrap_err();

    mix_multi_qubit_kraus_map(qureg, &[0, 0], &[m]).unwrap_err();
    mix_multi_qubit_kraus_map(qureg, &[1, 1], &[m]).unwrap_err();
    mix_multi_qubit_kraus_map(qureg, &[2, 2], &[m]).unwrap_err();

    mix_multi_qubit_kraus_map(qureg, &[-1, 0], &[m]).unwrap_err();
    mix_multi_qubit_kraus_map(qureg, &[0, 4], &[m]).unwrap_err();
}

#[test]
fn mix_multi_qubit_kraus_map_03() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<3>(env);
    qureg.init_zero_state();
    let m = &mut ComplexMatrixN::try_new(2).unwrap();
    init_complex_matrix_n(
        m,
        &[
            &[0., 0., 0., 1.],
            &[0., 1., 0., 0.],
            &[0., 0., 1., 0.],
            &[1., 0., 0., 0.],
        ],
        &[
            &[0., 0., 0., 0.],
            &[0., 0., 0., 0.],
            &[0., 0., 0., 0.],
            &[0., 0., 0., 0.],
        ],
    )
    .unwrap();

    mix_multi_qubit_kraus_map(qureg, &[1, 2], &[]).unwrap_err();
    mix_multi_qubit_kraus_map(qureg, &[0, 1], &[m, m]).unwrap_err();
    mix_multi_qubit_kraus_map(qureg, &[0, 2], &[m, m, m]).unwrap_err();
}

#[test]
fn calc_hilbert_schmidt_distance_01() {
    let env = &QuestEnv::new();
    let a = &mut create_density_qureg::<2>(env);
    a.init_classical_state(0).unwrap();
    let b = &mut create_density_qureg::<2>(env);
    b.init_classical_state(1).unwrap();

    let _ = calc_hilbert_schmidt_distance(a, b).unwrap();
}

// #[test]
// fn calc_hilbert_schmidt_distance_02() {
//     let env = &QuestEnv::new();
//     let a = &mut Qureg::try_new_density(1, env).unwrap();
//     init_classical_state(a, 0).unwrap();
//     let b = &mut create_density_qureg::<2>(env);
//     init_classical_state(b, 1).unwrap();

//     let _ = calc_hilbert_schmidt_distance(a, b).unwrap_err();
// }

#[test]
fn calc_hilbert_schmidt_distance_03() {
    let env = &QuestEnv::new();
    let a = &mut create_density_qureg::<2>(env);
    a.init_classical_state(0).unwrap();
    let b = &mut create_qureg::<2>(env);
    b.init_classical_state(1).unwrap();

    let _ = calc_hilbert_schmidt_distance(a, b).unwrap_err();
}

#[test]
fn mix_nontp_kraus_map_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<2>(env);
    qureg.init_zero_state();

    let m = &ComplexMatrix2::new([[0., 1.], [0., 0.]], [[0., 0.], [0., 0.]]);

    mix_nontp_kraus_map(qureg, 0, &[m]).unwrap();
    mix_nontp_kraus_map(qureg, 1, &[m]).unwrap();

    mix_nontp_kraus_map(qureg, -1, &[m]).unwrap_err();
    mix_nontp_kraus_map(qureg, 4, &[m]).unwrap_err();

    mix_nontp_kraus_map(qureg, 0, &[]).unwrap_err();
    // The maps must consists of not more then 4 Kraus operators
    mix_nontp_kraus_map(qureg, 0, &[m, m, m, m, m]).unwrap_err();
}

#[test]
fn mix_nontp_two_qubit_kraus_map_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<3>(env);
    qureg.init_zero_state();

    let m = &ComplexMatrix4::new(
        [
            [0., 0., 0., 1.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 0.],
        ],
        [
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ],
    );
    mix_nontp_two_qubit_kraus_map(qureg, 0, 1, &[m]).unwrap();
    mix_nontp_two_qubit_kraus_map(qureg, 1, 2, &[m]).unwrap();
    mix_nontp_two_qubit_kraus_map(qureg, 0, 2, &[m]).unwrap();

    mix_nontp_two_qubit_kraus_map(qureg, 0, 0, &[m]).unwrap_err();
    mix_nontp_two_qubit_kraus_map(qureg, 1, 1, &[m]).unwrap_err();
    mix_nontp_two_qubit_kraus_map(qureg, 2, 2, &[m]).unwrap_err();

    mix_nontp_two_qubit_kraus_map(qureg, -1, 0, &[m]).unwrap_err();
    mix_nontp_two_qubit_kraus_map(qureg, 0, 4, &[m]).unwrap_err();
}

#[test]
fn mix_nontp_multi_qubit_kraus_map_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<3>(env);
    qureg.init_zero_state();
    let m = &mut ComplexMatrixN::try_new(2).unwrap();
    init_complex_matrix_n(
        m,
        &[
            &[0., 0., 0., 1.],
            &[0., 1., 0., 0.],
            &[0., 0., 1., 0.],
            &[0., 0., 0., 0.],
        ],
        &[
            &[0., 0., 0., 0.],
            &[0., 0., 0., 0.],
            &[0., 0., 0., 0.],
            &[0., 0., 0., 0.],
        ],
    )
    .unwrap();

    mix_nontp_multi_qubit_kraus_map(qureg, &[1, 2], &[m]).unwrap();
    mix_nontp_multi_qubit_kraus_map(qureg, &[0, 1], &[m]).unwrap();
    mix_nontp_multi_qubit_kraus_map(qureg, &[0, 2], &[m]).unwrap();
    mix_nontp_multi_qubit_kraus_map(qureg, &[2, 0], &[m]).unwrap();

    mix_nontp_multi_qubit_kraus_map(qureg, &[0, 0], &[m]).unwrap_err();
    mix_nontp_multi_qubit_kraus_map(qureg, &[1, 1], &[m]).unwrap_err();
    mix_nontp_multi_qubit_kraus_map(qureg, &[2, 2], &[m]).unwrap_err();

    mix_nontp_multi_qubit_kraus_map(qureg, &[-1, 0], &[m]).unwrap_err();
    mix_nontp_multi_qubit_kraus_map(qureg, &[0, 4], &[m]).unwrap_err();
}

#[test]
fn mix_nontp_multi_qubit_kraus_map_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_density_qureg::<3>(env);
    qureg.init_zero_state();
    let m = &mut ComplexMatrixN::try_new(2).unwrap();
    init_complex_matrix_n(
        m,
        &[
            &[0., 0., 0., 1.],
            &[0., 1., 0., 0.],
            &[0., 0., 1., 0.],
            &[0., 0., 0., 0.],
        ],
        &[
            &[0., 0., 0., 0.],
            &[0., 0., 0., 0.],
            &[0., 0., 0., 0.],
            &[0., 0., 0., 0.],
        ],
    )
    .unwrap();

    mix_nontp_multi_qubit_kraus_map(qureg, &[1, 2], &[]).unwrap_err();
    // The maps must consists of not more then (2N)^2 Kraus operators
    mix_nontp_multi_qubit_kraus_map(
        qureg,
        &[0, 1],
        &[m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m],
    )
    .unwrap_err();
}

#[test]
fn apply_matrix4_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    let m = &ComplexMatrix4::new(
        [
            [0., 1., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ],
        [
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ],
    );
    apply_matrix4(qureg, 0, 1, m).unwrap();
    apply_matrix4(qureg, 1, 0, m).unwrap();

    apply_matrix4(qureg, 0, 0, m).unwrap_err();
    apply_matrix4(qureg, 1, 1, m).unwrap_err();

    apply_matrix4(qureg, -1, 1, m).unwrap_err();
    apply_matrix4(qureg, 3, 1, m).unwrap_err();
    apply_matrix4(qureg, 0, 3, m).unwrap_err();
    apply_matrix4(qureg, 0, -3, m).unwrap_err();

    apply_matrix4(qureg, 3, -3, m).unwrap_err();
}

#[test]
fn apply_matrix_n_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();

    let mtr = &mut ComplexMatrixN::try_new(3).unwrap();
    let empty = &[0., 0., 0., 0., 0., 0., 0., 0.];
    init_complex_matrix_n(
        mtr,
        &[
            &[0., 0., 0., 0., 0., 0., 0., 1.],
            &[0., 1., 0., 0., 0., 0., 0., 0.],
            &[0., 0., 1., 0., 0., 0., 0., 0.],
            &[0., 0., 0., 1., 0., 0., 0., 0.],
            &[0., 0., 0., 0., 1., 0., 0., 0.],
            &[0., 0., 0., 0., 0., 1., 0., 0.],
            &[0., 0., 0., 0., 0., 0., 1., 0.],
            &[1., 0., 0., 0., 0., 0., 0., 0.],
        ],
        &[empty, empty, empty, empty, empty, empty, empty, empty],
    )
    .unwrap();

    apply_matrix_n(qureg, &[0, 1, 2], mtr).unwrap();
    apply_matrix_n(qureg, &[1, 0, 2], mtr).unwrap();
    apply_matrix_n(qureg, &[2, 1, 0], mtr).unwrap();

    apply_matrix_n(qureg, &[0, 1, 0], mtr).unwrap_err();
    apply_matrix_n(qureg, &[0, 1, 1], mtr).unwrap_err();
    apply_matrix_n(qureg, &[2, 3, 1], mtr).unwrap_err();

    apply_matrix_n(qureg, &[-2, 0, 1], mtr).unwrap_err();
    apply_matrix_n(qureg, &[1, 0, 4], mtr).unwrap_err();

    apply_matrix_n(qureg, &[1, 0], mtr).unwrap_err();
    apply_matrix_n(qureg, &[1, 0, 2, 3], mtr).unwrap_err();
}

#[test]
fn apply_multi_controlled_matrix_n_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::<'_, 4>::try_new(env).unwrap();
    qureg.init_zero_state();

    let u = &mut ComplexMatrixN::try_new(2).unwrap();
    let zero_row = &[0., 0., 0., 0.];
    init_complex_matrix_n(
        u,
        &[
            &[0., 0., 0., 1.],
            &[0., 1., 0., 0.],
            &[0., 0., 1., 0.],
            &[1., 0., 0., 0.],
        ],
        &[zero_row, zero_row, zero_row, zero_row],
    )
    .unwrap();

    apply_multi_controlled_matrix_n(qureg, &[0, 1], &[2, 3], u).unwrap();
    apply_multi_controlled_matrix_n(qureg, &[0, 1], &[3, 2], u).unwrap();
    apply_multi_controlled_matrix_n(qureg, &[1, 0], &[2, 3], u).unwrap();
    apply_multi_controlled_matrix_n(qureg, &[0, 2], &[1, 3], u).unwrap();

    apply_multi_controlled_matrix_n(qureg, &[0, 0], &[2, 3], u).unwrap_err();
    apply_multi_controlled_matrix_n(qureg, &[0, 1], &[2, 2], u).unwrap_err();

    apply_multi_controlled_matrix_n(qureg, &[-1, 0], &[2, 3], u).unwrap_err();
    apply_multi_controlled_matrix_n(qureg, &[0, 4], &[2, 3], u).unwrap_err();
    apply_multi_controlled_matrix_n(qureg, &[0, 1], &[1, 4], u).unwrap_err();
    apply_multi_controlled_matrix_n(qureg, &[99, 99], &[99, 99], u)
        .unwrap_err();
}

#[test]
fn apply_qft_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();

    apply_qft(qureg, &[0, 1]).unwrap();
    apply_qft(qureg, &[1, 0]).unwrap();
    apply_qft(qureg, &[1, 2]).unwrap();
    apply_qft(qureg, &[0, 2]).unwrap();
}

#[test]
fn apply_qft_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();

    apply_qft(qureg, &[0, 0]).unwrap_err();
    apply_qft(qureg, &[1, 1]).unwrap_err();
    apply_qft(qureg, &[-1, 0]).unwrap_err();
    apply_qft(qureg, &[4, 0]).unwrap_err();
}

#[test]
fn apply_projector_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();
    apply_projector(qureg, 0, 0).unwrap();
    qureg.init_zero_state();
    apply_projector(qureg, 1, 0).unwrap();
    qureg.init_zero_state();
    apply_projector(qureg, 0, 1).unwrap();
    qureg.init_zero_state();
    apply_projector(qureg, 1, 1).unwrap();
}

#[test]
fn apply_projector_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();
    apply_projector(qureg, 0, -1).unwrap_err();
    apply_projector(qureg, 0, 3).unwrap_err();
    apply_projector(qureg, 2, 0).unwrap_err();
    apply_projector(qureg, -1, 0).unwrap_err();
}

#[test]
fn multi_rotate_pauli_01() {
    use PauliOpType::PAULI_X;

    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();

    multi_rotate_pauli(qureg, &[0, 1], &[PAULI_X, PAULI_X], 0.).unwrap();
    multi_rotate_pauli(qureg, &[1, 2], &[PAULI_X, PAULI_X], 0.).unwrap();
    multi_rotate_pauli(qureg, &[2, 0], &[PAULI_X, PAULI_X], 0.).unwrap();

    multi_rotate_pauli(qureg, &[0, 0], &[PAULI_X, PAULI_X], 0.).unwrap_err();
    multi_rotate_pauli(qureg, &[1, 1], &[PAULI_X, PAULI_X], 0.).unwrap_err();
    multi_rotate_pauli(qureg, &[2, 2], &[PAULI_X, PAULI_X], 0.).unwrap_err();

    multi_rotate_pauli(qureg, &[0, 3], &[PAULI_X, PAULI_X], 0.).unwrap_err();
    multi_rotate_pauli(qureg, &[-1, 0], &[PAULI_X, PAULI_X], 0.).unwrap_err();
    multi_rotate_pauli(qureg, &[0, 1, 2], &[PAULI_X, PAULI_X], 0.).unwrap_err();
    multi_rotate_pauli(qureg, &[0, 1, 2, 3], &[PAULI_X, PAULI_X], 0.)
        .unwrap_err();
    multi_rotate_pauli(qureg, &[0, 1], &[PAULI_X], 0.).unwrap_err();
}

#[test]
fn apply_phase_func_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    apply_phase_func(
        qureg,
        &[0, 1],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[0., 2.],
    )
    .unwrap();
    apply_phase_func(
        qureg,
        &[1, 0],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[0., 2.],
    )
    .unwrap();

    apply_phase_func(
        qureg,
        &[0, 0],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[0., 2.],
    )
    .unwrap_err();
    apply_phase_func(
        qureg,
        &[1, 1],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[0., 2.],
    )
    .unwrap_err();

    apply_phase_func(
        qureg,
        &[0, 2],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[0., 2.],
    )
    .unwrap_err();
    apply_phase_func(
        qureg,
        &[-1, 0],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[0., 2.],
    )
    .unwrap_err();
}

#[test]
fn apply_phase_func_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    // exponents contains a fractional number despite \p encoding <b>=</b>
    // ::TWOS_COMPLEMENT
    apply_phase_func(
        qureg,
        &[0, 1],
        BitEncoding::TWOS_COMPLEMENT,
        &[0.5, 0.5],
        &[0., 0.5],
    )
    .unwrap_err();
}

#[test]
fn apply_phase_func_overrides_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();
    pauli_x(qureg, 1).unwrap();

    apply_phase_func_overrides(
        qureg,
        &[0, 1],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[-2., 2.],
        &[0],
        &[0.],
    )
    .unwrap();

    apply_phase_func_overrides(
        qureg,
        &[0, 0],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[-2., 2.],
        &[0],
        &[0.],
    )
    .unwrap_err();

    apply_phase_func_overrides(
        qureg,
        &[-1, 0],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[-2., 2.],
        &[0],
        &[0.],
    )
    .unwrap_err();

    apply_phase_func_overrides(
        qureg,
        &[0, 4],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[-2., 2.],
        &[0],
        &[0.],
    )
    .unwrap_err();

    apply_phase_func_overrides(
        qureg,
        &[0, 1],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[-2., 2.],
        &[1],
        &[0.],
    )
    .unwrap_err();

    apply_phase_func_overrides(
        qureg,
        &[0, 9],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[1., 2.],
        &[0],
        &[0.],
    )
    .unwrap_err();
}

#[test]
fn apply_multi_var_phase_func_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();
    pauli_x(qureg, 1).unwrap();

    apply_multi_var_phase_func(
        qureg,
        &[0, 1],
        &[1, 1],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[2., 2.],
        &[1, 1],
    )
    .unwrap();

    apply_multi_var_phase_func(
        qureg,
        &[0, 0],
        &[1, 1],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[2., 2.],
        &[1, 1],
    )
    .unwrap_err();

    apply_multi_var_phase_func(
        qureg,
        &[-1, 1],
        &[1, 1],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[2., 2.],
        &[1, 1],
    )
    .unwrap_err();

    apply_multi_var_phase_func(
        qureg,
        &[0, 4],
        &[1, 1],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[2., 2.],
        &[1, 1],
    )
    .unwrap_err();

    apply_multi_var_phase_func(
        qureg,
        &[0, 1],
        &[1, 3],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[2., 2.],
        &[1, 1],
    )
    .unwrap_err();

    apply_multi_var_phase_func(
        qureg,
        &[0, 1],
        &[1, 1],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[-2., 2.],
        &[1, 1],
    )
    .unwrap_err();

    apply_multi_var_phase_func(
        qureg,
        &[0, 1],
        &[1, 1],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[2., 2.],
        &[1, 0],
    )
    .unwrap_err();
}

#[test]
fn apply_multi_var_phase_func_overrides_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();
    pauli_x(qureg, 1).unwrap();

    apply_multi_var_phase_func_overrides(
        qureg,
        &[0, 1],
        &[1, 1],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[2., 2.],
        &[1, 1],
        &[0, 1, 0, 1],
        &[0., 0.],
    )
    .unwrap();

    apply_multi_var_phase_func_overrides(
        qureg,
        &[0, 0],
        &[1, 1],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[2., 2.],
        &[1, 1],
        &[1, 0, 0, 1],
        &[0.],
    )
    .unwrap_err();

    apply_multi_var_phase_func_overrides(
        qureg,
        &[-1, 0],
        &[1, 1],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[2., 2.],
        &[1, 1],
        &[1, 0, 0, 1],
        &[0.],
    )
    .unwrap_err();

    apply_multi_var_phase_func_overrides(
        qureg,
        &[0, 4],
        &[1, 1],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[2., 2.],
        &[1, 1],
        &[1, 0, 0, 1],
        &[0.],
    )
    .unwrap_err();

    apply_multi_var_phase_func_overrides(
        qureg,
        &[0, 4],
        &[1, 1],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[2., 2.],
        &[1, 1],
        &[1, 0, 0, 1],
        &[0.],
    )
    .unwrap_err();

    apply_multi_var_phase_func_overrides(
        qureg,
        &[0, 1],
        &[1, 1],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[2., 2.],
        &[0, 1],
        &[1, 0, 0, 1],
        &[0.],
    )
    .unwrap_err();

    apply_multi_var_phase_func_overrides(
        qureg,
        &[0, 1],
        &[1, 1],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[-2., 2.],
        &[1, 1],
        &[1, 0, 1, 0],
        &[0.],
    )
    .unwrap_err();

    apply_multi_var_phase_func_overrides(
        qureg,
        &[0, 1],
        &[1, 1],
        BitEncoding::UNSIGNED,
        &[0.5, 0.5],
        &[2., 2.],
        &[1, 1],
        &[0, 9, 0, 1],
        &[0.],
    )
    .unwrap_err();
}

#[test]
fn appply_named_phase_func_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    apply_named_phase_func(
        qureg,
        &[0, 1],
        &[1, 1],
        BitEncoding::UNSIGNED,
        PhaseFunc::DISTANCE,
    )
    .unwrap();

    apply_named_phase_func(
        qureg,
        &[0, 0],
        &[1, 1],
        BitEncoding::UNSIGNED,
        PhaseFunc::DISTANCE,
    )
    .unwrap_err();

    apply_named_phase_func(
        qureg,
        &[-1, 0],
        &[1, 1],
        BitEncoding::UNSIGNED,
        PhaseFunc::DISTANCE,
    )
    .unwrap_err();

    apply_named_phase_func(
        qureg,
        &[0, 4],
        &[1, 1],
        BitEncoding::UNSIGNED,
        PhaseFunc::DISTANCE,
    )
    .unwrap_err();

    apply_named_phase_func(
        qureg,
        &[0, 1],
        &[1, 3],
        BitEncoding::UNSIGNED,
        PhaseFunc::DISTANCE,
    )
    .unwrap_err();
}

#[test]
fn apply_named_phase_func_overrides_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();

    apply_named_phase_func_overrides(
        qureg,
        &[0, 1],
        &[1, 1],
        BitEncoding::UNSIGNED,
        PhaseFunc::DISTANCE,
        &[0, 1, 0, 1],
        &[0., 0.],
    )
    .unwrap();

    apply_named_phase_func_overrides(
        qureg,
        &[0, 0],
        &[1, 1],
        BitEncoding::UNSIGNED,
        PhaseFunc::DISTANCE,
        &[0, 1, 0, 1],
        &[0., 0.],
    )
    .unwrap_err();

    apply_named_phase_func_overrides(
        qureg,
        &[-1, 0],
        &[1, 1],
        BitEncoding::UNSIGNED,
        PhaseFunc::DISTANCE,
        &[0, 1, 0, 1],
        &[0., 0.],
    )
    .unwrap_err();

    apply_named_phase_func_overrides(
        qureg,
        &[0, 4],
        &[1, 1],
        BitEncoding::UNSIGNED,
        PhaseFunc::DISTANCE,
        &[0, 1, 0, 1],
        &[0., 0.],
    )
    .unwrap_err();

    apply_named_phase_func_overrides(
        qureg,
        &[0, 1],
        &[1, 9],
        BitEncoding::UNSIGNED,
        PhaseFunc::DISTANCE,
        &[0, 1, 0, 1],
        &[0., 0.],
    )
    .unwrap_err();

    apply_named_phase_func_overrides(
        qureg,
        &[0, 1],
        &[1, 1],
        BitEncoding::UNSIGNED,
        PhaseFunc::DISTANCE,
        &[0, 9, 0, 1],
        &[0., 0.],
    )
    .unwrap_err();
}

#[test]
fn apply_param_named_phase_func_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();

    apply_param_named_phase_func(
        qureg,
        &[0, 1],
        &[1, 1],
        BitEncoding::UNSIGNED,
        PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
        &[0., 0., 0., 0.],
    )
    .unwrap();

    apply_param_named_phase_func(
        qureg,
        &[0, 0],
        &[1, 1],
        BitEncoding::UNSIGNED,
        PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
        &[0., 0., 0., 0.],
    )
    .unwrap_err();

    apply_param_named_phase_func(
        qureg,
        &[-1, 0],
        &[1, 1],
        BitEncoding::UNSIGNED,
        PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
        &[0., 0., 0., 0.],
    )
    .unwrap_err();

    apply_param_named_phase_func(
        qureg,
        &[0, 4],
        &[1, 1],
        BitEncoding::UNSIGNED,
        PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
        &[0., 0., 0., 0.],
    )
    .unwrap_err();

    apply_param_named_phase_func(
        qureg,
        &[0, 4],
        &[1, 1],
        BitEncoding::UNSIGNED,
        PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
        // wrong number of parameters
        &[0., 0.],
    )
    .unwrap_err();

    apply_param_named_phase_func(
        qureg,
        &[0, 4],
        &[1, 9],
        BitEncoding::UNSIGNED,
        PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
        &[0., 0., 0., 0.],
    )
    .unwrap_err();
}

#[test]
fn apply_param_named_phase_func_overrides_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();

    apply_param_named_phase_func_overrides(
        qureg,
        &[0, 1],
        &[1, 1],
        BitEncoding::UNSIGNED,
        PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
        &[0., 0., 0., 0.],
        &[0, 1, 0, 1],
        &[0., 0.],
    )
    .unwrap();

    apply_param_named_phase_func_overrides(
        qureg,
        &[0, 0],
        &[1, 1],
        BitEncoding::UNSIGNED,
        PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
        &[0., 0., 0., 0.],
        &[0, 1, 0, 1],
        &[0., 0.],
    )
    .unwrap_err();

    apply_param_named_phase_func_overrides(
        qureg,
        &[-1, 0],
        &[1, 1],
        BitEncoding::UNSIGNED,
        PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
        &[0., 0., 0., 0.],
        &[0, 1, 0, 1],
        &[0., 0.],
    )
    .unwrap_err();

    apply_param_named_phase_func_overrides(
        qureg,
        &[0, 4],
        &[1, 1],
        BitEncoding::UNSIGNED,
        PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
        &[0., 0., 0., 0.],
        &[0, 1, 0, 1],
        &[0., 0.],
    )
    .unwrap_err();

    apply_param_named_phase_func_overrides(
        qureg,
        &[0, 0],
        &[1, 9],
        BitEncoding::UNSIGNED,
        PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
        &[0., 0., 0., 0.],
        &[0, 1, 0, 1],
        &[0., 0.],
    )
    .unwrap_err();

    apply_param_named_phase_func_overrides(
        qureg,
        &[0, 0],
        &[1, 1],
        BitEncoding::UNSIGNED,
        PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
        // wrong number of parameters
        &[0., 0.],
        &[0, 1, 0, 1],
        &[0., 0.],
    )
    .unwrap_err();

    apply_param_named_phase_func_overrides(
        qureg,
        &[0, 0],
        &[1, 1],
        BitEncoding::UNSIGNED,
        PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
        &[0., 0., 0., 0.],
        &[0, 9, 0, 1],
        &[0., 0.],
    )
    .unwrap_err();
}

#[test]
fn calc_expec_pauli_prod_01() {
    use PauliOpType::PAULI_X;
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();
    let workspace = &mut create_qureg::<2>(env);

    calc_expec_pauli_prod(qureg, &[0, 1], &[PAULI_X, PAULI_X], workspace)
        .unwrap();
    calc_expec_pauli_prod(qureg, &[1, 0], &[PAULI_X, PAULI_X], workspace)
        .unwrap();

    calc_expec_pauli_prod(qureg, &[0, 1], &[PAULI_X], workspace).unwrap_err();
    calc_expec_pauli_prod(qureg, &[0, 0], &[PAULI_X, PAULI_X], workspace)
        .unwrap_err();

    calc_expec_pauli_prod(qureg, &[-1, 0], &[PAULI_X, PAULI_X], workspace)
        .unwrap_err();
    calc_expec_pauli_prod(qureg, &[4, 0], &[PAULI_X, PAULI_X], workspace)
        .unwrap_err();
    calc_expec_pauli_prod(qureg, &[0, -1], &[PAULI_X, PAULI_X], workspace)
        .unwrap_err();
    calc_expec_pauli_prod(qureg, &[0, 4], &[PAULI_X, PAULI_X], workspace)
        .unwrap_err();
}

// #[test]
// fn calc_expec_pauli_prod_02() {
//     use PauliOpType::PAULI_X;
//     let env = &QuestEnv::new();
//     let qureg = &mut create_qureg::<2>(env);
//     qureg.init_zero_state();
//     let workspace = &mut create_qureg::<3>(env);

//     calc_expec_pauli_prod(qureg, &[0, 1], &[PAULI_X, PAULI_X], workspace)
//         .unwrap_err();
// }

#[test]
fn calc_expec_pauli_sum_01() {
    use PauliOpType::{
        PAULI_X,
        PAULI_Z,
    };
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();
    let workspace = &mut create_qureg::<2>(env);

    let all_pauli_codes = &[PAULI_X, PAULI_Z, PAULI_Z, PAULI_X];
    let term_coeffs = &[0.5, 0.5];

    calc_expec_pauli_sum(qureg, all_pauli_codes, term_coeffs, workspace)
        .unwrap();
}

// #[test]
// fn calc_expec_pauli_sum_02() {
//     use PauliOpType::{
//         PAULI_X,
//         PAULI_Z,
//     };
//     let env = &QuestEnv::new();
//     let qureg = &mut create_qureg::<2>(env);
//     qureg.init_zero_state();
//     let workspace = &mut create_qureg::<3>(env);

//     let all_pauli_codes = &[PAULI_X, PAULI_Z, PAULI_Z, PAULI_X];
//     let term_coeffs = &[0.5, 0.5];

//     calc_expec_pauli_sum(qureg, all_pauli_codes, term_coeffs, workspace)
//         .unwrap_err();
// }

#[test]
fn calc_expec_pauli_hamil_01() {
    use PauliOpType::{
        PAULI_X,
        PAULI_Z,
    };
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);
    qureg.init_zero_state();
    let workspace = &mut create_qureg::<2>(env);

    let hamil = &mut PauliHamil::try_new(2, 2).unwrap();
    init_pauli_hamil(hamil, &[0.5, 0.5], &[PAULI_X, PAULI_X, PAULI_X, PAULI_Z])
        .unwrap();

    calc_expec_pauli_hamil(qureg, hamil, workspace).unwrap();
}

// #[test]
// fn calc_expec_pauli_hamil_02() {
//     use PauliOpType::{
//         PAULI_X,
//         PAULI_Z,
//     };
//     let env = &QuestEnv::new();
//     let qureg = &mut create_qureg::<2>(env);
//     qureg.init_zero_state();
//     let workspace = &mut create_qureg::<3>(env);

//     let hamil = &mut PauliHamil::try_new(2, 2).unwrap();
//     init_pauli_hamil(hamil, &[0.5, 0.5], &[PAULI_X, PAULI_X, PAULI_X,
// PAULI_Z])         .unwrap();

//     calc_expec_pauli_hamil(qureg, hamil, workspace).unwrap_err();
// }

#[test]
fn two_qubit_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();
    pauli_x(qureg, 0).unwrap();

    let u = &ComplexMatrix4::new(
        [
            [0., 0., 0., 1.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
        ],
        [
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ],
    );

    two_qubit_unitary(qureg, 0, 1, u).unwrap();
    two_qubit_unitary(qureg, 1, 2, u).unwrap();

    two_qubit_unitary(qureg, 0, 0, u).unwrap_err();
    two_qubit_unitary(qureg, -1, 0, u).unwrap_err();
    two_qubit_unitary(qureg, 0, 4, u).unwrap_err();
}

#[test]
fn two_qubit_unitary_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();
    pauli_x(qureg, 0).unwrap();

    // This matrix is not unitary
    let u = &ComplexMatrix4::new(
        [
            [11., 0., 0., 1.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
        ],
        [
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ],
    );

    two_qubit_unitary(qureg, 0, 1, u).unwrap_err();
    two_qubit_unitary(qureg, 1, 2, u).unwrap_err();

    two_qubit_unitary(qureg, 0, 0, u).unwrap_err();
    two_qubit_unitary(qureg, -1, 0, u).unwrap_err();
    two_qubit_unitary(qureg, 0, 4, u).unwrap_err();
}

#[test]
fn controlled_two_qubit_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();
    pauli_x(qureg, 0).unwrap();

    let u = &ComplexMatrix4::new(
        [
            [0., 0., 0., 1.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
        ],
        [
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ],
    );

    controlled_two_qubit_unitary(qureg, 0, 1, 2, u).unwrap();
    controlled_two_qubit_unitary(qureg, 1, 2, 0, u).unwrap();
    controlled_two_qubit_unitary(qureg, 2, 0, 1, u).unwrap();

    controlled_two_qubit_unitary(qureg, 0, 1, 1, u).unwrap_err();
    controlled_two_qubit_unitary(qureg, 0, 0, 1, u).unwrap_err();

    controlled_two_qubit_unitary(qureg, -1, 0, 1, u).unwrap_err();
    controlled_two_qubit_unitary(qureg, 4, 0, 1, u).unwrap_err();

    controlled_two_qubit_unitary(qureg, 0, -1, 0, u).unwrap_err();
    controlled_two_qubit_unitary(qureg, 0, 0, 4, u).unwrap_err();
}

#[test]
fn controlled_two_qubit_unitary_02() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<3>(env);
    qureg.init_zero_state();
    pauli_x(qureg, 0).unwrap();

    // This matrix is not unitary
    let u = &ComplexMatrix4::new(
        [
            [11., 0., 0., 1.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
        ],
        [
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ],
    );

    controlled_two_qubit_unitary(qureg, 0, 1, 2, u).unwrap_err();
    controlled_two_qubit_unitary(qureg, 1, 2, 0, u).unwrap_err();
    controlled_two_qubit_unitary(qureg, 2, 0, 1, u).unwrap_err();

    controlled_two_qubit_unitary(qureg, 0, 1, 1, u).unwrap_err();
    controlled_two_qubit_unitary(qureg, 0, 0, 1, u).unwrap_err();

    controlled_two_qubit_unitary(qureg, -1, 0, 1, u).unwrap_err();
    controlled_two_qubit_unitary(qureg, 4, 0, 1, u).unwrap_err();

    controlled_two_qubit_unitary(qureg, 0, -1, 0, u).unwrap_err();
    controlled_two_qubit_unitary(qureg, 0, 0, 4, u).unwrap_err();
}

#[test]
fn multi_controlled_two_qubit_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::<'_, 4>::try_new(env).unwrap();
    qureg.init_zero_state();
    pauli_x(qureg, 0).unwrap();
    pauli_x(qureg, 1).unwrap();

    let u = &ComplexMatrix4::new(
        [
            [0., 0., 0., 1.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
        ],
        [
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ],
    );

    multi_controlled_two_qubit_unitary(qureg, &[0, 1], 2, 3, u).unwrap();
    multi_controlled_two_qubit_unitary(qureg, &[1, 0], 3, 2, u).unwrap();
    multi_controlled_two_qubit_unitary(qureg, &[1, 2], 0, 3, u).unwrap();
    multi_controlled_two_qubit_unitary(qureg, &[3, 0], 2, 1, u).unwrap();

    multi_controlled_two_qubit_unitary(qureg, &[0, 0], 1, 2, u).unwrap_err();
    multi_controlled_two_qubit_unitary(qureg, &[0, 1], 1, 1, u).unwrap_err();

    multi_controlled_two_qubit_unitary(qureg, &[-1, 1], 2, 3, u).unwrap_err();
    multi_controlled_two_qubit_unitary(qureg, &[5, 1], 2, 3, u).unwrap_err();

    multi_controlled_two_qubit_unitary(qureg, &[0, 1], -1, 3, u).unwrap_err();
    multi_controlled_two_qubit_unitary(qureg, &[0, 1], 5, 3, u).unwrap_err();
}

#[test]
fn multi_controlled_two_qubit_unitary_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::<'_, 4>::try_new(env).unwrap();
    qureg.init_zero_state();
    pauli_x(qureg, 0).unwrap();
    pauli_x(qureg, 1).unwrap();

    // This matrix is not unitary
    let u = &ComplexMatrix4::new(
        [
            [11., 0., 0., 1.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [1., 0., 0., 0.],
        ],
        [
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ],
    );

    multi_controlled_two_qubit_unitary(qureg, &[0, 1], 2, 3, u).unwrap_err();
    multi_controlled_two_qubit_unitary(qureg, &[1, 0], 3, 2, u).unwrap_err();
    multi_controlled_two_qubit_unitary(qureg, &[1, 2], 0, 3, u).unwrap_err();
    multi_controlled_two_qubit_unitary(qureg, &[3, 0], 2, 1, u).unwrap_err();

    multi_controlled_two_qubit_unitary(qureg, &[0, 0], 1, 2, u).unwrap_err();
    multi_controlled_two_qubit_unitary(qureg, &[0, 1], 1, 1, u).unwrap_err();

    multi_controlled_two_qubit_unitary(qureg, &[-1, 1], 2, 3, u).unwrap_err();
    multi_controlled_two_qubit_unitary(qureg, &[5, 1], 2, 3, u).unwrap_err();

    multi_controlled_two_qubit_unitary(qureg, &[0, 1], -1, 3, u).unwrap_err();
    multi_controlled_two_qubit_unitary(qureg, &[0, 1], 5, 3, u).unwrap_err();
}

#[test]
fn multi_controlled_multi_qubit_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::<'_, 4>::try_new(env).unwrap();
    qureg.init_zero_state();
    pauli_x(qureg, 0).unwrap();
    pauli_x(qureg, 1).unwrap();

    let u = &mut ComplexMatrixN::try_new(2).unwrap();
    let zero_row = &[0., 0., 0., 0.];
    init_complex_matrix_n(
        u,
        &[
            &[0., 0., 0., 1.],
            &[0., 1., 0., 0.],
            &[0., 0., 1., 0.],
            &[1., 0., 0., 0.],
        ],
        &[zero_row, zero_row, zero_row, zero_row],
    )
    .unwrap();

    let ctrls = &[0, 1];
    let targs = &[2, 3];
    multi_controlled_multi_qubit_unitary(qureg, ctrls, targs, u).unwrap();

    // Check if the register is now in the state `|1111>`
    let amp = get_real_amp(qureg, 15).unwrap();
    assert!((amp - 1.).abs() < EPSILON);
}

#[test]
fn apply_pauli_sum_01() {
    use PauliOpType::{
        PAULI_I,
        PAULI_X,
    };
    let env = &QuestEnv::new();
    let in_qureg = &mut create_qureg::<2>(env);
    in_qureg.init_zero_state();
    let out_qureg = &mut create_qureg::<2>(env);
    let all_pauli_codes = &[PAULI_I, PAULI_X, PAULI_X, PAULI_I];
    let term_coeffs = &[SQRT_2.recip(), SQRT_2.recip()];

    apply_pauli_sum(in_qureg, all_pauli_codes, term_coeffs, out_qureg).unwrap();
}

// #[test]
// fn apply_pauli_sum_02() {
//     use PauliOpType::{
//         PAULI_I,
//         PAULI_X,
//     };
//     let env = &QuestEnv::new();
//     let in_qureg = &mut create_qureg::<2>(env);
//     in_qureg.init_zero_state();
//     let out_qureg = &mut create_qureg::<3>(env);
//     let all_pauli_codes = &[PAULI_I, PAULI_X, PAULI_X, PAULI_I];
//     let term_coeffs = &[SQRT_2.recip(), SQRT_2.recip()];

//     // in_ and out_qureg' dimensions are different
//     apply_pauli_sum(in_qureg, all_pauli_codes, term_coeffs, out_qureg)
//         .unwrap_err();
// }

#[test]
fn apply_pauli_sum_03() {
    use PauliOpType::{
        PAULI_I,
        PAULI_X,
    };
    let env = &QuestEnv::new();
    let in_qureg = &mut create_qureg::<2>(env);
    in_qureg.init_zero_state();
    let out_qureg = &mut create_qureg::<2>(env);

    // wrong number of codes
    let all_pauli_codes = &[PAULI_I, PAULI_X, PAULI_X];
    let term_coeffs = &[SQRT_2.recip(), SQRT_2.recip()];

    apply_pauli_sum(in_qureg, all_pauli_codes, term_coeffs, out_qureg).unwrap();
}

#[test]
fn apply_pauli_hamil_01() {
    use PauliOpType::{
        PAULI_I,
        PAULI_X,
    };
    let env = &QuestEnv::new();
    let in_qureg = &mut create_qureg::<2>(env);
    in_qureg.init_zero_state();
    let out_qureg = &mut create_qureg::<2>(env);

    let hamil = &mut PauliHamil::try_new(2, 2).unwrap();
    let coeffs = &[SQRT_2.recip(), SQRT_2.recip()];
    let codes = &[PAULI_I, PAULI_X, PAULI_X, PAULI_I];
    init_pauli_hamil(hamil, coeffs, codes).unwrap();

    apply_pauli_hamil(in_qureg, hamil, out_qureg).unwrap();
}

// #[test]
// fn apply_pauli_hamil_02() {
//     use PauliOpType::{
//         PAULI_I,
//         PAULI_X,
//     };
//     let env = &QuestEnv::new();
//     let in_qureg = &mut create_qureg::<2>(env);
//     in_qureg.init_zero_state();
//     // out_qureg is of different dimension
//     let out_qureg = &mut create_qureg::<3>(env);

//     let hamil = &mut PauliHamil::try_new(2, 2).unwrap();
//     let coeffs = &[SQRT_2.recip(), SQRT_2.recip()];
//     let codes = &[PAULI_I, PAULI_X, PAULI_X, PAULI_I];
//     init_pauli_hamil(hamil, coeffs, codes).unwrap();

//     apply_pauli_hamil(in_qureg, hamil, out_qureg).unwrap_err();
// }

#[test]
fn apply_trotter_circuit_01() {
    use PauliOpType::PAULI_X;

    let env = &QuestEnv::new();
    let qureg = &mut Qureg::<'_, 1>::try_new(env).unwrap();
    qureg.init_zero_state();

    let hamil = &mut PauliHamil::try_new(1, 1).unwrap();
    let coeffs = &[1.];
    let codes = &[PAULI_X];
    init_pauli_hamil(hamil, coeffs, codes).unwrap();

    apply_trotter_circuit(qureg, hamil, 0., 1, 1).unwrap();
    apply_trotter_circuit(qureg, hamil, 0., 2, 1).unwrap();
    apply_trotter_circuit(qureg, hamil, 0., 1, 2).unwrap();

    apply_trotter_circuit(qureg, hamil, 0., -1, 1).unwrap_err();
    apply_trotter_circuit(qureg, hamil, 0., 1, 0).unwrap_err();
    apply_trotter_circuit(qureg, hamil, 0., 1, -1).unwrap_err();
}

#[test]
fn set_weighted_qureg_01() {
    let env = &QuestEnv::new();
    let qureg1 = &mut Qureg::<'_, 1>::try_new(env).unwrap();
    qureg1.init_zero_state();
    let qureg2 = &mut Qureg::<'_, 1>::try_new(env).unwrap();
    qureg2.init_zero_state();
    pauli_x(qureg2, 0).unwrap();

    let out = &mut Qureg::<'_, 1>::try_new(env).unwrap();
    out.init_zero_state();

    let fac1 = Qcomplex::new(SQRT_2.recip(), 0.);
    let fac2 = Qcomplex::new(SQRT_2.recip(), 0.);
    let fac_out = Qcomplex::zero();

    set_weighted_qureg(fac1, qureg1, fac2, qureg2, fac_out, out).unwrap();
}

// #[test]
// fn set_weighted_qureg_02() {
//     let env = &QuestEnv::new();
//     let qureg1 = &mut Qureg::<'_, 1>::try_new(env).unwrap();
//     init_zero_state(qureg1);
//     // dimensions are not equal
//     let qureg2 = &mut create_qureg::<2>(env);
//     init_zero_state(qureg2);
//     pauli_x(qureg2, 0).unwrap();

//     let out = &mut Qureg::<'_, 1>::try_new(env).unwrap();
//     init_zero_state(out);

//     let fac1 = Qcomplex::new(SQRT_2.recip(), 0.);
//     let fac2 = Qcomplex::new(SQRT_2.recip(), 0.);
//     let fac_out = Qcomplex::zero();

//     set_weighted_qureg(fac1, qureg1, fac2, qureg2, fac_out,
// out).unwrap_err(); }

// #[test]
// fn set_weighted_qureg_03() {
//     let env = &QuestEnv::new();
//     let qureg1 = &mut Qureg::<'_, 1>::try_new(env).unwrap();
//     init_zero_state(qureg1);
//     let qureg2 = &mut Qureg::<'_, 1>::try_new(env).unwrap();
//     init_zero_state(qureg2);
//     pauli_x(qureg2, 0).unwrap();

//     // dimensions are not equal
//     let out = &mut create_qureg::<2>(env);
//     init_zero_state(out);

//     let fac1 = Qcomplex::new(SQRT_2.recip(), 0.);
//     let fac2 = Qcomplex::new(SQRT_2.recip(), 0.);
//     let fac_out = Qcomplex::zero();

//     set_weighted_qureg(fac1, qureg1, fac2, qureg2, fac_out,
// out).unwrap_err(); }

// #[test]
// fn set_weighted_qureg_04() {
//     let env = &QuestEnv::new();
//     let qureg1 = &mut Qureg::<'_, 1>::try_new(env).unwrap();
//     init_zero_state(qureg1);
//     // all quregs should either state vectors of density matrices
//     let qureg2 = &mut Qureg::try_new_density(1, env).unwrap();
//     init_zero_state(qureg2);
//     pauli_x(qureg2, 0).unwrap();

//     let out = &mut create_qureg::<2>(env);
//     init_zero_state(out);

//     let fac1 = Qcomplex::new(SQRT_2.recip(), 0.);
//     let fac2 = Qcomplex::new(SQRT_2.recip(), 0.);
//     let fac_out = Qcomplex::zero();

//     set_weighted_qureg(fac1, qureg1, fac2, qureg2, fac_out,
// out).unwrap_err(); }

#[test]
fn multi_controlled_multi_rotate_z_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::<'_, 4>::try_new(env).unwrap();

    // Initialize `|1111>`
    qureg.init_zero_state();
    (0..4).try_for_each(|i| pauli_x(qureg, i)).unwrap();

    multi_controlled_multi_rotate_z(qureg, &[0, 1], &[2, 3], 0.).unwrap();
    multi_controlled_multi_rotate_z(qureg, &[0, 2], &[1, 3], 0.).unwrap();
    multi_controlled_multi_rotate_z(qureg, &[2, 1], &[3, 0], 0.).unwrap();

    multi_controlled_multi_rotate_z(qureg, &[0, 0], &[2, 3], 0.).unwrap_err();
    multi_controlled_multi_rotate_z(qureg, &[0, 1], &[2, 2], 0.).unwrap_err();

    multi_controlled_multi_rotate_z(qureg, &[-1, 1], &[2, 3], 0.).unwrap_err();
    multi_controlled_multi_rotate_z(qureg, &[0, 4], &[2, 3], 0.).unwrap_err();
    multi_controlled_multi_rotate_z(qureg, &[0, 1], &[-1, 3], 0.).unwrap_err();
    multi_controlled_multi_rotate_z(qureg, &[0, 1], &[2, 4], 0.).unwrap_err();

    multi_controlled_multi_rotate_z(qureg, &[0, 1], &[0, 3], 0.).unwrap_err();
    multi_controlled_multi_rotate_z(qureg, &[0, 3], &[2, 3], 0.).unwrap_err();
}

#[test]
fn multi_controlled_multi_rotate_pauli_01() {
    use PauliOpType::PAULI_Z;

    let env = &QuestEnv::new();
    let qureg = &mut Qureg::<'_, 4>::try_new(env).unwrap();

    // Initialize `|1111>`
    qureg.init_zero_state();
    (0..4).try_for_each(|i| pauli_x(qureg, i)).unwrap();

    let tar_paul = &[PAULI_Z, PAULI_Z];

    multi_controlled_multi_rotate_pauli(qureg, &[0, 1], &[2, 3], tar_paul, 0.)
        .unwrap();
    multi_controlled_multi_rotate_pauli(qureg, &[0, 2], &[1, 3], tar_paul, 0.)
        .unwrap();
    multi_controlled_multi_rotate_pauli(qureg, &[2, 1], &[3, 0], tar_paul, 0.)
        .unwrap();

    multi_controlled_multi_rotate_pauli(qureg, &[0, 0], &[2, 3], tar_paul, 0.)
        .unwrap_err();
    multi_controlled_multi_rotate_pauli(qureg, &[0, 1], &[2, 2], tar_paul, 0.)
        .unwrap_err();

    multi_controlled_multi_rotate_pauli(qureg, &[-1, 1], &[2, 3], tar_paul, 0.)
        .unwrap_err();
    multi_controlled_multi_rotate_pauli(qureg, &[0, 4], &[2, 3], tar_paul, 0.)
        .unwrap_err();
    multi_controlled_multi_rotate_pauli(qureg, &[0, 1], &[-1, 3], tar_paul, 0.)
        .unwrap_err();
    multi_controlled_multi_rotate_pauli(qureg, &[0, 1], &[2, 4], tar_paul, 0.)
        .unwrap_err();

    multi_controlled_multi_rotate_pauli(qureg, &[0, 1], &[0, 3], tar_paul, 0.)
        .unwrap_err();
    multi_controlled_multi_rotate_pauli(qureg, &[0, 3], &[2, 3], tar_paul, 0.)
        .unwrap_err();
}

#[test]
fn check_array_length_init_state_from_amps() {
    let env = &QuestEnv::new();
    let qureg = &mut create_qureg::<2>(env);

    let reals = [0.; 4];
    let imags = [0.; 4];
    qureg.init_state_from_amps(&reals, &imags).unwrap();

    let reals = [0.; 3];
    let imags = [0.; 4];
    qureg.init_state_from_amps(&reals, &imags).unwrap_err();

    let reals = [0.; 4];
    let imags = [0.; 3];
    qureg.init_state_from_amps(&reals, &imags).unwrap_err();

    let reals = [0.; 5];
    let imags = [0.; 5];
    qureg.init_state_from_amps(&reals, &imags).unwrap();
}
