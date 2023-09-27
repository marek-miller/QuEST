#![allow(clippy::cast_sign_loss)]

use num::{
    Complex,
    Zero,
};

use super::*;

#[test]
fn create_qureg_01() -> Result<(), QuestError> {
    let env = &QuestEnv::new();
    let _ = Qureg::try_new(1, env)?;
    let _ = Qureg::try_new(5, env)?;

    let _ = Qureg::try_new(0, env).unwrap_err();
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
        let _ = Qureg::try_new_density(1, env)?;
        let _ = Qureg::try_new_density(5, env)?;

        let _ = Qureg::try_new_density(0, env).unwrap_err();
    }
    Ok(())
}

#[test]
fn get_matrix_n_elem_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();

    qureg.init_zero_state();
    let amp = qureg.get_imag_amp(0).unwrap();
    assert_eq!(amp, 0.);

    let mtr = &mut ComplexMatrixN::try_new(1).unwrap();
    init_complex_matrix_n(
        mtr,
        &[&[1., 2.], &[3., 4.]],
        &[&[11., 12.], &[13., 14.]],
    )
    .unwrap();
    qureg.apply_matrix_n(&[0], mtr).unwrap();
    let amp = qureg.get_imag_amp(0).unwrap();
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
fn init_complex_matrix_from_slice_01() {
    let mut m = ComplexMatrixN::try_new(1).unwrap();
    init_complex_matrix_from_slice(
        &mut m,
        &[
            Complex::new(1., 11.),
            Complex::new(2., 12.),
            Complex::new(3., 13.),
            Complex::new(4., 14.),
        ],
    )
    .unwrap();

    assert_eq!(m.row_real_as_slice(0), &[1., 2.]);
    assert_eq!(m.row_real_as_slice(1), &[3., 4.]);

    assert_eq!(m.row_imag_as_slice(0), &[11., 12.]);
    assert_eq!(m.row_imag_as_slice(1), &[13., 14.]);
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
    let mut qureg = Qureg::try_new(2, env).unwrap();
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
    let mut qureg = Qureg::try_new(2, env).unwrap();
    let mut op = DiagonalOp::try_new(2, env).unwrap();

    qureg.init_plus_state();
    init_diagonal_op(&mut op, &[1., 2., 3., 4.], &[5., 6., 7., 8.]).unwrap();

    let _ = calc_expec_diagonal_op(&qureg, &op).unwrap();

    let mut op = DiagonalOp::try_new(1, env).unwrap();
    init_diagonal_op(&mut op, &[1., 2.], &[5., 6.]).unwrap();
    let _ = calc_expec_diagonal_op(&qureg, &op).unwrap_err();
}

#[test]
fn create_subdiagonal_op_01() {
    let _ = SubDiagonalOp::try_new(1).unwrap();
    let _ = SubDiagonalOp::try_new(0).unwrap_err();
    let _ = SubDiagonalOp::try_new(-1).unwrap_err();
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
    let mut qureg = Qureg::try_new(3, env).unwrap();

    let re = [1., 2., 3., 4.];
    let im = [1., 2., 3., 4.];

    qureg.set_amps(0, &re, &im).unwrap();
    assert!((qureg.get_real_amp(0).unwrap() - 1.).abs() < EPSILON);

    qureg.set_amps(3, &re, &im).unwrap();

    qureg.set_amps(9, &re, &im).unwrap_err();
    qureg.set_amps(7, &re, &im).unwrap_err();
    qureg.set_amps(-1, &re, &im).unwrap_err();
}

#[test]
fn set_amps_02() {
    let env = &QuestEnv::new();
    let mut qureg = Qureg::try_new(2, env).unwrap();

    let re = [1.];
    let im = [1., 2.];

    let res = qureg.set_amps(0, &re, &im).unwrap_err();
    assert_eq!(res, QuestError::ArrayLengthError);

    let re = [1., 2.];
    let im = [1.];

    let res = qureg.set_amps(0, &re, &im).unwrap_err();
    assert_eq!(res, QuestError::ArrayLengthError);
}

#[test]
fn set_density_amps_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(3, env).unwrap();

    let re = &[1., 2., 3., 4.];
    let im = &[1., 2., 3., 4.];

    qureg.set_density_amps(0, 0, re, im).unwrap();
    assert!((qureg.get_density_amp(0, 0).unwrap().re - 1.).abs() < EPSILON);

    qureg.set_density_amps(1, 3, re, im).unwrap();

    qureg.set_amps(0, re, im).unwrap_err();

    qureg.set_density_amps(0, 9, re, im).unwrap_err();
    qureg.set_density_amps(8, 7, re, im).unwrap_err();
    qureg.set_density_amps(0, -1, re, im).unwrap_err();
}

#[test]
fn set_density_amps_02() {
    let env = &QuestEnv::new();
    let mut qureg = Qureg::try_new_density(2, env).unwrap();

    let re = [1.];
    let im = [1., 2.];

    let res = qureg.set_density_amps(0, 0, &re, &im).unwrap_err();
    assert_eq!(res, QuestError::ArrayLengthError);

    let re = [1., 2.];
    let im = [1.];

    let res = qureg.set_density_amps(0, 0, &re, &im).unwrap_err();
    assert_eq!(res, QuestError::ArrayLengthError);
}

#[test]
fn phase_shift_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();

    qureg.phase_shift(0, 0.0).unwrap();
    qureg.phase_shift(1, 0.5).unwrap();
    qureg.phase_shift(2, 1.0).unwrap();

    qureg.phase_shift(3, 0.0).unwrap_err();
    qureg.phase_shift(-11, 0.0).unwrap_err();
}

#[test]
fn controlled_phase_shift_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();

    qureg.controlled_phase_shift(0, 1, 0.5).unwrap();
    qureg.controlled_phase_shift(0, 2, 0.5).unwrap();

    qureg.controlled_phase_shift(0, 3, 0.5).unwrap_err();
    qureg.controlled_phase_shift(-1, 1, 0.5).unwrap_err();
}

#[test]
fn multi_controlled_phase_shift_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.multi_controlled_phase_shift(&[0, 1, 2], 0.5).unwrap();
    qureg.multi_controlled_phase_shift(&[2, 1, 0], 0.5).unwrap();

    qureg
        .multi_controlled_phase_shift(&[0, 1, 0], 0.5)
        .unwrap_err();
    qureg
        .multi_controlled_phase_shift(&[0, 1, 1], 0.5)
        .unwrap_err();

    qureg
        .multi_controlled_phase_shift(&[0, 4, 3, 4], 0.5)
        .unwrap_err();
}

#[test]
fn controlled_phase_flip_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();

    qureg.controlled_phase_flip(0, 1).unwrap();
    qureg.controlled_phase_flip(0, 2).unwrap();

    qureg.controlled_phase_flip(0, 3).unwrap_err();
    qureg.controlled_phase_flip(-1, 1).unwrap_err();
}

#[test]
fn multi_controlled_phase_flip_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(4, env).unwrap();
    qureg.multi_controlled_phase_flip(&[0, 1, 3]).unwrap();
    qureg.multi_controlled_phase_flip(&[0, 1, 3]).unwrap();

    qureg
        .multi_controlled_phase_flip(&[0, 4, 3, 4])
        .unwrap_err();
    qureg.multi_controlled_phase_flip(&[0, 7, -1]).unwrap_err();
}

#[test]
fn s_gate_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    qureg.s_gate(0).unwrap();
    assert!((qureg.get_imag_amp(0).unwrap()).abs() < EPSILON);

    qureg.pauli_x(0).unwrap();
    qureg.s_gate(0).unwrap();

    let amp = qureg.get_imag_amp(1).unwrap();
    assert!((amp - 1.).abs() < EPSILON);

    qureg.s_gate(-1).unwrap_err();
    qureg.s_gate(3).unwrap_err();
}

#[test]
fn t_gate_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    qureg.t_gate(0).unwrap();
    qureg.t_gate(-1).unwrap_err();
    qureg.t_gate(3).unwrap_err();
}

#[test]
fn get_amp_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_plus_state();

    qureg.get_amp(0).unwrap();
    qureg.get_amp(1).unwrap();
    qureg.get_amp(2).unwrap();
    qureg.get_amp(3).unwrap();

    qureg.get_amp(4).unwrap_err();
    qureg.get_amp(-1).unwrap_err();
}

#[test]
fn get_real_amp_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_plus_state();

    qureg.get_real_amp(0).unwrap();
    qureg.get_real_amp(1).unwrap();
    qureg.get_real_amp(2).unwrap();
    qureg.get_real_amp(3).unwrap();

    qureg.get_real_amp(4).unwrap_err();
    qureg.get_real_amp(-1).unwrap_err();
}

#[test]
fn get_imag_amp_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_plus_state();

    qureg.get_imag_amp(0).unwrap();
    qureg.get_imag_amp(1).unwrap();
    qureg.get_imag_amp(2).unwrap();
    qureg.get_imag_amp(3).unwrap();

    qureg.get_imag_amp(4).unwrap_err();
    qureg.get_imag_amp(-1).unwrap_err();
}

#[test]
fn get_prob_amp_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_plus_state();

    qureg.get_prob_amp(0).unwrap();
    qureg.get_prob_amp(1).unwrap();
    qureg.get_prob_amp(2).unwrap();
    qureg.get_prob_amp(3).unwrap();

    qureg.get_prob_amp(4).unwrap_err();
    qureg.get_prob_amp(-1).unwrap_err();
}

#[test]
fn get_density_amp_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();

    qureg.get_density_amp(0, 0).unwrap_err();
    qureg.get_density_amp(1, 0).unwrap_err();
    qureg.get_density_amp(2, 0).unwrap_err();
    qureg.get_density_amp(3, 0).unwrap_err();
    qureg.get_density_amp(-1, 5).unwrap_err();
    qureg.get_density_amp(4, 0).unwrap_err();
}

#[test]
fn get_density_amp_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(2, env).unwrap();

    qureg.get_density_amp(0, 0).unwrap();
    qureg.get_density_amp(1, 0).unwrap();
    qureg.get_density_amp(2, 0).unwrap();
    qureg.get_density_amp(3, 0).unwrap();
    qureg.get_density_amp(-1, 0).unwrap_err();
    qureg.get_density_amp(4, 0).unwrap_err();
}

#[test]
fn compact_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    let norm = SQRT_2.recip();
    let alpha = Qcomplex::new(0., norm);
    let beta = Qcomplex::new(0., norm);

    qureg.compact_unitary(0, alpha, beta).unwrap();
    qureg.compact_unitary(1, alpha, beta).unwrap();

    qureg.compact_unitary(4, alpha, beta).unwrap_err();
    qureg.compact_unitary(-1, alpha, beta).unwrap_err();
}

#[test]
fn compact_unitary_02() {
    // env_logger::init();
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    // this doesn't define a unitary matrix
    let alpha = Qcomplex::new(1., 2.);
    let beta = Qcomplex::new(2., 1.);

    qureg.compact_unitary(0, alpha, beta).unwrap_err();
    qureg.compact_unitary(1, alpha, beta).unwrap_err();

    qureg.compact_unitary(4, alpha, beta).unwrap_err();
    qureg.compact_unitary(-1, alpha, beta).unwrap_err();
}

#[test]
fn unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    let norm = SQRT_2.recip();
    let mtr = ComplexMatrix2::new(
        [[norm, norm], [norm, -norm]],
        [[0., 0.], [0., 0.]],
    );
    qureg.unitary(0, &mtr).unwrap();
    qureg.unitary(1, &mtr).unwrap();
    qureg.unitary(2, &mtr).unwrap_err();
    qureg.unitary(-1, &mtr).unwrap_err();
}

#[test]
fn unitary_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    // This isn't a unitary
    let mtr = ComplexMatrix2::new([[1., 2.], [0., 0.]], [[0., 0.], [1., 2.]]);
    qureg.unitary(0, &mtr).unwrap_err();
    qureg.unitary(1, &mtr).unwrap_err();
    qureg.unitary(2, &mtr).unwrap_err();
    qureg.unitary(-1, &mtr).unwrap_err();
}

#[test]
fn rotate_x_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    let theta = PI;
    qureg.rotate_x(0, theta).unwrap();
    qureg.rotate_x(1, theta).unwrap();
    qureg.rotate_x(2, theta).unwrap();

    qureg.rotate_x(3, theta).unwrap_err();
    qureg.rotate_x(-1, theta).unwrap_err();
}

#[test]
fn rotate_y_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    let theta = PI;
    qureg.rotate_y(0, theta).unwrap();
    qureg.rotate_y(1, theta).unwrap();
    qureg.rotate_y(2, theta).unwrap();

    qureg.rotate_y(3, theta).unwrap_err();
    qureg.rotate_y(-1, theta).unwrap_err();
}

#[test]
fn rotate_z_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    let theta = PI;
    qureg.rotate_z(0, theta).unwrap();
    qureg.rotate_z(1, theta).unwrap();
    qureg.rotate_z(2, theta).unwrap();

    qureg.rotate_z(3, theta).unwrap_err();
    qureg.rotate_z(-1, theta).unwrap_err();
}

#[test]
fn rotate_around_axis_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.init_zero_state();

    let angle = 0.;
    let axis = &Vector::new(0., 0., 1.);
    qureg.rotate_around_axis(0, angle, axis).unwrap();
    qureg.rotate_around_axis(1, angle, axis).unwrap();
    qureg.rotate_around_axis(2, angle, axis).unwrap();

    qureg.rotate_around_axis(3, angle, axis).unwrap_err();
    qureg.rotate_around_axis(-1, angle, axis).unwrap_err();
}

#[test]
fn rotate_around_axis_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.init_zero_state();

    let angle = 0.;
    // zero vector should throw an exception
    let axis = &Vector::new(0., 0., 0.);
    qureg.rotate_around_axis(0, angle, axis).unwrap_err();
    qureg.rotate_around_axis(1, angle, axis).unwrap_err();
    qureg.rotate_around_axis(2, angle, axis).unwrap_err();

    qureg.rotate_around_axis(3, angle, axis).unwrap_err();
    qureg.rotate_around_axis(-1, angle, axis).unwrap_err();
}

#[test]
fn controlled_rotate_x_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();

    qureg.controlled_rotate_x(1, 0, 0.5).unwrap();
    qureg.controlled_rotate_x(1, 2, 0.5).unwrap();

    qureg.controlled_rotate_x(1, 1, 0.5).unwrap_err();
    qureg.controlled_rotate_x(2, 2, 0.5).unwrap_err();
    qureg.controlled_rotate_x(-1, 2, 0.5).unwrap_err();
    qureg.controlled_rotate_x(2, -1, 0.5).unwrap_err();
    qureg.controlled_rotate_x(0, 4, 0.5).unwrap_err();
    qureg.controlled_rotate_x(4, 0, 0.5).unwrap_err();
}

#[test]
fn controlled_rotate_y_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();

    qureg.controlled_rotate_y(1, 0, 0.5).unwrap();
    qureg.controlled_rotate_y(1, 2, 0.5).unwrap();

    qureg.controlled_rotate_y(1, 1, 0.5).unwrap_err();
    qureg.controlled_rotate_y(2, 2, 0.5).unwrap_err();
    qureg.controlled_rotate_y(-1, 2, 0.5).unwrap_err();
    qureg.controlled_rotate_y(2, -1, 0.5).unwrap_err();
    qureg.controlled_rotate_y(0, 4, 0.5).unwrap_err();
    qureg.controlled_rotate_y(4, 0, 0.5).unwrap_err();
}

#[test]
fn controlled_rotate_z_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();

    qureg.controlled_rotate_z(1, 0, 0.5).unwrap();
    qureg.controlled_rotate_z(1, 2, 0.5).unwrap();

    qureg.controlled_rotate_z(1, 1, 0.5).unwrap_err();
    qureg.controlled_rotate_z(2, 2, 0.5).unwrap_err();
    qureg.controlled_rotate_z(-1, 2, 0.5).unwrap_err();
    qureg.controlled_rotate_z(2, -1, 0.5).unwrap_err();
    qureg.controlled_rotate_z(0, 4, 0.5).unwrap_err();
    qureg.controlled_rotate_z(4, 0, 0.5).unwrap_err();
}

#[test]
fn controlled_rotate_around_axis_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    let vector = &Vector::new(0., 0., 1.);

    qureg
        .controlled_rotate_around_axis(1, 0, 0.5, vector)
        .unwrap();
    qureg
        .controlled_rotate_around_axis(1, 2, 0.5, vector)
        .unwrap();

    qureg
        .controlled_rotate_around_axis(1, 1, 0.5, vector)
        .unwrap_err();
    qureg
        .controlled_rotate_around_axis(2, 2, 0.5, vector)
        .unwrap_err();
    qureg
        .controlled_rotate_around_axis(-1, 2, 0.5, vector)
        .unwrap_err();
    qureg
        .controlled_rotate_around_axis(2, -1, 0.5, vector)
        .unwrap_err();
    qureg
        .controlled_rotate_around_axis(0, 4, 0.5, vector)
        .unwrap_err();
    qureg
        .controlled_rotate_around_axis(4, 0, 0.5, vector)
        .unwrap_err();
}

#[test]
fn controlled_rotate_around_axis_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    // vector cannot be zero
    let vector = &Vector::new(0., 0., 0.);

    qureg
        .controlled_rotate_around_axis(1, 0, 0.5, vector)
        .unwrap_err();
    qureg
        .controlled_rotate_around_axis(1, 2, 0.5, vector)
        .unwrap_err();
}

#[test]
fn controlled_compact_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    let norm = SQRT_2.recip();
    let alpha = Qcomplex::new(0., norm);
    let beta = Qcomplex::new(0., norm);

    qureg.controlled_compact_unitary(0, 1, alpha, beta).unwrap();
    qureg.controlled_compact_unitary(1, 0, alpha, beta).unwrap();

    qureg
        .controlled_compact_unitary(1, 1, alpha, beta)
        .unwrap_err();
    qureg
        .controlled_compact_unitary(2, 2, alpha, beta)
        .unwrap_err();
    qureg
        .controlled_compact_unitary(4, 1, alpha, beta)
        .unwrap_err();
    qureg
        .controlled_compact_unitary(-1, 1, alpha, beta)
        .unwrap_err();
}

#[test]
fn controlled_compact_unitary_02() {
    // env_logger::init();
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    // this doesn't define a unitary matrix
    let alpha = Qcomplex::new(1., 2.);
    let beta = Qcomplex::new(2., 1.);

    qureg
        .controlled_compact_unitary(0, 1, alpha, beta)
        .unwrap_err();
    qureg
        .controlled_compact_unitary(1, 0, alpha, beta)
        .unwrap_err();

    qureg
        .controlled_compact_unitary(1, 1, alpha, beta)
        .unwrap_err();
    qureg
        .controlled_compact_unitary(4, 1, alpha, beta)
        .unwrap_err();
    qureg
        .controlled_compact_unitary(-1, 2, alpha, beta)
        .unwrap_err();
}

#[test]
fn controlled_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    let norm = SQRT_2.recip();
    let mtr = &ComplexMatrix2::new(
        [[norm, norm], [norm, -norm]],
        [[0., 0.], [0., 0.]],
    );

    qureg.controlled_unitary(0, 1, mtr).unwrap();
    qureg.controlled_unitary(1, 0, mtr).unwrap();

    qureg.controlled_unitary(1, 1, mtr).unwrap_err();
    qureg.controlled_unitary(2, 2, mtr).unwrap_err();
    qureg.controlled_unitary(4, 1, mtr).unwrap_err();
    qureg.controlled_unitary(-1, 1, mtr).unwrap_err();
}

#[test]
fn controlled_unitary_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    // this doesn't define a unitary matrix
    let mtr = &ComplexMatrix2::new([[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]);

    qureg.controlled_unitary(0, 1, mtr).unwrap_err();
    qureg.controlled_unitary(1, 0, mtr).unwrap_err();

    qureg.controlled_unitary(1, 1, mtr).unwrap_err();
    qureg.controlled_unitary(2, 2, mtr).unwrap_err();
    qureg.controlled_unitary(4, 1, mtr).unwrap_err();
    qureg.controlled_unitary(-1, 1, mtr).unwrap_err();
}

#[test]
fn multi_controlled_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.init_zero_state();

    let norm = SQRT_2.recip();
    let mtr = &ComplexMatrix2::new(
        [[norm, norm], [norm, -norm]],
        [[0., 0.], [0., 0.]],
    );

    qureg.multi_controlled_unitary(&[0], 2, mtr).unwrap();
    qureg.multi_controlled_unitary(&[0, 1], 2, mtr).unwrap();
    qureg.multi_controlled_unitary(&[1, 0], 2, mtr).unwrap();
    qureg.multi_controlled_unitary(&[1, 2], 0, mtr).unwrap();

    qureg.multi_controlled_unitary(&[1, 1], 1, mtr).unwrap_err();
    qureg.multi_controlled_unitary(&[1, 1], 4, mtr).unwrap_err();
    qureg
        .multi_controlled_unitary(&[-1, 1], 0, mtr)
        .unwrap_err();
    qureg
        .multi_controlled_unitary(&[1, 1, 1], 0, mtr)
        .unwrap_err();
    qureg
        .multi_controlled_unitary(&[0, 1, 2, 3], 0, mtr)
        .unwrap_err();
}

#[test]
fn multi_controlled_unitary_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.init_zero_state();

    // this doesn't define a unitary matrix
    let mtr = &ComplexMatrix2::new([[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]);

    qureg.multi_controlled_unitary(&[0, 1], 2, mtr).unwrap_err();
    qureg.multi_controlled_unitary(&[1, 2], 0, mtr).unwrap_err();
    qureg.multi_controlled_unitary(&[1, 1], 1, mtr).unwrap_err();
    qureg.multi_controlled_unitary(&[1, 1], 4, mtr).unwrap_err();
}

#[test]
fn pauli_x_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    qureg.pauli_x(0).unwrap();
    qureg.pauli_x(1).unwrap();

    qureg.pauli_x(2).unwrap_err();
    qureg.pauli_x(-1).unwrap_err();
}

#[test]
fn pauli_y_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    qureg.pauli_y(0).unwrap();
    qureg.pauli_y(1).unwrap();

    qureg.pauli_y(2).unwrap_err();
    qureg.pauli_y(-1).unwrap_err();
}

#[test]
fn pauli_z_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    qureg.pauli_z(0).unwrap();
    qureg.pauli_z(1).unwrap();

    qureg.pauli_z(2).unwrap_err();
    qureg.pauli_z(-1).unwrap_err();
}

#[test]
fn hadamard_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    qureg.hadamard(0).unwrap();
    qureg.hadamard(1).unwrap();
    qureg.hadamard(2).unwrap_err();
    qureg.hadamard(-1).unwrap_err();
}

#[test]
fn controlled_not_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();
    qureg.pauli_x(1).unwrap();

    qureg.controlled_not(1, 0).unwrap();
    qureg.controlled_not(0, 1).unwrap();

    qureg.controlled_not(0, 0).unwrap_err();
    qureg.controlled_not(1, 1).unwrap_err();
    qureg.controlled_not(1, 2).unwrap_err();
    qureg.controlled_not(2, 4).unwrap_err();
    qureg.controlled_not(2, -1).unwrap_err();
}

#[test]
fn multi_qubit_not_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    qureg.multi_qubit_not(&[0, 1]).unwrap();
    qureg.multi_qubit_not(&[1, 0]).unwrap();
    qureg.multi_qubit_not(&[0, 0]).unwrap_err();
    qureg.multi_qubit_not(&[1, 1]).unwrap_err();
    qureg.multi_qubit_not(&[4, 1]).unwrap_err();
    qureg.multi_qubit_not(&[0, -1]).unwrap_err();
}

#[test]
fn multi_controlled_multi_qubit_not_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(4, env).unwrap();
    qureg.init_zero_state();

    qureg
        .multi_controlled_multi_qubit_not(&[0, 1], &[2, 3])
        .unwrap();
    qureg
        .multi_controlled_multi_qubit_not(&[1, 0], &[3, 2])
        .unwrap();
    qureg
        .multi_controlled_multi_qubit_not(&[1, 0], &[3])
        .unwrap();
    qureg
        .multi_controlled_multi_qubit_not(&[1], &[3, 0])
        .unwrap();

    qureg
        .multi_controlled_multi_qubit_not(&[1, 0], &[0])
        .unwrap_err();
    qureg
        .multi_controlled_multi_qubit_not(&[0, 0], &[1])
        .unwrap_err();
    qureg
        .multi_controlled_multi_qubit_not(&[0, 0], &[-1])
        .unwrap_err();
    qureg
        .multi_controlled_multi_qubit_not(&[4, 1], &[0])
        .unwrap_err();
    qureg
        .multi_controlled_multi_qubit_not(&[0, 1], &[4])
        .unwrap_err();
}

#[test]
fn controlled_pauli_y_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    qureg.controlled_pauli_y(1, 0).unwrap();
    qureg.controlled_pauli_y(0, 1).unwrap();

    qureg.controlled_pauli_y(0, 0).unwrap_err();
    qureg.controlled_pauli_y(1, 1).unwrap_err();
    qureg.controlled_pauli_y(1, 2).unwrap_err();
    qureg.controlled_pauli_y(2, 4).unwrap_err();
    qureg.controlled_pauli_y(2, -1).unwrap_err();
}

#[test]
fn calc_prob_of_outcome_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    let _ = qureg.calc_prob_of_outcome(0, 0).unwrap();
    let _ = qureg.calc_prob_of_outcome(0, 1).unwrap();
    let _ = qureg.calc_prob_of_outcome(1, 0).unwrap();
    let _ = qureg.calc_prob_of_outcome(1, 1).unwrap();

    let _ = qureg.calc_prob_of_outcome(0, 2).unwrap_err();
    let _ = qureg.calc_prob_of_outcome(0, -2).unwrap_err();
    let _ = qureg.calc_prob_of_outcome(1, 3).unwrap_err();
    let _ = qureg.calc_prob_of_outcome(4, 0).unwrap_err();
}

#[test]
fn calc_prob_of_all_outcomes_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.init_zero_state();

    let outcome_probs = &mut vec![0.; 4];
    qureg
        .calc_prob_of_all_outcomes(outcome_probs, &[1, 2])
        .unwrap();
    qureg
        .calc_prob_of_all_outcomes(outcome_probs, &[0, 1])
        .unwrap();
    qureg
        .calc_prob_of_all_outcomes(outcome_probs, &[0, 2])
        .unwrap();

    qureg
        .calc_prob_of_all_outcomes(outcome_probs, &[1, 3])
        .unwrap_err();
    qureg
        .calc_prob_of_all_outcomes(outcome_probs, &[0, 0])
        .unwrap_err();
    qureg
        .calc_prob_of_all_outcomes(outcome_probs, &[4, 0])
        .unwrap_err();
    qureg
        .calc_prob_of_all_outcomes(outcome_probs, &[0, -1])
        .unwrap_err();
}

#[test]
fn calc_prob_of_all_outcomes_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.init_zero_state();

    let outcome_probs = &mut vec![0.; 3];

    qureg
        .calc_prob_of_all_outcomes(outcome_probs, &[1, 2])
        .unwrap_err();
}

#[test]
fn collapse_to_outcome_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();

    qureg.init_zero_state();
    qureg.collapse_to_outcome(0, 0).unwrap();

    qureg.init_zero_state();
    qureg.collapse_to_outcome(0, 1).unwrap_err();

    qureg.init_zero_state();
    qureg.collapse_to_outcome(-1, 0).unwrap_err();
    qureg.collapse_to_outcome(3, 0).unwrap_err();
    qureg.collapse_to_outcome(1, 3).unwrap_err();
    qureg.collapse_to_outcome(4, 3).unwrap_err();
}

#[test]
fn measure_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();

    qureg.init_zero_state();

    let _ = qureg.measure(0).unwrap();
    let _ = qureg.measure(1).unwrap();
    let _ = qureg.measure(-1).unwrap_err();
    let _ = qureg.measure(3).unwrap_err();
}

#[test]
fn measure_with_stats_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();

    // Prepare a triplet state `|00> + |11>`
    qureg.init_zero_state();

    let prob = &mut -1.;
    let _ = qureg.measure_with_stats(0, prob).unwrap();
    let _ = qureg.measure_with_stats(1, prob).unwrap();
    let _ = qureg.measure_with_stats(-1, prob).unwrap_err();
    let _ = qureg.measure_with_stats(3, prob).unwrap_err();
}

#[test]
fn calc_inner_product_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();
    let other_qureg = &mut Qureg::try_new(2, env).unwrap();
    other_qureg.init_zero_state();

    let _ = calc_inner_product(qureg, other_qureg).unwrap();
}

// #[test]
// fn calc_inner_product_02() {
//     let env = &QuestEnv::new();
//     let qureg = &mut Qureg::try_new(2,env).unwrap();
//     qureg.init_zero_state();
//     let other_qureg = &mut Qureg::try_new(1, env).unwrap();
//     other_qureg.init_zero_state();

//     let _ = calc_inner_product(qureg, other_qureg).unwrap_err();
// }

#[test]
fn calc_inner_product_03() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();
    let other_qureg = &mut Qureg::try_new_density(2, env).unwrap();
    other_qureg.init_zero_state();

    let _ = calc_inner_product(qureg, other_qureg).unwrap_err();
}

#[test]
fn calc_density_inner_product_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(2, env).unwrap();
    qureg.init_zero_state();
    let other_qureg = &mut Qureg::try_new_density(2, env).unwrap();
    other_qureg.init_zero_state();

    let _ = calc_density_inner_product(qureg, other_qureg).unwrap();
}

// #[test]
// fn calc_density_inner_product_02() {
//     let env = &QuestEnv::new();
//     let qureg = &mut Qureg::try_new_density(2,env).unwrap();
//     qureg.init_zero_state();
//     let other_qureg = &mut Qureg::try_new_density(1, env).unwrap();
//     other_qureg.init_zero_state();

//     let _ = calc_density_inner_product(qureg,  other_qureg).unwrap_err();
// }

#[test]
fn calc_density_inner_product_03() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(2, env).unwrap();
    qureg.init_zero_state();
    let other_qureg = &mut Qureg::try_new(2, env).unwrap();
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
    let qureg = &mut Qureg::try_new(2, env).unwrap();

    qureg.start_recording_qasm();
    qureg.hadamard(0).and(qureg.controlled_not(0, 1)).unwrap();
    qureg.stop_recording_qasm();

    qureg.print_recorded_qasm();
}

#[test]
fn mix_dephasing_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(2, env).unwrap();
    qureg.init_plus_state();

    qureg.mix_dephasing(0, 0.5).unwrap();
    qureg.mix_dephasing(1, 0.0).unwrap();

    qureg.mix_dephasing(0, 0.75).unwrap_err();
    qureg.mix_dephasing(2, 0.25).unwrap_err();
    qureg.mix_dephasing(0, -0.25).unwrap_err();
    qureg.mix_dephasing(-10, 0.25).unwrap_err();
}

#[test]
fn mix_dephasing_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_plus_state();

    // qureg is not a density matrix
    qureg.mix_dephasing(0, 0.5).unwrap_err();
    qureg.mix_dephasing(1, 0.0).unwrap_err();
}

#[test]
fn mix_two_qubit_dephasing_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(3, env).unwrap();
    qureg.init_plus_state();

    qureg.mix_two_qubit_dephasing(0, 1, 0.75).unwrap();
    qureg.mix_two_qubit_dephasing(0, 2, 0.75).unwrap();
    qureg.mix_two_qubit_dephasing(1, 2, 0.75).unwrap();
    qureg.mix_two_qubit_dephasing(1, 0, 0.75).unwrap();
    qureg.mix_two_qubit_dephasing(2, 1, 0.75).unwrap();

    qureg.mix_two_qubit_dephasing(0, 1, 0.99).unwrap_err();
    qureg.mix_two_qubit_dephasing(2, 1, 0.99).unwrap_err();

    qureg.mix_two_qubit_dephasing(4, 0, 0.1).unwrap_err();
    qureg.mix_two_qubit_dephasing(0, 4, 0.1).unwrap_err();

    qureg.mix_two_qubit_dephasing(-1, 0, 0.1).unwrap_err();
    qureg.mix_two_qubit_dephasing(0, -1, 0.1).unwrap_err();
}

#[test]
fn mix_two_qubit_dephasing_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.init_plus_state();

    // qureg is not a density matrix
    qureg.mix_two_qubit_dephasing(0, 1, 0.75).unwrap_err();
    qureg.mix_two_qubit_dephasing(0, 2, 0.75).unwrap_err();
}

#[test]
fn mix_depolarising_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(2, env).unwrap();
    qureg.init_zero_state();

    qureg.mix_depolarising(0, 0.00).unwrap();
    qureg.mix_depolarising(0, 0.75).unwrap();
    qureg.mix_depolarising(1, 0.75).unwrap();

    qureg.mix_depolarising(0, 0.99).unwrap_err();
    qureg.mix_depolarising(1, 0.99).unwrap_err();
    qureg.mix_depolarising(0, -0.99).unwrap_err();
    qureg.mix_depolarising(-1, 0.99).unwrap_err();
    qureg.mix_depolarising(-1, -0.99).unwrap_err();
}

#[test]
fn mix_damping_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(2, env).unwrap();
    qureg.init_plus_state();

    qureg.mix_damping(0, 1.).unwrap();
    qureg.mix_damping(0, 0.).unwrap();
    qureg.mix_damping(1, 1.).unwrap();
    qureg.mix_damping(1, 0.).unwrap();

    qureg.mix_damping(0, 10.).unwrap_err();
    qureg.mix_damping(0, -10.).unwrap_err();
    qureg.mix_damping(1, 10.).unwrap_err();
    qureg.mix_damping(1, -10.).unwrap_err();
    qureg.mix_damping(3, 0.5).unwrap_err();
    qureg.mix_damping(-3, 0.5).unwrap_err();
}

#[test]
fn mix_damping_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_plus_state();

    // qureg is not a density matrix
    qureg.mix_damping(1, 0.).unwrap_err();
    qureg.mix_damping(0, 0.).unwrap_err();
    // QuEST seg faults here:
    qureg.mix_damping(0, 0.5).unwrap_err();
}

#[test]
fn mix_two_qubit_depolarising_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(3, env).unwrap();
    qureg.init_plus_state();

    qureg.mix_two_qubit_depolarising(0, 1, 15. / 16.).unwrap();
    qureg.mix_two_qubit_depolarising(0, 2, 15. / 16.).unwrap();
    qureg.mix_two_qubit_depolarising(1, 2, 15. / 16.).unwrap();
    qureg.mix_two_qubit_depolarising(1, 0, 15. / 16.).unwrap();
    qureg.mix_two_qubit_depolarising(2, 1, 15. / 16.).unwrap();

    qureg.mix_two_qubit_depolarising(0, 1, 0.99).unwrap_err();
    qureg.mix_two_qubit_depolarising(2, 1, 0.99).unwrap_err();

    qureg.mix_two_qubit_depolarising(4, 0, 0.1).unwrap_err();
    qureg.mix_two_qubit_depolarising(0, 4, 0.1).unwrap_err();

    qureg.mix_two_qubit_depolarising(-1, 0, 0.1).unwrap_err();
    qureg.mix_two_qubit_depolarising(0, -1, 0.1).unwrap_err();
}

#[test]
fn mix_two_qubit_depolarising_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.init_plus_state();

    // qureg is not a density matrix
    qureg.mix_two_qubit_depolarising(0, 1, 0.75).unwrap_err();
    qureg.mix_two_qubit_depolarising(0, 2, 0.75).unwrap_err();
}

#[test]
fn mix_pauli_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(2, env).unwrap();
    qureg.init_zero_state();

    let (prob_x, prob_y, prob_z) = (0.25, 0.25, 0.25);
    qureg.mix_pauli(0, prob_x, prob_y, prob_z).unwrap();
    qureg.mix_pauli(1, prob_x, prob_y, prob_z).unwrap();

    qureg.mix_pauli(2, prob_x, prob_y, prob_z).unwrap_err();
    qureg.mix_pauli(-2, prob_x, prob_y, prob_z).unwrap_err();
}

#[test]
fn mix_pauli_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(2, env).unwrap();
    qureg.init_zero_state();

    // this is not a prob distribution
    let (prob_x, prob_y, prob_z) = (0.5, 0.5, 0.5);
    qureg.mix_pauli(0, prob_x, prob_y, prob_z).unwrap_err();
    qureg.mix_pauli(1, prob_x, prob_y, prob_z).unwrap_err();
}

#[test]
fn mix_pauli_03() {
    let env = &QuestEnv::new();
    // not a density matrix
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    let (prob_x, prob_y, prob_z) = (0.25, 0.25, 0.25);
    qureg.mix_pauli(0, prob_x, prob_y, prob_z).unwrap_err();
    qureg.mix_pauli(1, prob_x, prob_y, prob_z).unwrap_err();
}

#[test]
fn mix_density_matrix_01() {
    let env = &QuestEnv::new();
    let combine_qureg = &mut Qureg::try_new_density(2, env).unwrap();
    let other_qureg = &mut Qureg::try_new_density(2, env).unwrap();

    combine_qureg.init_zero_state();
    other_qureg.init_zero_state();

    combine_qureg.mix_density_matrix(0.0, other_qureg).unwrap();
    combine_qureg.mix_density_matrix(0.5, other_qureg).unwrap();
    combine_qureg.mix_density_matrix(0.99, other_qureg).unwrap();

    combine_qureg
        .mix_density_matrix(1.01, other_qureg)
        .unwrap_err();
    combine_qureg
        .mix_density_matrix(-1.01, other_qureg)
        .unwrap_err();
}

#[test]
fn mix_density_matrix_02() {
    let env = &QuestEnv::new();
    // this is not a density matrix
    let combine_qureg = &mut Qureg::try_new(2, env).unwrap();
    let other_qureg = &mut Qureg::try_new_density(2, env).unwrap();

    combine_qureg.init_zero_state();
    other_qureg.init_zero_state();

    combine_qureg
        .mix_density_matrix(0.0, other_qureg)
        .unwrap_err();
}

#[test]
fn mix_density_matrix_03() {
    let env = &QuestEnv::new();
    let combine_qureg = &mut Qureg::try_new_density(2, env).unwrap();
    // this is not a density matrix
    let other_qureg = &mut Qureg::try_new(2, env).unwrap();

    combine_qureg.init_zero_state();
    other_qureg.init_zero_state();

    combine_qureg
        .mix_density_matrix(0.0, other_qureg)
        .unwrap_err();
}

// #[test]
// fn mix_density_matrix_04() {
//     let env = &QuestEnv::new();
//     // dimensions don't match
//     let combine_qureg = &mut Qureg::try_new_density(2,env).unwrap();
//     let other_qureg = &mut Qureg::try_new_density(3,env).unwrap();

//     combine_qureg.init_zero_state();
//     other_qureg.init_zero_state();

//     combine_qureg.mix_density_matrix( 0.0, other_qureg).unwrap_err();
// }

#[test]
fn calc_purity_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(2, env).unwrap();
    qureg.init_zero_state();

    let _ = qureg.calc_purity().unwrap();

    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();
    let _ = qureg.calc_purity().unwrap_err();
}

#[test]
fn calc_fidelity_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(2, env).unwrap();
    let pure_state = &mut Qureg::try_new(2, env).unwrap();

    qureg.init_zero_state();
    pure_state.init_zero_state();

    let _ = qureg.calc_fidelity(pure_state).unwrap();
}

// #[test]
// fn calc_fidelity_02() {
//     let env = &QuestEnv::new();
//     let qureg = &mut Qureg::try_new_density(3,env).unwrap();
//     let pure_state = &mut Qureg::try_new(2,env).unwrap();

//     qureg.init_zero_state();
//     pure_state.init_zero_state();

//     let _ = qureg.calc_fidelity( pure_state).unwrap_err();
// }

#[test]
fn calc_fidelity_03() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    let pure_state = &mut Qureg::try_new_density(2, env).unwrap();

    qureg.init_zero_state();
    pure_state.init_zero_state();

    let _ = qureg.calc_fidelity(pure_state).unwrap_err();
}

#[test]
fn swap_gate_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    qureg.swap_gate(0, 1).unwrap();
    qureg.swap_gate(1, 0).unwrap();

    qureg.swap_gate(0, 0).unwrap_err();
    qureg.swap_gate(1, 1).unwrap_err();
}

#[test]
fn swap_gate_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    qureg.swap_gate(0, 2).unwrap_err();
    qureg.swap_gate(2, 0).unwrap_err();

    qureg.swap_gate(-1, 0).unwrap_err();
    qureg.swap_gate(0, -1).unwrap_err();

    qureg.swap_gate(4, 0).unwrap_err();
    qureg.swap_gate(0, 4).unwrap_err();
    qureg.swap_gate(4, 4).unwrap_err();
}

#[test]
fn swap_gate_03() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    // QuEST seg faults here
    qureg.swap_gate(4, -4).unwrap_err();
    qureg.swap_gate(-4, 4).unwrap_err();
    qureg.swap_gate(-4, -4).unwrap_err();
}

#[test]
fn sqrt_swap_gate_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    qureg.sqrt_swap_gate(0, 1).unwrap();
    qureg.sqrt_swap_gate(1, 0).unwrap();

    qureg.sqrt_swap_gate(0, 0).unwrap_err();
    qureg.sqrt_swap_gate(1, 1).unwrap_err();
}

#[test]
fn sqrt_swap_gate_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    qureg.sqrt_swap_gate(0, 2).unwrap_err();
    qureg.sqrt_swap_gate(2, 0).unwrap_err();

    qureg.sqrt_swap_gate(-1, 0).unwrap_err();
    qureg.sqrt_swap_gate(0, -1).unwrap_err();

    qureg.sqrt_swap_gate(4, 0).unwrap_err();
    qureg.sqrt_swap_gate(0, 4).unwrap_err();
    qureg.sqrt_swap_gate(4, 4).unwrap_err();
}

#[test]
fn sqrt_swap_gate_03() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    // QuEST seg faults here
    qureg.sqrt_swap_gate(4, -4).unwrap_err();
    qureg.sqrt_swap_gate(-4, 4).unwrap_err();
    qureg.sqrt_swap_gate(-4, -4).unwrap_err();
}

#[test]
fn multi_rotate_z_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    qureg.multi_rotate_z(&[0, 1], PI).unwrap();
    qureg.multi_rotate_z(&[0, 1, 2], PI).unwrap_err();
    qureg.multi_rotate_z(&[0, 2], PI).unwrap_err();
    qureg.multi_rotate_z(&[0, 0], PI).unwrap_err();
}

#[test]
fn multi_state_controlled_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.init_zero_state();

    let u = &ComplexMatrix2::new([[0., 1.], [1., 0.]], [[0., 0.], [0., 0.]]);
    qureg
        .multi_state_controlled_unitary(&[1, 2], &[0, 0], 0, u)
        .unwrap();
    qureg
        .multi_state_controlled_unitary(&[0, 2], &[0, 0], 1, u)
        .unwrap();
    qureg
        .multi_state_controlled_unitary(&[0, 1], &[0, 0], 2, u)
        .unwrap();

    qureg
        .multi_state_controlled_unitary(&[0, 1], &[0, 0], 0, u)
        .unwrap_err();
    qureg
        .multi_state_controlled_unitary(&[0, 1], &[0, 0], 1, u)
        .unwrap_err();
    qureg
        .multi_state_controlled_unitary(&[0, 0], &[0, 0], 1, u)
        .unwrap_err();

    qureg
        .multi_state_controlled_unitary(&[0, 0], &[0, 0], 3, u)
        .unwrap_err();
    qureg
        .multi_state_controlled_unitary(&[0, 0], &[0, 0], -1, u)
        .unwrap_err();
    qureg
        .multi_state_controlled_unitary(&[4, 0], &[0, 0], 1, u)
        .unwrap_err();
    qureg
        .multi_state_controlled_unitary(&[4, -1], &[0, 0], 1, u)
        .unwrap_err();
}

#[test]
fn multi_qubit_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
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

    qureg.multi_qubit_unitary(&[0, 1], u).unwrap();
    qureg.multi_qubit_unitary(&[1, 0], u).unwrap();

    qureg.multi_qubit_unitary(&[0, 0], u).unwrap_err();
    qureg.multi_qubit_unitary(&[1, 1], u).unwrap_err();

    qureg.multi_qubit_unitary(&[0, 2], u).unwrap_err();
    qureg.multi_qubit_unitary(&[1, -1], u).unwrap_err();
}

#[test]
fn multi_qubit_unitary_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
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

    qureg.multi_qubit_unitary(&[0, 1], u).unwrap_err();
    qureg.multi_qubit_unitary(&[1, 0], u).unwrap_err();

    qureg.multi_qubit_unitary(&[0, 0], u).unwrap_err();
    qureg.multi_qubit_unitary(&[1, 1], u).unwrap_err();

    qureg.multi_qubit_unitary(&[0, 2], u).unwrap_err();
    qureg.multi_qubit_unitary(&[1, -1], u).unwrap_err();
}

#[test]
fn controlled_multi_qubit_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
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

    qureg.controlled_multi_qubit_unitary(2, &[0, 1], u).unwrap();
    qureg.controlled_multi_qubit_unitary(2, &[1, 0], u).unwrap();

    qureg
        .controlled_multi_qubit_unitary(2, &[0, 0], u)
        .unwrap_err();
    qureg
        .controlled_multi_qubit_unitary(2, &[1, 1], u)
        .unwrap_err();

    qureg
        .controlled_multi_qubit_unitary(2, &[0, 2], u)
        .unwrap_err();
    qureg
        .controlled_multi_qubit_unitary(2, &[1, -1], u)
        .unwrap_err();

    qureg
        .controlled_multi_qubit_unitary(-1, &[0, 1], u)
        .unwrap_err();
    qureg
        .controlled_multi_qubit_unitary(4, &[0, 1], u)
        .unwrap_err();
}

#[test]
fn controlled_multi_qubit_unitary_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
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

    qureg
        .controlled_multi_qubit_unitary(2, &[0, 1], u)
        .unwrap_err();
    qureg
        .controlled_multi_qubit_unitary(2, &[1, 0], u)
        .unwrap_err();

    qureg
        .controlled_multi_qubit_unitary(2, &[0, 0], u)
        .unwrap_err();
    qureg
        .controlled_multi_qubit_unitary(2, &[1, 1], u)
        .unwrap_err();

    qureg
        .controlled_multi_qubit_unitary(2, &[0, 2], u)
        .unwrap_err();
    qureg
        .controlled_multi_qubit_unitary(2, &[1, -1], u)
        .unwrap_err();

    qureg
        .controlled_multi_qubit_unitary(-1, &[0, 1], u)
        .unwrap_err();
    qureg
        .controlled_multi_qubit_unitary(4, &[0, 1], u)
        .unwrap_err();
}

#[test]
fn muti_controlled_multi_qubit_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(4, env).unwrap();
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

    qureg
        .multi_controlled_multi_qubit_unitary(&[2, 3], &[0, 1], u)
        .unwrap();
    qureg
        .multi_controlled_multi_qubit_unitary(&[2, 3], &[1, 0], u)
        .unwrap();

    qureg
        .multi_controlled_multi_qubit_unitary(&[2, 3], &[0, 0], u)
        .unwrap_err();
    qureg
        .multi_controlled_multi_qubit_unitary(&[2, 3], &[1, 1], u)
        .unwrap_err();

    qureg
        .multi_controlled_multi_qubit_unitary(&[2, 3], &[0, 2], u)
        .unwrap_err();
    qureg
        .multi_controlled_multi_qubit_unitary(&[2, 3], &[1, -1], u)
        .unwrap_err();

    qureg
        .multi_controlled_multi_qubit_unitary(&[2, 3], &[0, 2, 1], u)
        .unwrap_err();
    qureg
        .multi_controlled_multi_qubit_unitary(&[2, 3], &[1, -1, 5], u)
        .unwrap_err();
}

#[test]
fn muti_controlled_multi_qubit_unitary_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(4, env).unwrap();
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

    qureg
        .multi_controlled_multi_qubit_unitary(&[0, 1], &[2, 3], u)
        .unwrap();
    qureg
        .multi_controlled_multi_qubit_unitary(&[1, 0], &[2, 3], u)
        .unwrap();

    qureg
        .multi_controlled_multi_qubit_unitary(&[0, 0], &[2, 3], u)
        .unwrap_err();
    qureg
        .multi_controlled_multi_qubit_unitary(&[1, 1], &[2, 3], u)
        .unwrap_err();

    qureg
        .multi_controlled_multi_qubit_unitary(&[0, 2], &[2, 3], u)
        .unwrap_err();
    qureg
        .multi_controlled_multi_qubit_unitary(&[1, -1], &[2, 3], u)
        .unwrap_err();

    qureg
        .multi_controlled_multi_qubit_unitary(&[0, 2, 1], &[2, 3], u)
        .unwrap_err();
    qureg
        .multi_controlled_multi_qubit_unitary(&[1, -1, 5], &[2, 3], u)
        .unwrap_err();
}

#[test]
fn muti_controlled_multi_qubit_unitary_03() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(4, env).unwrap();
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

    qureg
        .multi_controlled_multi_qubit_unitary(&[0, 1], &[2, 3], u)
        .unwrap_err();
    qureg
        .multi_controlled_multi_qubit_unitary(&[1, 0], &[2, 3], u)
        .unwrap_err();

    qureg
        .multi_controlled_multi_qubit_unitary(&[0, 0], &[2, 3], u)
        .unwrap_err();
    qureg
        .multi_controlled_multi_qubit_unitary(&[1, 1], &[2, 3], u)
        .unwrap_err();

    qureg
        .multi_controlled_multi_qubit_unitary(&[0, 2], &[2, 3], u)
        .unwrap_err();
    qureg
        .multi_controlled_multi_qubit_unitary(&[1, -1], &[2, 3], u)
        .unwrap_err();

    qureg
        .multi_controlled_multi_qubit_unitary(&[0, 2, 1], &[2, 3], u)
        .unwrap_err();
    qureg
        .multi_controlled_multi_qubit_unitary(&[1, -1, 5], &[2, 3], u)
        .unwrap_err();
}

#[test]
fn apply_matrix2_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    let m = &ComplexMatrix2::new([[0., 1.], [1., 0.]], [[0., 0.], [0., 0.]]);

    qureg.apply_matrix2(0, m).unwrap();
    qureg.apply_matrix2(1, m).unwrap();

    qureg.apply_matrix2(-1, m).unwrap_err();
    qureg.apply_matrix2(2, m).unwrap_err();
}

#[test]
fn mix_kraus_map_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(2, env).unwrap();
    qureg.init_zero_state();

    let m = &ComplexMatrix2::new([[0., 1.], [1., 0.]], [[0., 0.], [0., 0.]]);

    qureg.mix_kraus_map(0, &[m]).unwrap();
    qureg.mix_kraus_map(1, &[m]).unwrap();

    qureg.mix_kraus_map(-1, &[m]).unwrap_err();
    qureg.mix_kraus_map(2, &[m]).unwrap_err();
}

#[test]
fn mix_kraus_map_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(2, env).unwrap();
    qureg.init_zero_state();

    // This is not a CPTP map
    let m1 = &ComplexMatrix2::new([[0., 1.], [1., 0.]], [[0., 0.], [0., 0.]]);
    let m2 = &ComplexMatrix2::new([[0., 1.], [1., 1.]], [[0., 0.], [0., 0.]]);

    qureg.mix_kraus_map(0, &[m1, m2]).unwrap_err();
    qureg.mix_kraus_map(1, &[m1, m2]).unwrap_err();
}

#[test]
fn mix_kraus_map_03() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(2, env).unwrap();
    qureg.init_zero_state();

    let m1 = &ComplexMatrix2::new([[0., 1.], [1., 0.]], [[0., 0.], [0., 0.]]);

    qureg.mix_kraus_map(0, &[]).unwrap_err();
    qureg.mix_kraus_map(0, &[m1, m1]).unwrap_err();
    qureg.mix_kraus_map(0, &[m1, m1, m1]).unwrap_err();
    qureg.mix_kraus_map(0, &[m1, m1, m1, m1]).unwrap_err();
}

#[test]
fn mix_two_qubit_kraus_map_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(3, env).unwrap();
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
    qureg.mix_two_qubit_kraus_map(0, 1, &[m]).unwrap();
    qureg.mix_two_qubit_kraus_map(1, 2, &[m]).unwrap();
    qureg.mix_two_qubit_kraus_map(0, 2, &[m]).unwrap();

    qureg.mix_two_qubit_kraus_map(0, 0, &[m]).unwrap_err();
    qureg.mix_two_qubit_kraus_map(1, 1, &[m]).unwrap_err();
    qureg.mix_two_qubit_kraus_map(2, 2, &[m]).unwrap_err();

    qureg.mix_two_qubit_kraus_map(-1, 0, &[m]).unwrap_err();
    qureg.mix_two_qubit_kraus_map(0, 4, &[m]).unwrap_err();
}

#[test]
fn mix_two_qubit_kraus_map_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(3, env).unwrap();
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
    qureg.mix_two_qubit_kraus_map(0, 1, &[m]).unwrap_err();
    qureg.mix_two_qubit_kraus_map(1, 2, &[m]).unwrap_err();
    qureg.mix_two_qubit_kraus_map(0, 2, &[m]).unwrap_err();
}

#[test]
fn mix_two_qubit_kraus_map_03() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(3, env).unwrap();
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
    qureg.mix_two_qubit_kraus_map(0, 1, &[]).unwrap_err();
    qureg.mix_two_qubit_kraus_map(1, 2, &[m, m]).unwrap_err();
    qureg.mix_two_qubit_kraus_map(0, 2, &[m, m, m]).unwrap_err();
}

#[test]
fn mix_multi_qubit_kraus_map_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(3, env).unwrap();
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

    qureg.mix_multi_qubit_kraus_map(&[1, 2], &[m]).unwrap();
    qureg.mix_multi_qubit_kraus_map(&[0, 1], &[m]).unwrap();
    qureg.mix_multi_qubit_kraus_map(&[0, 2], &[m]).unwrap();
    qureg.mix_multi_qubit_kraus_map(&[2, 0], &[m]).unwrap();

    qureg.mix_multi_qubit_kraus_map(&[0, 0], &[m]).unwrap_err();
    qureg.mix_multi_qubit_kraus_map(&[1, 1], &[m]).unwrap_err();
    qureg.mix_multi_qubit_kraus_map(&[2, 2], &[m]).unwrap_err();

    qureg.mix_multi_qubit_kraus_map(&[-1, 0], &[m]).unwrap_err();
    qureg.mix_multi_qubit_kraus_map(&[0, 4], &[m]).unwrap_err();
}

#[test]
fn mix_multi_qubit_kraus_map_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(3, env).unwrap();
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

    qureg.mix_multi_qubit_kraus_map(&[1, 2], &[m]).unwrap_err();
    qureg.mix_multi_qubit_kraus_map(&[0, 1], &[m]).unwrap_err();
    qureg.mix_multi_qubit_kraus_map(&[0, 2], &[m]).unwrap_err();
    qureg.mix_multi_qubit_kraus_map(&[2, 0], &[m]).unwrap_err();

    qureg.mix_multi_qubit_kraus_map(&[0, 0], &[m]).unwrap_err();
    qureg.mix_multi_qubit_kraus_map(&[1, 1], &[m]).unwrap_err();
    qureg.mix_multi_qubit_kraus_map(&[2, 2], &[m]).unwrap_err();

    qureg.mix_multi_qubit_kraus_map(&[-1, 0], &[m]).unwrap_err();
    qureg.mix_multi_qubit_kraus_map(&[0, 4], &[m]).unwrap_err();
}

#[test]
fn mix_multi_qubit_kraus_map_03() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(3, env).unwrap();
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

    qureg.mix_multi_qubit_kraus_map(&[1, 2], &[]).unwrap_err();
    qureg
        .mix_multi_qubit_kraus_map(&[0, 1], &[m, m])
        .unwrap_err();
    qureg
        .mix_multi_qubit_kraus_map(&[0, 2], &[m, m, m])
        .unwrap_err();
}

#[test]
fn calc_hilbert_schmidt_distance_01() {
    let env = &QuestEnv::new();
    let a = &mut Qureg::try_new_density(2, env).unwrap();
    a.init_classical_state(0).unwrap();
    let b = &mut Qureg::try_new_density(2, env).unwrap();
    b.init_classical_state(1).unwrap();

    let _ = calc_hilbert_schmidt_distance(a, b).unwrap();
}

// #[test]
// fn calc_hilbert_schmidt_distance_02() {
//     let env = &QuestEnv::new();
//     let a = &mut Qureg::try_new_density(1, env).unwrap();
//     init_classical_state(a, 0).unwrap();
//     let b = &mut Qureg::try_new_density(2,env).unwrap();
//     init_classical_state(b, 1).unwrap();

//     let _ = calc_hilbert_schmidt_distance(a, b).unwrap_err();
// }

#[test]
fn calc_hilbert_schmidt_distance_03() {
    let env = &QuestEnv::new();
    let a = &mut Qureg::try_new_density(2, env).unwrap();
    a.init_classical_state(0).unwrap();
    let b = &mut Qureg::try_new(2, env).unwrap();
    b.init_classical_state(1).unwrap();

    let _ = calc_hilbert_schmidt_distance(a, b).unwrap_err();
}

#[test]
fn mix_nontp_kraus_map_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(2, env).unwrap();
    qureg.init_zero_state();

    let m = &ComplexMatrix2::new([[0., 1.], [0., 0.]], [[0., 0.], [0., 0.]]);

    qureg.mix_nontp_kraus_map(0, &[m]).unwrap();
    qureg.mix_nontp_kraus_map(1, &[m]).unwrap();

    qureg.mix_nontp_kraus_map(-1, &[m]).unwrap_err();
    qureg.mix_nontp_kraus_map(4, &[m]).unwrap_err();

    qureg.mix_nontp_kraus_map(0, &[]).unwrap_err();
    // The maps must consists of not more then 4 Kraus operators
    qureg.mix_nontp_kraus_map(0, &[m, m, m, m, m]).unwrap_err();
}

#[test]
fn mix_nontp_two_qubit_kraus_map_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(3, env).unwrap();
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
    qureg.mix_nontp_two_qubit_kraus_map(0, 1, &[m]).unwrap();
    qureg.mix_nontp_two_qubit_kraus_map(1, 2, &[m]).unwrap();
    qureg.mix_nontp_two_qubit_kraus_map(0, 2, &[m]).unwrap();

    qureg.mix_nontp_two_qubit_kraus_map(0, 0, &[m]).unwrap_err();
    qureg.mix_nontp_two_qubit_kraus_map(1, 1, &[m]).unwrap_err();
    qureg.mix_nontp_two_qubit_kraus_map(2, 2, &[m]).unwrap_err();

    qureg
        .mix_nontp_two_qubit_kraus_map(-1, 0, &[m])
        .unwrap_err();
    qureg.mix_nontp_two_qubit_kraus_map(0, 4, &[m]).unwrap_err();
}

#[test]
fn mix_nontp_multi_qubit_kraus_map_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(3, env).unwrap();
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

    qureg
        .mix_nontp_multi_qubit_kraus_map(&[1, 2], &[m])
        .unwrap();
    qureg
        .mix_nontp_multi_qubit_kraus_map(&[0, 1], &[m])
        .unwrap();
    qureg
        .mix_nontp_multi_qubit_kraus_map(&[0, 2], &[m])
        .unwrap();
    qureg
        .mix_nontp_multi_qubit_kraus_map(&[2, 0], &[m])
        .unwrap();

    qureg
        .mix_nontp_multi_qubit_kraus_map(&[0, 0], &[m])
        .unwrap_err();
    qureg
        .mix_nontp_multi_qubit_kraus_map(&[1, 1], &[m])
        .unwrap_err();
    qureg
        .mix_nontp_multi_qubit_kraus_map(&[2, 2], &[m])
        .unwrap_err();

    qureg
        .mix_nontp_multi_qubit_kraus_map(&[-1, 0], &[m])
        .unwrap_err();
    qureg
        .mix_nontp_multi_qubit_kraus_map(&[0, 4], &[m])
        .unwrap_err();
}

#[test]
fn mix_nontp_multi_qubit_kraus_map_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new_density(3, env).unwrap();
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

    qureg
        .mix_nontp_multi_qubit_kraus_map(&[1, 2], &[])
        .unwrap_err();
    // The maps must consists of not more then (2N)^2 Kraus operators
    qureg
        .mix_nontp_multi_qubit_kraus_map(
            &[0, 1],
            &[m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m],
        )
        .unwrap_err();
}

#[test]
fn apply_matrix4_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
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
    qureg.apply_matrix4(0, 1, m).unwrap();
    qureg.apply_matrix4(1, 0, m).unwrap();

    qureg.apply_matrix4(0, 0, m).unwrap_err();
    qureg.apply_matrix4(1, 1, m).unwrap_err();

    qureg.apply_matrix4(-1, 1, m).unwrap_err();
    qureg.apply_matrix4(3, 1, m).unwrap_err();
    qureg.apply_matrix4(0, 3, m).unwrap_err();
    qureg.apply_matrix4(0, -3, m).unwrap_err();

    qureg.apply_matrix4(3, -3, m).unwrap_err();
}

#[test]
fn apply_matrix_n_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
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

    qureg.apply_matrix_n(&[0, 1, 2], mtr).unwrap();
    qureg.apply_matrix_n(&[1, 0, 2], mtr).unwrap();
    qureg.apply_matrix_n(&[2, 1, 0], mtr).unwrap();

    qureg.apply_matrix_n(&[0, 1, 0], mtr).unwrap_err();
    qureg.apply_matrix_n(&[0, 1, 1], mtr).unwrap_err();
    qureg.apply_matrix_n(&[2, 3, 1], mtr).unwrap_err();

    qureg.apply_matrix_n(&[-2, 0, 1], mtr).unwrap_err();
    qureg.apply_matrix_n(&[1, 0, 4], mtr).unwrap_err();

    qureg.apply_matrix_n(&[1, 0], mtr).unwrap_err();
    qureg.apply_matrix_n(&[1, 0, 2, 3], mtr).unwrap_err();
}

#[test]
fn apply_multi_controlled_matrix_n_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(4, env).unwrap();
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

    qureg
        .apply_multi_controlled_matrix_n(&[0, 1], &[2, 3], u)
        .unwrap();
    qureg
        .apply_multi_controlled_matrix_n(&[0, 1], &[3, 2], u)
        .unwrap();
    qureg
        .apply_multi_controlled_matrix_n(&[1, 0], &[2, 3], u)
        .unwrap();
    qureg
        .apply_multi_controlled_matrix_n(&[0, 2], &[1, 3], u)
        .unwrap();

    qureg
        .apply_multi_controlled_matrix_n(&[0, 0], &[2, 3], u)
        .unwrap_err();
    qureg
        .apply_multi_controlled_matrix_n(&[0, 1], &[2, 2], u)
        .unwrap_err();

    qureg
        .apply_multi_controlled_matrix_n(&[-1, 0], &[2, 3], u)
        .unwrap_err();
    qureg
        .apply_multi_controlled_matrix_n(&[0, 4], &[2, 3], u)
        .unwrap_err();
    qureg
        .apply_multi_controlled_matrix_n(&[0, 1], &[1, 4], u)
        .unwrap_err();
    qureg
        .apply_multi_controlled_matrix_n(&[99, 99], &[99, 99], u)
        .unwrap_err();
}

#[test]
fn apply_qft_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.init_zero_state();

    qureg.apply_qft(&[0, 1]).unwrap();
    qureg.apply_qft(&[1, 0]).unwrap();
    qureg.apply_qft(&[1, 2]).unwrap();
    qureg.apply_qft(&[0, 2]).unwrap();
}

#[test]
fn apply_qft_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.init_zero_state();

    qureg.apply_qft(&[0, 0]).unwrap_err();
    qureg.apply_qft(&[1, 1]).unwrap_err();
    qureg.apply_qft(&[-1, 0]).unwrap_err();
    qureg.apply_qft(&[4, 0]).unwrap_err();
}

#[test]
fn apply_projector_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();
    qureg.apply_projector(0, 0).unwrap();
    qureg.init_zero_state();
    qureg.apply_projector(1, 0).unwrap();
    qureg.init_zero_state();
    qureg.apply_projector(0, 1).unwrap();
    qureg.init_zero_state();
    qureg.apply_projector(1, 1).unwrap();
}

#[test]
fn apply_projector_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();
    qureg.apply_projector(0, -1).unwrap_err();
    qureg.apply_projector(0, 3).unwrap_err();
    qureg.apply_projector(2, 0).unwrap_err();
    qureg.apply_projector(-1, 0).unwrap_err();
}

#[test]
fn multi_rotate_pauli_01() {
    use PauliOpType::PAULI_X;

    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.init_zero_state();

    qureg
        .multi_rotate_pauli(&[0, 1], &[PAULI_X, PAULI_X], 0.)
        .unwrap();
    qureg
        .multi_rotate_pauli(&[1, 2], &[PAULI_X, PAULI_X], 0.)
        .unwrap();
    qureg
        .multi_rotate_pauli(&[2, 0], &[PAULI_X, PAULI_X], 0.)
        .unwrap();

    qureg
        .multi_rotate_pauli(&[0, 0], &[PAULI_X, PAULI_X], 0.)
        .unwrap_err();
    qureg
        .multi_rotate_pauli(&[1, 1], &[PAULI_X, PAULI_X], 0.)
        .unwrap_err();
    qureg
        .multi_rotate_pauli(&[2, 2], &[PAULI_X, PAULI_X], 0.)
        .unwrap_err();

    qureg
        .multi_rotate_pauli(&[0, 3], &[PAULI_X, PAULI_X], 0.)
        .unwrap_err();
    qureg
        .multi_rotate_pauli(&[-1, 0], &[PAULI_X, PAULI_X], 0.)
        .unwrap_err();
    qureg
        .multi_rotate_pauli(&[0, 1, 2], &[PAULI_X, PAULI_X], 0.)
        .unwrap_err();
    qureg
        .multi_rotate_pauli(&[0, 1, 2, 3], &[PAULI_X, PAULI_X], 0.)
        .unwrap_err();
    qureg
        .multi_rotate_pauli(&[0, 1], &[PAULI_X], 0.)
        .unwrap_err();
}

#[test]
fn apply_phase_func_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    qureg
        .apply_phase_func(
            &[0, 1],
            BitEncoding::UNSIGNED,
            &[0.5, 0.5],
            &[0., 2.],
        )
        .unwrap();
    qureg
        .apply_phase_func(
            &[1, 0],
            BitEncoding::UNSIGNED,
            &[0.5, 0.5],
            &[0., 2.],
        )
        .unwrap();

    qureg
        .apply_phase_func(
            &[0, 0],
            BitEncoding::UNSIGNED,
            &[0.5, 0.5],
            &[0., 2.],
        )
        .unwrap_err();
    qureg
        .apply_phase_func(
            &[1, 1],
            BitEncoding::UNSIGNED,
            &[0.5, 0.5],
            &[0., 2.],
        )
        .unwrap_err();

    qureg
        .apply_phase_func(
            &[0, 2],
            BitEncoding::UNSIGNED,
            &[0.5, 0.5],
            &[0., 2.],
        )
        .unwrap_err();
    qureg
        .apply_phase_func(
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
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    // exponents contains a fractional number despite \p encoding <b>=</b>
    // ::TWOS_COMPLEMENT
    qureg
        .apply_phase_func(
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
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.init_zero_state();
    qureg.pauli_x(1).unwrap();

    qureg
        .apply_phase_func_overrides(
            &[0, 1],
            BitEncoding::UNSIGNED,
            &[0.5, 0.5],
            &[-2., 2.],
            &[0],
            &[0.],
        )
        .unwrap();

    qureg
        .apply_phase_func_overrides(
            &[0, 0],
            BitEncoding::UNSIGNED,
            &[0.5, 0.5],
            &[-2., 2.],
            &[0],
            &[0.],
        )
        .unwrap_err();

    qureg
        .apply_phase_func_overrides(
            &[-1, 0],
            BitEncoding::UNSIGNED,
            &[0.5, 0.5],
            &[-2., 2.],
            &[0],
            &[0.],
        )
        .unwrap_err();

    qureg
        .apply_phase_func_overrides(
            &[0, 4],
            BitEncoding::UNSIGNED,
            &[0.5, 0.5],
            &[-2., 2.],
            &[0],
            &[0.],
        )
        .unwrap_err();

    qureg
        .apply_phase_func_overrides(
            &[0, 1],
            BitEncoding::UNSIGNED,
            &[0.5, 0.5],
            &[-2., 2.],
            &[1],
            &[0.],
        )
        .unwrap_err();

    qureg
        .apply_phase_func_overrides(
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
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.init_zero_state();
    qureg.pauli_x(1).unwrap();

    qureg
        .apply_multi_var_phase_func(
            &[0, 1],
            &[1, 1],
            BitEncoding::UNSIGNED,
            &[0.5, 0.5],
            &[2., 2.],
            &[1, 1],
        )
        .unwrap();

    qureg
        .apply_multi_var_phase_func(
            &[0, 0],
            &[1, 1],
            BitEncoding::UNSIGNED,
            &[0.5, 0.5],
            &[2., 2.],
            &[1, 1],
        )
        .unwrap_err();

    qureg
        .apply_multi_var_phase_func(
            &[-1, 1],
            &[1, 1],
            BitEncoding::UNSIGNED,
            &[0.5, 0.5],
            &[2., 2.],
            &[1, 1],
        )
        .unwrap_err();

    qureg
        .apply_multi_var_phase_func(
            &[0, 4],
            &[1, 1],
            BitEncoding::UNSIGNED,
            &[0.5, 0.5],
            &[2., 2.],
            &[1, 1],
        )
        .unwrap_err();

    qureg
        .apply_multi_var_phase_func(
            &[0, 1],
            &[1, 3],
            BitEncoding::UNSIGNED,
            &[0.5, 0.5],
            &[2., 2.],
            &[1, 1],
        )
        .unwrap_err();

    qureg
        .apply_multi_var_phase_func(
            &[0, 1],
            &[1, 1],
            BitEncoding::UNSIGNED,
            &[0.5, 0.5],
            &[-2., 2.],
            &[1, 1],
        )
        .unwrap_err();

    qureg
        .apply_multi_var_phase_func(
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
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.init_zero_state();
    qureg.pauli_x(1).unwrap();

    qureg
        .apply_multi_var_phase_func_overrides(
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

    qureg
        .apply_multi_var_phase_func_overrides(
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

    qureg
        .apply_multi_var_phase_func_overrides(
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

    qureg
        .apply_multi_var_phase_func_overrides(
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

    qureg
        .apply_multi_var_phase_func_overrides(
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

    qureg
        .apply_multi_var_phase_func_overrides(
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

    qureg
        .apply_multi_var_phase_func_overrides(
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

    qureg
        .apply_multi_var_phase_func_overrides(
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
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    qureg
        .apply_named_phase_func(
            &[0, 1],
            &[1, 1],
            BitEncoding::UNSIGNED,
            PhaseFunc::DISTANCE,
        )
        .unwrap();

    qureg
        .apply_named_phase_func(
            &[0, 0],
            &[1, 1],
            BitEncoding::UNSIGNED,
            PhaseFunc::DISTANCE,
        )
        .unwrap_err();

    qureg
        .apply_named_phase_func(
            &[-1, 0],
            &[1, 1],
            BitEncoding::UNSIGNED,
            PhaseFunc::DISTANCE,
        )
        .unwrap_err();

    qureg
        .apply_named_phase_func(
            &[0, 4],
            &[1, 1],
            BitEncoding::UNSIGNED,
            PhaseFunc::DISTANCE,
        )
        .unwrap_err();

    qureg
        .apply_named_phase_func(
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
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.init_zero_state();

    qureg
        .apply_named_phase_func_overrides(
            &[0, 1],
            &[1, 1],
            BitEncoding::UNSIGNED,
            PhaseFunc::DISTANCE,
            &[0, 1, 0, 1],
            &[0., 0.],
        )
        .unwrap();

    qureg
        .apply_named_phase_func_overrides(
            &[0, 0],
            &[1, 1],
            BitEncoding::UNSIGNED,
            PhaseFunc::DISTANCE,
            &[0, 1, 0, 1],
            &[0., 0.],
        )
        .unwrap_err();

    qureg
        .apply_named_phase_func_overrides(
            &[-1, 0],
            &[1, 1],
            BitEncoding::UNSIGNED,
            PhaseFunc::DISTANCE,
            &[0, 1, 0, 1],
            &[0., 0.],
        )
        .unwrap_err();

    qureg
        .apply_named_phase_func_overrides(
            &[0, 4],
            &[1, 1],
            BitEncoding::UNSIGNED,
            PhaseFunc::DISTANCE,
            &[0, 1, 0, 1],
            &[0., 0.],
        )
        .unwrap_err();

    qureg
        .apply_named_phase_func_overrides(
            &[0, 1],
            &[1, 9],
            BitEncoding::UNSIGNED,
            PhaseFunc::DISTANCE,
            &[0, 1, 0, 1],
            &[0., 0.],
        )
        .unwrap_err();

    qureg
        .apply_named_phase_func_overrides(
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
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();

    qureg
        .apply_param_named_phase_func(
            &[0, 1],
            &[1, 1],
            BitEncoding::UNSIGNED,
            PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
            &[0., 0., 0., 0.],
        )
        .unwrap();

    qureg
        .apply_param_named_phase_func(
            &[0, 0],
            &[1, 1],
            BitEncoding::UNSIGNED,
            PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
            &[0., 0., 0., 0.],
        )
        .unwrap_err();

    qureg
        .apply_param_named_phase_func(
            &[-1, 0],
            &[1, 1],
            BitEncoding::UNSIGNED,
            PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
            &[0., 0., 0., 0.],
        )
        .unwrap_err();

    qureg
        .apply_param_named_phase_func(
            &[0, 4],
            &[1, 1],
            BitEncoding::UNSIGNED,
            PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
            &[0., 0., 0., 0.],
        )
        .unwrap_err();

    qureg
        .apply_param_named_phase_func(
            &[0, 4],
            &[1, 1],
            BitEncoding::UNSIGNED,
            PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
            // wrong number of parameters
            &[0., 0.],
        )
        .unwrap_err();

    qureg
        .apply_param_named_phase_func(
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
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.init_zero_state();

    qureg
        .apply_param_named_phase_func_overrides(
            &[0, 1],
            &[1, 1],
            BitEncoding::UNSIGNED,
            PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
            &[0., 0., 0., 0.],
            &[0, 1, 0, 1],
            &[0., 0.],
        )
        .unwrap();

    qureg
        .apply_param_named_phase_func_overrides(
            &[0, 0],
            &[1, 1],
            BitEncoding::UNSIGNED,
            PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
            &[0., 0., 0., 0.],
            &[0, 1, 0, 1],
            &[0., 0.],
        )
        .unwrap_err();

    qureg
        .apply_param_named_phase_func_overrides(
            &[-1, 0],
            &[1, 1],
            BitEncoding::UNSIGNED,
            PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
            &[0., 0., 0., 0.],
            &[0, 1, 0, 1],
            &[0., 0.],
        )
        .unwrap_err();

    qureg
        .apply_param_named_phase_func_overrides(
            &[0, 4],
            &[1, 1],
            BitEncoding::UNSIGNED,
            PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
            &[0., 0., 0., 0.],
            &[0, 1, 0, 1],
            &[0., 0.],
        )
        .unwrap_err();

    qureg
        .apply_param_named_phase_func_overrides(
            &[0, 0],
            &[1, 9],
            BitEncoding::UNSIGNED,
            PhaseFunc::SCALED_INVERSE_SHIFTED_NORM,
            &[0., 0., 0., 0.],
            &[0, 1, 0, 1],
            &[0., 0.],
        )
        .unwrap_err();

    qureg
        .apply_param_named_phase_func_overrides(
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

    qureg
        .apply_param_named_phase_func_overrides(
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
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();
    let workspace = &mut Qureg::try_new(2, env).unwrap();

    qureg
        .calc_expec_pauli_prod(&[0, 1], &[PAULI_X, PAULI_X], workspace)
        .unwrap();
    qureg
        .calc_expec_pauli_prod(&[1, 0], &[PAULI_X, PAULI_X], workspace)
        .unwrap();

    qureg
        .calc_expec_pauli_prod(&[0, 1], &[PAULI_X], workspace)
        .unwrap_err();
    qureg
        .calc_expec_pauli_prod(&[0, 0], &[PAULI_X, PAULI_X], workspace)
        .unwrap_err();

    qureg
        .calc_expec_pauli_prod(&[-1, 0], &[PAULI_X, PAULI_X], workspace)
        .unwrap_err();
    qureg
        .calc_expec_pauli_prod(&[4, 0], &[PAULI_X, PAULI_X], workspace)
        .unwrap_err();
    qureg
        .calc_expec_pauli_prod(&[0, -1], &[PAULI_X, PAULI_X], workspace)
        .unwrap_err();
    qureg
        .calc_expec_pauli_prod(&[0, 4], &[PAULI_X, PAULI_X], workspace)
        .unwrap_err();
}

// #[test]
// fn calc_expec_pauli_prod_02() {
//     use PauliOpType::PAULI_X;
//     let env = &QuestEnv::new();
//     let qureg = &mut Qureg::try_new(2,env).unwrap();
//     qureg.init_zero_state();
//     let workspace = &mut Qureg::try_new(3,env).unwrap();

//     qureg.calc_expec_pauli_prod( &[0, 1], &[PAULI_X, PAULI_X], workspace)
//         .unwrap_err();
// }

#[test]
fn calc_expec_pauli_sum_01() {
    use PauliOpType::{
        PAULI_X,
        PAULI_Z,
    };
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();
    let workspace = &mut Qureg::try_new(2, env).unwrap();

    let all_pauli_codes = &[PAULI_X, PAULI_Z, PAULI_Z, PAULI_X];
    let term_coeffs = &[0.5, 0.5];

    qureg
        .calc_expec_pauli_sum(all_pauli_codes, term_coeffs, workspace)
        .unwrap();
}

// #[test]
// fn calc_expec_pauli_sum_02() {
//     use PauliOpType::{
//         PAULI_X,
//         PAULI_Z,
//     };
//     let env = &QuestEnv::new();
//     let qureg = &mut Qureg::try_new(2,env).unwrap();
//     qureg.init_zero_state();
//     let workspace = &mut Qureg::try_new(3,env).unwrap();

//     let all_pauli_codes = &[PAULI_X, PAULI_Z, PAULI_Z, PAULI_X];
//     let term_coeffs = &[0.5, 0.5];

//     qureg.calc_expec_pauli_sum( all_pauli_codes, term_coeffs, workspace)
//         .unwrap_err();
// }

#[test]
fn calc_expec_pauli_hamil_01() {
    use PauliOpType::{
        PAULI_X,
        PAULI_Z,
    };
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    qureg.init_zero_state();
    let workspace = &mut Qureg::try_new(2, env).unwrap();

    let hamil = &mut PauliHamil::try_new(2, 2).unwrap();
    init_pauli_hamil(hamil, &[0.5, 0.5], &[PAULI_X, PAULI_X, PAULI_X, PAULI_Z])
        .unwrap();

    qureg.calc_expec_pauli_hamil(hamil, workspace).unwrap();
}

#[test]
fn two_qubit_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.init_zero_state();
    qureg.pauli_x(0).unwrap();

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

    qureg.two_qubit_unitary(0, 1, u).unwrap();
    qureg.two_qubit_unitary(1, 2, u).unwrap();

    qureg.two_qubit_unitary(0, 0, u).unwrap_err();
    qureg.two_qubit_unitary(-1, 0, u).unwrap_err();
    qureg.two_qubit_unitary(0, 4, u).unwrap_err();
}

#[test]
fn two_qubit_unitary_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.init_zero_state();
    qureg.pauli_x(0).unwrap();

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

    qureg.two_qubit_unitary(0, 1, u).unwrap_err();
    qureg.two_qubit_unitary(1, 2, u).unwrap_err();

    qureg.two_qubit_unitary(0, 0, u).unwrap_err();
    qureg.two_qubit_unitary(-1, 0, u).unwrap_err();
    qureg.two_qubit_unitary(0, 4, u).unwrap_err();
}

#[test]
fn controlled_two_qubit_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.init_zero_state();
    qureg.pauli_x(0).unwrap();

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

    qureg.controlled_two_qubit_unitary(0, 1, 2, u).unwrap();
    qureg.controlled_two_qubit_unitary(1, 2, 0, u).unwrap();
    qureg.controlled_two_qubit_unitary(2, 0, 1, u).unwrap();

    qureg.controlled_two_qubit_unitary(0, 1, 1, u).unwrap_err();
    qureg.controlled_two_qubit_unitary(0, 0, 1, u).unwrap_err();

    qureg.controlled_two_qubit_unitary(-1, 0, 1, u).unwrap_err();
    qureg.controlled_two_qubit_unitary(4, 0, 1, u).unwrap_err();

    qureg.controlled_two_qubit_unitary(0, -1, 0, u).unwrap_err();
    qureg.controlled_two_qubit_unitary(0, 0, 4, u).unwrap_err();
}

#[test]
fn controlled_two_qubit_unitary_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(3, env).unwrap();
    qureg.init_zero_state();
    qureg.pauli_x(0).unwrap();

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

    qureg.controlled_two_qubit_unitary(0, 1, 2, u).unwrap_err();
    qureg.controlled_two_qubit_unitary(1, 2, 0, u).unwrap_err();
    qureg.controlled_two_qubit_unitary(2, 0, 1, u).unwrap_err();

    qureg.controlled_two_qubit_unitary(0, 1, 1, u).unwrap_err();
    qureg.controlled_two_qubit_unitary(0, 0, 1, u).unwrap_err();

    qureg.controlled_two_qubit_unitary(-1, 0, 1, u).unwrap_err();
    qureg.controlled_two_qubit_unitary(4, 0, 1, u).unwrap_err();

    qureg.controlled_two_qubit_unitary(0, -1, 0, u).unwrap_err();
    qureg.controlled_two_qubit_unitary(0, 0, 4, u).unwrap_err();
}

#[test]
fn multi_controlled_two_qubit_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(4, env).unwrap();
    qureg.init_zero_state();
    qureg.pauli_x(0).unwrap();
    qureg.pauli_x(1).unwrap();

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

    qureg
        .multi_controlled_two_qubit_unitary(&[0, 1], 2, 3, u)
        .unwrap();
    qureg
        .multi_controlled_two_qubit_unitary(&[1, 0], 3, 2, u)
        .unwrap();
    qureg
        .multi_controlled_two_qubit_unitary(&[1, 2], 0, 3, u)
        .unwrap();
    qureg
        .multi_controlled_two_qubit_unitary(&[3, 0], 2, 1, u)
        .unwrap();

    qureg
        .multi_controlled_two_qubit_unitary(&[0, 0], 1, 2, u)
        .unwrap_err();
    qureg
        .multi_controlled_two_qubit_unitary(&[0, 1], 1, 1, u)
        .unwrap_err();

    qureg
        .multi_controlled_two_qubit_unitary(&[-1, 1], 2, 3, u)
        .unwrap_err();
    qureg
        .multi_controlled_two_qubit_unitary(&[5, 1], 2, 3, u)
        .unwrap_err();

    qureg
        .multi_controlled_two_qubit_unitary(&[0, 1], -1, 3, u)
        .unwrap_err();
    qureg
        .multi_controlled_two_qubit_unitary(&[0, 1], 5, 3, u)
        .unwrap_err();
}

#[test]
fn multi_controlled_two_qubit_unitary_02() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(4, env).unwrap();
    qureg.init_zero_state();
    qureg.pauli_x(0).unwrap();
    qureg.pauli_x(1).unwrap();

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

    qureg
        .multi_controlled_two_qubit_unitary(&[0, 1], 2, 3, u)
        .unwrap_err();
    qureg
        .multi_controlled_two_qubit_unitary(&[1, 0], 3, 2, u)
        .unwrap_err();
    qureg
        .multi_controlled_two_qubit_unitary(&[1, 2], 0, 3, u)
        .unwrap_err();
    qureg
        .multi_controlled_two_qubit_unitary(&[3, 0], 2, 1, u)
        .unwrap_err();

    qureg
        .multi_controlled_two_qubit_unitary(&[0, 0], 1, 2, u)
        .unwrap_err();
    qureg
        .multi_controlled_two_qubit_unitary(&[0, 1], 1, 1, u)
        .unwrap_err();

    qureg
        .multi_controlled_two_qubit_unitary(&[-1, 1], 2, 3, u)
        .unwrap_err();
    qureg
        .multi_controlled_two_qubit_unitary(&[5, 1], 2, 3, u)
        .unwrap_err();

    qureg
        .multi_controlled_two_qubit_unitary(&[0, 1], -1, 3, u)
        .unwrap_err();
    qureg
        .multi_controlled_two_qubit_unitary(&[0, 1], 5, 3, u)
        .unwrap_err();
}

#[test]
fn multi_controlled_multi_qubit_unitary_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(4, env).unwrap();
    qureg.init_zero_state();
    qureg.pauli_x(0).unwrap();
    qureg.pauli_x(1).unwrap();

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
    qureg
        .multi_controlled_multi_qubit_unitary(ctrls, targs, u)
        .unwrap();

    // Check if the register is now in the state `|1111>`
    let amp = qureg.get_real_amp(15).unwrap();
    assert!((amp - 1.).abs() < EPSILON);
}

#[test]
fn apply_pauli_sum_01() {
    use PauliOpType::{
        PAULI_I,
        PAULI_X,
    };
    let env = &QuestEnv::new();
    let in_qureg = &mut Qureg::try_new(2, env).unwrap();
    in_qureg.init_zero_state();
    let out_qureg = &mut Qureg::try_new(2, env).unwrap();
    let all_pauli_codes = &[PAULI_I, PAULI_X, PAULI_X, PAULI_I];
    let term_coeffs = &[SQRT_2.recip(), SQRT_2.recip()];

    apply_pauli_sum(in_qureg, all_pauli_codes, term_coeffs, out_qureg).unwrap();
}

#[test]
fn apply_pauli_sum_03() {
    use PauliOpType::{
        PAULI_I,
        PAULI_X,
    };
    let env = &QuestEnv::new();
    let in_qureg = &mut Qureg::try_new(2, env).unwrap();
    in_qureg.init_zero_state();
    let out_qureg = &mut Qureg::try_new(2, env).unwrap();

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
    let in_qureg = &mut Qureg::try_new(2, env).unwrap();
    in_qureg.init_zero_state();
    let out_qureg = &mut Qureg::try_new(2, env).unwrap();

    let hamil = &mut PauliHamil::try_new(2, 2).unwrap();
    let coeffs = &[SQRT_2.recip(), SQRT_2.recip()];
    let codes = &[PAULI_I, PAULI_X, PAULI_X, PAULI_I];
    init_pauli_hamil(hamil, coeffs, codes).unwrap();

    apply_pauli_hamil(in_qureg, hamil, out_qureg).unwrap();
}

#[test]
fn apply_trotter_circuit_01() {
    use PauliOpType::PAULI_X;

    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(1, env).unwrap();
    qureg.init_zero_state();

    let hamil = &mut PauliHamil::try_new(1, 1).unwrap();
    let coeffs = &[1.];
    let codes = &[PAULI_X];
    init_pauli_hamil(hamil, coeffs, codes).unwrap();

    qureg.apply_trotter_circuit(hamil, 0., 1, 1).unwrap();
    qureg.apply_trotter_circuit(hamil, 0., 2, 1).unwrap();
    qureg.apply_trotter_circuit(hamil, 0., 1, 2).unwrap();

    qureg.apply_trotter_circuit(hamil, 0., -1, 1).unwrap_err();
    qureg.apply_trotter_circuit(hamil, 0., 1, 0).unwrap_err();
    qureg.apply_trotter_circuit(hamil, 0., 1, -1).unwrap_err();
}

#[test]
fn set_weighted_qureg_01() {
    let env = &QuestEnv::new();
    let qureg1 = &mut Qureg::try_new(1, env).unwrap();
    qureg1.init_zero_state();
    let qureg2 = &mut Qureg::try_new(1, env).unwrap();
    qureg2.init_zero_state();
    qureg2.pauli_x(0).unwrap();

    let out = &mut Qureg::try_new(1, env).unwrap();
    out.init_zero_state();

    let fac1 = Qcomplex::new(SQRT_2.recip(), 0.);
    let fac2 = Qcomplex::new(SQRT_2.recip(), 0.);
    let fac_out = Qcomplex::zero();

    set_weighted_qureg(fac1, qureg1, fac2, qureg2, fac_out, out).unwrap();
}

#[test]
fn set_weighted_qureg_04() {
    let env = &QuestEnv::new();
    let qureg1 = &mut Qureg::try_new(2, env).unwrap();
    qureg1.init_zero_state();
    // all quregs should either state vectors of density matrices
    let qureg2 = &mut Qureg::try_new_density(2, env).unwrap();
    qureg2.init_zero_state();
    qureg2.pauli_x(0).unwrap();

    let out = &mut Qureg::try_new(2, env).unwrap();
    out.init_zero_state();

    let fac1 = Qcomplex::new(SQRT_2.recip(), 0.);
    let fac2 = Qcomplex::new(SQRT_2.recip(), 0.);
    let fac_out = Qcomplex::zero();

    set_weighted_qureg(fac1, qureg1, fac2, qureg2, fac_out, out).unwrap_err();
}

#[test]
fn multi_controlled_multi_rotate_z_01() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(4, env).unwrap();

    // Initialize `|1111>`
    qureg.init_zero_state();
    (0..4).try_for_each(|i| qureg.pauli_x(i)).unwrap();

    qureg
        .multi_controlled_multi_rotate_z(&[0, 1], &[2, 3], 0.)
        .unwrap();
    qureg
        .multi_controlled_multi_rotate_z(&[0, 2], &[1, 3], 0.)
        .unwrap();
    qureg
        .multi_controlled_multi_rotate_z(&[2, 1], &[3, 0], 0.)
        .unwrap();

    qureg
        .multi_controlled_multi_rotate_z(&[0, 0], &[2, 3], 0.)
        .unwrap_err();
    qureg
        .multi_controlled_multi_rotate_z(&[0, 1], &[2, 2], 0.)
        .unwrap_err();

    qureg
        .multi_controlled_multi_rotate_z(&[-1, 1], &[2, 3], 0.)
        .unwrap_err();
    qureg
        .multi_controlled_multi_rotate_z(&[0, 4], &[2, 3], 0.)
        .unwrap_err();
    qureg
        .multi_controlled_multi_rotate_z(&[0, 1], &[-1, 3], 0.)
        .unwrap_err();
    qureg
        .multi_controlled_multi_rotate_z(&[0, 1], &[2, 4], 0.)
        .unwrap_err();

    qureg
        .multi_controlled_multi_rotate_z(&[0, 1], &[0, 3], 0.)
        .unwrap_err();
    qureg
        .multi_controlled_multi_rotate_z(&[0, 3], &[2, 3], 0.)
        .unwrap_err();
}

#[test]
fn multi_controlled_multi_rotate_pauli_01() {
    use PauliOpType::PAULI_Z;

    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(4, env).unwrap();

    // Initialize `|1111>`
    qureg.init_zero_state();
    (0..4).try_for_each(|i| qureg.pauli_x(i)).unwrap();

    let tar_paul = &[PAULI_Z, PAULI_Z];

    qureg
        .multi_controlled_multi_rotate_pauli(&[0, 1], &[2, 3], tar_paul, 0.)
        .unwrap();
    qureg
        .multi_controlled_multi_rotate_pauli(&[0, 2], &[1, 3], tar_paul, 0.)
        .unwrap();
    qureg
        .multi_controlled_multi_rotate_pauli(&[2, 1], &[3, 0], tar_paul, 0.)
        .unwrap();

    qureg
        .multi_controlled_multi_rotate_pauli(&[0, 0], &[2, 3], tar_paul, 0.)
        .unwrap_err();
    qureg
        .multi_controlled_multi_rotate_pauli(&[0, 1], &[2, 2], tar_paul, 0.)
        .unwrap_err();

    qureg
        .multi_controlled_multi_rotate_pauli(&[-1, 1], &[2, 3], tar_paul, 0.)
        .unwrap_err();
    qureg
        .multi_controlled_multi_rotate_pauli(&[0, 4], &[2, 3], tar_paul, 0.)
        .unwrap_err();
    qureg
        .multi_controlled_multi_rotate_pauli(&[0, 1], &[-1, 3], tar_paul, 0.)
        .unwrap_err();
    qureg
        .multi_controlled_multi_rotate_pauli(&[0, 1], &[2, 4], tar_paul, 0.)
        .unwrap_err();

    qureg
        .multi_controlled_multi_rotate_pauli(&[0, 1], &[0, 3], tar_paul, 0.)
        .unwrap_err();
    qureg
        .multi_controlled_multi_rotate_pauli(&[0, 3], &[2, 3], tar_paul, 0.)
        .unwrap_err();
}

#[test]
fn check_array_length_init_state_from_amps() {
    let env = &QuestEnv::new();
    let qureg = &mut Qureg::try_new(2, env).unwrap();

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
