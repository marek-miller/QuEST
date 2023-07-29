use std::thread;

use quest_bind::{
    get_quest_seeds,
    get_real_amp,
    init_zero_state,
    pauli_x,
    seed_quest,
    seed_quest_default,
    QuestEnv,
    Qureg,
    EPSILON,
};

// This won't compile unless QuestEnv is Send
#[test]
fn questenv_is_send_01() {
    let env = QuestEnv::new();

    thread::scope(|s| {
        s.spawn(move || env.report_quest_env());
    });
}

#[test]
fn questenv_is_send_02() {
    let env = QuestEnv::new();
    let handle = thread::spawn(move || {
        env.report_quest_env();
        env
    });

    let env = handle.join().unwrap();
    env.report_quest_env();
}

#[test]
fn questenv_seed_from_other_thread() {
    let seeds = [0, 1, 2];

    let mut env = QuestEnv::new();

    let handle = thread::spawn(move || {
        seed_quest(&mut env, &seeds);
        env
    });

    let env = handle.join().unwrap();
    assert_eq!(get_quest_seeds(&env), seeds);
}

#[test]
fn questenv_seed_quest_default_01() {
    let mut env = QuestEnv::new();
    seed_quest_default(&mut env);

    let handle = thread::spawn(move || {
        let seeds = get_quest_seeds(&env);
        let copy_seeds = seeds.to_vec();
        (copy_seeds, env)
    });

    let (copy_seeds, env) = handle.join().unwrap();
    let check_seeds = get_quest_seeds(&env);

    assert_eq!(copy_seeds, check_seeds);
}

fn compute_qureg(env: &QuestEnv) -> bool {
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    init_zero_state(qureg);
    pauli_x(qureg, 0).unwrap();
    let amp = get_real_amp(qureg, 1).unwrap();

    (amp - 1.).abs() < EPSILON
}

#[test]
fn send_questenv_01() {
    let env = QuestEnv::new();
    assert!(compute_qureg(&env));

    let handle = thread::spawn(move || {
        assert!(compute_qureg(&env));
        env
    });

    let env = handle.join().unwrap();
    assert!(compute_qureg(&env));
}

#[test]
fn send_questenv_02() {
    let env = QuestEnv::new();
    assert!(compute_qureg(&env));

    let env = thread::spawn(move || {
        assert!(compute_qureg(&env));
        env
    })
    .join()
    .unwrap();

    let env = thread::spawn(move || {
        assert!(compute_qureg(&env));
        env
    })
    .join()
    .unwrap();

    assert!(compute_qureg(&env));
}
