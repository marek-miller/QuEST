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

// This won't compile unless QuestEnv is Sync
#[test]
fn questenv_is_sync() {
    let env = QuestEnv::new();

    thread::scope(|s| {
        s.spawn(|| env.report_quest_env());
    });
}

// Assure no data race happens here
#[test]
fn questenv_sync() {
    let env = QuestEnv::new();

    thread::scope(|s| {
        s.spawn(|| env.sync());
        s.spawn(|| env.sync());
        env.sync();
    });
}

#[test]
fn questenv_seed_01() {
    let mut env = QuestEnv::new();
    seed_quest(&mut env, &[1, 2, 3]);

    thread::scope(|s| {
        s.spawn(|| {
            let seeds = get_quest_seeds(&env);
            assert_eq!(seeds, &[1, 2, 3])
        });
    });

    seed_quest(&mut env, &[3, 2, 1]);
    thread::scope(|s| {
        s.spawn(|| {
            let seeds = get_quest_seeds(&env);
            assert_eq!(seeds, &[3, 2, 1])
        });
    });
}

#[test]
fn questenv_seed_quest_default_01() {
    let mut env = QuestEnv::new();
    seed_quest_default(&mut env);

    let seeds = get_quest_seeds(&env);
    let mut seeds_check = get_quest_seeds(&env);
    assert_eq!(seeds, seeds_check);

    thread::scope(|s| {
        s.spawn(|| seeds_check = get_quest_seeds(&env));
    });
    assert_eq!(seeds, seeds_check);
}

fn compute_qureg(env: &QuestEnv) -> bool {
    let qureg = &mut Qureg::try_new(2, env).unwrap();
    init_zero_state(qureg);
    pauli_x(qureg, 0).unwrap();
    let amp = get_real_amp(qureg, 1).unwrap();

    (amp - 1.).abs() < EPSILON
}

#[test]
fn independent_qubit_ops_01() {
    let env = QuestEnv::new();

    assert!(compute_qureg(&env));

    thread::scope(|s| {
        s.spawn(|| assert!(compute_qureg(&env)));
    });
}

#[test]
fn independent_qubit_ops_02() {
    let env = QuestEnv::new();

    thread::scope(|s| {
        s.spawn(|| assert!(compute_qureg(&env)));
        s.spawn(|| assert!(compute_qureg(&env)));
    });

    assert!(compute_qureg(&env));
}

#[test]
fn independent_qubit_ops_03() {
    const NUM_REPS: usize = 100;

    let env = QuestEnv::new();

    thread::scope(|s| {
        s.spawn(|| {
            for _ in 0..NUM_REPS {
                assert!(compute_qureg(&env))
            }
        });

        s.spawn(|| {
            for _ in 0..NUM_REPS {
                assert!(compute_qureg(&env))
            }
        });
    });

    assert!(compute_qureg(&env));
}
