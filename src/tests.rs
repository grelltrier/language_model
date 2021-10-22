use super::*;

// Check if the two Vecs are equal
fn cmp(a: Vec<(&str, f32)>, b: Vec<(&str, f32)>) -> bool {
    println!("len a: {}, len b: {}", a.len(), b.len());

    println!();
    println!("a:");
    for &(word, precentage) in &a {
        println!("  ({}, {})", word, precentage);
    }
    println!("b:");
    for &(word, precentage) in &b {
        println!("  ({}, {})", word, precentage);
    }

    if a.len() != b.len() {
        return false;
    }
    for idx in 0..a.len() {
        if b[idx].0 != b[idx].0 {
            return false;
        }
        if (a[idx].1 - b[idx].1).abs() > 0.00001 {
            return false;
        }
    }
    true
}

#[test]
/// Test case D1
/// Convert the text files to the language model and write it to a
/// compressed binary file
fn test_convert_and_load_model() {
    // Convert the text files to the language model and write it to a compressed file
    convert_text_to_cmprssd_bin(true).unwrap();
    println!("Loading language model from file...");
    let fname_language_model = "ngrams_test/language_model.bin";

    // Read it from the compressed binary file
    let language_model = LanguageModel::read(fname_language_model).unwrap();
    println!("Done loading");

    // Check if language model was loaded correctly
    let mut correct_symt = IndexSet::new();
    correct_symt.insert("a".to_string());
    correct_symt.insert("b".to_string());
    let correct_unigrams = vec![(-0.6931472, 0, 1), (-0.6931472, 1, 2)];
    let correct_bigrams = vec![
        (1, -0.40546507, 0, 2),
        (0, -0.40546507, 2, 1),
        (1, -1.0986123, 3, 1),
    ];
    let correct_trigrams = vec![
        (0, -0.6931472, 1),
        (1, -0.6931472, 2),
        (1, -0.6931472, 0),
        (0, 0.0, 1),
    ];
    let correct_lm = LanguageModel {
        symt: correct_symt,
        unigrams: correct_unigrams,
        bigrams: correct_bigrams,
        trigrams: correct_trigrams,
    };
    assert!(language_model == correct_lm);
}

#[test]
/// Test case D2
/// Test transitioning to the next state (no backoff required)
fn test_valid_transitions() {
    convert_text_to_cmprssd_bin(true).unwrap();
    println!("Loading language model from file...");
    let fname_language_model = "ngrams_test/language_model.bin";
    let language_model = LanguageModel::read(fname_language_model).unwrap();
    println!("Done loading");
    println!();

    let mut start_state;
    let mut dest_state;
    let mut correct_state;

    // Transitions from the initial state (state 0)
    println!("Transitions from the initial state (state 0)");
    start_state = LMState::default();
    correct_state = get_test_state_no(0);
    println!("correct: {:?}", correct_state);
    println!("dest   : {:?}", start_state);
    println!();
    assert!(start_state == correct_state);

    // Transition from state 0 to 1
    println!("Transition from state 0 to 1");
    dest_state = language_model.get_next_state(start_state, "a");
    correct_state = get_test_state_no(1);
    println!("correct: {:?}", correct_state);
    println!("dest   : {:?}", dest_state);
    println!();
    assert!(dest_state == correct_state);

    // Transition from state 0 to 2
    println!("Transition from state 0 to 2");
    dest_state = language_model.get_next_state(start_state, "b");
    correct_state = get_test_state_no(2);
    println!("correct: {:?}", correct_state);
    println!("dest   : {:?}", dest_state);
    println!();
    assert!(dest_state == correct_state);

    // Transition from state 1 to 3
    println!("Transition from state 1 to 3");
    start_state = get_test_state_no(1);
    dest_state = language_model.get_next_state(start_state, "b");
    correct_state = get_test_state_no(3);
    println!("correct: {:?}", correct_state);
    println!("dest   : {:?}", dest_state);
    println!();
    assert!(dest_state == correct_state);

    // Transition from state 2 to 4
    println!("Transition from state 2 to 4");
    start_state = get_test_state_no(2);
    dest_state = language_model.get_next_state(start_state, "a");
    correct_state = get_test_state_no(4);
    println!("correct: {:?}", correct_state);
    println!("dest   : {:?}", dest_state);
    println!();
    assert!(dest_state == correct_state);

    // Transition from state 2 to 5
    println!("Transition from state 2 to 5");
    start_state = get_test_state_no(2);
    dest_state = language_model.get_next_state(start_state, "b");
    correct_state = get_test_state_no(5);
    println!("correct: {:?}", correct_state);
    println!("dest   : {:?}", dest_state);
    println!();
    assert!(dest_state == correct_state);

    // Transition from state 3 to 4
    println!("Transition from state 3 to 4");
    start_state = get_test_state_no(3);
    dest_state = language_model.get_next_state(start_state, "a");
    correct_state = get_test_state_no(4);
    println!("correct: {:?}", correct_state);
    println!("dest   : {:?}", dest_state);
    println!();
    assert!(dest_state == correct_state);

    // Transition from state 3 to 5
    println!("Transition from state 3 to 5");
    start_state = get_test_state_no(3);
    dest_state = language_model.get_next_state(start_state, "b");
    correct_state = get_test_state_no(5);
    println!("correct: {:?}", correct_state);
    println!("dest   : {:?}", dest_state);
    println!();
    assert!(dest_state == correct_state);

    // Transition from state 4 to 3
    println!("Transition from state 4 to 3");
    start_state = get_test_state_no(4);
    dest_state = language_model.get_next_state(start_state, "b");
    correct_state = get_test_state_no(3);
    println!("correct: {:?}", correct_state);
    println!("dest   : {:?}", dest_state);
    println!();
    assert!(dest_state == correct_state);

    // Transition from state 5 to 4
    println!("Transition from state 5 to 4");
    start_state = get_test_state_no(5);
    dest_state = language_model.get_next_state(start_state, "a");
    correct_state = get_test_state_no(4);
    println!("correct: {:?}", correct_state);
    println!("dest   : {:?}", dest_state);
    println!();
    assert!(dest_state == correct_state);
}

#[test]
/// Test case D3
/// Test transitioning to the next state (backoff required)
fn test_invalid_transitions() {
    convert_text_to_cmprssd_bin(true).unwrap();
    println!("Loading language model from file...");
    let fname_language_model = "ngrams_test/language_model.bin";
    let language_model = LanguageModel::read(fname_language_model).unwrap();
    println!("Done loading");
    println!();

    let mut lm_state;
    let mut correct_state;

    // Start in state 1 and read input label "a"
    println!("Start in state 1 and read input label \"a\"");
    lm_state = get_test_state_no(1);
    lm_state = language_model.get_next_state(lm_state, "a");
    correct_state = get_test_state_no(1);
    println!("correct: {:?}", correct_state);
    println!("dest   : {:?}", lm_state);
    println!();
    assert!(lm_state == correct_state);

    // Start in state 4 and read input label "a"
    println!("Start in state 4 and read input label \"a\"");
    lm_state = get_test_state_no(4);
    lm_state = language_model.get_next_state(lm_state, "a");
    correct_state = get_test_state_no(1);
    println!("correct: {:?}", correct_state);
    println!("dest   : {:?}", lm_state);
    println!();
    assert!(lm_state == correct_state);

    // Start in state 5 and read input label "b"
    println!("Start in state 5 and read input label \"b\"");
    lm_state = get_test_state_no(5);
    lm_state = language_model.get_next_state(lm_state, "b");
    correct_state = get_test_state_no(5);
    println!("correct: {:?}", correct_state);
    println!("dest   : {:?}", lm_state);
    println!();
    assert!(lm_state == correct_state);
}

#[test]
/// Test case D4
/// Backoff to the state associated with the suffix
fn test_backoff() {
    convert_text_to_cmprssd_bin(true).unwrap();
    println!("Loading language model from file...");
    let fname_language_model = "ngrams_test/language_model.bin";
    let language_model = LanguageModel::read(fname_language_model).unwrap();
    println!("Done loading");
    println!();

    let mut lm_state;
    let mut correct_state;

    // Backoff from the initial state (state 0)
    println!("Backoff from the initial state (state 0)");
    correct_state = get_test_state_no(0);
    lm_state = get_test_state_no(0);
    lm_state = language_model.backoff(lm_state);
    println!("correct: {:?}", correct_state);
    println!("dest   : {:?}", lm_state);
    println!();
    assert!(lm_state == correct_state);

    // Backoff from state 1
    println!("Backoff from state 1");
    correct_state = get_test_state_no(0);
    lm_state = get_test_state_no(1);
    lm_state = language_model.backoff(lm_state);
    println!("correct: {:?}", correct_state);
    println!("dest   : {:?}", lm_state);
    println!();
    assert!(lm_state == correct_state);

    // Backoff from state 2
    println!("Backoff from state 2");
    correct_state = get_test_state_no(0);
    lm_state = get_test_state_no(2);
    lm_state = language_model.backoff(lm_state);
    println!("correct: {:?}", correct_state);
    println!("dest   : {:?}", lm_state);
    println!();
    assert!(lm_state == correct_state);

    // Backoff from state 3
    println!("Backoff from state 3");
    correct_state = get_test_state_no(2);
    lm_state = get_test_state_no(3);
    lm_state = language_model.backoff(lm_state);
    println!("correct: {:?}", correct_state);
    println!("dest   : {:?}", lm_state);
    println!();
    assert!(lm_state == correct_state);

    // Backoff from state 4
    println!("Backoff from state 4");
    correct_state = get_test_state_no(1);
    lm_state = get_test_state_no(4);
    lm_state = language_model.backoff(lm_state);
    println!("correct: {:?}", correct_state);
    println!("dest   : {:?}", lm_state);
    println!();
    assert!(lm_state == correct_state);

    // Backoff from state 5
    println!("Backoff from state 5");
    correct_state = get_test_state_no(2);
    lm_state = get_test_state_no(5);
    lm_state = language_model.backoff(lm_state);
    println!("correct: {:?}", correct_state);
    println!("dest   : {:?}", lm_state);
    println!();
    assert!(lm_state == correct_state);
}

#[test]
/// Test case D5
/// Test transitions, backoffs and predictions
fn test_transitions_and_backoffs() {
    convert_text_to_cmprssd_bin(true).unwrap();
    println!("Loading language model from file...");
    let fname_language_model = "ngrams_test/language_model.bin";
    let language_model = LanguageModel::read(fname_language_model).unwrap();
    println!("Done loading");
    println!();

    let mut predictions;
    let mut lm_state;
    let mut correct_prediction;
    let mut correct_state;

    // Start in initial state
    lm_state = LMState::default();
    correct_state = LMState {
        last_processed_label: 0,
        ngrams_offset: 0,
        ngrams_no: usize::MAX,
        context_len: LMContext::Zero,
    };
    assert!(lm_state == correct_state);
    predictions = language_model.predict(lm_state, 10);
    correct_prediction = vec![("a", -0.6931472), ("b", -0.6931472)];
    assert!(cmp(predictions, correct_prediction));
    println!("Initial state okay");
    println!();

    // Transition to state 1
    lm_state = language_model.get_next_state(lm_state, "a");
    correct_state = LMState {
        last_processed_label: 0,
        ngrams_offset: 0,
        ngrams_no: 1,
        context_len: LMContext::One,
    };
    assert!(lm_state == correct_state);
    predictions = language_model.predict(lm_state, 10);
    correct_prediction = vec![("b", -0.40546507), ("a", -1.60943796)];
    assert!(cmp(predictions, correct_prediction));
    println!("State 1 okay");
    println!();

    // Transition to state 3
    lm_state = language_model.get_next_state(lm_state, "b");
    correct_state = LMState {
        last_processed_label: 1,
        ngrams_offset: 0,
        ngrams_no: 2,
        context_len: LMContext::Two,
    };
    assert!(lm_state == correct_state);
    predictions = language_model.predict(lm_state, 10);
    correct_prediction = vec![("a", -0.6931472), ("b", -0.6931472)];
    assert!(cmp(predictions, correct_prediction));
    println!("State 3 okay");
    println!();

    // Transition to state 5
    lm_state = language_model.get_next_state(lm_state, "b");
    correct_state = LMState {
        last_processed_label: 1,
        ngrams_offset: 3,
        ngrams_no: 1,
        context_len: LMContext::Two,
    };
    assert!(lm_state == correct_state);
    predictions = language_model.predict(lm_state, 10);
    correct_prediction = vec![("a", 0.0), ("b", -2.01490306)];
    assert!(cmp(predictions, correct_prediction));
    println!("State 5 okay");
    println!();

    // Backoff to state 2
    lm_state = language_model.backoff(lm_state);
    correct_state = LMState {
        last_processed_label: 1,
        ngrams_offset: 1,
        ngrams_no: 2,
        context_len: LMContext::One,
    };
    assert!(lm_state == correct_state);
    predictions = language_model.predict(lm_state, 10);
    correct_prediction = vec![("a", -0.40546507), ("b", -1.0986123)];
    assert!(cmp(predictions, correct_prediction));
    println!("Backoff okay");
    println!();
}

fn get_test_state_no(state_no: usize) -> LMState {
    let last_processed_label;
    let ngrams_offset;
    let ngrams_no;
    let context_len;

    match state_no {
        0 => {
            last_processed_label = 0;
            ngrams_offset = 0;
            ngrams_no = usize::MAX;
            context_len = LMContext::Zero
        }
        1 => {
            last_processed_label = 0;
            ngrams_offset = 0;
            ngrams_no = 1;
            context_len = LMContext::One
        }
        2 => {
            last_processed_label = 1;
            ngrams_offset = 1;
            ngrams_no = 2;
            context_len = LMContext::One
        }
        3 => {
            last_processed_label = 1;
            ngrams_offset = 0;
            ngrams_no = 2;
            context_len = LMContext::Two
        }
        4 => {
            last_processed_label = 0;
            ngrams_offset = 2;
            ngrams_no = 1;
            context_len = LMContext::Two
        }
        5 => {
            last_processed_label = 1;
            ngrams_offset = 3;
            ngrams_no = 1;
            context_len = LMContext::Two
        }
        _ => {
            println!("Asked for the invalid state no: {}", state_no);
            panic!()
        }
    }

    LMState {
        last_processed_label,
        ngrams_offset,
        ngrams_no,
        context_len,
    }
}
