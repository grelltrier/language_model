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
/// Test the language model
fn test_model() {
    convert_text_to_cmprssd_bin(true).unwrap();
    println!("Loading language model from file...");
    let fname_language_model = "./ngrams_test/language_model.bin";
    let language_model = LanguageModel::read(fname_language_model).unwrap();
    println!("Done loading");
    println!();

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
