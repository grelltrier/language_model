use language_model::*;

fn main() {
    //convert_text_to_cmprssd_bin(false).unwrap();
    println!("Loading language model from file...");
    let fname_language_model = "./ngrams/language_model.bin";
    let language_model = LanguageModel::read(fname_language_model).unwrap();
    println!("Done loading");
    println!();

    let entered_text = vec!["hello", "how", "are", "you"];
    print_predicions(entered_text, &language_model);

    println!();
    println!();

    let entered_text = vec!["the", "man", "walked", "down", "the", "street"];
    print_predicions(entered_text, &language_model);
}

fn print_predicions(entered_text: Vec<&str>, language_model: &LanguageModel) {
    println!("Entered text:");
    println!("{:?}", entered_text);
    println!();
    let mut entered_text = entered_text.iter();
    let mut lm_state = LMState::default();

    for no_prediction in 0..=entered_text.len() {
        println!("Current state {:?}", lm_state);
        println!();
        let predictions = language_model.predict(lm_state, 10);
        println!("Prediction after reading {} input symbols", no_prediction);
        println!("{:?}", predictions);
        println!();

        if let Some(symbol) = entered_text.next() {
            lm_state = language_model.get_next_state(lm_state, symbol);
            println!("Entered symbol {}", symbol);
            println!();
        }
    }
}
