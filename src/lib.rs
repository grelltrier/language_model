#[macro_use]
extern crate serde_derive;
extern crate bincode;

use flate2::{bufread::GzDecoder, write::GzEncoder, Compression};
use indexmap::IndexSet;
use std::{cmp::Ordering, collections::HashMap, fs::File, io::BufReader, iter::FromIterator};

pub mod utilities;
use utilities::*;

const BACKOFF_WEIGHT: f32 = -0.916290731874155; // ln(0.4)

type Symbol = String;
type Label = u32;
type LogProb = f32;
type Offset = u32;
type NoOfNgrams = u16;

type Unigram = (LogProb, Offset, NoOfNgrams);
type Bigram = (Label, LogProb, Offset, NoOfNgrams);
type Trigram = (Label, LogProb, Offset);

#[derive(Copy, Clone, PartialEq, Debug)]
enum LMContext {
    Zero,
    One,
    Two,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct LMState {
    last_processed_label: Label,
    ngrams_offset: usize,
    ngrams_no: usize,
    context_len: LMContext,
}

impl Default for LMState {
    fn default() -> Self {
        Self {
            last_processed_label: 0,
            ngrams_offset: 0,
            ngrams_no: usize::MAX,
            context_len: LMContext::Zero,
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct LanguageModel {
    symt: IndexSet<String>,
    unigrams: Vec<Unigram>,
    bigrams: Vec<Bigram>,
    trigrams: Vec<Trigram>,
}
impl LanguageModel {
    pub fn read_from_text(
        fname_symt: &str,
        fname_unigrams: &str,
        fname_bigrams: &str,
        fname_trigrams: &str,
    ) -> Self {
        let mut symt = IndexSet::new();
        for symbol in SymtIterator::new(fname_symt) {
            symt.insert(symbol);
        }
        symt.shrink_to_fit();
        let mut unigrams = Vec::new();
        for unigram in UnigramIterator::new(fname_unigrams) {
            unigrams.push(unigram);
        }
        unigrams.shrink_to_fit();
        let mut bigrams = Vec::new();
        for bigram in BigramIterator::new(fname_bigrams) {
            bigrams.push(bigram);
        }
        bigrams.shrink_to_fit();
        let mut trigrams = Vec::new();
        for trigram in TrigramIterator::new(fname_trigrams) {
            trigrams.push(trigram);
        }
        trigrams.shrink_to_fit();

        Self {
            symt,
            unigrams,
            bigrams,
            trigrams,
        }
    }

    pub fn write(&self, fname: &str) -> Result<(), Box<bincode::ErrorKind>> {
        let file = File::create(fname).unwrap();
        let encoder = GzEncoder::new(file, Compression::default());
        bincode::serialize_into(encoder, self)
    }

    pub fn read(fname: &str) -> Result<Self, Box<bincode::ErrorKind>> {
        let file = File::open(fname).unwrap();
        let buf_reader = BufReader::new(file);
        let decoder = GzDecoder::new(buf_reader);
        bincode::deserialize_from(decoder)
    }

    pub fn predict(&self, lm_state: LMState, max_no_predictions: usize) -> Vec<(&String, LogProb)> {
        let mut lm_state = lm_state;

        let mut predictions = HashMap::with_capacity(max_no_predictions);
        let mut backoff = 0;

        if lm_state.context_len == LMContext::Two {
            for (label, log_prob, _) in self
                .trigrams
                .iter()
                .skip(lm_state.ngrams_offset)
                .take(lm_state.ngrams_no)
            {
                predictions.entry(*label).or_insert(*log_prob);
                // predictions.insert(*label, *log_prob);
            }
            // TODO: this condition could be improved by backing off if the lowest likelihood is lower than the backoff likelihood
            if predictions.len() < max_no_predictions {
                lm_state = self.backoff(lm_state);
                backoff += 1;
            }
        }
        if lm_state.context_len == LMContext::One {
            let backoff_penalty = backoff as f32 * BACKOFF_WEIGHT;
            for (label, log_prob, _, _) in self
                .bigrams
                .iter()
                .skip(lm_state.ngrams_offset)
                .take(lm_state.ngrams_no)
            {
                // The probabilities are added because they are the neg log probs
                predictions
                    .entry(*label)
                    .or_insert(*log_prob + backoff_penalty);
                //predictions.insert(*label, *log_prob + backoff_penalty);
            }
            if predictions.len() < max_no_predictions {
                lm_state = self.backoff(lm_state);
                backoff += 1;
            }
        }

        if lm_state.context_len == LMContext::Zero {
            let backoff_penalty = backoff as f32 * BACKOFF_WEIGHT;
            for (idx, (log_prob, _, _)) in self.unigrams.iter().enumerate() {
                // The probabilities are added because they are the neg log probs
                predictions
                    .entry(idx as u32)
                    .or_insert(*log_prob + backoff_penalty);
                //predictions.insert(idx as u32, *log_prob + backoff_penalty);
            }
        }

        let mut predictions = Vec::from_iter(predictions);
        predictions.sort_by(|&(_, a), &(_, b)| b.partial_cmp(&a).unwrap_or(Ordering::Equal));

        // Translate the labels into symbols
        let mut final_predictions = Vec::new();
        let mut symbol;
        for (label, log_prob) in predictions.iter().take(max_no_predictions) {
            symbol = &self.symt[*label as usize];
            final_predictions.push((symbol, *log_prob));
        }
        final_predictions
    }

    pub fn get_next_state(&self, lm_state: LMState, symbol: &str) -> LMState {
        let mut lm_state = lm_state;

        // Try to translate the symbol into a label
        // If we can't find the symbol, it is not a known word so the next state is the starting state
        let label = match self.symt.get_index_of(symbol) {
            Some(known_label) => known_label,
            None => return LMState::default(),
        };

        if lm_state.context_len == LMContext::Two {
            if let Some(new_state) = self.try_finding_trigram_trs(label as u32, lm_state) {
                return new_state;
            } else {
                lm_state = self.backoff(lm_state);
            }
        }
        if lm_state.context_len == LMContext::One {
            if let Some(new_state) = self.try_finding_bigram_trs(label as u32, lm_state) {
                return new_state;
            } else {
                lm_state = self.backoff(lm_state);
            }
        }
        self.finding_unigram_trs(label, lm_state)
    }

    fn backoff(&self, start_state: LMState) -> LMState {
        // Destructure the state
        let LMState {
            last_processed_label,
            context_len,
            ..
        } = start_state;

        match context_len {
            // It doesn't make sense to backoff when being in the start state. This should never happen, but it is also not an error
            LMContext::Zero => start_state,
            // If we use only one word and backoff, we land in the start state of the language model
            LMContext::One => LMState::default(),
            // To backup from a context of two, we look up where to find the unigram, that represents the last processed label
            LMContext::Two => {
                let last_processed_label = last_processed_label as usize;
                LMState {
                    last_processed_label: 0,
                    ngrams_offset: self.unigrams[last_processed_label].1 as usize,
                    ngrams_no: self.unigrams[last_processed_label as usize].2 as usize,
                    context_len: LMContext::One,
                }
            }
        }
    }

    fn try_finding_trigram_trs(&self, label: u32, lm_state: LMState) -> Option<LMState> {
        let ngrams_offset = lm_state.ngrams_offset;
        let ngrams_no = lm_state.ngrams_no;

        match self.trigrams[ngrams_offset..ngrams_offset + ngrams_no]
            .binary_search_by_key(&(label as u32), |&(a, _, _)| a)
        {
            Ok(idx) => {
                let offset_in_bigrams = self.trigrams[ngrams_offset + idx].2 as usize;
                Some(LMState {
                    last_processed_label: label,
                    ngrams_offset: self.bigrams[offset_in_bigrams].2 as usize,
                    ngrams_no: self.bigrams[offset_in_bigrams].3 as usize,
                    context_len: LMContext::Two,
                })
            }
            Err(_) => None,
        }
    }

    fn try_finding_bigram_trs(&self, label: u32, lm_state: LMState) -> Option<LMState> {
        let ngrams_offset = lm_state.ngrams_offset;
        let ngrams_no = lm_state.ngrams_no;

        match self.bigrams[ngrams_offset..ngrams_offset + ngrams_no]
            .binary_search_by_key(&(label as u32), |&(a, _, _, _)| a)
        {
            Ok(idx) => Some(LMState {
                last_processed_label: label,
                ngrams_offset: self.bigrams[ngrams_offset + idx].2 as usize,
                ngrams_no: self.bigrams[ngrams_offset + idx].3 as usize,
                context_len: LMContext::Two,
            }),
            Err(_) => None,
        }
    }

    fn finding_unigram_trs(&self, label: usize, lm_state: LMState) -> LMState {
        let ngrams_offset = lm_state.ngrams_offset;
        LMState {
            last_processed_label: label as u32,
            ngrams_offset: self.unigrams[ngrams_offset].1 as usize,
            ngrams_no: self.unigrams[ngrams_offset].2 as usize,
            context_len: LMContext::One,
        }
    }
}

pub fn convert_text_to_cmprssd_bin(test_mode: bool) -> Result<(), Box<bincode::ErrorKind>> {
    let folder = if test_mode {
        "./ngrams_test/"
    } else {
        "./ngrams/"
    };
    let fname_symt = format!("{}symt.txt", folder);
    let fname_unigrams = format!("{}{}gms.txt", folder, 1);
    let fname_bigrams = format!("{}{}gms.txt", folder, 2);
    let fname_trigrams = format!("{}{}gms.txt", folder, 3);
    let fname_write_bin = format!("{}language_model.bin", folder);

    println!("Test mode: {}", test_mode);
    println!("Reading ngrams from the files:");
    println!("symt     : {}", fname_symt);
    println!("unigrams : {}", fname_unigrams);
    println!("bigrams  : {}", fname_bigrams);
    println!("trigrams : {}", fname_trigrams);
    println!();

    let language_model = LanguageModel::read_from_text(
        &fname_symt,
        &fname_unigrams,
        &fname_bigrams,
        &fname_trigrams,
    );

    println!("Language model successfully read from file");
    language_model.write(&fname_write_bin)?;
    println!("Language model successfully written to file");
    Ok(())
}
