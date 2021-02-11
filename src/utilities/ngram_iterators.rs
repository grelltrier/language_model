use std::fs::File;
use std::io::Lines;
use std::io::{BufRead, BufReader};

use super::*;

struct LinesIterator {
    lines: Lines<BufReader<File>>,
}

impl LinesIterator {
    fn new(filename: &str) -> Self {
        // Open the file in read-only mode.
        let file = File::open(filename).unwrap();
        let buf_reader = BufReader::new(file);
        let lines = buf_reader.lines();
        LinesIterator { lines }
    }
}

impl Iterator for LinesIterator {
    type Item = String;
    fn next(&mut self) -> Option<String> {
        if let Some(Ok(line)) = self.lines.next() {
            Some(line)
        } else {
            None
        }
    }
}

pub struct SymtIterator {
    lines_iterator: LinesIterator,
}

impl SymtIterator {
    pub fn new(filename: &str) -> Self {
        Self {
            lines_iterator: LinesIterator::new(filename),
        }
    }
}

impl Iterator for SymtIterator {
    type Item = Symbol;
    fn next(&mut self) -> Option<Self::Item> {
        // If the end of the file was reached, return None
        if let Some(line) = self.lines_iterator.next() {
            let symbol = line.trim().parse::<Symbol>().unwrap();
            Some(symbol)
        } else {
            None
        }
    }
}

pub struct UnigramIterator {
    lines_iterator: LinesIterator,
}

impl UnigramIterator {
    pub fn new(filename: &str) -> Self {
        Self {
            lines_iterator: LinesIterator::new(filename),
        }
    }
}

impl Iterator for UnigramIterator {
    type Item = Unigram;
    fn next(&mut self) -> Option<Self::Item> {
        // If the end of the file was reached, return None
        if let Some(line) = self.lines_iterator.next() {
            let mut token = line.split_whitespace();
            let log_prob = token.next().unwrap().trim().parse::<LogProb>().unwrap();
            let offset = token.next().unwrap().trim().parse::<Offset>().unwrap();
            let no_of_ngrams = token.next().unwrap().trim().parse::<NoOfNgrams>().unwrap();

            // There should not be more tokens in the line
            // If there are, there most likely is a problem
            if token.next().is_some() {
                return None;
            }

            Some((log_prob, offset, no_of_ngrams))
        } else {
            None
        }
    }
}

pub struct BigramIterator {
    lines_iterator: LinesIterator,
}

impl BigramIterator {
    pub fn new(filename: &str) -> Self {
        Self {
            lines_iterator: LinesIterator::new(filename),
        }
    }
}

impl Iterator for BigramIterator {
    type Item = Bigram;
    fn next(&mut self) -> Option<Self::Item> {
        // If the end of the file was reached, return None
        if let Some(line) = self.lines_iterator.next() {
            let mut token = line.split_whitespace();
            let label = token.next().unwrap().trim().parse::<Label>().unwrap();
            let log_prob = token.next().unwrap().trim().parse::<LogProb>().unwrap();
            let offset = token.next().unwrap().trim().parse::<Offset>().unwrap();
            let no_of_ngrams = token.next().unwrap().trim().parse::<NoOfNgrams>().unwrap();

            // There should not be more tokens in the line
            // If there are, there most likely is a problem
            if token.next().is_some() {
                return None;
            }

            Some((label, log_prob, offset, no_of_ngrams))
        } else {
            None
        }
    }
}

pub struct TrigramIterator {
    lines_iterator: LinesIterator,
}

impl TrigramIterator {
    pub fn new(filename: &str) -> Self {
        Self {
            lines_iterator: LinesIterator::new(filename),
        }
    }
}

impl Iterator for TrigramIterator {
    type Item = Trigram;
    fn next(&mut self) -> Option<Self::Item> {
        // If the end of the file was reached, return None
        if let Some(line) = self.lines_iterator.next() {
            let mut token = line.split_whitespace();
            let label = token.next().unwrap().trim().parse::<Label>().unwrap();
            let log_prob = token.next().unwrap().trim().parse::<LogProb>().unwrap();
            let offset = token.next().unwrap().trim().parse::<Offset>().unwrap();

            // There should not be more tokens in the line
            // If there are, there most likely is a problem
            if token.next().is_some() {
                return None;
            }

            Some((label, log_prob, offset))
        } else {
            None
        }
    }
}
