use std::fs::OpenOptions;
use std::io::Write;
use pyo3::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::ops::Neg;

const LAYOUTS: [&str; 4] = ["1234567890", "qwertyuiop", "asdfghjkl", "zxcvbnm"];

const MATCH:         i32 = 1;
const GAP:           i32 = -1;
const ADD:           i32 = -2;
const DELETE:        i32 = -2;
const MISMATCH:      i32 = -15;

type SVec<T> = SmallVec<[T; 64]>;

fn distance(x: u8, y: u8) -> i32 {
    match (x.is_ascii_uppercase(), y.is_ascii_uppercase()) {
        (false, true)  => return MISMATCH,
        (true, false)  => return MISMATCH,
        _ => {}
    }

    let find_coords = |c: u8| {
        LAYOUTS
            .iter()
            .enumerate()
            .find_map(|(idx, layout)| {
                layout
                    .find(c.to_ascii_lowercase() as char)
                    .map(|p| (idx as i32, p as i32))
            })
    };
    let x_coords = find_coords(x);
    let y_coords = find_coords(y);

    match (x_coords, y_coords) {
        (Some(x_coords), Some(y_coords)) => {
            let level_dif = (x_coords.0 - y_coords.0).abs();
            let pos_dif = (x_coords.1 - y_coords.1).abs();

            MATCH - (level_dif + pos_dif)
        }
        _ => MISMATCH,
    }
}

fn nw_score<'a>(x: &'a str, y: &'a str) -> SVec<i32> {
    let mut cur: SVec<i32> = ((y.len() as i32).neg()..=0).rev().collect();
    for x_char in x.bytes() {
        let prev = cur.clone();
        for (j, y_char) in y.bytes().enumerate() {
            let add = prev[j + 1] + ADD;
            let delete = cur[j] + DELETE;
            let change = prev[j] + if x_char != y_char {
                distance(x_char, y_char)
            } else {
                MATCH
            };

            cur[j + 1] = *[add, delete, change].iter().max().unwrap();
        }

        cur[0] = prev[0] + GAP
    }

    cur
}

fn rust_hirschberg(x: &str, y: &str) -> i32 {
    let x_len = x.len();
    let y_len = y.len();

    if x_len <= 1 || y_len <= 1 {
        nw_score(x, y).into_iter().max().unwrap()
    } else {
        let mid = x_len / 2;
        let (xf, xs) = x.split_at(mid);

        let xs_rev_bytes: SVec<u8> = xs.bytes().rev().collect();
        let xs_rev = std::str::from_utf8(xs_rev_bytes.as_slice()).unwrap();

        let y_rev_bytes: SVec<u8> = y.bytes().rev().collect();
        let y_rev = std::str::from_utf8(y_rev_bytes.as_slice()).unwrap();

        let fh = nw_score(xf, y);
        let sh = nw_score(xs_rev, y_rev);

        let sums: SVec<i32> = fh
            .into_iter()
            .zip(sh.into_iter().rev())
            .map(|(f, s)| f + s)
            .collect();
        let (split_idx, _) = sums
            .into_iter()
            .enumerate()
            .max_by(|f, s| f.1.cmp(&s.1))
            .unwrap();

        let (yf, ys) = y.split_at(split_idx);
        let x_score = rust_hirschberg(xf, yf);
        let y_score = rust_hirschberg(xs, ys);

        x_score + y_score
    }
}

#[pyfunction]
fn try_correct(dictionary: Vec<&str>, text: &str) -> String {
    let (answer_idx, score) = dictionary
        .par_iter()
        .map(|word| rust_hirschberg(word, text))
        .enumerate()
        .max_by(|f, s| f.1.cmp(&s.1))
        .unwrap();

    let threshold= text.len() as i32;
    if score > threshold || score < -threshold  {
        text.to_string()
    } else {
        dictionary[answer_idx].to_string()
    }
}

#[pyfunction]
fn try_test_correct(dictionary: Vec<&str>, text: &str, answer: &str) -> String {
    let (answer_idx, score) = dictionary
        .par_iter()
        .map(|word| rust_hirschberg(word, text))
        .enumerate()
        .max_by(|f, s| f.1.cmp(&s.1))
        .unwrap();

    if answer != dictionary[answer_idx] {
        let mut file = OpenOptions::new().append(true).create(true).open("/home/naymoll/Downloads/some.txt").unwrap();
        writeln!(file, "Text: {text} | Answer: {answer} | Score: {score} | Predict: {}", dictionary[answer_idx]).unwrap();
    }

    let threshold= text.len() as i32;
    if score > threshold || score < -threshold  {
        text.to_string()
    } else {
        dictionary[answer_idx].to_string()
    }
}

#[pymodule]
fn hirschberg(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(try_correct, m)?)?;
    m.add_function(wrap_pyfunction!(try_test_correct, m)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::path::Path;

    use crate::try_correct;

    fn read_from_csv<P: AsRef<Path>>(path: P) -> Vec<String> {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_path(path)
            .unwrap();

        let mut dict = HashSet::new();
        for record in reader.records() {
            for field in record.unwrap().iter() {
                if !dict.contains(field) {
                    dict.insert(field.to_string());
                }
            }
        }

        dict.into_iter().collect()
    }

    #[test]
    fn try_correct_test() {
        let rust_dictionary =
            read_from_csv("/home/naymoll/Projects/NLP/projects/msokolov/assets/dictionary.csv");
        let test_tokens =
            read_from_csv("/home/naymoll/Projects/NLP/projects/msokolov/assets/incorrect.csv");

        let dictionary: Vec<_> = rust_dictionary.iter().map(|s| s.as_str()).collect();
        for token in test_tokens {
            let answer = try_correct(dictionary.clone(), token.as_str());
            println!("{answer:?}")
        }
    }
}
