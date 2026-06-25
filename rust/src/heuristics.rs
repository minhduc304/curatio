//! Heuristic filters — mirror of src/curate/heuristics.py for parity (NFR5). TDD §4.2.
//!
//! Pure function; rules checked in a fixed order, first failure's reason returned.
//! Constants mirror CurationConfig defaults (src/models.py).

const MIN_CHARS: usize = 200;
const MAX_CHARS: usize = 100_000;
const MAX_SYMBOL_RATIO: f64 = 0.10;
const MIN_MEAN_LINE_LEN: f64 = 20.0;

const BOILERPLATE_MARKERS: [&str; 4] = [
    "lorem ipsum",
    "all rights reserved",
    "terms of service",
    "enable javascript",
];

/// Split like Python's str.splitlines() for the common cases (\n and \r\n):
/// split on '\n', drop a trailing '\r' from each piece. Line terminators are
/// excluded from the returned lengths, matching CPython.
fn split_lines(text: &str) -> Vec<&str> {
    text.split('\n')
        .map(|ln| ln.strip_suffix('\r').unwrap_or(ln))
        .collect()
}

/// Return Some(reason) to reject, or None to pass. Matches the Python rules and
/// ordering exactly so decisions are identical (SC6).
pub fn check(text: &str) -> Option<&'static str> {
    let n = text.chars().count();

    if n < MIN_CHARS {
        return Some("too_short");
    }
    if n > MAX_CHARS {
        return Some("too_long");
    }

    // Symbol ratio: non-alphanumeric, non-whitespace chars over total.
    let symbols = text
        .chars()
        .filter(|c| !c.is_alphanumeric() && !c.is_whitespace())
        .count();
    if symbols as f64 / n as f64 > MAX_SYMBOL_RATIO {
        return Some("symbol_ratio");
    }

    // Mean line length: nav bars / link lists have many tiny lines.
    let lines = split_lines(text);
    let lines: &[&str] = if lines.is_empty() { &[text] } else { &lines };
    let total: usize = lines.iter().map(|ln| ln.chars().count()).sum();
    let mean_line_len = total as f64 / lines.len() as f64;
    if mean_line_len < MIN_MEAN_LINE_LEN {
        return Some("short_lines");
    }

    let lowered = text.to_lowercase();
    if BOILERPLATE_MARKERS.iter().any(|m| lowered.contains(m)) {
        return Some("boilerplate");
    }

    // Language: target is English. whatlang's Lang::Eng <-> langid's "en".
    let english = whatlang::detect(text)
        .map(|info| info.lang() == whatlang::Lang::Eng)
        .unwrap_or(false);
    if !english {
        return Some("wrong_language");
    }

    None
}
