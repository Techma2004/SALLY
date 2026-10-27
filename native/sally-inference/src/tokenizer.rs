//! Byte-level BPE tokenizer built from GGUF metadata, a streaming UTF-8 decoder,
//! and the Llama 3 instruct chat template.
//!
//! Supported: `tokenizer.ggml.model == "gpt2"` with the Llama 3 pre-tokenizer
//! (`tokenizer.ggml.pre` == "llama-bpe", or absent). Anything else is an
//! `UnsupportedArchitecture` error naming the value; there is no silent fallback.
//!
//! Fidelity notes (checked against llama.cpp `src/llama-vocab.cpp`, master):
//! - Pre-tokenizer pattern: llama.cpp's LLAMA3 expression, with the contraction
//!   alternatives matched ASCII-case-insensitively, as `'[sS]|'[tT]|...` does there.
//! - A pre-token that is itself a vocabulary entry is emitted directly
//!   (`ignore_merges`, which llama.cpp enables for llama-bpe).
//! - Otherwise merges are applied lowest rank first, leftmost on ties.
//! - After merging, a symbol missing from the vocabulary falls back to one token
//!   per byte-character. llama.cpp silently drops a byte that has no token;
//!   here construction fails with `InvalidModel` unless all 256 byte tokens exist,
//!   so the fallback can never drop input.
//! - Special tokens (`token_type == 3`) are matched literally in the input text,
//!   leftmost, longest first, before BPE. This means user text containing e.g.
//!   `<|eot_id|>` becomes the control token (same as llama.cpp with parse_special).
//! - If `tokenizer.ggml.pre` is absent, llama.cpp would use its "default" regex and
//!   warn; this crate treats absent as llama-bpe as instructed. Do not rely on
//!   that for models that are not Llama 3 family.

use crate::error::EngineError;
use crate::gguf::Metadata;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

const TOKEN_TYPE_CONTROL: i64 = 3;
const MAX_VOCAB: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Copy)]
struct ChatIds {
    eot: u32,
}

#[derive(Debug)]
pub struct Tokenizer {
    /// Raw bytes per token (special tokens: their literal text).
    pieces: Vec<Vec<u8>>,
    control: Vec<bool>,
    /// Byte-encoded token text -> id (normal tokens and specials alike).
    text_to_id: HashMap<String, u32>,
    /// "left right" -> rank.
    ranks: HashMap<String, u32>,
    /// Special tokens, longest text first.
    specials: Vec<(Vec<u8>, u32)>,
    special_first_byte: [bool; 256],
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    bos: Option<u32>,
    eos: Option<u32>,
    eog: Vec<u32>,
    chat: Option<ChatIds>,
}

fn invalid(msg: impl Into<String>) -> EngineError {
    EngineError::InvalidModel(msg.into())
}

/// GPT-2 `bytes_to_unicode`: printable bytes map to themselves, the rest to U+0100...
fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next = 256u32;
    for b in 0..=255u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174;
        let cp = if printable {
            b
        } else {
            next += 1;
            next - 1
        };
        // All code points here are valid scalar values (< 0x200).
        table[b as usize] = char::from_u32(cp).unwrap_or('\u{FFFD}');
    }
    table
}

impl Tokenizer {
    pub fn from_gguf(m: &Metadata) -> Result<Tokenizer, EngineError> {
        let model = m
            .get_str("tokenizer.ggml.model")
            .ok_or_else(|| invalid("missing tokenizer.ggml.model"))?;
        if model != "gpt2" {
            return Err(EngineError::UnsupportedArchitecture(format!(
                "tokenizer.ggml.model {model:?} (only \"gpt2\" byte-level BPE is supported)"
            )));
        }
        if m.get("tokenizer.ggml.pre").is_some() && m.get_str("tokenizer.ggml.pre").is_none() {
            return Err(invalid("tokenizer.ggml.pre is not a string"));
        }
        match m.get_str("tokenizer.ggml.pre") {
            None | Some("llama-bpe") => {}
            Some(other) => {
                return Err(EngineError::UnsupportedArchitecture(format!(
                    "tokenizer.ggml.pre {other:?} (only \"llama-bpe\" is supported)"
                )))
            }
        }

        let tokens = m
            .get_str_array("tokenizer.ggml.tokens")
            .ok_or_else(|| invalid("missing or non-string tokenizer.ggml.tokens"))?;
        if tokens.is_empty() || tokens.len() > MAX_VOCAB {
            return Err(invalid(format!(
                "implausible vocabulary size {}",
                tokens.len()
            )));
        }
        let types = match m.get("tokenizer.ggml.token_type") {
            None => None,
            Some(_) => Some(
                m.get_int_array("tokenizer.ggml.token_type")
                    .ok_or_else(|| invalid("tokenizer.ggml.token_type is not an integer array"))?,
            ),
        };
        if let Some(t) = &types {
            if t.len() != tokens.len() {
                return Err(invalid(format!(
                    "token_type has {} entries for {} tokens",
                    t.len(),
                    tokens.len()
                )));
            }
        }
        let merges = m
            .get_str_array("tokenizer.ggml.merges")
            .ok_or_else(|| invalid("missing or non-string tokenizer.ggml.merges"))?;

        let byte_to_char = bytes_to_unicode();
        let char_to_byte: HashMap<char, u8> = byte_to_char
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();

        let mut pieces = Vec::with_capacity(tokens.len());
        let mut control = Vec::with_capacity(tokens.len());
        let mut text_to_id: HashMap<String, u32> = HashMap::with_capacity(tokens.len());
        let mut specials: Vec<(Vec<u8>, u32)> = Vec::new();
        for (i, t) in tokens.iter().enumerate() {
            let id = u32::try_from(i).map_err(|_| invalid("vocabulary too large"))?;
            let is_control = types
                .as_ref()
                .map(|v| v[i] == TOKEN_TYPE_CONTROL)
                .unwrap_or(false);
            if is_control {
                pieces.push(t.as_bytes().to_vec());
                if !t.is_empty() {
                    specials.push((t.as_bytes().to_vec(), id));
                }
            } else {
                let mut bytes = Vec::with_capacity(t.len());
                let mut buf = [0u8; 4];
                for c in t.chars() {
                    match char_to_byte.get(&c) {
                        Some(&b) => bytes.push(b),
                        None => bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes()),
                    }
                }
                pieces.push(bytes);
            }
            control.push(is_control);
            // First entry wins on duplicates (llama.cpp does not overwrite either).
            text_to_id.entry((*t).to_string()).or_insert(id);
        }

        // Every byte must have a single-character token so unknown symbols can always fall back.
        for (b, c) in byte_to_char.iter().enumerate() {
            if !text_to_id.contains_key(c.to_string().as_str()) {
                return Err(invalid(format!(
                    "vocabulary has no token for byte 0x{b:02X} (expected {c:?}); not a byte-level BPE vocabulary"
                )));
            }
        }

        let mut ranks = HashMap::with_capacity(merges.len());
        for (rank, merge) in merges.iter().enumerate() {
            let mut parts = merge.split(' ');
            let (Some(a), Some(b), None) = (parts.next(), parts.next(), parts.next()) else {
                return Err(invalid(format!(
                    "merge #{rank} is not \"left right\": {merge:?}"
                )));
            };
            if a.is_empty() || b.is_empty() {
                return Err(invalid(format!(
                    "merge #{rank} has an empty side: {merge:?}"
                )));
            }
            let rank = u32::try_from(rank).map_err(|_| invalid("too many merges"))?;
            ranks.entry((*merge).to_string()).or_insert(rank);
        }

        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
        let mut special_first_byte = [false; 256];
        for (s, _) in &specials {
            special_first_byte[s[0] as usize] = true;
        }

        let id_of = |key: &str| -> Result<Option<u32>, EngineError> {
            match m.get(key) {
                None => Ok(None),
                Some(_) => {
                    let v = m
                        .get_u32(key)
                        .ok_or_else(|| invalid(format!("{key} is not a u32")))?;
                    if (v as usize) >= tokens.len() {
                        return Err(invalid(format!("{key} {v} is outside the vocabulary")));
                    }
                    Ok(Some(v))
                }
            }
        };
        let bos = id_of("tokenizer.ggml.bos_token_id")?;
        let eos = id_of("tokenizer.ggml.eos_token_id")?;

        let lookup = |s: &str| text_to_id.get(s).copied();
        let mut eog: Vec<u32> = eos.into_iter().collect();
        for name in ["<|eot_id|>", "<|end_of_text|>"] {
            if let Some(id) = lookup(name) {
                if !eog.contains(&id) {
                    eog.push(id);
                }
            }
        }
        let chat = match (
            lookup("<|start_header_id|>"),
            lookup("<|end_header_id|>"),
            lookup("<|eot_id|>"),
        ) {
            (Some(_), Some(_), Some(eot)) => Some(ChatIds { eot }),
            _ => None,
        };

        Ok(Tokenizer {
            pieces,
            control,
            text_to_id,
            ranks,
            specials,
            special_first_byte,
            byte_to_char,
            char_to_byte,
            bos,
            eos,
            eog,
            chat,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.bos
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.eos
    }

    /// True for the metadata EOS token, `<|eot_id|>` and `<|end_of_text|>`.
    pub fn is_eos(&self, id: u32) -> bool {
        self.eog.contains(&id)
    }

    /// Raw bytes of a token. Control tokens give an empty vec; out-of-range ids too.
    pub fn decode_piece(&self, id: u32) -> Vec<u8> {
        self.decode_piece_special(id, false)
    }

    /// Like [`decode_piece`](Self::decode_piece); with `special` true, control tokens
    /// yield their literal text.
    pub fn decode_piece_special(&self, id: u32, special: bool) -> Vec<u8> {
        let i = id as usize;
        match self.pieces.get(i) {
            Some(p) if !self.control[i] || special => p.clone(),
            _ => Vec::new(),
        }
    }

    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_bos {
            if let Some(b) = self.bos {
                out.push(b);
            }
        }
        let bytes = text.as_bytes();
        let mut seg_start = 0;
        let mut i = 0;
        while i < bytes.len() {
            if self.special_first_byte[bytes[i] as usize] {
                if let Some((s, id)) = self
                    .specials
                    .iter()
                    .find(|(s, _)| bytes[i..].starts_with(s))
                {
                    self.encode_plain(&text[seg_start..i], &mut out);
                    out.push(*id);
                    i += s.len();
                    seg_start = i;
                    continue;
                }
            }
            i += 1;
        }
        self.encode_plain(&text[seg_start..], &mut out);
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let mut mapped = String::new();
        for word in pretokenize_llama_bpe(text) {
            mapped.clear();
            let mut buf = [0u8; 4];
            for c in word.chars() {
                // Pre-tokens are slices of valid UTF-8, so encode each char's bytes.
                for &b in c.encode_utf8(&mut buf).as_bytes() {
                    mapped.push(self.byte_to_char[b as usize]);
                }
            }
            self.bpe(&mapped, out);
        }
    }

    /// Merges by lowest rank first, leftmost on ties, in O(n log n): a doubly linked list
    /// of symbols plus a min-heap of candidate pairs (same scheme as llama.cpp's bigram queue).
    fn bpe(&self, word: &str, out: &mut Vec<u32>) {
        if let Some(&id) = self.text_to_id.get(word) {
            out.push(id);
            return;
        }
        // Symbol i covers word[start..end]; zero length means merged away.
        let mut starts: Vec<usize> = word.char_indices().map(|(i, _)| i).collect();
        let n = starts.len();
        starts.push(word.len());
        let mut end: Vec<usize> = starts[1..].to_vec();
        let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect();
        let mut next: Vec<usize> = (1..=n).collect();
        let mut alive = vec![true; n];

        let mut heap: BinaryHeap<Reverse<(u32, usize, usize, usize)>> = BinaryHeap::new();
        let mut key = String::new();
        // (rank, left idx, left end, right end): stale entries are detected by the ends.
        let mut push = |heap: &mut BinaryHeap<Reverse<(u32, usize, usize, usize)>>,
                        starts: &[usize],
                        end: &[usize],
                        l: usize,
                        r: usize| {
            key.clear();
            key.push_str(&word[starts[l]..end[l]]);
            key.push(' ');
            key.push_str(&word[starts[r]..end[r]]);
            if let Some(&rank) = self.ranks.get(key.as_str()) {
                heap.push(Reverse((rank, l, end[l], end[r])));
            }
        };
        for i in 0..n.saturating_sub(1) {
            push(&mut heap, &starts, &end, i, i + 1);
        }
        while let Some(Reverse((_, l, l_end, r_end))) = heap.pop() {
            if !alive[l] || end[l] != l_end {
                continue;
            }
            let r = next[l];
            if r >= n || !alive[r] || end[r] != r_end {
                continue;
            }
            // Merge r into l.
            end[l] = end[r];
            alive[r] = false;
            next[l] = next[r];
            if next[r] < n {
                prev[next[r]] = l;
            }
            if prev[l] < n {
                push(&mut heap, &starts, &end, prev[l], l);
            }
            if next[l] < n {
                push(&mut heap, &starts, &end, l, next[l]);
            }
        }
        let mut i = 0;
        while i < n {
            let sym = &word[starts[i]..end[i]];
            match self.text_to_id.get(sym) {
                Some(&id) => out.push(id),
                None => {
                    // Fall back to one token per byte-character (all 256 exist).
                    let mut buf = [0u8; 4];
                    for c in sym.chars() {
                        if let Some(&id) = self.text_to_id.get(&*c.encode_utf8(&mut buf)) {
                            out.push(id);
                        }
                    }
                }
            }
            i = next[i];
        }
    }

    /// Renders messages in the Llama 3 instruct format, ending with an open assistant
    /// header. `<|begin_of_text|>` is not included; use `encode(.., true)`.
    /// Roles are limited to system, user, assistant, tool.
    pub fn apply_chat(&self, msgs: &[ChatMessage]) -> Result<String, EngineError> {
        let Some(ids) = self.chat else {
            return Err(EngineError::UnsupportedArchitecture(
                "chat template: only the Llama 3 instruct format (<|start_header_id|>, <|end_header_id|>, <|eot_id|> tokens) is supported"
                    .into(),
            ));
        };
        let eot = String::from_utf8_lossy(&self.pieces[ids.eot as usize]).into_owned();
        let mut s = String::new();
        for m in msgs {
            if !matches!(m.role.as_str(), "system" | "user" | "assistant" | "tool") {
                return Err(EngineError::InvalidRequest(format!(
                    "unknown role {:?} (expected system, user, assistant or tool)",
                    m.role
                )));
            }
            s.push_str("<|start_header_id|>");
            s.push_str(&m.role);
            s.push_str("<|end_header_id|>\n\n");
            s.push_str(&m.content);
            s.push_str(&eot);
        }
        s.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        Ok(s)
    }

    #[doc(hidden)]
    pub fn byte_for_char(&self, c: char) -> Option<u8> {
        self.char_to_byte.get(&c).copied()
    }
}

/// Incremental UTF-8 decoder for byte-level tokens. Emits only complete code points,
/// holds an incomplete trailing sequence until it can be completed, and replaces
/// invalid bytes with U+FFFD without stalling.
#[derive(Debug, Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn new() -> Utf8Stream {
        Utf8Stream::default()
    }

    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        let mut start = 0;
        loop {
            match std::str::from_utf8(&self.pending[start..]) {
                Ok(s) => {
                    out.push_str(s);
                    start = self.pending.len();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    // Valid by construction of valid_up_to.
                    out.push_str(&String::from_utf8_lossy(
                        &self.pending[start..start + valid],
                    ));
                    start += valid;
                    match e.error_len() {
                        Some(n) => {
                            out.push('\u{FFFD}');
                            start += n;
                        }
                        // Incomplete sequence at the end (at most 3 bytes): hold it.
                        None => break,
                    }
                }
            }
        }
        self.pending.drain(..start);
        out
    }

    /// Emits any held bytes as a replacement character (end of stream).
    pub fn flush(&mut self) -> String {
        let out = if self.pending.is_empty() {
            String::new()
        } else {
            "\u{FFFD}".to_string()
        };
        self.pending.clear();
        out
    }
}

// ---- Llama 3 pre-tokenizer -------------------------------------------------

// APPROXIMATION: Rust's std has no Unicode general-category tables. \p{L}, \p{N}
// and \s are approximated with char::is_alphabetic (minus numerics), char::is_numeric
// and char::is_whitespace. This is exact for ASCII and nearly all text, but differs
// for a few code points: combining marks with the Alphabetic property (e.g. some
// Indic vowel signs) count as letters here but are \p{M} in the real pattern, and
// \p{Nl}/\p{No} edge cases may differ. Such differences change token boundaries
// only for rare scripts/symbols.

fn is_letter(c: char) -> bool {
    c.is_alphabetic() && !c.is_numeric()
}

fn is_number(c: char) -> bool {
    c.is_numeric()
}

fn is_newline(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// Length in chars of the match starting at `i` (always >= 1).
fn match_len(cs: &[char], i: usize) -> usize {
    let n = cs.len();
    let c = cs[i];

    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if c == '\'' && i + 1 < n {
        let c1 = cs[i + 1].to_ascii_lowercase();
        if matches!(c1, 's' | 't' | 'm' | 'd') {
            return 2;
        }
        if i + 2 < n {
            let c2 = cs[i + 2].to_ascii_lowercase();
            if matches!((c1, c2), ('r', 'e') | ('v', 'e') | ('l', 'l')) {
                return 3;
            }
        }
    }

    // [^\r\n\p{L}\p{N}]?\p{L}+
    let start = if is_letter(c) {
        Some(i)
    } else if !is_newline(c) && !is_number(c) && i + 1 < n && is_letter(cs[i + 1]) {
        Some(i + 1)
    } else {
        None
    };
    if let Some(mut j) = start {
        while j < n && is_letter(cs[j]) {
            j += 1;
        }
        return j - i;
    }

    // \p{N}{1,3}
    if is_number(c) {
        let mut j = i;
        while j < n && j - i < 3 && is_number(cs[j]) {
            j += 1;
        }
        return j - i;
    }

    // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
    let is_sym = |ch: char| !ch.is_whitespace() && !is_letter(ch) && !is_number(ch);
    let j = if c == ' ' { i + 1 } else { i };
    let mut k = j;
    while k < n && is_sym(cs[k]) {
        k += 1;
    }
    if k > j {
        while k < n && is_newline(cs[k]) {
            k += 1;
        }
        return k - i;
    }

    // Whitespace alternatives. c is whitespace here.
    let mut run = i;
    while run < n && cs[run].is_whitespace() {
        run += 1;
    }
    let run_len = run - i;
    // \s*[\r\n]+ : through the last newline of the run.
    if let Some(last_nl) = (i..run).rev().find(|&p| is_newline(cs[p])) {
        return last_nl - i + 1;
    }
    // \s+(?!\S) : whole run at end of text, else leave one char for the next token.
    if run == n {
        return run_len;
    }
    if run_len > 1 {
        return run_len - 1;
    }
    // \s+
    run_len.max(1)
}

/// Splits text with the Llama 3 ("llama-bpe") pre-tokenizer pattern:
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
/// implemented by hand (see the approximation note above). The pieces concatenate to the input.
pub fn pretokenize_llama_bpe(text: &str) -> Vec<String> {
    let cs: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < cs.len() {
        let len = match_len(&cs, i).max(1);
        out.push(cs[i..i + len].iter().collect());
        i += len;
    }
    out
}
