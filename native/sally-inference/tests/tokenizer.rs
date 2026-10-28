//! Tokenizer tests on a synthetic byte-level vocabulary, plus a gated real-model check.
#[path = "common/gguf_builder.rs"]
mod gguf_builder;

use gguf_builder::GgufBuilder;
use sally_inference::gguf::{GgufFile, Metadata};
use sally_inference::tokenizer::{pretokenize_llama_bpe, ChatMessage, Tokenizer, Utf8Stream};
use std::io::Cursor;

/// GPT-2 bytes_to_unicode, written independently of the crate.
fn byte_chars() -> Vec<char> {
    let mut printable: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut table = vec!['\0'; 256];
    for &b in &printable {
        table[b as usize] = char::from_u32(b).unwrap();
    }
    let mut n = 0;
    for b in 0..256u32 {
        if !printable.contains(&b) {
            table[b as usize] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    printable.clear();
    table
}

// Vocabulary layout: 0..=255 byte tokens, then "he"(256) "ll"(257) "hell"(258)
// "hello"(259), then the specials starting with <|begin_of_text|>(260).
const HELLO: u32 = 259;
const BOS: u32 = 260;
const EOT: u32 = 264;

fn vocab() -> (Vec<String>, Vec<i32>) {
    let mut tokens: Vec<String> = byte_chars().iter().map(|c| c.to_string()).collect();
    let mut types = vec![1; 256];
    for t in ["he", "ll", "hell", "hello"] {
        tokens.push(t.to_string());
        types.push(1);
    }
    for t in [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ] {
        tokens.push(t.to_string());
        types.push(3);
    }
    (tokens, types)
}

fn builder(model: &str, pre: Option<&str>) -> GgufBuilder {
    let (tokens, types) = vocab();
    let merges: Vec<String> = ["h e", "l l", "he ll", "hell o"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut b = GgufBuilder::new()
        .kv_str("general.architecture", "llama")
        .kv_str("tokenizer.ggml.model", model)
        .kv_str_array("tokenizer.ggml.tokens", &tokens)
        .kv_str_array("tokenizer.ggml.merges", &merges)
        .kv_i32_array("tokenizer.ggml.token_type", &types)
        .kv_u32("tokenizer.ggml.bos_token_id", BOS)
        .kv_u32("tokenizer.ggml.eos_token_id", EOT);
    if let Some(p) = pre {
        b = b.kv_str("tokenizer.ggml.pre", p);
    }
    b
}

fn metadata(b: GgufBuilder) -> Metadata {
    let bytes = b.build();
    let len = bytes.len() as u64;
    GgufFile::from_reader(Cursor::new(bytes), len)
        .unwrap()
        .metadata
}

fn tok() -> Tokenizer {
    Tokenizer::from_gguf(&metadata(builder("gpt2", Some("llama-bpe")))).unwrap()
}

fn decode(t: &Tokenizer, ids: &[u32]) -> Vec<u8> {
    ids.iter().flat_map(|&i| t.decode_piece(i)).collect()
}

fn byte_id(b: u8) -> u32 {
    b as u32
}

#[test]
fn encode_decode_roundtrip() {
    let t = tok();
    for text in [
        "hello",
        "héllo €",
        "emoji \u{1F600} and \u{1F1F3}\u{1F1EC}",
        "  leading and trailing  ",
        "line1\nline2\r\n\ttab",
        "12345 numbers 6",
        "",
        "日本語のテキスト",
    ] {
        let ids = t.encode(text, false);
        assert_eq!(decode(&t, &ids), text.as_bytes(), "text {text:?}");
        if !text.is_empty() {
            assert!(!ids.is_empty());
        }
    }
    assert_eq!(t.vocab_size(), 256 + 4 + 5);
}

#[test]
fn bpe_merges_apply_by_rank() {
    let t = tok();
    // "hello" is a whole vocabulary entry (llama.cpp ignore_merges fast path).
    assert_eq!(t.encode("hello", false), vec![HELLO]);
    // "hellos": he+ll -> hell -> hello, 's' stays a byte token.
    assert_eq!(t.encode("hellos", false), vec![HELLO, byte_id(b's')]);
    assert_eq!(t.encode("hehe", false), vec![256, 256]);
    // a word with no merges falls to one token per byte
    assert_eq!(t.encode("xy", false), vec![byte_id(b'x'), byte_id(b'y')]);
}

#[test]
fn add_bos_prepends_bos() {
    let t = tok();
    assert_eq!(t.bos_id(), Some(BOS));
    let with = t.encode("hello", true);
    assert_eq!(with, vec![BOS, HELLO]);
}

#[test]
fn special_token_matched_literally() {
    let t = tok();
    let ids = t.encode("a<|eot_id|>b", false);
    assert_eq!(ids, vec![byte_id(b'a'), EOT, byte_id(b'b')]);
    // Longest/leftmost: adjacent specials.
    let ids = t.encode("<|start_header_id|><|end_header_id|>", false);
    assert_eq!(ids, vec![256 + 6, 256 + 7]);
    // A near-miss is plain text.
    let ids = t.encode("<|eot_id|", false);
    assert!(!ids.contains(&EOT));
    // Control tokens decode to nothing by default, to their text on request.
    assert!(t.decode_piece(EOT).is_empty());
    assert_eq!(t.decode_piece_special(EOT, true), b"<|eot_id|>".to_vec());
}

#[test]
fn eos_detection() {
    let t = tok();
    assert_eq!(t.eos_id(), Some(EOT));
    assert!(t.is_eos(EOT));
    assert!(t.is_eos(256 + 5)); // <|end_of_text|>
    assert!(!t.is_eos(byte_id(b'a')));
}

#[test]
fn utf8_stream_holds_partial_codepoint() {
    let euro = "€".as_bytes();
    assert_eq!(euro.len(), 3);
    let mut s = Utf8Stream::new();
    assert_eq!(s.push(&euro[..2]), "");
    assert_eq!(s.push(&euro[2..]), "€");
    assert_eq!(s.flush(), "");
    // Complete text interleaved with a held prefix.
    let mut s = Utf8Stream::new();
    assert_eq!(s.push(b"ab\xE2\x82"), "ab");
    assert_eq!(s.push(b"\xACcd"), "€cd");
    // Emoji split byte by byte.
    let mut s = Utf8Stream::new();
    let mut out = String::new();
    for b in "\u{1F600}".as_bytes() {
        out.push_str(&s.push(&[*b]));
    }
    assert_eq!(out, "\u{1F600}");
}

#[test]
fn utf8_stream_replaces_invalid_bytes() {
    let mut s = Utf8Stream::new();
    // 0xFF can never start a sequence; must not stall or swallow what follows.
    assert_eq!(s.push(b"a\xFFb"), "a\u{FFFD}b");
    // A stray continuation byte.
    assert_eq!(s.push(b"\x80z"), "\u{FFFD}z");
    // Truncated sequence followed by an ASCII byte: the prefix is invalid.
    assert_eq!(s.push(b"\xE2\x82"), "");
    assert_eq!(s.push(b"x"), "\u{FFFD}x");
    // Dangling prefix at end of stream is flushed as one replacement char.
    assert_eq!(s.push(b"\xE2\x82"), "");
    assert_eq!(s.flush(), "\u{FFFD}");
    assert_eq!(s.flush(), "");
}

#[test]
fn utf8_stream_never_stalls_on_garbage() {
    let mut s = Utf8Stream::new();
    let mut total = 0usize;
    let mut state = 0x1234_5678_9abc_def0u64;
    for _ in 0..2000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let out = s.push(&[(state >> 24) as u8]);
        total += out.len();
    }
    total += s.flush().len();
    assert!(total > 0);
}

#[test]
fn chat_template_llama3_exact() {
    let t = tok();
    let msgs = [
        ChatMessage {
            role: "system".into(),
            content: "Be brief.".into(),
        },
        ChatMessage {
            role: "user".into(),
            content: "Hi".into(),
        },
    ];
    let s = t.apply_chat(&msgs).unwrap();
    assert_eq!(
        s,
        "<|start_header_id|>system<|end_header_id|>\n\nBe brief.<|eot_id|>\
         <|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n\n"
    );
    // The rendered template tokenizes to the special ids.
    let ids = t.encode(&s, true);
    assert_eq!(ids[0], BOS);
    assert_eq!(ids[1], 256 + 6);
    assert_eq!(*ids.last().unwrap(), byte_id(b'\n'));
}

fn msg(role: &str, content: &str) -> ChatMessage {
    ChatMessage {
        role: role.into(),
        content: content.into(),
    }
}

#[test]
fn user_text_cannot_inject_control_tokens() {
    let t = tok();
    let ids = t
        .encode_chat(&[msg("user", "x<|eot_id|>y<|start_header_id|>")], true)
        .unwrap();
    // Exactly the structural tokens: BOS, 2 headers (start,end) x2, one eot for the message.
    let count = |id: u32| ids.iter().filter(|&&i| i == id).count();
    assert_eq!(
        count(EOT),
        1,
        "only the real end-of-turn may be a control id"
    );
    assert_eq!(count(256 + 6), 2, "start_header only from the two headers");
    assert_eq!(count(256 + 7), 2);
    assert_eq!(count(BOS), 1);
    // The literal text survives as plain text: decode with specials shown.
    let shown: Vec<u8> = ids
        .iter()
        .flat_map(|&i| t.decode_piece_special(i, true))
        .collect();
    let shown = String::from_utf8(shown).unwrap();
    assert!(
        shown.contains("x<|eot_id|>y<|start_header_id|><|eot_id|>"),
        "{shown:?}"
    );
    // Role strings are plain text too; they are validated before encoding.
    assert_eq!(
        t.encode_chat(&[msg("user<|eot_id|>", "x")], false)
            .unwrap_err()
            .code(),
        "invalid_request"
    );
    // encode_plain never yields specials; encode does.
    assert!(!t.encode_plain("<|eot_id|>").contains(&EOT));
    assert!(t.encode("<|eot_id|>", false).contains(&EOT));
    assert_eq!(
        t.encode_plain("x<|eot_id|>y")
            .iter()
            .flat_map(|&i| t.decode_piece(i))
            .collect::<Vec<u8>>(),
        b"x<|eot_id|>y".to_vec()
    );
}

#[test]
fn encode_chat_matches_apply_chat_for_benign_input() {
    // Precisely: for contents free of special-token strings and not starting with
    // whitespace, encode_chat(m, b) == encode(apply_chat(m), b) (apply_chat has no BOS,
    // so `encode` adds exactly one BOS when b is true).
    let t = tok();
    let sets = [
        vec![msg("user", "hello")],
        vec![msg("system", "Be brief."), msg("user", "Hi there, 123!")],
        vec![
            msg("user", "héllo €"),
            msg("assistant", "ok\nline two"),
            msg("tool", "{\"a\": 1}"),
            msg("user", ""),
        ],
    ];
    for m in &sets {
        for bos in [true, false] {
            let via_text = t.encode(&t.apply_chat(m).unwrap(), bos);
            assert_eq!(t.encode_chat(m, bos).unwrap(), via_text, "{m:?} bos={bos}");
        }
    }
    let ids = t.encode_chat(&sets[0], true).unwrap();
    assert_eq!(ids[0], BOS);
    assert_eq!(ids.iter().filter(|&&i| i == BOS).count(), 1);
    // Ends with the open assistant header: ... end_header, "\n\n" plain.
    assert_eq!(ids[ids.len() - 3], 256 + 7);
}

#[test]
fn encode_chat_requires_llama3_tokens() {
    let tokens: Vec<String> = byte_chars().iter().map(|c| c.to_string()).collect();
    let b = GgufBuilder::new()
        .kv_str("tokenizer.ggml.model", "gpt2")
        .kv_str("tokenizer.ggml.pre", "llama-bpe")
        .kv_str_array("tokenizer.ggml.tokens", &tokens)
        .kv_str_array("tokenizer.ggml.merges", &[]);
    let t = Tokenizer::from_gguf(&metadata(b)).unwrap();
    assert_eq!(
        t.encode_chat(&[msg("user", "x")], true).unwrap_err().code(),
        "unsupported_architecture"
    );
}

#[test]
fn unknown_role_is_invalid_request() {
    let t = tok();
    let e = t
        .apply_chat(&[ChatMessage {
            role: "wizard".into(),
            content: "x".into(),
        }])
        .unwrap_err();
    assert_eq!(e.code(), "invalid_request");
    assert!(e.to_string().contains("wizard"));
    for role in ["system", "user", "assistant", "tool"] {
        assert!(t
            .apply_chat(&[ChatMessage {
                role: role.into(),
                content: "x".into()
            }])
            .is_ok());
    }
}

#[test]
fn chat_unsupported_without_llama3_tokens() {
    // Vocab with no header tokens at all.
    let tokens: Vec<String> = byte_chars().iter().map(|c| c.to_string()).collect();
    let b = GgufBuilder::new()
        .kv_str("tokenizer.ggml.model", "gpt2")
        .kv_str("tokenizer.ggml.pre", "llama-bpe")
        .kv_str_array("tokenizer.ggml.tokens", &tokens)
        .kv_str_array("tokenizer.ggml.merges", &[]);
    let t = Tokenizer::from_gguf(&metadata(b)).unwrap();
    let e = t
        .apply_chat(&[ChatMessage {
            role: "user".into(),
            content: "x".into(),
        }])
        .unwrap_err();
    assert_eq!(e.code(), "unsupported_architecture");
}

#[test]
fn unknown_tokenizer_model_unsupported() {
    let e = Tokenizer::from_gguf(&metadata(builder("llama", None))).unwrap_err();
    assert_eq!(e.code(), "unsupported_architecture");
    assert!(e.to_string().contains("llama"));
    let e = Tokenizer::from_gguf(&metadata(builder("gpt2", Some("qwen2")))).unwrap_err();
    assert_eq!(e.code(), "unsupported_architecture");
    assert!(e.to_string().contains("qwen2"));
    // pre absent is accepted as llama-bpe (documented).
    assert!(Tokenizer::from_gguf(&metadata(builder("gpt2", None))).is_ok());
    // no tokenizer metadata at all
    let b = GgufBuilder::new().kv_str("general.architecture", "llama");
    let e = Tokenizer::from_gguf(&metadata(b)).unwrap_err();
    assert_eq!(e.code(), "invalid_model");
}

#[test]
fn malformed_vocab_is_invalid_model() {
    // Missing byte tokens: vocabulary of just two entries.
    let b = GgufBuilder::new()
        .kv_str("tokenizer.ggml.model", "gpt2")
        .kv_str_array("tokenizer.ggml.tokens", &["a".to_string(), "b".to_string()])
        .kv_str_array("tokenizer.ggml.merges", &[]);
    assert_eq!(
        Tokenizer::from_gguf(&metadata(b)).unwrap_err().code(),
        "invalid_model"
    );
    // Merge entry without a separator.
    let (tokens, types) = vocab();
    let b = GgufBuilder::new()
        .kv_str("tokenizer.ggml.model", "gpt2")
        .kv_str_array("tokenizer.ggml.tokens", &tokens)
        .kv_i32_array("tokenizer.ggml.token_type", &types)
        .kv_str_array("tokenizer.ggml.merges", &["nospace".to_string()]);
    assert_eq!(
        Tokenizer::from_gguf(&metadata(b)).unwrap_err().code(),
        "invalid_model"
    );
    // token_type length mismatch.
    let b = GgufBuilder::new()
        .kv_str("tokenizer.ggml.model", "gpt2")
        .kv_str_array("tokenizer.ggml.tokens", &tokens)
        .kv_i32_array("tokenizer.ggml.token_type", &[1, 1])
        .kv_str_array("tokenizer.ggml.merges", &[]);
    assert_eq!(
        Tokenizer::from_gguf(&metadata(b)).unwrap_err().code(),
        "invalid_model"
    );
    // pre present but not a string.
    let b = builder("gpt2", None).kv_u32("tokenizer.ggml.pre", 7);
    assert_eq!(
        Tokenizer::from_gguf(&metadata(b)).unwrap_err().code(),
        "invalid_model"
    );
    // bos id out of range.
    let b = builder("gpt2", Some("llama-bpe")).kv_u32("tokenizer.ggml.padding_token_id", 1);
    assert!(Tokenizer::from_gguf(&metadata(b)).is_ok());
}

fn split(s: &str) -> Vec<String> {
    pretokenize_llama_bpe(s)
}

#[test]
fn pretokenizer_splits_match_pattern() {
    // Derived by hand from
    // (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}
    //   | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
    assert_eq!(
        split("Hello world's 12345  x\n\nok"),
        vec!["Hello", " world", "'s", " ", "123", "45", " ", " x", "\n\n", "ok"]
    );
    // Contractions are case-insensitive; a letter run takes one optional leading non-letter.
    assert_eq!(split("DON'T"), vec!["DON", "'T"]);
    assert_eq!(split("we've"), vec!["we", "'ve"]);
    assert_eq!(split("'s"), vec!["'s"]);
    // Punctuation run takes an optional leading space and trailing newlines.
    assert_eq!(split("a ...\n\nb"), vec!["a", " ...\n\n", "b"]);
    assert_eq!(split("(x)"), vec!["(x", ")"]);
    // Digits: groups of at most three, never with a leading space attached.
    assert_eq!(split("1234567"), vec!["123", "456", "7"]);
    assert_eq!(split("x 12"), vec!["x", " ", "12"]);
    // Whitespace: \s*[\r\n]+ consumes through the last newline of the run,
    // \s+(?!\S) leaves one space to attach to the next word.
    assert_eq!(split("a \n b"), vec!["a", " \n", " b"]);
    assert_eq!(split("a   b"), vec!["a", "  ", " b"]);
    assert_eq!(split("a   "), vec!["a", "   "]);
    assert_eq!(split("\r\n"), vec!["\r\n"]);
    // Letter run can start after a symbol that is not newline/letter/number.
    assert_eq!(split("$abc"), vec!["$abc"]);
    assert_eq!(split("\nabc"), vec!["\n", "abc"]);
    // Non-ASCII letters and numbers.
    assert_eq!(split("héllo €5"), vec!["héllo", " €", "5"]);
    assert_eq!(split(""), Vec::<String>::new());
}

#[test]
fn pretokenizer_is_a_partition() {
    for s in [
        "Hello world's 12345  x\n\nok",
        "  \t\n \r\n x\u{00A0}y\u{3000}z",
        "日本語 テキスト 123４５６",
        "a'b'c''s'",
    ] {
        assert_eq!(split(s).concat(), s);
    }
}

#[test]
fn real_model_tokenizer_matches_known_ids() {
    let Ok(path) = std::env::var("SALLY_TEST_MODEL") else {
        println!("skipped: SALLY_TEST_MODEL not set");
        return;
    };
    let g = GgufFile::open(std::path::Path::new(&path)).unwrap();
    let t = Tokenizer::from_gguf(&g.metadata).unwrap();
    for text in [
        "The capital of France is",
        "Hello world",
        "héllo € \u{1F600}",
        "  indented\n\ncode();",
        "12345 and 3.14159",
    ] {
        let ids = t.encode(text, false);
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| t.decode_piece(i)).collect();
        assert_eq!(bytes, text.as_bytes(), "round trip {text:?}");
    }
    assert_eq!(t.encode("Hello world", false), vec![9906, 1917]);
}

/// Naive reference BPE (lowest rank first, leftmost on ties) for the synthetic merges.
fn reference_bpe(word: &str) -> Vec<String> {
    let merges = ["h e", "l l", "he ll", "hell o"];
    let mut syms: Vec<String> = word.chars().map(|c| c.to_string()).collect();
    loop {
        let mut best: Option<(usize, usize)> = None;
        for i in 0..syms.len().saturating_sub(1) {
            let k = format!("{} {}", syms[i], syms[i + 1]);
            if let Some(r) = merges.iter().position(|m| *m == k) {
                if best.map(|(br, _)| r < br).unwrap_or(true) {
                    best = Some((r, i));
                }
            }
        }
        let Some((_, i)) = best else { break };
        let r = syms.remove(i + 1);
        syms[i].push_str(&r);
    }
    syms
}

#[test]
fn bpe_matches_reference_on_random_words() {
    let t = tok();
    let vocab_ids: std::collections::HashMap<String, u32> = {
        let (tokens, _) = vocab();
        tokens
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s, i as u32))
            .collect()
    };
    let alphabet: Vec<char> = "helo".chars().collect();
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..500 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let len = 1 + (state % 12) as usize;
        let mut w = String::new();
        let mut s = state;
        for _ in 0..len {
            w.push(alphabet[(s % 4) as usize]);
            s /= 4;
            if s == 0 {
                s = state | 1;
            }
        }
        let expected: Vec<u32> = reference_bpe(&w).iter().map(|p| vocab_ids[p]).collect();
        assert_eq!(t.encode(&w, false), expected, "word {w:?}");
    }
}

fn assert_fast(label: &str, t: &Tokenizer, text: &str) -> Vec<u32> {
    let start = std::time::Instant::now();
    let ids = t.encode(text, false);
    let el = start.elapsed();
    assert!(el.as_secs_f64() < 2.0, "{label} took {el:?}");
    ids
}

#[test]
fn encode_large_inputs_is_not_quadratic() {
    let t = tok();
    // ~50k chars of ordinary text.
    let prose = "hello world, it's 12345 hehe!\n".repeat(1700);
    assert!(prose.chars().count() >= 50_000);
    let ids = assert_fast("prose", &t, &prose);
    assert_eq!(decode(&t, &ids), prose.as_bytes());
    // One 50k-char pre-token with many merges (worst case for naive BPE).
    let ids = assert_fast("one long word", &t, &"he".repeat(25_000));
    assert_eq!(ids, vec![256; 25_000]);
    let ids = assert_fast("long hello run", &t, &"hello".repeat(10_000));
    assert_eq!(decode(&t, &ids), "hello".repeat(10_000).as_bytes());
    // Long whitespace and symbol runs, and many specials.
    for (label, text) in [
        ("spaces", " ".repeat(50_000)),
        ("alternating ws", " \t".repeat(25_000)),
        ("newline mix", " \n".repeat(25_000)),
        ("symbols", "<|".repeat(25_000)),
        ("multibyte", "\u{1F600}".repeat(12_500)),
    ] {
        let ids = assert_fast(label, &t, &text);
        assert_eq!(decode(&t, &ids), text.as_bytes(), "{label}");
    }
    let ids = assert_fast("specials", &t, &"<|eot_id|>".repeat(5_000));
    assert_eq!(ids, vec![EOT; 5_000]);
}
