use sally_inference::architecture::llama::LlamaModel;
use sally_inference::architecture::Architecture;
use sally_inference::cancel::CancelToken;
use sally_inference::config::Config;
use sally_inference::error::EngineError;
use sally_inference::generate::{generate, GenParams};
use sally_inference::gguf::GgufFile;
use sally_inference::memory::{fmt_bytes, open_cache, Budget};
use sally_inference::model_registry;
use sally_inference::telemetry::{mem_available_bytes, peak_rss_bytes, Metrics};
use sally_inference::tensor_store::TensorStore;
use sally_inference::tokenizer::{ChatMessage, Tokenizer};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "usage:
  sally-inference inspect <model.gguf>
  sally-inference generate --model <model.gguf> --prompt <text> [--chat]
      [--max-tokens N] [--temperature T] [--top-k K] [--top-p P] [--seed S]
      [--threads N] [--context N] [--memory-mb N] [--log-json]";

/// 0 ok; 1 I/O, resource or internal error; 2 invalid/unsupported model; 64 usage.
fn exit_code(e: &EngineError) -> u8 {
    match e {
        EngineError::InvalidModel(_) | EngineError::UnsupportedArchitecture(_) => 2,
        _ => 1,
    }
}

fn fail(e: &EngineError) -> ExitCode {
    eprintln!("error [{}]: {e}", e.code());
    ExitCode::from(exit_code(e))
}

fn usage_error(msg: &str) -> ExitCode {
    eprintln!("{msg}\n{USAGE}");
    ExitCode::from(64)
}

struct GenerateArgs {
    model: PathBuf,
    prompt: String,
    chat: bool,
    params: GenParams,
    threads: Option<usize>,
    context: Option<usize>,
    memory_mb: Option<usize>,
    log_json: bool,
}

fn parse_generate(args: &[String]) -> Result<GenerateArgs, String> {
    let mut model = None;
    let mut prompt = None;
    let mut out = GenerateArgs {
        model: PathBuf::new(),
        prompt: String::new(),
        chat: false,
        params: GenParams::default(),
        threads: None,
        context: None,
        memory_mb: None,
        log_json: false,
    };
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        if flag == "--chat" {
            out.chat = true;
            continue;
        }
        if flag == "--log-json" {
            out.log_json = true;
            continue;
        }
        let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        let num = |what: &str| -> Result<f64, String> {
            value
                .parse::<f64>()
                .map_err(|_| format!("{what}: {value:?} is not a number"))
        };
        let int = |what: &str| -> Result<usize, String> {
            value
                .parse::<usize>()
                .map_err(|_| format!("{what}: {value:?} is not a non-negative integer"))
        };
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--prompt" => prompt = Some(value.clone()),
            "--max-tokens" => out.params.max_tokens = int("--max-tokens")?,
            "--temperature" => out.params.temperature = num("--temperature")? as f32,
            "--top-k" => out.params.top_k = int("--top-k")?,
            "--top-p" => out.params.top_p = num("--top-p")? as f32,
            "--seed" => {
                out.params.seed = value
                    .parse::<u64>()
                    .map_err(|_| format!("--seed: {value:?} is not an unsigned integer"))?
            }
            "--threads" => out.threads = Some(int("--threads")?),
            "--context" => out.context = Some(int("--context")?),
            "--memory-mb" => out.memory_mb = Some(int("--memory-mb")?),
            other => return Err(format!("unknown option {other}")),
        }
    }
    out.model = model.ok_or("--model is required")?;
    out.prompt = prompt.ok_or("--prompt is required")?;
    Ok(out)
}

fn run_generate(a: GenerateArgs) -> Result<(), EngineError> {
    let loading = Instant::now();
    let file = GgufFile::open(&a.model)?;
    let arch = file.architecture().unwrap_or("").to_string();
    model_registry::lookup(&arch)?;

    let defaults = Config::default();
    let threads = a.threads.unwrap_or(defaults.threads).max(1);
    let context = a.context.unwrap_or(defaults.context_len);

    let tok = Tokenizer::from_gguf(&file.metadata)?;
    let store = TensorStore::open(&file, &a.model)?;
    let mut model = LlamaModel::load(store, &file.metadata, threads)?;
    let ctx = context.min(model.context_len());

    let weight_bytes: u64 = file.tensors.iter().map(|t| t.n_bytes).sum();
    let budget_mb = a.memory_mb.unwrap_or(defaults.memory_budget_mb);
    let budget = Budget::new(budget_mb.saturating_mul(1024 * 1024));
    let _scratch = budget.reserve("scratch buffers", model.scratch_bytes())?;
    let (mut cache, _kv) = open_cache(&model, ctx, &budget)?;
    let load_ms = loading.elapsed().as_secs_f64() * 1000.0;

    eprintln!(
        "[sally] memory: weights_mapped={} kv_cache={} scratch={} budget_used={} of {}",
        fmt_bytes(weight_bytes as usize),
        fmt_bytes(cache.bytes()),
        fmt_bytes(model.scratch_bytes()),
        fmt_bytes(budget.used()),
        fmt_bytes(budget.limit()),
    );
    if let Some(avail) = mem_available_bytes() {
        if weight_bytes > avail {
            eprintln!(
                "[sally] warning: the model's weights ({}) are larger than the memory currently \
                 available ({}). They are memory-mapped and will be paged from disk, so expect \
                 heavy disk reads and slow generation.",
                fmt_bytes(weight_bytes as usize),
                fmt_bytes(avail as usize),
            );
        }
    }

    let prompt = if a.chat {
        let msgs = [ChatMessage {
            role: "user".into(),
            content: a.prompt.clone(),
        }];
        tok.encode_chat(&msgs, true)?
    } else {
        tok.encode(&a.prompt, true)
    };

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let stats = generate(
        &mut model,
        &tok,
        &mut cache,
        &prompt,
        &a.params,
        &CancelToken::new(),
        &mut |text| {
            let _ = out.write_all(text.as_bytes());
            let _ = out.flush();
        },
    )?;
    let _ = writeln!(out);

    eprintln!(
        "[sally] architecture={arch} threads={threads} context={ctx} prompt_tokens={} \
         generated_tokens={} load_ms={load_ms:.0} prefill_tok_s={:.2} decode_tok_s={:.2} finish={}",
        stats.prompt_tokens,
        stats.generated_tokens(),
        stats.prefill_tokens_per_sec(),
        stats.decode_tokens_per_sec(),
        stats.finish.as_str(),
    );
    if let Some(peak) = peak_rss_bytes() {
        eprintln!("[sally] peak_rss={}", fmt_bytes(peak as usize));
    }
    if a.log_json {
        eprintln!(
            "{}",
            Metrics::from_stats(&stats, load_ms, cache.bytes()).log_json()
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [cmd, path] if cmd == "inspect" => match GgufFile::open(Path::new(path)) {
            Ok(file) => {
                print!("{}", file.report());
                ExitCode::SUCCESS
            }
            Err(e) => fail(&e),
        },
        [cmd, rest @ ..] if cmd == "generate" => match parse_generate(rest) {
            Ok(a) => match run_generate(a) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => fail(&e),
            },
            Err(msg) => usage_error(&msg),
        },
        _ => usage_error("unknown or missing command"),
    }
}
