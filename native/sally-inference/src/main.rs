use sally_inference::error::EngineError;
use sally_inference::gguf::GgufFile;
use std::path::Path;
use std::process::ExitCode;

const USAGE: &str = "usage: sally-inference inspect <model.gguf>";

/// 0 ok; 1 I/O or internal error; 2 invalid/unsupported model; 64 usage.
fn exit_code(e: &EngineError) -> u8 {
    match e {
        EngineError::InvalidModel(_) | EngineError::UnsupportedArchitecture(_) => 2,
        _ => 1,
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [cmd, path] if cmd == "inspect" => match GgufFile::open(Path::new(path)) {
            Ok(file) => {
                print!("{}", file.report());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error [{}]: {e}", e.code());
                ExitCode::from(exit_code(&e))
            }
        },
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(64)
        }
    }
}
