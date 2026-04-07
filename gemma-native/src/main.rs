use std::fs;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::pin::pin;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use serde::{Deserialize, Serialize};

// --- Constants ---

const MAGIC: &[u8; 8] = b"GMNAPAK\0";
const FOOTER_SIZE: u64 = 24;
const N_CTX: u32 = 4096;
const MAX_TOKENS: i32 = 2048;
const MAX_TOOL_ROUNDS: usize = 10;

static VERBOSE: AtomicBool = AtomicBool::new(false);

fn verbose() -> bool { VERBOSE.load(Ordering::Relaxed) }

macro_rules! vlog {
    ($($arg:tt)*) => { if verbose() { eprintln!($($arg)*); } }
}

const SYSTEM_PROMPT: &str = r#"You are a helpful assistant with access to tools. You can use tools by including tool calls in your response.

Available tools:

1. **bash** - Execute a shell command
   Call: <tool_call>{"name": "bash", "arguments": {"command": "your command here"}}</tool_call>

2. **read_file** - Read the contents of a file
   Call: <tool_call>{"name": "read_file", "arguments": {"path": "/path/to/file"}}</tool_call>

3. **write_file** - Write content to a file
   Call: <tool_call>{"name": "write_file", "arguments": {"path": "/path/to/file", "content": "file content"}}</tool_call>

Rules:
- You may include multiple tool calls in one response.
- After tool results are returned, continue your response to the user.
- If no tools are needed, just respond normally.
- Always explain what you're doing before using tools."#;

// Special tokens to strip from output
const STRIP_TAGS: &[&str] = &[
    "<end_of_turn>",
    "<start_of_turn>model",
    "<start_of_turn>user",
    "<start_of_turn>system",
    "<start_of_turn>tool",
    "<start_of_turn>",
];

/// Strip Gemma special tokens from a response string
fn strip_special_tokens(text: &str) -> String {
    let mut out = text.to_string();
    for tag in STRIP_TAGS {
        out = out.replace(tag, "");
    }
    out.trim().to_string()
}

// --- Model Registry ---

struct ModelEntry {
    alias: &'static str,
    repo: &'static str,
    file: &'static str,
    size: &'static str,
}

const MODELS: &[ModelEntry] = &[
    ModelEntry { alias: "e4b",       repo: "bartowski/google_gemma-4-E4B-it-GGUF", file: "google_gemma-4-E4B-it-Q4_K_M.gguf",  size: "~5.4 GB" },
    ModelEntry { alias: "e4b-q8",    repo: "bartowski/google_gemma-4-E4B-it-GGUF", file: "google_gemma-4-E4B-it-Q8_0.gguf",    size: "~8.0 GB" },
    ModelEntry { alias: "e4b-q4ks",  repo: "bartowski/google_gemma-4-E4B-it-GGUF", file: "google_gemma-4-E4B-it-Q4_K_S.gguf",  size: "~5.2 GB" },
    ModelEntry { alias: "e4b-q3km",  repo: "bartowski/google_gemma-4-E4B-it-GGUF", file: "google_gemma-4-E4B-it-Q3_K_M.gguf",  size: "~4.9 GB" },
    ModelEntry { alias: "e4b-iq4xs", repo: "bartowski/google_gemma-4-E4B-it-GGUF", file: "google_gemma-4-E4B-it-IQ4_XS.gguf",  size: "~5.1 GB" },
    ModelEntry { alias: "e2b",       repo: "bartowski/google_gemma-4-E2B-it-GGUF", file: "google_gemma-4-E2B-it-Q4_K_M.gguf",  size: "~2.0 GB" },
    ModelEntry { alias: "e2b-q8",    repo: "bartowski/google_gemma-4-E2B-it-GGUF", file: "google_gemma-4-E2B-it-Q8_0.gguf",    size: "~3.0 GB" },
    ModelEntry { alias: "e2b-q4ks",  repo: "bartowski/google_gemma-4-E2B-it-GGUF", file: "google_gemma-4-E2B-it-Q4_K_S.gguf",  size: "~1.9 GB" },
    ModelEntry { alias: "e2b-q3km",  repo: "bartowski/google_gemma-4-E2B-it-GGUF", file: "google_gemma-4-E2B-it-Q3_K_M.gguf",  size: "~1.8 GB" },
    ModelEntry { alias: "e2b-iq4xs", repo: "bartowski/google_gemma-4-E2B-it-GGUF", file: "google_gemma-4-E2B-it-IQ4_XS.gguf",  size: "~1.8 GB" },
];

fn find_model(alias: &str) -> Option<&'static ModelEntry> {
    MODELS.iter().find(|m| m.alias == alias)
}

fn print_models() {
    println!("Available models (for build.sh --model=ALIAS):\n");
    println!("  {:<16} {:<52} {}", "ALIAS", "GGUF FILE", "SIZE");
    println!("  {}", "-".repeat(80));
    for m in MODELS {
        let default = if m.alias == "e4b" { " (default)" } else { "" };
        println!("  {:<16} {:<52} {}{}", m.alias, m.file, m.size, default);
    }
    println!("\n  Or pass a local .gguf file path directly.");
}

// --- Tool Call Parsing & Execution ---

#[derive(Deserialize, Debug)]
struct ToolCall {
    name: String,
    arguments: serde_json::Value,
}

#[derive(Serialize)]
struct ToolResult {
    name: String,
    success: bool,
    output: String,
}

fn parse_tool_calls(response: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut search = response;
    while let Some(start) = search.find("<tool_call>") {
        let after = &search[start + 11..];
        if let Some(end) = after.find("</tool_call>") {
            let json_str = after[..end].trim();
            if let Ok(call) = serde_json::from_str::<ToolCall>(json_str) {
                calls.push(call);
            }
            search = &after[end + 12..];
        } else {
            break;
        }
    }
    calls
}

fn execute_tool(call: &ToolCall) -> ToolResult {
    match call.name.as_str() {
        "bash" => {
            let cmd = call.arguments.get("command")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            eprintln!("[tool] bash: {cmd}");
            match Command::new("sh").arg("-c").arg(cmd).output() {
                Ok(out) => {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    let mut output = String::new();
                    if !stdout.is_empty() { output.push_str(&stdout); }
                    if !stderr.is_empty() {
                        if !output.is_empty() { output.push('\n'); }
                        output.push_str("[stderr] ");
                        output.push_str(&stderr);
                    }
                    if output.len() > 4000 {
                        output.truncate(4000);
                        output.push_str("\n[...truncated]");
                    }
                    ToolResult { name: "bash".into(), success: out.status.success(), output }
                }
                Err(e) => ToolResult { name: "bash".into(), success: false, output: format!("error: {e}") },
            }
        }
        "read_file" => {
            let path = call.arguments.get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            eprintln!("[tool] read_file: {path}");
            match fs::read_to_string(path) {
                Ok(content) => {
                    let mut output = content;
                    if output.len() > 8000 {
                        output.truncate(8000);
                        output.push_str("\n[...truncated]");
                    }
                    ToolResult { name: "read_file".into(), success: true, output }
                }
                Err(e) => ToolResult { name: "read_file".into(), success: false, output: format!("error: {e}") },
            }
        }
        "write_file" => {
            let path = call.arguments.get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let content = call.arguments.get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            eprintln!("[tool] write_file: {path} ({} bytes)", content.len());
            match fs::write(path, content) {
                Ok(()) => ToolResult { name: "write_file".into(), success: true, output: format!("wrote {} bytes to {path}", content.len()) },
                Err(e) => ToolResult { name: "write_file".into(), success: false, output: format!("error: {e}") },
            }
        }
        _ => ToolResult { name: call.name.clone(), success: false, output: format!("unknown tool: {}", call.name) },
    }
}

// --- GGUF Extraction ---

fn find_embedded_gguf() -> Result<Option<(PathBuf, u64, u64)>> {
    let exe = std::env::current_exe().context("cannot resolve own executable path")?;
    let mut f = fs::File::open(&exe).context("cannot open own executable")?;
    let file_len = f.metadata()?.len();
    if file_len < FOOTER_SIZE {
        return Ok(None);
    }
    f.seek(SeekFrom::End(-(FOOTER_SIZE as i64)))?;
    let mut footer = [0u8; 24];
    f.read_exact(&mut footer)?;
    let magic = &footer[16..24];
    if magic != MAGIC {
        return Ok(None);
    }
    let offset = u64::from_le_bytes(footer[0..8].try_into().unwrap());
    let length = u64::from_le_bytes(footer[8..16].try_into().unwrap());
    vlog!("  embedded GGUF: offset={offset}, length={length} ({:.2} GB)", length as f64 / 1e9);
    Ok(Some((exe, offset, length)))
}

fn extract_gguf(exe_path: &PathBuf, offset: u64, length: u64) -> Result<tempfile::NamedTempFile> {
    eprint!("Extracting embedded weights ({:.2} GB)...", length as f64 / 1e9);
    let t0 = Instant::now();
    let mut src = fs::File::open(exe_path)?;
    src.seek(SeekFrom::Start(offset))?;

    let tmp = tempfile::Builder::new()
        .prefix("gemma-native-")
        .suffix(".gguf")
        .tempfile()
        .context("cannot create temp file for GGUF")?;

    {
        let mut writer = io::BufWriter::new(tmp.as_file());
        let mut remaining = length;
        let mut buf = vec![0u8; 8 * 1024 * 1024];
        while remaining > 0 {
            let to_read = remaining.min(buf.len() as u64) as usize;
            let n = src.read(&mut buf[..to_read])?;
            if n == 0 { bail!("unexpected EOF extracting GGUF"); }
            writer.write_all(&buf[..n])?;
            remaining -= n as u64;
        }
        writer.flush()?;
    }
    eprintln!(" done ({:.1}s)", t0.elapsed().as_secs_f64());
    Ok(tmp)
}

// --- Gemma 4 Chat Template ---

fn apply_chat_template(messages: &[(String, String)], add_generation_prompt: bool) -> String {
    let mut out = String::new();
    for (role, content) in messages {
        let tag = match role.as_str() {
            "system" => "system",
            "user" => "user",
            "assistant" | "model" => "model",
            "tool" => "tool",
            other => other,
        };
        out.push_str(&format!("<start_of_turn>{tag}\n{content}<end_of_turn>\n"));
    }
    if add_generation_prompt {
        out.push_str("<start_of_turn>model\n");
    }
    out
}

// --- Model Loading & Generation ---

fn load_model(backend: &LlamaBackend, gguf_path: &str) -> Result<LlamaModel> {
    eprint!("Loading model...");
    let t0 = Instant::now();
    let model_params = pin!(LlamaModelParams::default().with_n_gpu_layers(1000));
    let model = LlamaModel::load_from_file(backend, gguf_path, &model_params)
        .map_err(|e| anyhow::anyhow!("model load failed: {e}"))?;
    eprintln!(" done ({:.1}s)", t0.elapsed().as_secs_f64());
    Ok(model)
}

fn generate_streaming(
    model: &LlamaModel,
    backend: &LlamaBackend,
    messages: &[(String, String)],
) -> Result<(String, usize, f64)> {
    let prompt = apply_chat_template(messages, true);

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(N_CTX).unwrap()));

    let mut ctx = model.new_context(backend, ctx_params)
        .map_err(|e| anyhow::anyhow!("context creation failed: {e}"))?;

    let tokens = model.str_to_token(&prompt, AddBos::Always)
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

    let prompt_tokens = tokens.len();
    vlog!("[gen] prompt tokens: {prompt_tokens}");

    let t0 = Instant::now();
    let mut batch = LlamaBatch::new(N_CTX as usize, 1);
    let last_idx = (tokens.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens.into_iter()) {
        batch.add(token, i, &[0], i == last_idx)?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| anyhow::anyhow!("prompt decode failed: {e}"))?;
    let prompt_time = t0.elapsed().as_secs_f64();
    vlog!("[gen] prompt: {prompt_time:.2}s ({:.0} tok/s)", prompt_tokens as f64 / prompt_time);

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::temp(0.7),
        LlamaSampler::dist(42),
    ]);

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut n_cur = batch.n_tokens();
    let mut response = String::new();
    let mut gen_tokens: usize = 0;
    let gen_start = Instant::now();

    // Buffer for detecting and suppressing special tokens during streaming
    let mut print_buf = String::new();

    while n_cur <= MAX_TOKENS {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let piece = model.token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| anyhow::anyhow!("token decode failed: {e}"))?;
        response.push_str(&piece);
        gen_tokens += 1;

        // Buffer output to detect and suppress special tags during streaming
        print_buf.push_str(&piece);

        // Check if buffer contains a complete special tag to suppress
        let mut suppressed = false;
        for tag in STRIP_TAGS {
            if print_buf.contains(tag) {
                // Flush everything before the tag, skip the tag itself
                if let Some(pos) = print_buf.find(tag) {
                    let before = &print_buf[..pos];
                    if !before.is_empty() {
                        print!("{before}");
                        io::stdout().flush()?;
                    }
                    print_buf = print_buf[pos + tag.len()..].to_string();
                    suppressed = true;
                    break;
                }
            }
        }

        // If no tag is being built up, flush the buffer
        if !suppressed {
            // Check if buffer could be the start of a special tag
            let might_be_tag = print_buf.starts_with('<') &&
                STRIP_TAGS.iter().any(|t| t.starts_with(print_buf.as_str()));
            if !might_be_tag {
                print!("{print_buf}");
                io::stdout().flush()?;
                print_buf.clear();
            }
        }

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("decode step failed: {e}"))?;
        n_cur += 1;
    }

    // Flush remaining buffer
    if !print_buf.is_empty() {
        let cleaned = strip_special_tokens(&print_buf);
        if !cleaned.is_empty() {
            print!("{cleaned}");
            io::stdout().flush()?;
        }
    }

    let gen_time = gen_start.elapsed().as_secs_f64();
    let total_time = t0.elapsed().as_secs_f64();
    vlog!("[gen] {gen_tokens} tok in {gen_time:.2}s ({:.1} tok/s)", gen_tokens as f64 / gen_time);

    // Clean special tokens from the stored response too
    let clean_response = strip_special_tokens(&response);
    Ok((clean_response, gen_tokens, total_time))
}

// --- Agent Loop (handles tool calls) ---

fn agent_turn(
    model: &LlamaModel,
    backend: &LlamaBackend,
    messages: &mut Vec<(String, String)>,
) -> Result<(usize, f64)> {
    let mut total_toks = 0;
    let mut total_time = 0.0;

    for round in 0..MAX_TOOL_ROUNDS {
        let (response, toks, elapsed) = generate_streaming(model, backend, messages)?;
        total_toks += toks;
        total_time += elapsed;

        let tool_calls = parse_tool_calls(&response);
        messages.push(("assistant".into(), response));

        if tool_calls.is_empty() {
            break;
        }

        vlog!("[agent] round {}: {} tool call(s)", round + 1, tool_calls.len());

        let mut results = Vec::new();
        for call in &tool_calls {
            let result = execute_tool(call);
            let status = if result.success { "ok" } else { "err" };
            println!("\n[{} -> {}]\n{}", call.name, status, result.output);
            results.push(result);
        }

        let results_json = serde_json::to_string_pretty(&results)?;
        messages.push(("tool".into(), results_json));
        println!();
    }

    Ok((total_toks, total_time))
}

// --- Main ---

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--list-models") {
        print_models();
        return Ok(());
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("Usage: gemma-native [OPTIONS] [GGUF_PATH]");
        println!();
        println!("Options:");
        println!("  --model=ALIAS    Model alias (see --list-models)");
        println!("  --list-models    List available model aliases");
        println!("  --verbose        Show llama.cpp internals and debug info");
        println!("  -h, --help       Show this help");
        println!();
        println!("Built-in tools: bash, read_file, write_file");
        println!("Shell escape: prefix input with ! (e.g. !ls -la)");
        return Ok(());
    }

    if args.iter().any(|a| a == "--verbose" || a == "-v") {
        VERBOSE.store(true, Ordering::Relaxed);
    }

    let model_arg = args.iter()
        .find(|a| a.starts_with("--model="))
        .map(|a| a.strip_prefix("--model=").unwrap().to_string())
        .or_else(|| args.iter().position(|a| a == "--model").and_then(|i| args.get(i + 1).cloned()));

    let total_start = Instant::now();

    // Resolve GGUF path
    let gguf_path: PathBuf;
    let _tmp_file: Option<tempfile::NamedTempFile>;

    if let Some(ref alias) = model_arg {
        if let Some(entry) = find_model(alias) {
            vlog!("model alias '{alias}' -> {} [{}]", entry.file, entry.size);
        }
    }

    if let Some((exe_path, offset, length)) = find_embedded_gguf()? {
        let tmp = extract_gguf(&exe_path, offset, length)?;
        gguf_path = tmp.path().to_path_buf();
        _tmp_file = Some(tmp);
    } else {
        let path = args.iter()
            .find(|a| !a.starts_with("--") && **a != args[0])
            .cloned()
            .or(model_arg.clone())
            .or_else(|| std::env::var("GEMMA_GGUF").ok())
            .unwrap_or_else(|| {
                eprintln!("No embedded weights found.");
                eprintln!("Usage: gemma-native <path-to-model.gguf>");
                eprintln!("       GEMMA_GGUF=/path/to/model.gguf gemma-native");
                eprintln!("       gemma-native --list-models");
                std::process::exit(1);
            });
        gguf_path = PathBuf::from(&path);
        if !gguf_path.exists() {
            bail!("GGUF not found: {}", gguf_path.display());
        }
        _tmp_file = None;
    }

    // Initialize backend and suppress llama.cpp logs
    let mut backend = LlamaBackend::init()
        .map_err(|e| anyhow::anyhow!("backend init failed: {e}"))?;
    if !verbose() {
        backend.void_logs();
    }

    // Load model
    let model = load_model(&backend, gguf_path.to_str().unwrap())?;

    let startup_time = total_start.elapsed().as_secs_f64();
    eprintln!("Ready ({startup_time:.1}s)\n");

    // Interactive agent loop
    let mut messages: Vec<(String, String)> = vec![
        ("system".into(), SYSTEM_PROMPT.into()),
    ];

    println!("Gemma 4 Agent (tools: bash, read_file, write_file)");
    println!("Type 'quit' to exit, '!' prefix for direct shell\n");

    loop {
        print!("gemma> ");
        io::stdout().flush()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        }
        let input = input.trim();
        if input.is_empty() { continue; }
        if input == "quit" || input == "exit" { break; }

        if let Some(cmd) = input.strip_prefix('!') {
            let output = Command::new("sh").arg("-c").arg(cmd).output()?;
            io::stdout().write_all(&output.stdout)?;
            io::stderr().write_all(&output.stderr)?;
            continue;
        }

        messages.push(("user".into(), input.to_string()));

        println!();
        let (toks, elapsed) = agent_turn(&model, &backend, &mut messages)?;
        let tps = if elapsed > 0.0 { toks as f64 / elapsed } else { 0.0 };

        let ctx_bytes = serde_json::to_string(&messages)?.len();
        println!("\n[{elapsed:.1}s | {toks} tok | {tps:.1} tok/s | ctx: {:.1}KB ~{}tok]\n",
            ctx_bytes as f64 / 1024.0, ctx_bytes / 4);
    }

    Ok(())
}
