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

// --- Constants ---

const MAGIC: &[u8; 8] = b"GMNAPAK\0";
const FOOTER_SIZE: u64 = 24;
const N_CTX: u32 = 8192;
const MAX_TOKENS: i32 = 2048;
const MAX_TOOL_ROUNDS: usize = 10;

static VERBOSE: AtomicBool = AtomicBool::new(false);
fn verbose() -> bool { VERBOSE.load(Ordering::Relaxed) }
macro_rules! vlog {
    ($($arg:tt)*) => { if verbose() { eprintln!($($arg)*); } }
}

// --- Tool System Prompt ---
// Use plain-text JSON format that any instruction model can parse reliably.
// The Gemma 4 native <|tool> tokens require special token IDs that str_to_token
// doesn't handle, so we use a universal JSON-based approach instead.

const SYSTEM_PROMPT: &str = r#"You are a helpful assistant with access to tools. When you need to use a tool, output a JSON tool call wrapped in markers like this:

<tool_call>
{"name": "bash", "args": {"command": "date"}}
</tool_call>

Available tools:
- bash: Execute a shell command. Args: {"command": "..."}
- read_file: Read a file. Args: {"path": "..."}
- write_file: Write a file. Args: {"path": "...", "content": "..."}

IMPORTANT: When a question requires real-time information (like the current time, date, files on disk, etc.), you MUST use a tool. Do NOT say you cannot access this information — use the bash tool instead. You may use multiple tool calls in one response. After you receive tool results, incorporate them into your answer."#;

// --- Special tokens to strip from display ---

const STRIP_TAGS: &[&str] = &[
    "<end_of_turn>",
    "<start_of_turn>model",
    "<start_of_turn>user",
    "<start_of_turn>system",
    "<start_of_turn>tool",
    "<start_of_turn>",
    "<turn|>",
    "<|turn>model",
    "<|turn>user",
    "<|turn>system",
    "<|turn>",
];

fn strip_special_tokens(text: &str) -> String {
    let mut out = text.to_string();
    for tag in STRIP_TAGS {
        out = out.replace(tag, "");
    }
    // Also strip tool_call blocks from display (they're handled separately)
    while let Some(start) = out.find("<|tool_call>") {
        if let Some(end) = out[start..].find("<tool_call|>") {
            out.replace_range(start..start + end + 12, "");
        } else {
            break;
        }
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

// --- Tool Call Parsing (JSON-based) ---

#[derive(serde::Deserialize, Debug)]
struct ToolCallJson {
    name: String,
    args: serde_json::Value,
}

struct ParsedToolCall {
    name: String,
    args: serde_json::Value,
}

/// Parse <tool_call>{"name":"bash","args":{"command":"date"}}</tool_call> from response
fn parse_tool_calls(response: &str) -> Vec<ParsedToolCall> {
    let mut calls = Vec::new();
    let mut search = response;
    while let Some(start) = search.find("<tool_call>") {
        let after = &search[start + 11..];
        if let Some(end) = after.find("</tool_call>") {
            let json_str = after[..end].trim();
            if let Ok(call) = serde_json::from_str::<ToolCallJson>(json_str) {
                calls.push(ParsedToolCall { name: call.name, args: call.args });
            }
            search = &after[end + 12..];
        } else {
            break;
        }
    }
    calls
}

// --- Tool Execution ---

fn execute_tool(call: &ParsedToolCall) -> (bool, String) {
    let get_arg = |key: &str| -> String {
        call.args.get(key).and_then(|v| v.as_str()).unwrap_or("").to_string()
    };

    match call.name.as_str() {
        "bash" => {
            let cmd = get_arg("command");
            eprintln!("[tool] bash: {cmd}");
            match Command::new("sh").arg("-c").arg(&cmd).output() {
                Ok(out) => {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    let mut output = String::new();
                    if !stdout.is_empty() { output.push_str(&stdout); }
                    if !stderr.is_empty() {
                        if !output.is_empty() { output.push('\n'); }
                        output.push_str(&stderr);
                    }
                    if output.len() > 4000 {
                        output.truncate(4000);
                        output.push_str("\n[...truncated]");
                    }
                    (out.status.success(), output)
                }
                Err(e) => (false, format!("error: {e}")),
            }
        }
        "read_file" => {
            let path = get_arg("path");
            eprintln!("[tool] read_file: {path}");
            match fs::read_to_string(&path) {
                Ok(mut content) => {
                    if content.len() > 8000 {
                        content.truncate(8000);
                        content.push_str("\n[...truncated]");
                    }
                    (true, content)
                }
                Err(e) => (false, format!("error: {e}")),
            }
        }
        "write_file" => {
            let path = get_arg("path");
            let content = get_arg("content");
            eprintln!("[tool] write_file: {path} ({} bytes)", content.len());
            match fs::write(&path, &content) {
                Ok(()) => (true, format!("wrote {} bytes to {path}", content.len())),
                Err(e) => (false, format!("error: {e}")),
            }
        }
        _ => (false, format!("unknown tool: {}", call.name)),
    }
}

// --- GGUF Extraction ---

fn find_embedded_gguf() -> Result<Option<(PathBuf, u64, u64)>> {
    let exe = std::env::current_exe().context("cannot resolve own executable path")?;
    let mut f = fs::File::open(&exe).context("cannot open own executable")?;
    let file_len = f.metadata()?.len();
    if file_len < FOOTER_SIZE { return Ok(None); }
    f.seek(SeekFrom::End(-(FOOTER_SIZE as i64)))?;
    let mut footer = [0u8; 24];
    f.read_exact(&mut footer)?;
    if &footer[16..24] != MAGIC { return Ok(None); }
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
    let tmp = tempfile::Builder::new().prefix("gemma-agent-").suffix(".gguf").tempfile()
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
// Uses <start_of_turn>/<end_of_turn> for turn structure (as the GGUF tokenizer expects)
// Tool definitions go in the system turn using <|tool>...<tool|>

fn build_prompt(messages: &[(String, String)], add_generation_prompt: bool) -> String {
    let mut out = String::new();
    for (role, content) in messages {
        let tag = match role.as_str() {
            "system" => "system",
            "user" => "user",
            "assistant" | "model" => "model",
            "model_with_tools" => "model", // model turn that includes tool call+response
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

fn generate(
    model: &LlamaModel,
    backend: &LlamaBackend,
    prompt: &str,
) -> Result<(String, usize, f64)> {
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(NonZeroU32::new(N_CTX).unwrap()))
        .with_n_batch(N_CTX);
    let mut ctx = model.new_context(backend, ctx_params)
        .map_err(|e| anyhow::anyhow!("context creation failed: {e}"))?;

    let tokens = model.str_to_token(prompt, AddBos::Always)
        .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

    let n_tokens = tokens.len();
    vlog!("[gen] prompt tokens: {n_tokens}");

    let t0 = Instant::now();
    let mut batch = LlamaBatch::new(N_CTX as usize, 1);
    let last_idx = (n_tokens - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens.into_iter()) {
        batch.add(token, i, &[0], i == last_idx)?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| anyhow::anyhow!("prompt decode failed: {e}"))?;
    vlog!("[gen] prompt decoded in {:.1}s", t0.elapsed().as_secs_f64());

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::min_p(0.05, 1),
        LlamaSampler::temp(0.7),
        LlamaSampler::dist(42),
    ]);

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut n_cur = batch.n_tokens();
    let mut response = String::new();
    let mut gen_tokens: usize = 0;

    while n_cur <= MAX_TOKENS {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if model.is_eog_token(token) { break; }

        let piece = model.token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| anyhow::anyhow!("token decode failed: {e}"))?;
        response.push_str(&piece);
        gen_tokens += 1;

        // Stop on tool call completion
        if response.contains("</tool_call>") {
            break;
        }
        // Stop on end-of-turn markers
        if response.contains("<end_of_turn>") || response.contains("<turn|>") {
            break;
        }
        // Stop if model starts hallucinating new turns (only after some real content)
        let stripped = strip_special_tokens(&response);
        if stripped.len() > 5 && (
            response.contains("<start_of_turn>user")
            || response.contains("<start_of_turn>system")
            || response.contains("<|turn>user")
        ) {
            break;
        }
        // Degenerate repetition detector: if model keeps generating the same tag
        if gen_tokens > 20 && response.matches("<start_of_turn>").count() > 3 {
            break;
        }

        // Stream cleaned output (suppress tool call syntax while building)
        if !response.contains("<|tool_call>") {
            print!("{piece}");
            io::stdout().flush()?;
        }

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("decode step failed: {e}"))?;
        n_cur += 1;
    }

    let total_time = t0.elapsed().as_secs_f64();
    Ok((response, gen_tokens, total_time))
}

// --- Agent Turn ---

fn agent_turn(
    model: &LlamaModel,
    backend: &LlamaBackend,
    messages: &mut Vec<(String, String)>,
) -> Result<(usize, f64)> {
    let mut total_toks = 0;
    let mut total_time = 0.0;

    for round in 0..MAX_TOOL_ROUNDS {
        let prompt = build_prompt(messages, true);
        vlog!("[agent] round {} prompt:\n{}", round + 1, &prompt[prompt.len().saturating_sub(200)..]);

        let (response, toks, elapsed) = generate(model, backend, &prompt)?;
        total_toks += toks;
        total_time += elapsed;

        let tool_calls = parse_tool_calls(&response);

        if tool_calls.is_empty() {
            // No tool calls — this is the final response
            let clean = strip_special_tokens(&response);
            messages.push(("assistant".into(), clean));
            break;
        }

        vlog!("[agent] round {}: {} tool call(s)", round + 1, tool_calls.len());

        // Store the model's response (with tool calls) as assistant message
        messages.push(("assistant".into(), response.clone()));

        // Execute tools and build result message
        let mut results_text = String::from("Tool results:\n");
        for call in &tool_calls {
            let (success, output) = execute_tool(call);
            let status = if success { "ok" } else { "error" };
            println!("\n[tool:{} -> {}] {}", call.name, status, output.lines().next().unwrap_or(""));
            vlog!("[tool] full output:\n{output}");
            results_text.push_str(&format!("\n{}({}): {}\n", call.name, status, output));
        }

        // Add tool results as a user message so the model can incorporate them
        messages.push(("user".into(), results_text));
        println!();
    }

    Ok((total_toks, total_time))
}

// --- CLI arg helpers ---

fn get_arg_value(args: &[String], prefix: &str) -> Option<String> {
    // --flag=value
    args.iter()
        .find(|a| a.starts_with(prefix))
        .and_then(|a| a.strip_prefix(prefix))
        .map(|s| s.to_string())
        // --flag value
        .or_else(|| {
            let bare = prefix.trim_end_matches('=');
            args.iter().position(|a| a == bare).and_then(|i| args.get(i + 1).cloned())
        })
}

// --- Main ---

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--list-models") {
        print_models();
        return Ok(());
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("Usage: gemma-agent [OPTIONS] [GGUF_PATH]");
        println!();
        println!("Options:");
        println!("  --model=ALIAS      Model alias (see --list-models)");
        println!("  --prompt=\"...\"      Send a single prompt and exit");
        println!("  --list-models      List available model aliases");
        println!("  --verbose          Show debug info");
        println!("  -h, --help         Show this help");
        println!();
        println!("Built-in tools: bash, read_file, write_file");
        return Ok(());
    }

    if args.iter().any(|a| a == "--verbose" || a == "-v") {
        VERBOSE.store(true, Ordering::Relaxed);
    }

    let model_arg = get_arg_value(&args, "--model=");
    let prompt_arg = get_arg_value(&args, "--prompt=");

    let total_start = Instant::now();

    // Resolve GGUF path
    let gguf_path: PathBuf;
    let _tmp_file: Option<tempfile::NamedTempFile>;

    if let Some(ref alias) = model_arg {
        if let Some(entry) = find_model(alias) {
            vlog!("model: '{alias}' -> {} [{}]", entry.file, entry.size);
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
                eprintln!("Run ./build.sh to create a self-contained binary,");
                eprintln!("or pass a GGUF file path:");
                eprintln!("  gemma-agent <path-to-model.gguf>");
                std::process::exit(1);
            });
        // Check if user passed a model alias instead of a file path
        if find_model(&path).is_some() {
            bail!("'{}' is a model alias, not a file path.\n\
                   Use: ./build.sh --model={}\n\
                   Or download the GGUF and pass the file path directly.", path, path);
        }
        gguf_path = PathBuf::from(&path);
        if !gguf_path.exists() {
            bail!("GGUF file not found: {}", gguf_path.display());
        }
        _tmp_file = None;
    }

    // Initialize backend
    let mut backend = LlamaBackend::init()
        .map_err(|e| anyhow::anyhow!("backend init failed: {e}"))?;
    if !verbose() { backend.void_logs(); }

    let model = load_model(&backend, gguf_path.to_str().unwrap())?;

    let startup_time = total_start.elapsed().as_secs_f64();
    eprintln!("Ready ({startup_time:.1}s)\n");

    let mut messages: Vec<(String, String)> = vec![
        ("system".into(), SYSTEM_PROMPT.into()),
    ];

    // Single prompt mode
    if let Some(prompt) = prompt_arg {
        messages.push(("user".into(), prompt.clone()));
        eprintln!("[prompt] {prompt}\n");
        let (toks, elapsed) = agent_turn(&model, &backend, &mut messages)?;
        let tps = if elapsed > 0.0 { toks as f64 / elapsed } else { 0.0 };
        eprintln!("\n[{elapsed:.1}s | {toks} tok | {tps:.1} tok/s]");
        return Ok(());
    }

    // Interactive loop
    println!("Gemma 4 Agent (tools: bash, read_file, write_file)");
    println!("Type 'quit' to exit, '!' prefix for direct shell\n");

    loop {
        print!("gemma> ");
        io::stdout().flush()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 { break; }
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

        let ctx_bytes: usize = messages.iter().map(|(_, c)| c.len()).sum();
        println!("\n[{elapsed:.1}s | {toks} tok | {tps:.1} tok/s | ctx: {:.1}KB ~{}tok]\n",
            ctx_bytes as f64 / 1024.0, ctx_bytes / 4);
    }

    Ok(())
}
