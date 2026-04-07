use std::io::{self, Read, Seek, SeekFrom, Write};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::pin::pin;
use std::process::Command;
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
const FOOTER_SIZE: u64 = 24; // 8 (offset) + 8 (length) + 8 (magic)
const N_CTX: u32 = 4096;
const MAX_TOKENS: i32 = 2048;
const SYSTEM_PROMPT: &str = "You are a helpful assistant.";

// --- Model Registry ---

struct ModelEntry {
    alias: &'static str,
    repo: &'static str,
    file: &'static str,
    size: &'static str,
}

const MODELS: &[ModelEntry] = &[
    // E4B variants (default)
    ModelEntry { alias: "e4b",          repo: "bartowski/google_gemma-4-E4B-it-GGUF", file: "google_gemma-4-E4B-it-Q4_K_M.gguf",  size: "~5.4 GB" },
    ModelEntry { alias: "e4b-q8",       repo: "bartowski/google_gemma-4-E4B-it-GGUF", file: "google_gemma-4-E4B-it-Q8_0.gguf",    size: "~8.0 GB" },
    ModelEntry { alias: "e4b-q4ks",     repo: "bartowski/google_gemma-4-E4B-it-GGUF", file: "google_gemma-4-E4B-it-Q4_K_S.gguf",  size: "~5.2 GB" },
    ModelEntry { alias: "e4b-q3km",     repo: "bartowski/google_gemma-4-E4B-it-GGUF", file: "google_gemma-4-E4B-it-Q3_K_M.gguf",  size: "~4.9 GB" },
    ModelEntry { alias: "e4b-iq4xs",    repo: "bartowski/google_gemma-4-E4B-it-GGUF", file: "google_gemma-4-E4B-it-IQ4_XS.gguf",  size: "~5.1 GB" },
    // E2B variants (smaller)
    ModelEntry { alias: "e2b",          repo: "bartowski/google_gemma-4-E2B-it-GGUF", file: "google_gemma-4-E2B-it-Q4_K_M.gguf",  size: "~2.0 GB" },
    ModelEntry { alias: "e2b-q8",       repo: "bartowski/google_gemma-4-E2B-it-GGUF", file: "google_gemma-4-E2B-it-Q8_0.gguf",    size: "~3.0 GB" },
    ModelEntry { alias: "e2b-q4ks",     repo: "bartowski/google_gemma-4-E2B-it-GGUF", file: "google_gemma-4-E2B-it-Q4_K_S.gguf",  size: "~1.9 GB" },
    ModelEntry { alias: "e2b-q3km",     repo: "bartowski/google_gemma-4-E2B-it-GGUF", file: "google_gemma-4-E2B-it-Q3_K_M.gguf",  size: "~1.8 GB" },
    ModelEntry { alias: "e2b-iq4xs",    repo: "bartowski/google_gemma-4-E2B-it-GGUF", file: "google_gemma-4-E2B-it-IQ4_XS.gguf",  size: "~1.8 GB" },
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

// --- Phase timing helper ---

struct Phase {
    name: String,
    start: Instant,
}

impl Phase {
    fn begin(name: &str) -> Self {
        let bar = "=".repeat(60);
        eprintln!("\n{bar}");
        eprintln!("[PHASE] {name}");
        eprintln!("{bar}");
        Self { name: name.to_string(), start: Instant::now() }
    }
    fn done(self) {
        let elapsed = self.start.elapsed();
        eprintln!("[DONE]  {} ({:.2}s)\n", self.name, elapsed.as_secs_f64());
    }
}

// --- GGUF Extraction ---

fn find_embedded_gguf() -> Result<Option<(PathBuf, u64, u64)>> {
    let exe = std::env::current_exe().context("cannot resolve own executable path")?;
    let mut f = std::fs::File::open(&exe).context("cannot open own executable")?;
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
    eprintln!("  embedded GGUF found: offset={offset}, length={length} ({:.2} GB)", length as f64 / 1e9);
    Ok(Some((exe, offset, length)))
}

fn extract_gguf(exe_path: &PathBuf, offset: u64, length: u64) -> Result<tempfile::NamedTempFile> {
    let phase = Phase::begin("Extracting embedded GGUF weights");

    let mut src = std::fs::File::open(exe_path)?;
    src.seek(SeekFrom::Start(offset))?;

    let tmp = tempfile::Builder::new()
        .prefix("gemma-native-")
        .suffix(".gguf")
        .tempfile()
        .context("cannot create temp file for GGUF")?;

    {
        let mut writer = io::BufWriter::new(tmp.as_file());
        let mut remaining = length;
        let mut buf = vec![0u8; 8 * 1024 * 1024]; // 8MB chunks
        let mut written: u64 = 0;
        let report_interval = length / 10;
        let mut next_report = report_interval;

        while remaining > 0 {
            let to_read = remaining.min(buf.len() as u64) as usize;
            let n = src.read(&mut buf[..to_read])?;
            if n == 0 { bail!("unexpected EOF extracting GGUF"); }
            writer.write_all(&buf[..n])?;
            remaining -= n as u64;
            written += n as u64;
            if written >= next_report {
                let pct = (written as f64 / length as f64) * 100.0;
                eprintln!("  extracting... {pct:.0}% ({:.2} GB / {:.2} GB)", written as f64 / 1e9, length as f64 / 1e9);
                next_report += report_interval;
            }
        }
        writer.flush()?;
    }
    eprintln!("  extracted {:.2} GB to {}", length as f64 / 1e9, tmp.path().display());
    phase.done();
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
    let phase = Phase::begin(&format!("Loading model from {gguf_path}"));

    let model_params = pin!(LlamaModelParams::default().with_n_gpu_layers(1000));
    eprintln!("  n_gpu_layers: 1000 (offload everything)");

    let model = LlamaModel::load_from_file(backend, gguf_path, &model_params)
        .map_err(|e| anyhow::anyhow!("model load failed: {e}"))?;

    eprintln!("  vocab size: {}", model.n_vocab());
    phase.done();
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
    eprintln!("[gen] prompt tokens: {prompt_tokens}");

    // Feed prompt
    let t0 = Instant::now();
    let mut batch = LlamaBatch::new(N_CTX as usize, 1);
    let last_idx = (tokens.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens.into_iter()) {
        batch.add(token, i, &[0], i == last_idx)?;
    }
    ctx.decode(&mut batch)
        .map_err(|e| anyhow::anyhow!("prompt decode failed: {e}"))?;
    let prompt_time = t0.elapsed().as_secs_f64();
    eprintln!("[gen] prompt processed in {prompt_time:.2}s ({:.0} tok/s)", prompt_tokens as f64 / prompt_time);

    // Sample
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

    while n_cur <= MAX_TOKENS {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let piece = model.token_to_piece(token, &mut decoder, true, None)
            .map_err(|e| anyhow::anyhow!("token decode failed: {e}"))?;
        print!("{piece}");
        io::stdout().flush()?;
        response.push_str(&piece);
        gen_tokens += 1;

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("decode step failed: {e}"))?;
        n_cur += 1;
    }

    let _gen_time = gen_start.elapsed().as_secs_f64();
    let total_time = t0.elapsed().as_secs_f64();
    Ok((response, gen_tokens, total_time))
}

// --- Main ---

fn main() -> Result<()> {
    // Parse CLI args
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--list-models") {
        print_models();
        return Ok(());
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("Usage: gemma-native [OPTIONS] [GGUF_PATH]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --model=ALIAS    Model alias (see --list-models)");
        eprintln!("  --list-models    List available model aliases");
        eprintln!("  -h, --help       Show this help");
        eprintln!();
        eprintln!("If no GGUF path or --model is given, uses embedded weights.");
        eprintln!("Set GEMMA_GGUF env var as fallback.");
        return Ok(());
    }
    let model_arg = args.iter()
        .find(|a| a.starts_with("--model="))
        .map(|a| a.strip_prefix("--model=").unwrap().to_string())
        .or_else(|| args.iter().position(|a| a == "--model").and_then(|i| args.get(i + 1).cloned()));

    let total_start = Instant::now();
    eprintln!(r#"
   ╔═══════════════════════════════════════════╗
   ║     gemma-native v0.1.0                   ║
   ║     Self-contained Gemma 4 Agent          ║
   ╚═══════════════════════════════════════════╝
"#);

    // Phase 1: Resolve GGUF path
    let phase = Phase::begin("Resolving model weights");

    let gguf_path: PathBuf;
    let _tmp_file: Option<tempfile::NamedTempFile>; // prevent drop

    // If --model=alias given, show which model it maps to
    if let Some(ref alias) = model_arg {
        if let Some(entry) = find_model(alias) {
            eprintln!("  model alias '{alias}' -> {} ({}) [{}]", entry.file, entry.repo, entry.size);
            eprintln!("  note: --model selects for build.sh; at runtime, pass the GGUF path");
        }
    }

    if let Some((exe_path, offset, length)) = find_embedded_gguf()? {
        phase.done();
        let tmp = extract_gguf(&exe_path, offset, length)?;
        gguf_path = tmp.path().to_path_buf();
        _tmp_file = Some(tmp);
    } else {
        // Fall back to positional arg, --model path, or env var
        let path = args.get(1)
            .filter(|a| !a.starts_with("--"))
            .cloned()
            .or(model_arg.clone())
            .or_else(|| std::env::var("GEMMA_GGUF").ok())
            .unwrap_or_else(|| {
                eprintln!("  no embedded weights found");
                eprintln!("  usage: gemma-native <path-to-model.gguf>");
                eprintln!("  or:    gemma-native --model=e4b  (for build.sh)");
                eprintln!("  or:    GEMMA_GGUF=/path/to/model.gguf gemma-native");
                eprintln!();
                eprintln!("  run --list-models to see available aliases");
                std::process::exit(1);
            });
        gguf_path = PathBuf::from(&path);
        if !gguf_path.exists() {
            bail!("GGUF not found: {}", gguf_path.display());
        }
        let size = std::fs::metadata(&gguf_path)?.len();
        eprintln!("  using external GGUF: {} ({:.2} GB)", gguf_path.display(), size as f64 / 1e9);
        _tmp_file = None;
        phase.done();
    }

    // Phase 2: Initialize backend
    let phase = Phase::begin("Initializing llama.cpp backend");
    let backend = LlamaBackend::init()
        .map_err(|e| anyhow::anyhow!("backend init failed: {e}"))?;
    phase.done();

    // Phase 3: Load model
    let model = load_model(&backend, gguf_path.to_str().unwrap())?;

    let startup_time = total_start.elapsed().as_secs_f64();
    eprintln!("============================================================");
    eprintln!("[READY] Total startup: {startup_time:.2}s");
    eprintln!("============================================================");

    // Phase 4: Interactive loop
    let mut messages: Vec<(String, String)> = vec![
        ("system".into(), SYSTEM_PROMPT.into()),
    ];

    println!("\nGemma 4 Native Chat (type 'quit' to exit)\n");

    loop {
        print!("gemma> ");
        io::stdout().flush()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            break; // EOF
        }
        let input = input.trim();
        if input.is_empty() { continue; }
        if input == "quit" || input == "exit" { break; }

        // Shell escape
        if let Some(cmd) = input.strip_prefix('!') {
            let output = Command::new("sh").arg("-c").arg(cmd).output()?;
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stdout.is_empty() { print!("{stdout}"); }
            if !stderr.is_empty() { eprint!("{stderr}"); }
            continue;
        }

        messages.push(("user".into(), input.to_string()));

        println!();
        let (response, toks, elapsed) = generate_streaming(&model, &backend, &messages)?;
        let tps = if elapsed > 0.0 { toks as f64 / elapsed } else { 0.0 };
        messages.push(("assistant".into(), response));

        let ctx_bytes = serde_json::to_string(&messages)?.len();
        println!("\n[{elapsed:.1}s | {toks} tok | {tps:.1} tok/s | ctx: {:.1}KB ~{}tok]",
            ctx_bytes as f64 / 1024.0, ctx_bytes / 4);
        println!();
    }

    eprintln!("\n[total session: {:.1}s]", total_start.elapsed().as_secs_f64());
    Ok(())
}
