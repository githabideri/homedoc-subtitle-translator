# Usage

`setzer` translates subtitle files with a homedoc-style CLI. All flags are
available through the `homedoc-subtitle-translator` alias as well.

## Flags

| Flag | Type | Default | Env Var | Description |
| --- | --- | --- | --- | --- |
| `--in` | path | required | – | Input subtitle (`.srt`, `.vtt`, `.tsv`). |
| `--out` | path | required | – | Output directory. |
| `--flat` / `--no-flat` | bool | `--no-flat` | – | Write directly into `--out` or into a timestamped subfolder. |
| `--source` | text | `auto` | – | Source language hint. |
| `--target` | text | `English` | – | Target language. |
| `--batch-per-chunk` | int | `1` | – | Number of cues per LLM request when chunking. |
| `--max-chars` | int | `4000` | – | Planning size for chunk generation. |
| `--no-translate-bracketed` | bool | disabled | – | Preserve bracketed tags such as `[MUSIC]`. |
| `--server` | URL | `http://127.0.0.1:11434` | `HOMEDOC_LLM_SERVER` | Ollama-compatible server URL. |
| `--model` | text | `gemma3:12b` | `HOMEDOC_LLM_MODEL` | Model identifier to request. |
| `--llm-mode` | choice | `auto` | `HOMEDOC_LLM_MODE` | Prefer `chat`, `generate`, or auto-switching. |
| `--stream` / `--no-stream` | bool | env / `True` | `HOMEDOC_STREAM` | Stream responses line-by-line. |
| `--timeout` | float | `60` | `HOMEDOC_HTTP_TIMEOUT` | HTTP timeout in seconds. |
| `--no-llm` | bool | disabled | – | Skip the LLM and reuse original text. |
| `--debug` | bool | disabled | – | Verbose logging and raw payload capture. |
| `--version` | flag | – | – | Print version and exit. |

Environment variables provide defaults when the related flags are omitted. If
`HOMEDOC_STREAM` is `0` or `false`, streaming is disabled unless `--stream` is
explicitly provided. When `HOMEDOC_TZ` is set, folder mode uses that timezone
for the `<YYYYMMDD-HHMMSS>` output name.

## Examples

### Minimal dry run

```bash
setzer --in demo.srt --out ./out --no-llm --flat
```

### Translate an SRT file (timestamped folder)

```bash
setzer --in drama.srt --out ./translated --server http://127.0.0.1:11434 --model gemma3:12b
```

### Flat output placement

```bash
setzer --in talk.vtt --out ./translated --flat
```

### Batch mode

```bash
setzer --in lessons.srt --out ./translated --batch-per-chunk 8 --max-chars 6000
```

### Preserve bracketed tags

```bash
setzer --in concert.srt --out ./translated --no-translate-bracketed
```

### LLM mode controls

```bash
setzer --in doc.vtt --out ./translated --llm-mode chat
setzer --in doc.vtt --out ./translated --llm-mode generate
```

### Toggle streaming

```bash
setzer --in demo.tsv --out ./translated --stream
setzer --in demo.tsv --out ./translated --no-stream
```

### Short timeout demo

```bash
setzer --in drama.srt --out ./translated --server http://127.0.0.1:65535 --timeout 1
# -> Expect a timeout error and non-zero exit code.
```

## GUI Wrapper

`setzer-gui` launches a minimal Tk application with the same core translation
logic. Use the GUI to plan chunks, run full translations, or process a single
chunk at a time. The `Abort` button stops queued work before the next LLM call.
