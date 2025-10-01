# homedoc-subtitle-translator

Translate `.srt`, `.vtt`, and `.tsv` subtitle files with a local Ollama-compatible
large language model. The project follows the homedoc toolkit style with a
CLI-first workflow and an optional Tk GUI wrapper.

## Run or Install

```bash
# install with pipx (recommended)
pipx install .

# or run from a local checkout
pip install --user .

# run without installing (from repo root)
python setzer_cli.py --help
```

Quick examples:

```bash
# dry run (no LLM calls)
setzer --in samples/demo.srt --out ./out --no-llm --flat

# translate with explicit server/model and streaming enabled
setzer \
  --in samples/demo.srt \
  --out ./out \
  --server http://127.0.0.1:11434 \
  --model gemma3:12b \
  --stream

# disable streaming
setzer --in demo.vtt --out ./out --no-stream
```

See [USAGE.md](USAGE.md) for the full flag reference and examples.

## Update setzer

To upgrade an existing pipx installation, reinstall the package:

```bash
pipx reinstall homedoc-subtitle-translator
```

If you installed with `pip`, run:

```bash
pip install --user --upgrade homedoc-subtitle-translator
```

## Remove setzer

Uninstall with the same tool you used originally:

```bash
# pipx install
pipx uninstall homedoc-subtitle-translator

# pip install
pip uninstall homedoc-subtitle-translator
```

## Usage

The CLI keeps homedoc-style defaults: environment variables provide fallback
values for the LLM server, model, and streaming options. Run `setzer --help` to
see all arguments, or the alias `homedoc-subtitle-translator` for the same
behaviour.

## Outputs

Every CLI invocation writes:

- `report.<ext>` — rewritten subtitle file matching the input format.
- `homedoc.log` — timestamped log of the run.
- `llm_raw.txt` — raw LLM payloads when streaming or `--debug` is active.

By default results are placed in `--out/<YYYYMMDD-HHMMSS>/`. Use `--flat` to
write directly into the specified output directory.

## Notes & Safety

- `--debug` logs raw LLM payloads into `llm_raw.txt`. Inspect before sharing
  outputs beyond your local environment.
- No network calls are made beyond the configured Ollama-compatible endpoint.
- The optional `setzer-gui` command provides a thin Tk wrapper that calls the
  same core translation routines.

## License

Distributed under the terms of the GPL-3.0-or-later license. See [LICENSE](LICENSE).
