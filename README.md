# HomeDoc — Subtitle Translator aka setzer

![Version](https://img.shields.io/badge/version-0.1.2-blue?style=flat-square)
![License](https://img.shields.io/badge/license-GPL--3.0--or--later-brightgreen?style=flat-square)

Part of the HomeDoc scripts that use Ollama-compatible large language models to do useful work completely local. HomeDoc — Subtitle Translator aka setzer (from "Übersetzer" - translator - in German) translates subtitle files (`.srt`, `.vtt`, and `.tsv`) from any language into any other language, although the quality and capability depends on the respective model. The project follows the homedoc toolkit style with a CLI-first workflow and an optional Tk GUI wrapper. The settings are displayed in GUI and can be either used in CLI workflow or to load old setting.

# Disclaimer

This and other scripts (as well as accompanying texts/files/documentation) are written by LLMs (mostly GPT-5), so be aware of potential security issues or plain nonsense; never run code that you haven't inspected. I tried to minimize the potential damage by sticking to the very simple approach of single file scripts (or in this case triple file scripts) with as little dependencies as possible.

If you want to commit, feel free to fork, mess around and put "ai slop" on my "ai slop", or maybe deslop it enirely, but there is no garantuee that I will incorporate changes.

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

# custom output template and format override
setzer --in drama.srt --out ./out --flat --outfmt vtt --outfile "~/subs/{basename}.{dst}.{fmt}"
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
values for the LLM server, model, and streaming options. Run `setzer --help` (or see [USAGE.md](USAGE.md)) to
see all arguments, or the alias `homedoc-subtitle-translator` for the same
behaviour.

## Outputs

Every CLI invocation writes:

- Translated subtitles following the `{basename}.{dst}.{fmt}` template by
  default. Adjust with `--outfile` to include placeholders such as
  `{basename}`, `{src}`, `{dst}`, `{fmt}`, and `{ts}` or to point to a custom
  folder.
- `homedoc.log` — timestamped log of the run.
- `llm_raw.txt` — raw LLM payloads when streaming or `--debug` is active.

By default results are placed in `--out/<YYYYMMDD-HHMMSS>/`. Use `--flat` to
only write output file into the specified output directory.

## Notes & Safety

- `--debug` logs raw LLM payloads into `llm_raw.txt`. Inspect before sharing
  outputs beyond your local environment.
- No network calls are made beyond the configured Ollama-compatible endpoint.
- The optional `setzer-gui` command provides a thin Tk wrapper that calls the
  same core translation routines.

## License

Distributed under the terms of the GPL-3.0-or-later license. See [LICENSE](LICENSE).
