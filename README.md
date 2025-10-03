# HomeDoc — Subtitle Translator aka setzer

![Version](https://img.shields.io/badge/version-0.1.2-blue?style=flat-square)
![License](https://img.shields.io/badge/license-GPL--3.0--or--later-brightgreen?style=flat-square)

Do useful work fully locally with Ollama-compatible large language models. No cloud needed.

HomeDoc — Subtitle Translator aka setzer (from "Übersetzer" - translator - in German) translates subtitle files (`.srt`, `.vtt`, and `.tsv`) from any language into any other language, although the quality and capability depends on the respective model. The project follows the homedoc toolkit style with a CLI-first workflow and an optional Tk GUI wrapper. The settings are displayed in GUI and can be either used in CLI workflow or to load old setting. It should work on all platforms (Linux, MacOS and Windows) as long as you have Ollama API accessible (either on the machine itself or on LAN).

<img width="600" alt="Screenshot of the HomeDoc - subtitle translator aka setzer version 0.1.2" src="https://github.com/user-attachments/assets/69cda40d-a494-48b6-9896-f75dfd66d447" />

This is how the GUI looks like in (Fedora Linux) Cinnamon, while being busy chugging through the Czechoslovak classic "Král Šumavy" to produce some lovely německy subtitles.

## Disclaimer

This and other scripts (as well as accompanying texts/files/documentation) are written by LLMs (mostly GPT-5), so be aware of potential security issues or plain nonsense; never run code that you haven't inspected. I tried to minimize the potential damage by sticking to the very simple approach of single file scripts (or in this case triple file scripts) with as little dependencies as possible.

If you want to commit, feel free to fork, mess around and put "ai slop" on my "ai slop", or maybe deslop it enirely, but there is no garantuee that I will incorporate changes.

## Prerequisites

- **Python 3.9** or newer (matches the package metadata in
  `pyproject.toml`). Use your system interpreter or a virtual environment.
- **Ollama-compatible server** accessible via HTTP, either from a local
  Ollama installation (default `http://127.0.0.1:11434`) or a reachable host on
  your LAN/WAN. Configure the URL with `--server` or the `SETZER_SERVER`
  environment variable when running the CLI.



## Run or Install

```bash
# verify python version (3.9 or higher)
python3 --version

# verify ollama accessibility (for local Ollama installation, change if needed)
curl http://127.0.0.1:11434

# alternatively open that link in browser, should display "Ollama is running" either way

# fetch the sources
git clone https://github.com/homedoc-ai/homedoc-subtitle-translator.git
cd homedoc-subtitle-translator

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
