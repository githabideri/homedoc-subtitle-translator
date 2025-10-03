# Changelog

All notable changes to this project will be documented in this file.

## [0.1.2] - 2025-10-03
### Added
- Allow overriding the rendered subtitle format with `--outfmt` and custom file templates via `--outfile`, including placeholders such as `{basename}`, `{src}`, `{dst}`, `{fmt}`, and `{ts}`.
- Teach the GUI to surface the available template placeholders, mirror the new CLI options, and preload settings when launched with CLI-style arguments.

### Changed
- Reuse the shared `resolve_outfile` helper when translating from the GUI so template handling matches the CLI path.

## [0.1.1] - 2025-10-02
### Added
- Display a live CLI command preview in the GUI so users can copy the equivalent terminal invocation.
- Provide a `--cues-per-request`/`--batch-per-chunk` flag to control how many subtitle cues are sent to the LLM per request.

### Changed
- Refined subtitle cleanup heuristics to strip noisy tokens from LLM responses before writing outputs.

## [0.1.0] - 2025-10-01
### Added
- Initial release with a CLI for translating `.srt`, `.vtt`, and `.tsv` subtitles using a local Ollama-compatible LLM.
- Optional Tk GUI wrapper sharing the same translation pipeline and environment-variable defaults.
- Streaming support, bracketed text preservation, and run logging with `homedoc.log`/`llm_raw.txt` artifacts.
