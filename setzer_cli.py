"""CLI entry point for homedoc-subtitle-translator."""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional

from setzer_core import (
    Chunk,
    Transcript,
    build_output,
    make_chunks,
    read_transcript,
    translate_range,
)

__version__ = "0.1.0"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_value(name: str, default: str) -> str:
    return os.getenv(name, default)


class Logger:
    def __init__(self, *, file_path: Optional[Path], verbose: bool) -> None:
        self.file_path = file_path
        self.verbose = verbose
        self._handle = None
        if file_path is not None:
            try:
                self._handle = file_path.open("a", encoding="utf-8")
            except OSError:
                self._handle = None

    def log(self, message: str) -> None:
        timestamp = dt.datetime.now().isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"
        print(line)
        if self._handle is not None:
            try:
                self._handle.write(line + "\n")
                self._handle.flush()
            except OSError:
                self._handle = None

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            finally:
                self._handle = None


def _build_parser() -> argparse.ArgumentParser:
    server_default = _env_value("HOMEDOC_LLM_SERVER", "http://127.0.0.1:11434")
    model_default = _env_value("HOMEDOC_LLM_MODEL", "gemma3:12b")
    mode_default = _env_value("HOMEDOC_LLM_MODE", "auto")
    stream_default = _env_bool("HOMEDOC_STREAM", True)
    timeout_default = float(_env_value("HOMEDOC_HTTP_TIMEOUT", "60"))

    parser = argparse.ArgumentParser(
        description="Translate subtitle files using a local Ollama-compatible LLM.",
    )
    parser.add_argument("--in", dest="input_path", required=True, help="Input subtitle file (.srt/.vtt/.tsv)")
    parser.add_argument("--out", dest="output_dir", required=True, help="Output directory for generated files")
    parser.add_argument(
        "--flat",
        action="store_true",
        default=False,
        help="Write outputs directly into --out (default: %(default)s)",
    )
    parser.add_argument(
        "--no-flat",
        dest="flat",
        action="store_false",
        help="Write into timestamped folder within --out (default)",
    )
    parser.add_argument("--source", default="auto", help="Source language (default: %(default)s)")
    parser.add_argument("--target", default="English", help="Target language (default: %(default)s)")
    parser.add_argument(
        "--batch-per-chunk",
        type=int,
        default=1,
        help="Number of cues to send per LLM request (default: %(default)s)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=4000,
        help="Maximum characters per chunk when planning (default: %(default)s)",
    )
    parser.add_argument(
        "--no-translate-bracketed",
        dest="translate_bracketed",
        action="store_false",
        default=True,
        help="Preserve bracketed tags like [MUSIC] without translation",
    )
    parser.add_argument(
        "--server",
        default=server_default,
        help=f"LLM server URL (default: {server_default})",
    )
    parser.add_argument(
        "--model",
        default=model_default,
        help=f"LLM model tag (default: {model_default})",
    )
    parser.add_argument(
        "--llm-mode",
        choices=["auto", "generate", "chat"],
        default=mode_default,
        help=f"LLM mode (default: {mode_default})",
    )
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        default=stream_default,
        help="Enable streaming responses",
    )
    stream_group.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable streaming responses",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=timeout_default,
        help=f"HTTP timeout in seconds (default: {timeout_default})",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM calls and reuse original text",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging and capture raw LLM responses",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _resolve_output_directory(base: Path, flat: bool) -> Path:
    if flat:
        base.mkdir(parents=True, exist_ok=True)
        return base
    tz_name = os.getenv("HOMEDOC_TZ")
    now = dt.datetime.now()
    if tz_name:
        try:
            from zoneinfo import ZoneInfo

            tz = ZoneInfo(tz_name)
            now = dt.datetime.now(tz)
        except Exception:
            pass
    folder_name = now.strftime("%Y%m%d-%H%M%S")
    target = base / folder_name
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_file(path: Path, content: str) -> None:
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Unable to write {path.name}: {exc}") from exc


def _output_filename(fmt: str) -> str:
    mapping = {"srt": "report.srt", "vtt": "report.vtt", "tsv": "report.tsv"}
    try:
        return mapping[fmt]
    except KeyError as exc:
        raise RuntimeError(f"Unknown transcript format: {fmt}") from exc


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    try:
        transcript = read_transcript(str(input_path))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        chunks = make_chunks(transcript.cues, args.max_chars)
    except Exception as exc:
        print(f"Error preparing chunks: {exc}", file=sys.stderr)
        return 1

    target_dir = _resolve_output_directory(output_dir, args.flat)
    log_path = target_dir / "homedoc.log"
    logger = Logger(file_path=log_path, verbose=args.debug)
    raw_lines: List[str] = []

    def raw_handler(payload: str) -> None:
        raw_lines.append(payload)

    logger.log(f"Loaded transcript with {len(transcript.cues)} cues in {transcript.fmt.upper()} format")
    logger.log(f"Planned {len(chunks)} chunk(s) with max {args.max_chars} characters")

    try:
        translate_range(
            transcript,
            chunks,
            server=args.server,
            model=args.model,
            source=args.source,
            target=args.target,
            batch_n=args.batch_per_chunk,
            translate_bracketed=args.translate_bracketed,
            llm_mode=args.llm_mode,
            stream=args.stream,
            timeout=args.timeout,
            no_llm=args.no_llm,
            logger=logger.log,
            raw_handler=raw_handler if (args.debug or args.stream) else None,
            verbose=args.debug,
        )
    except Exception as exc:
        logger.log(f"Translation failed: {exc}")
        logger.close()
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    timestamp = dt.datetime.now(dt.timezone.utc).astimezone()
    vtt_note = f"translated-with model={args.model} time={timestamp.isoformat()}"
    try:
        result = build_output(transcript, vtt_note=vtt_note if transcript.fmt == "vtt" else None)
    except Exception as exc:
        logger.log(f"Failed to render output: {exc}")
        logger.close()
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_path = target_dir / _output_filename(transcript.fmt)
    try:
        _write_file(output_path, result)
    except Exception as exc:
        logger.log(str(exc))
        logger.close()
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    logger.log(f"Wrote output to {output_path}")

    if args.debug or args.stream:
        raw_path = target_dir / "llm_raw.txt"
        try:
            raw_path.write_text("\n".join(raw_lines), encoding="utf-8")
            logger.log(f"Captured raw LLM payloads in {raw_path}")
        except OSError:
            logger.log("Unable to write llm_raw.txt")

    logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
