"""Core logic for homedoc-subtitle-translator.

This module keeps all parsing, formatting, chunking, and LLM communication in a
single stdlib-only location so the CLI and GUI can share it without re-
implementing behaviours.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class Cue:
    index: int
    start: str
    end: str
    text: str
    translated: Optional[str] = None


@dataclass
class Transcript:
    fmt: str
    cues: List[Cue]
    header: str = ""
    tsv_header: Optional[List[str]] = None
    tsv_cols: Optional[Tuple[int, int, int]] = None


@dataclass
class Chunk:
    cid: int
    start_idx: int
    end_idx: int
    charcount: int
    status: str = "pending"
    err: Optional[str] = None


class TranscriptError(RuntimeError):
    """Raised for parsing and formatting problems."""


class LLMError(RuntimeError):
    """Raised when the LLM could not be contacted or returned malformed data."""


_FORMAT_GUESS_RE = {
    "srt": re.compile(r"^\s*\d+\s*\n\s*\d\d:\d\d:\d\d[,\.]\d\d\d\s*-->", re.MULTILINE),
    "vtt": re.compile(r"^\s*WEBVTT", re.IGNORECASE),
    "tsv": re.compile(r"\t"),
}


def detect_format(text: str, filename: str = "") -> str:
    """Heuristic format detection based on extension and content."""

    ext = os.path.splitext(filename.lower())[1]
    if ext in {".srt"}:
        return "srt"
    if ext in {".vtt"}:
        return "vtt"
    if ext in {".tsv", ".csv"}:
        return "tsv"

    stripped = text.lstrip()
    if _FORMAT_GUESS_RE["vtt"].search(stripped):
        return "vtt"
    if _FORMAT_GUESS_RE["srt"].search(stripped):
        return "srt"
    if _FORMAT_GUESS_RE["tsv"].search(stripped):
        return "tsv"
    raise TranscriptError("Unable to detect subtitle format from input content.")


def read_transcript(path: str) -> Transcript:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = handle.read()
    except OSError as exc:
        raise TranscriptError(f"Unable to read input file: {exc}") from exc

    if not data.strip():
        raise TranscriptError("Input file is empty.")

    fmt = detect_format(data, filename=path)
    if fmt == "srt":
        return parse_srt(data)
    if fmt == "vtt":
        return parse_vtt(data)
    if fmt == "tsv":
        return parse_tsv(data)
    raise TranscriptError(f"Unsupported subtitle format: {fmt}")




def _clean_lines(text: str) -> List[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def parse_srt(text: str) -> Transcript:
    lines = _clean_lines(text)
    cues: List[Cue] = []
    block: List[str] = []

    def flush(block_lines: List[str]) -> None:
        if not block_lines:
            return
        filtered = [line.replace("\ufeff", "") for line in block_lines]
        while filtered and not filtered[0].strip():
            filtered.pop(0)
        while filtered and not filtered[-1].strip():
            filtered.pop()
        if len(filtered) < 2:
            return
        first = filtered[0].strip()
        second = filtered[1] if len(filtered) > 1 else ""
        text_start = 2
        try:
            idx = int(first)
            time_line = second
        except ValueError:
            idx = len(cues) + 1
            if "-->" in filtered[0]:
                time_line = filtered[0]
                text_start = 1
            else:
                time_line = filtered[1]
        text_lines = filtered[text_start:]
        start, end = _split_times(time_line)
        cues.append(Cue(index=idx, start=start, end=end, text="\n".join(text_lines)))

    for line in lines:
        if line.strip() == "":
            flush(block)
            block = []
        else:
            block.append(line)
    flush(block)

    if not cues:
        raise TranscriptError("No cues parsed from SRT file.")

    return Transcript(fmt="srt", cues=cues)


def parse_vtt(text: str) -> Transcript:
    lines = _clean_lines(text)
    header_lines: List[str] = []
    cues: List[Cue] = []
    block: List[str] = []
    seen_header = False

    for line in lines:
        if not seen_header:
            header_lines.append(line)
            if line.strip() == "":
                seen_header = True
            continue
        if line.strip() == "":
            if block:
                _flush_vtt_block(block, cues)
                block = []
        else:
            block.append(line)
    if block:
        _flush_vtt_block(block, cues)

    header = "\n".join(header_lines).strip("\n")
    if not header.upper().startswith("WEBVTT"):
        header = "WEBVTT" + ("\n" + header if header else "")

    if not cues:
        raise TranscriptError("No cues parsed from VTT file.")

    return Transcript(fmt="vtt", cues=cues, header=header)


def _flush_vtt_block(block: List[str], cues: List[Cue]) -> None:
    if not block:
        return
    time_line = None
    text_start = 0
    if "-->" in block[0]:
        time_line = block[0]
        text_start = 1
    else:
        for idx, candidate in enumerate(block[1:], start=1):
            if "-->" in candidate:
                time_line = candidate
                text_start = idx + 1
                break
    if not time_line:
        return
    start, end = _split_times(time_line)
    text = "\n".join(block[text_start:])
    index = len(cues) + 1
    cues.append(Cue(index=index, start=start, end=end, text=text))


def parse_tsv(text: str) -> Transcript:
    import csv

    reader = csv.reader(_clean_lines(text), delimiter="\t")
    try:
        header = next(reader)
    except StopIteration as exc:
        raise TranscriptError("TSV appears empty.") from exc

    header_lower = [h.lower() for h in header]
    start_idx, end_idx, text_idx = _infer_tsv_columns(header_lower)

    cues: List[Cue] = []
    for idx, row in enumerate(reader, start=1):
        if not row:
            continue
        try:
            start = row[start_idx]
            end = row[end_idx]
            text_val = row[text_idx]
        except IndexError:
            raise TranscriptError(
                f"Row {idx} does not contain required columns (expected at least {text_idx+1} columns)."
            )
        cues.append(Cue(index=idx, start=start, end=end, text=text_val))

    if not cues:
        raise TranscriptError("No cues parsed from TSV file.")

    return Transcript(fmt="tsv", cues=cues, tsv_header=header, tsv_cols=(start_idx, end_idx, text_idx))


def _infer_tsv_columns(header_lower: List[str]) -> Tuple[int, int, int]:
    start_idx = _first_with_keywords(header_lower, ["start", "begin"])
    end_idx = _first_with_keywords(header_lower, ["end", "finish"])
    text_idx = _first_with_keywords(header_lower, ["text", "subtitle", "caption", "transcript"])

    if start_idx is None or end_idx is None or text_idx is None:
        # Fall back to first three columns.
        if len(header_lower) < 3:
            raise TranscriptError("TSV header must contain at least three columns.")
        return 0, 1, 2
    return start_idx, end_idx, text_idx


def _first_with_keywords(header: List[str], keywords: Iterable[str]) -> Optional[int]:
    for idx, value in enumerate(header):
        for key in keywords:
            if key in value:
                return idx
    return None


def _split_times(time_line: str) -> Tuple[str, str]:
    parts = time_line.split("-->")
    if len(parts) < 2:
        raise TranscriptError(f"Unable to parse cue timing line: {time_line!r}")
    start = parts[0].strip()
    end_part = parts[1].strip().split()[0]
    return start, end_part


def write_srt(transcript: Transcript) -> str:
    segments = []
    for idx, cue in enumerate(transcript.cues, start=1):
        text = cue.translated if cue.translated is not None else cue.text
        segments.append(f"{idx}\n{cue.start} --> {cue.end}\n{text}")
    return "\n\n".join(segments) + "\n"


def write_vtt(transcript: Transcript, note: Optional[str] = None) -> str:
    header = transcript.header or "WEBVTT"
    lines = [header.strip()]
    lines.append("")
    if note:
        lines.append(f"NOTE {note}")
        lines.append("")
    for cue in transcript.cues:
        text = cue.translated if cue.translated is not None else cue.text
        lines.append(f"{cue.start} --> {cue.end}")
        lines.extend(text.split("\n"))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_tsv(transcript: Transcript) -> str:
    import csv
    from io import StringIO

    buffer = StringIO()
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    header = transcript.tsv_header or ["start", "end", "text"]
    writer.writerow(header)
    start_idx, end_idx, text_idx = transcript.tsv_cols or (0, 1, 2)
    for cue in transcript.cues:
        row = [""] * max(len(header), text_idx + 1)
        row[start_idx] = cue.start
        row[end_idx] = cue.end
        row[text_idx] = cue.translated if cue.translated is not None else cue.text
        writer.writerow(row)
    return buffer.getvalue()


def build_output(transcript: Transcript, vtt_note: Optional[str] = None) -> str:
    if transcript.fmt == "srt":
        return write_srt(transcript)
    if transcript.fmt == "vtt":
        return write_vtt(transcript, note=vtt_note)
    if transcript.fmt == "tsv":
        return write_tsv(transcript)
    raise TranscriptError(f"Cannot build output for unknown format: {transcript.fmt}")


def make_chunks(cues: List[Cue], max_chars: int) -> List[Chunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    chunks: List[Chunk] = []
    cid = 1
    char_total = 0
    start_pos = 0
    for pos, cue in enumerate(cues):
        text_len = len(cue.text)
        if char_total and char_total + text_len > max_chars:
            chunks.append(
                Chunk(
                    cid=cid,
                    start_idx=start_pos + 1,
                    end_idx=pos,
                    charcount=char_total,
                )
            )
            cid += 1
            start_pos = pos
            char_total = 0
        char_total += text_len
    if cues:
        chunks.append(
            Chunk(
                cid=cid,
                start_idx=start_pos + 1,
                end_idx=len(cues),
                charcount=char_total,
            )
        )
    return chunks


def _http_json(url: str, payload: Dict[str, object], timeout: float, *, stream: bool,
               raw_handler: Optional[Callable[[str], None]] = None) -> str:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            if not stream:
                body = resp.read().decode("utf-8", errors="replace")
                if raw_handler:
                    raw_handler(body)
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError as exc:
                    raise LLMError(f"Malformed JSON response from server: {exc}") from exc
                return _extract_message(parsed)
            else:
                pieces: List[str] = []
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    if raw_handler:
                        raw_handler(line)
                    cleaned = line
                    if cleaned.startswith("data:"):
                        cleaned = cleaned[5:].strip()
                    if not cleaned or cleaned == "[DONE]":
                        continue
                    try:
                        parsed = json.loads(cleaned)
                    except json.JSONDecodeError:
                        continue
                    piece = _extract_stream_piece(parsed)
                    if piece:
                        pieces.append(piece)
                if not pieces:
                    raise LLMError("Streamed response contained no usable content.")
                return "".join(pieces)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        if raw_handler and detail:
            raw_handler(detail)
        raise LLMError(f"HTTP error {exc.code} from LLM server: {detail[:200]}") from exc
    except URLError as exc:
        raise LLMError(f"Failed to reach LLM server: {exc}") from exc


def _extract_message(payload: Dict[str, object]) -> str:
    if "message" in payload and isinstance(payload["message"], dict):
        content = payload["message"].get("content")
        if isinstance(content, str):
            return content
    if "messages" in payload and isinstance(payload["messages"], list):
        messages = payload["messages"]
        if messages:
            content = messages[-1].get("content")
            if isinstance(content, str):
                return content
    if "response" in payload and isinstance(payload["response"], str):
        return payload["response"]
    if "text" in payload and isinstance(payload["text"], str):
        return payload["text"]
    raise LLMError("LLM response missing message content.")


def _extract_stream_piece(payload: Dict[str, object]) -> str:
    if "message" in payload and isinstance(payload["message"], dict):
        content = payload["message"].get("content")
        if isinstance(content, str):
            return content
    if "response" in payload and isinstance(payload["response"], str):
        return payload["response"]
    if "text" in payload and isinstance(payload["text"], str):
        return payload["text"]
    return ""


_TAG_RE = re.compile(r"<[^>]+>")
_BRACKET_RE = re.compile(r"\[[^\]]+\]")
_LEADING_MARKER_RE = re.compile(
    r"^\s*(?:CUE|OUTPUT|TRANSLATION|TRANSLATED|RESPONSE|ANSWER)\s*:\s*",
    re.IGNORECASE,
)


def _protect_tags(text: str) -> Tuple[str, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    counter = 0
    def repl(match: re.Match[str]) -> str:
        nonlocal counter
        placeholder = f"__TAG{counter}__"
        mapping[placeholder] = match.group(0)
        counter += 1
        return placeholder
    protected = _TAG_RE.sub(repl, text)
    return protected, mapping


def _protect_brackets(text: str) -> Tuple[str, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    counter = 0
    def repl(match: re.Match[str]) -> str:
        nonlocal counter
        placeholder = f"__BR{counter}__"
        mapping[placeholder] = match.group(0)
        counter += 1
        return placeholder
    protected = _BRACKET_RE.sub(repl, text)
    return protected, mapping


def _restore_placeholders(text: str, mapping: Dict[str, str]) -> str:
    for placeholder, value in mapping.items():
        text = text.replace(placeholder, value)
    return text


def _cleanup_translation(text: str) -> str:
    if not text:
        return text

    cleaned = text.lstrip("\ufeff")
    match = _LEADING_MARKER_RE.match(cleaned)
    if match:
        cleaned = cleaned[match.end():]
        if cleaned.startswith("\r\n"):
            cleaned = cleaned[2:]
        elif cleaned.startswith("\n"):
            cleaned = cleaned[1:]
    return cleaned


def llm_translate_single(
    text: str,
    *,
    source: str,
    target: str,
    model: str,
    server: str,
    translate_bracketed: bool,
    llm_mode: str,
    stream: bool,
    timeout: float,
    raw_handler: Optional[Callable[[str], None]] = None,
) -> str:
    prepared, tag_map = _protect_tags(text)
    bracket_map: Dict[str, str] = {}
    if not translate_bracketed:
        prepared, bracket_map = _protect_brackets(prepared)

    prompt = (
        "Translate the following subtitle cue from {src} to {dst}. "
        "Preserve placeholders, formatting, and whitespace exactly."
    ).format(src=source or "auto-detected language", dst=target)

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise subtitle translator."},
            {
                "role": "user",
                "content": f"{prompt}\n\nCUE:\n{prepared}",
            },
        ],
        "stream": stream,
    }

    raw_result = _perform_llm_call(
        server=server,
        mode=llm_mode,
        body=body,
        generate_prompt=f"{prompt}\n\n{prepared}",
        stream=stream,
        timeout=timeout,
        raw_handler=raw_handler,
    )

    if not raw_result:
        return text
    cleaned = _cleanup_translation(raw_result)
    if not cleaned.strip():
        return text
    return _restore_placeholders(cleaned, {**tag_map, **bracket_map})


def llm_translate_batch(
    pairs: List[Tuple[str, str]],
    *,
    source: str,
    target: str,
    model: str,
    server: str,
    llm_mode: str,
    stream: bool,
    timeout: float,
    translate_bracketed: bool,
    raw_handler: Optional[Callable[[str], None]] = None,
) -> List[Tuple[str, str]]:
    protected_pairs: List[Tuple[str, str, Dict[str, str], Dict[str, str]]] = []
    inputs: List[str] = []
    for pid, text in pairs:
        prepared, tag_map = _protect_tags(text)
        bracket_map: Dict[str, str] = {}
        if not translate_bracketed:
            prepared, bracket_map = _protect_brackets(prepared)
        inputs.append(f"{pid}|||{prepared}")
        protected_pairs.append((pid, prepared, tag_map, bracket_map))

    instructions = (
        "Translate each input line from {src} to {dst}. "
        "Return one line per input in the form ID|||TRANSLATION. "
        "Preserve placeholders and whitespace."
    ).format(src=source or "auto-detected language", dst=target)

    joined = "\n".join(inputs)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You translate subtitles in bulk."},
            {
                "role": "user",
                "content": f"{instructions}\n\nINPUT:\n{joined}",
            },
        ],
        "stream": stream,
    }

    result = _perform_llm_call(
        server=server,
        mode=llm_mode,
        body=body,
        generate_prompt=f"{instructions}\n\n{joined}",
        stream=stream,
        timeout=timeout,
        raw_handler=raw_handler,
    )

    mapping: Dict[str, str] = {}
    for line in result.splitlines():
        if "|||" not in line:
            continue
        pid, translated = line.split("|||", 1)
        mapping[pid.strip()] = translated

    output: List[Tuple[str, str]] = []
    for pid, prepared, tag_map, bracket_map in protected_pairs:
        translated = mapping.get(pid)
        if translated is None:
            restored = _restore_placeholders(prepared, {**tag_map, **bracket_map})
        else:
            cleaned = _cleanup_translation(translated)
            if not cleaned.strip():
                restored = _restore_placeholders(prepared, {**tag_map, **bracket_map})
            else:
                restored = _restore_placeholders(cleaned, {**tag_map, **bracket_map})
        output.append((pid, restored))
    return output


def _perform_llm_call(
    *,
    server: str,
    mode: str,
    body: Dict[str, object],
    generate_prompt: str,
    stream: bool,
    timeout: float,
    raw_handler: Optional[Callable[[str], None]] = None,
) -> str:
    mode = (mode or "auto").lower()
    errors: List[str] = []

    def request_chat() -> str:
        url = server.rstrip("/") + "/api/chat"
        return _http_json(url, body, timeout, stream=stream, raw_handler=raw_handler)

    def request_generate() -> str:
        payload = {
            "model": body.get("model"),
            "prompt": generate_prompt,
            "stream": stream,
        }
        url = server.rstrip("/") + "/api/generate"
        return _http_json(url, payload, timeout, stream=stream, raw_handler=raw_handler)

    if mode == "chat":
        return request_chat()
    if mode == "generate":
        return request_generate()
    # auto: try chat then generate.
    try:
        return request_chat()
    except LLMError as exc:
        errors.append(str(exc))
        return request_generate()


def translate_range(
    transcript: Transcript,
    chunks: List[Chunk],
    *,
    server: str,
    model: str,
    source: str,
    target: str,
    batch_n: int,
    translate_bracketed: bool,
    llm_mode: str,
    stream: bool,
    timeout: float,
    no_llm: bool = False,
    logger: Optional[Callable[[str], None]] = None,
    raw_handler: Optional[Callable[[str], None]] = None,
    verbose: bool = False,
) -> None:
    if batch_n < 1:
        raise ValueError("batch_n must be >= 1")

    for chunk in chunks:
        if logger and verbose:
            logger(
                f"Processing chunk {chunk.cid} covering cues {chunk.start_idx}-{chunk.end_idx}"
            )
        start = chunk.start_idx - 1
        end = chunk.end_idx
        cues_slice = transcript.cues[start:end]
        try:
            if no_llm:
                for cue in cues_slice:
                    cue.translated = cue.text
                chunk.status = "done"
                continue
            if batch_n == 1:
                for cue in cues_slice:
                    translated = llm_translate_single(
                        cue.text,
                        source=source,
                        target=target,
                        model=model,
                        server=server,
                        translate_bracketed=translate_bracketed,
                        llm_mode=llm_mode,
                        stream=stream,
                        timeout=timeout,
                        raw_handler=raw_handler,
                    )
                    if translated:
                        cue.translated = translated
                    else:
                        cue.translated = cue.text
                        if logger:
                            logger(
                                f"Warning: empty translation for cue {cue.index}; reused original text"
                            )
            else:
                batch: List[Tuple[str, str]] = []
                for cue in cues_slice:
                    batch.append((str(cue.index), cue.text))
                    if len(batch) == batch_n:
                        missing = _apply_batch(
                            batch,
                            cues_slice,
                            source,
                            target,
                            model,
                            server,
                            llm_mode,
                            stream,
                            timeout,
                            translate_bracketed,
                            raw_handler,
                        )
                        if missing and logger:
                            logger(
                                "Warning: missing translations for IDs "
                                + ", ".join(missing)
                            )
                        batch = []
                if batch:
                    missing = _apply_batch(
                        batch,
                        cues_slice,
                        source,
                        target,
                        model,
                        server,
                        llm_mode,
                        stream,
                        timeout,
                        translate_bracketed,
                        raw_handler,
                    )
                    if missing and logger:
                        logger(
                            "Warning: missing translations for IDs "
                            + ", ".join(missing)
                        )
            chunk.status = "done"
        except LLMError as exc:
            chunk.status = "error"
            chunk.err = str(exc)
            if logger:
                logger(f"Error processing chunk {chunk.cid}: {exc}")
            raise RuntimeError(f"Chunk {chunk.cid} failed: {exc}") from exc


def _apply_batch(
    batch: List[Tuple[str, str]],
    cues_slice: List[Cue],
    source: str,
    target: str,
    model: str,
    server: str,
    llm_mode: str,
    stream: bool,
    timeout: float,
    translate_bracketed: bool,
    raw_handler: Optional[Callable[[str], None]],
) -> List[str]:
    translated_pairs = llm_translate_batch(
        batch,
        source=source,
        target=target,
        model=model,
        server=server,
        llm_mode=llm_mode,
        stream=stream,
        timeout=timeout,
        translate_bracketed=translate_bracketed,
        raw_handler=raw_handler,
    )
    mapping = {pid: text for pid, text in translated_pairs}
    cue_index = {str(cue.index): cue for cue in cues_slice}
    missing: List[str] = []
    for pid, _ in batch:
        cue = cue_index.get(pid)
        if not cue:
            missing.append(pid)
            continue
        translated = mapping.get(pid)
        if translated is None or not translated.strip():
            cue.translated = cue.text
            missing.append(pid)
        else:
            cue.translated = translated
    return missing

