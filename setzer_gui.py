"""Optional Tk GUI for homedoc-subtitle-translator."""
from __future__ import annotations

import datetime as dt
import queue
import re
import shlex
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext
from typing import List, Optional

from setzer_core import (
    Chunk,
    Transcript,
    build_output,
    make_chunks,
    read_transcript,
    translate_range,
)


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("homedoc-subtitle-translator")

        self.input_var = tk.StringVar()
        default_output = Path.cwd() / "results"
        self.output_var = tk.StringVar(value=str(default_output))
        self.source_var = tk.StringVar(value="auto")
        self.target_var = tk.StringVar(value="English")
        self.server_var = tk.StringVar(value="http://127.0.0.1:11434")
        self.model_var = tk.StringVar(value="gemma3:12b")
        self.cues_per_request_var = tk.IntVar(value=1)
        self.max_chars_var = tk.IntVar(value=4000)
        self.bracket_var = tk.BooleanVar(value=True)
        self.stream_var = tk.BooleanVar(value=True)
        self.flat_var = tk.BooleanVar(value=False)
        self.no_llm_var = tk.BooleanVar(value=False)
        self.cli_preview_var = tk.StringVar()

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.transcript: Optional[Transcript] = None
        self.chunks: List[Chunk] = []
        self.abort_event = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.input_path: Optional[Path] = None
        self._trace_tokens: List[str] = []

        self._build_layout()
        self._register_variable_traces()
        self._update_cli_preview()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(150, self._drain_logs)

    def _build_layout(self) -> None:
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        def add_row(label: str, widget: tk.Widget, row: int) -> None:
            tk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6))
            widget.grid(row=row, column=1, sticky="ew")

        frame.columnconfigure(1, weight=1)

        # Input path
        input_entry = tk.Entry(frame, textvariable=self.input_var)
        add_row("Input", input_entry, 0)
        tk.Button(frame, text="Browse", command=self._browse_input).grid(row=0, column=2, padx=(6, 0))

        # Output directory
        output_entry = tk.Entry(frame, textvariable=self.output_var)
        add_row("Output", output_entry, 1)
        tk.Button(frame, text="Browse", command=self._browse_output).grid(row=1, column=2, padx=(6, 0))

        add_row("Source", tk.Entry(frame, textvariable=self.source_var), 2)
        add_row("Target", tk.Entry(frame, textvariable=self.target_var), 3)
        add_row("Server", tk.Entry(frame, textvariable=self.server_var), 4)
        add_row("Model", tk.Entry(frame, textvariable=self.model_var), 5)
        cues_spin = tk.Spinbox(frame, from_=1, to=50, textvariable=self.cues_per_request_var, width=6)
        add_row("Cues/request", cues_spin, 6)
        add_row("Max chars", tk.Entry(frame, textvariable=self.max_chars_var), 7)

        options_frame = tk.Frame(frame)
        options_frame.grid(row=8, column=0, columnspan=3, sticky="w", pady=(4, 4))
        tk.Checkbutton(options_frame, text="Translate bracketed", variable=self.bracket_var).pack(side=tk.LEFT)
        tk.Checkbutton(options_frame, text="Stream", variable=self.stream_var).pack(side=tk.LEFT, padx=(10, 0))
        tk.Checkbutton(options_frame, text="Flat output", variable=self.flat_var).pack(side=tk.LEFT, padx=(10, 0))
        tk.Checkbutton(options_frame, text="Dry run (no LLM)", variable=self.no_llm_var).pack(side=tk.LEFT, padx=(10, 0))

        self.chunk_list = tk.Listbox(frame, height=6)
        frame.rowconfigure(9, weight=1)
        self.chunk_list.grid(row=9, column=0, columnspan=3, sticky="nsew", pady=(6, 6))

        button_frame = tk.Frame(frame)
        button_frame.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(4, 4))
        tk.Button(button_frame, text="Build/Update chunks", command=self.build_chunks).pack(side=tk.LEFT)
        tk.Button(button_frame, text="Translate ALL", command=self.translate_all).pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(button_frame, text="Translate selected chunk", command=self.translate_selected).pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(button_frame, text="Abort", command=self.abort).pack(side=tk.LEFT, padx=(6, 0))

        self.console = scrolledtext.ScrolledText(frame, height=12, state=tk.DISABLED)
        self.console.grid(row=11, column=0, columnspan=3, sticky="nsew")
        frame.rowconfigure(11, weight=2)

        preview_label = tk.Label(frame, text="CLI preview")
        preview_label.grid(row=12, column=0, sticky="w", pady=(6, 0))
        preview_entry = tk.Entry(
            frame,
            textvariable=self.cli_preview_var,
            state="readonly",
        )
        preview_entry.grid(row=12, column=1, columnspan=2, sticky="ew", pady=(6, 0))

    def _register_variable_traces(self) -> None:
        variables: List[tk.Variable] = [
            self.input_var,
            self.output_var,
            self.source_var,
            self.target_var,
            self.server_var,
            self.model_var,
            self.cues_per_request_var,
            self.max_chars_var,
            self.bracket_var,
            self.stream_var,
            self.flat_var,
            self.no_llm_var,
        ]
        for var in variables:
            token = var.trace_add("write", self._update_cli_preview)
            self._trace_tokens.append(token)

    def _update_cli_preview(self, *_: object) -> None:
        self.cli_preview_var.set(self._format_cli_command())

    def _format_cli_command(self) -> str:
        args: List[str] = ["python", "-m", "setzer_cli"]

        input_path = self.input_var.get().strip() or "<input.srt>"
        output_dir = self.output_var.get().strip() or "<output-directory>"
        args.extend(["--in", input_path])
        args.extend(["--out", output_dir])

        source = self.source_var.get().strip() or "auto"
        target = self.target_var.get().strip() or "English"
        server = self.server_var.get().strip() or "http://127.0.0.1:11434"
        model = self.model_var.get().strip() or "gemma3:12b"
        cues = max(1, int(self.cues_per_request_var.get() or 1))
        max_chars = max(1, int(self.max_chars_var.get() or 4000))

        args.extend(["--source", source])
        args.extend(["--target", target])
        args.extend(["--server", server])
        args.extend(["--model", model])
        args.extend(["--cues-per-request", str(cues)])
        args.extend(["--max-chars", str(max_chars)])

        if self.flat_var.get():
            args.append("--flat")
        else:
            args.append("--no-flat")

        if not self.bracket_var.get():
            args.append("--no-translate-bracketed")

        if self.stream_var.get():
            args.append("--stream")
        else:
            args.append("--no-stream")

        args.extend(["--llm-mode", "auto"])
        args.extend(["--timeout", "60"])

        if self.no_llm_var.get():
            args.append("--no-llm")

        return shlex.join(args)

    def _browse_input(self) -> None:
        filename = filedialog.askopenfilename(filetypes=[("Subtitles", "*.srt *.vtt *.tsv"), ("All", "*.*")])
        if filename:
            self.input_var.set(filename)

    def _browse_output(self) -> None:
        directory = filedialog.askdirectory()
        if directory:
            self.output_var.set(directory)

    def log(self, message: str) -> None:
        timestamp = dt.datetime.now().isoformat(timespec="seconds")
        self.log_queue.put(f"[{timestamp}] {message}")

    def _drain_logs(self) -> None:
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.console.configure(state=tk.NORMAL)
            self.console.insert(tk.END, line + "\n")
            self.console.configure(state=tk.DISABLED)
            self.console.see(tk.END)
        self.root.after(150, self._drain_logs)

    def build_chunks(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Busy", "Wait for the current job to finish or abort it first.")
            return
        input_path = self.input_var.get().strip()
        if not input_path:
            messagebox.showerror("Missing input", "Please choose an input subtitle file.")
            return
        try:
            transcript = read_transcript(input_path)
            chunks = make_chunks(transcript.cues, int(self.max_chars_var.get()))
        except Exception as exc:
            self.log(f"Failed to load transcript: {exc}")
            messagebox.showerror("Error", str(exc))
            return
        self.input_path = Path(input_path)
        self.transcript = transcript
        self.chunks = chunks
        self.chunk_list.delete(0, tk.END)
        for chunk in chunks:
            self.chunk_list.insert(tk.END, f"Chunk {chunk.cid}: cues {chunk.start_idx}-{chunk.end_idx} ({chunk.charcount} chars)")
        self.log(f"Loaded {len(transcript.cues)} cues and planned {len(chunks)} chunk(s)")

    def translate_all(self) -> None:
        if not self._ensure_ready():
            return
        self._start_worker(self.chunks)

    def translate_selected(self) -> None:
        if not self._ensure_ready():
            return
        selection = self.chunk_list.curselection()
        if not selection:
            messagebox.showinfo("Select chunk", "Choose a chunk from the list first.")
            return
        index = selection[0]
        chunk = self.chunks[index]
        self._start_worker([chunk])

    def _ensure_ready(self) -> bool:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("Busy", "Worker already running. Abort it first.")
            return False
        if self.transcript is None or not self.chunks:
            messagebox.showinfo("Need chunks", "Build chunks before running translation.")
            return False
        if not self.output_var.get().strip():
            messagebox.showinfo("Output", "Choose an output directory first.")
            return False
        return True

    def _start_worker(self, subset: List[Chunk]) -> None:
        self.abort_event.clear()
        try:
            batch_n = max(1, int(self.cues_per_request_var.get()))
        except (TypeError, ValueError):
            messagebox.showerror(
                "Invalid batch size",
                "Cues per request must be an integer >= 1.",
            )
            return
        args = dict(
            server=self.server_var.get().strip(),
            model=self.model_var.get().strip(),
            source=self.source_var.get().strip(),
            target=self.target_var.get().strip(),
            batch_n=batch_n,
            translate_bracketed=self.bracket_var.get(),
            llm_mode="auto",
            stream=self.stream_var.get(),
            timeout=60.0,
            no_llm=self.no_llm_var.get(),
        )
        output_dir = Path(self.output_var.get()).expanduser()
        flat = self.flat_var.get()
        model_tag = self.model_var.get().strip()

        def run() -> None:
            self.log("Starting translation job")
            processed_any = False
            for chunk in subset:
                if self.abort_event.is_set():
                    self.log("Translation aborted by user")
                    break
                try:
                    translate_range(
                        self.transcript,
                        [chunk],
                        logger=self.log,
                        raw_handler=None,
                        verbose=True,
                        **args,
                    )
                    processed_any = True
                    self.log(f"Finished chunk {chunk.cid}")
                except Exception as exc:
                    self.log(f"Chunk {chunk.cid} failed: {exc}")
                    self.root.after(0, lambda msg=str(exc): messagebox.showerror("Translation error", msg))
                    break
            if processed_any and not self.abort_event.is_set():
                self._write_outputs(output_dir, flat, model_tag)
            self.worker = None

        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def _write_outputs(self, output_dir: Path, flat: bool, model_tag: str) -> None:
        assert self.transcript is not None
        timestamp = dt.datetime.now(dt.timezone.utc).astimezone()
        target = self._resolve_output_directory(output_dir, flat, model_tag, timestamp)
        note = f"translated-with model={model_tag} time={timestamp.isoformat()}"
        content = build_output(self.transcript, vtt_note=note if self.transcript.fmt == "vtt" else None)
        filename = self._output_filename(self.transcript.fmt)
        path = target / filename
        try:
            path.write_text(content, encoding="utf-8")
            self.log(f"Wrote output to {path}")
        except OSError as exc:
            self.log(f"Unable to write output: {exc}")

    def _resolve_output_directory(
        self, base: Path, flat: bool, model_tag: str, timestamp: dt.datetime
    ) -> Path:
        base.mkdir(parents=True, exist_ok=True)
        if flat:
            return base
        if self.input_path is not None:
            orig = self.input_path.stem
        elif self.transcript is not None and self.transcript.cues:
            orig = f"job-{self.transcript.cues[0].index}"
        else:
            orig = "job"
        orig_slug = self._slugify(orig)
        model_slug = self._slugify(model_tag or "model")
        lang_slug = self._language_code()
        stamp = timestamp.strftime("%Y%m%d-%H%M%S")
        folder = f"{orig_slug}_{stamp}_{model_slug}_{lang_slug}"
        target = base / folder
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _output_filename(self, fmt: str) -> str:
        if self.input_path is not None:
            base = self.input_path.stem
        elif self.transcript is not None and self.transcript.cues:
            base = f"job-{self.transcript.cues[0].index}"
        else:
            base = "output"
        lang = self._language_code()
        ext_map = {"srt": ".srt", "vtt": ".vtt", "tsv": ".tsv"}
        ext = ext_map.get(fmt, ".txt")
        suffix = f"_{lang}" if lang else ""
        return f"{self._slugify(base)}{suffix}{ext}"

    def _language_code(self) -> str:
        target = self.target_var.get().strip()
        if not target:
            return "unknown"
        return self._slugify(target).lower() or "unknown"

    @staticmethod
    def _slugify(text: str) -> str:
        cleaned = re.sub(r"[\s]+", "_", text.strip())
        cleaned = re.sub(r"[^0-9A-Za-z._-]", "-", cleaned)
        cleaned = re.sub(r"[-_]{2,}", "_", cleaned)
        return cleaned.strip("_-.") or "unnamed"

    def abort(self) -> None:
        self.abort_event.set()
        self.log("Abort requested")

    def on_close(self) -> None:
        self.abort()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=1)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
