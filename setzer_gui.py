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
from typing import Callable, List, Optional

from setzer_core import (
    Chunk,
    Transcript,
    build_output_as,
    make_chunks,
    read_transcript,
    resolve_outfile,
    translate_range,
)


DEFAULT_OUTFILE_TEMPLATE = "{basename}.{dst}.{fmt}"


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("homedoc-subtitle-translator")

        self.input_var = tk.StringVar()
        default_output = Path.cwd() / "results"
        self.output_var = tk.StringVar(value=str(default_output))
        self.source_var = tk.StringVar(value="auto")
        self.target_var = tk.StringVar(value="English")
        self.outfmt_var = tk.StringVar(value="auto")
        self.outfile_template_var = tk.StringVar(value=DEFAULT_OUTFILE_TEMPLATE)
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
        self._update_template_hint()
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

        row = 0

        input_entry = tk.Entry(frame, textvariable=self.input_var)
        add_row("Input", input_entry, row)
        tk.Button(frame, text="Browse", command=self._browse_input).grid(row=row, column=2, padx=(6, 0))
        row += 1

        output_entry = tk.Entry(frame, textvariable=self.output_var)
        add_row("Output dir", output_entry, row)
        tk.Button(frame, text="Browse", command=self._browse_output).grid(row=row, column=2, padx=(6, 0))
        row += 1

        template_entry = tk.Entry(frame, textvariable=self.outfile_template_var)
        add_row("Output file", template_entry, row)
        self.template_entry = template_entry
        row += 1

        self.placeholder_hint = tk.Label(
            frame,
            text="Placeholders: {basename} {src} {dst} {fmt} {ts}",
            fg="#555555",
        )
        self.placeholder_hint.grid(row=row, column=1, columnspan=2, sticky="w", pady=(0, 6))
        row += 1

        fmt_menu = tk.OptionMenu(frame, self.outfmt_var, "auto", "srt", "vtt", "tsv")
        add_row("Format", fmt_menu, row)
        row += 1

        add_row("Source", tk.Entry(frame, textvariable=self.source_var), row)
        row += 1
        add_row("Target", tk.Entry(frame, textvariable=self.target_var), row)
        row += 1
        add_row("Server", tk.Entry(frame, textvariable=self.server_var), row)
        row += 1
        add_row("Model", tk.Entry(frame, textvariable=self.model_var), row)
        row += 1

        cues_spin = tk.Spinbox(frame, from_=1, to=50, textvariable=self.cues_per_request_var, width=6)
        add_row("Cues/request", cues_spin, row)
        row += 1
        add_row("Max chars", tk.Entry(frame, textvariable=self.max_chars_var), row)
        row += 1

        options_frame = tk.Frame(frame)
        options_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=(4, 4))
        tk.Checkbutton(options_frame, text="Translate bracketed", variable=self.bracket_var).pack(side=tk.LEFT)
        tk.Checkbutton(options_frame, text="Stream", variable=self.stream_var).pack(side=tk.LEFT, padx=(10, 0))
        tk.Checkbutton(options_frame, text="Flat output", variable=self.flat_var).pack(side=tk.LEFT, padx=(10, 0))
        tk.Checkbutton(options_frame, text="Dry run (no LLM)", variable=self.no_llm_var).pack(side=tk.LEFT, padx=(10, 0))

        row += 1

        self.chunk_list = tk.Listbox(frame, height=6)
        frame.rowconfigure(row, weight=1)
        self.chunk_list.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(6, 6))

        row += 1

        button_frame = tk.Frame(frame)
        button_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(4, 4))
        tk.Button(button_frame, text="Build/Update chunks", command=self.build_chunks).pack(side=tk.LEFT)
        tk.Button(button_frame, text="Translate ALL", command=self.translate_all).pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(button_frame, text="Translate selected chunk", command=self.translate_selected).pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(button_frame, text="Abort", command=self.abort).pack(side=tk.LEFT, padx=(6, 0))

        row += 1

        self.console = scrolledtext.ScrolledText(frame, height=12, state=tk.DISABLED)
        self.console.grid(row=row, column=0, columnspan=3, sticky="nsew")
        frame.rowconfigure(row, weight=2)

        row += 1

        preview_label = tk.Label(frame, text="CLI preview")
        preview_label.grid(row=row, column=0, sticky="nw", pady=(6, 0))
        self.cli_preview_widget = scrolledtext.ScrolledText(
            frame,
            height=5,
            wrap="word",
            state=tk.DISABLED,
        )
        self.cli_preview_widget.grid(row=row, column=1, columnspan=2, sticky="nsew", pady=(6, 0))
        frame.rowconfigure(row, weight=0)


    def _register_variable_traces(self) -> None:
        def register(var: tk.Variable, callback: Callable[..., None]) -> None:
            token = var.trace_add("write", callback)
            self._trace_tokens.append(token)

        core_vars: List[tk.Variable] = [
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
            self.no_llm_var,
            self.outfmt_var,
            self.outfile_template_var,
        ]
        for var in core_vars:
            register(var, self._update_cli_preview)

        def flat_handler(*_: object) -> None:
            self._update_template_hint()
            self._update_cli_preview()

        register(self.flat_var, flat_handler)

    def _update_cli_preview(self, *_: object) -> None:
        command = self._format_cli_command()
        self.cli_preview_var.set(command)
        self.cli_preview_widget.configure(state=tk.NORMAL)
        self.cli_preview_widget.delete("1.0", tk.END)
        self.cli_preview_widget.insert("1.0", command)
        self.cli_preview_widget.configure(state=tk.DISABLED)

    def _update_template_hint(self, *_: object) -> None:
        if not hasattr(self, "placeholder_hint"):
            return
        if self.flat_var.get():
            self.placeholder_hint.grid()
        else:
            self.placeholder_hint.grid_remove()

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
        outfmt = self.outfmt_var.get().strip() or "auto"
        template = self.outfile_template_var.get().strip()

        args.extend(["--source", source])
        args.extend(["--target", target])
        args.extend(["--server", server])
        args.extend(["--model", model])
        args.extend(["--cues-per-request", str(cues)])
        args.extend(["--max-chars", str(max_chars)])

        if outfmt != "auto":
            args.extend(["--outfmt", outfmt])

        if self.flat_var.get():
            args.append("--flat")
        else:
            args.append("--no-flat")

        if self.flat_var.get():
            if template:
                args.extend(["--outfile", template])
        elif template and template != DEFAULT_OUTFILE_TEMPLATE:
            args.extend(["--outfile", template])

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
        target_fmt = self.outfmt_var.get().strip() or "auto"
        if target_fmt == "auto":
            target_fmt = self.transcript.fmt
        try:
            content = build_output_as(
                self.transcript,
                target_fmt,
                vtt_note=note if target_fmt == "vtt" else None,
            )
        except Exception as exc:
            self.log(f"Unable to render output: {exc}")
            return

        template_value = self.outfile_template_var.get().strip() or DEFAULT_OUTFILE_TEMPLATE
        if template_value.startswith("~") or Path(template_value).is_absolute():
            template_path = template_value
        else:
            template_path = str(target / template_value)

        input_hint: Optional[Path]
        if self.input_path is not None:
            input_hint = self.input_path
        else:
            raw_input = self.input_var.get().strip()
            input_hint = Path(raw_input) if raw_input else None

        try:
            path = resolve_outfile(
                template_path,
                input_hint,
                self._language_token(self.source_var.get(), "auto"),
                self._language_token(self.target_var.get(), "unknown"),
                target_fmt,
            )
        except Exception as exc:
            self.log(f"Unable to resolve output path: {exc}")
            return

        try:
            path.write_text(content, encoding="utf-8")
            self.log(f"Wrote {target_fmt.upper()} output to {path}")
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
        lang_slug = self._language_token(self.target_var.get(), "unknown")
        stamp = timestamp.strftime("%Y%m%d-%H%M%S")
        folder = f"{orig_slug}_{stamp}_{model_slug}_{lang_slug}"
        target = base / folder
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _language_token(self, value: str, fallback: str) -> str:
        text = (value or "").strip()
        if not text:
            return fallback
        return self._slugify(text).lower() or fallback

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
