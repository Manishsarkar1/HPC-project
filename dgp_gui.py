#!/usr/bin/env python3
"""Desktop dashboard for the MPI4Py distributed graph-processing benchmark."""
from __future__ import annotations

import csv
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, TOP, X, Y, Canvas, StringVar, Tk, filedialog, messagebox, ttk


if "--worker" in sys.argv:
    sys.argv.remove("--worker")
    import dgp
    dgp.main()
    raise SystemExit


class GraphDashboard(Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Distributed Graph Processing | MPI Dashboard")
        self.minsize(1000, 700)
        self.results: list[dict[str, float | str]] = []
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.input_path = StringVar()
        self.partition = StringVar(value="hash")
        self.processes = StringVar(value="4")
        self.scale = StringVar(value="16")
        self.algorithms = {name: ttk.Checkbutton for name in ("bfs", "pagerank", "cc")}
        self.selection = {name: StringVar(value="1") for name in self.algorithms}
        self._build()
        self.after(80, self._poll)

    def _build(self) -> None:
        controls = ttk.LabelFrame(self, text="Benchmark setup", padding=12)
        controls.pack(fill=X, padx=12, pady=(12, 6))
        ttk.Label(controls, text="MPI processes").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(controls, from_=1, to=64, textvariable=self.processes, width=7).grid(row=0, column=1, padx=(6, 20))
        ttk.Label(controls, text="R-MAT scale").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(controls, from_=8, to=24, textvariable=self.scale, width=7).grid(row=0, column=3, padx=(6, 20))
        ttk.Label(controls, text="Partition").grid(row=0, column=4, sticky="w")
        ttk.Combobox(controls, textvariable=self.partition, values=("hash", "range"), state="readonly", width=9).grid(row=0, column=5, padx=(6, 20))
        for column, name in enumerate(("bfs", "pagerank", "cc"), start=6):
            ttk.Checkbutton(controls, text=name.upper(), variable=self.selection[name], onvalue="1", offvalue="0").grid(row=0, column=column, padx=5)

        ttk.Label(controls, text="Optional edge-list file").grid(row=1, column=0, pady=(12, 0), sticky="w")
        ttk.Entry(controls, textvariable=self.input_path, width=78).grid(row=1, column=1, columnspan=7, pady=(12, 0), padx=6, sticky="ew")
        ttk.Button(controls, text="Browse", command=self._browse).grid(row=1, column=8, pady=(12, 0), padx=4)
        self.run_button = ttk.Button(controls, text="Run benchmark", command=self._run)
        self.run_button.grid(row=1, column=9, pady=(12, 0), padx=(14, 0))

        metrics = ttk.LabelFrame(self, text="Benchmark results", padding=8)
        metrics.pack(fill=BOTH, expand=True, padx=12, pady=6)
        columns = ("algorithm", "parallel_s", "communication_s", "serial_s", "speedup", "imbalance", "sent_mb")
        self.table = ttk.Treeview(metrics, columns=columns, show="headings", height=7)
        headings = ("Algorithm", "Parallel time (s)", "Communication (s)", "Serial time (s)", "Speedup", "Compute imbalance", "Sent (MB)")
        for key, label in zip(columns, headings):
            self.table.heading(key, text=label)
            self.table.column(key, width=135, anchor="center")
        self.table.pack(side=LEFT, fill=BOTH, expand=True)
        ttk.Scrollbar(metrics, orient="vertical", command=self.table.yview).pack(side=RIGHT, fill=Y)

        charts = ttk.LabelFrame(self, text="Visual comparison", padding=8)
        charts.pack(fill=BOTH, expand=True, padx=12, pady=(6, 12))
        self.canvas = Canvas(charts, highlightthickness=0, background="#ffffff")
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._draw())

        self.status = StringVar(value="Choose settings and run the benchmark.")
        ttk.Label(self, textvariable=self.status, anchor="w", padding=(14, 0, 14, 10)).pack(fill=X)

    def _browse(self) -> None:
        path = filedialog.askopenfilename(title="Select edge-list input", filetypes=(("Text files", "*.txt *.edgelist *.csv"), ("All files", "*.*")))
        if path:
            self.input_path.set(path)

    def _command(self) -> list[str]:
        try:
            count = int(self.processes.get())
            scale = int(self.scale.get())
            if count < 1 or scale < 8:
                raise ValueError
        except ValueError as exc:
            raise ValueError("MPI processes must be at least 1 and graph scale must be at least 8.") from exc
        selected = [name for name, flag in self.selection.items() if flag.get() == "1"]
        if not selected:
            raise ValueError("Choose at least one algorithm.")
        command = ["mpiexec", "-n", str(count), sys.executable, "--worker", "--partition", self.partition.get(), "--algos", ",".join(selected), "--verify"]
        if self.input_path.get().strip():
            command.extend(["--input", self.input_path.get().strip()])
        else:
            command.extend(["--scale", str(scale)])
        return command

    def _run(self) -> None:
        try:
            command = self._command()
        except ValueError as error:
            messagebox.showerror("Invalid benchmark setup", str(error))
            return
        self.run_button.configure(state="disabled")
        self.status.set("Running MPI benchmark…")
        self.results.clear()
        for item in self.table.get_children():
            self.table.delete(item)
        self._draw()
        threading.Thread(target=self._execute, args=(command,), daemon=True).start()

    def _execute(self, command: list[str]) -> None:
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if completed.returncode:
                raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"mpiexec returned {completed.returncode}")
            rows = []
            for line in completed.stdout.splitlines():
                if not line.startswith("RESULT,"):
                    continue
                values = next(csv.reader([line]))
                rows.append({"algorithm": values[3], "parallel_s": float(values[4]), "communication_s": float(values[5]), "serial_s": float(values[6]), "speedup": float(values[7]), "imbalance": float(values[8]), "sent_mb": float(values[9])})
            if not rows:
                raise RuntimeError("The benchmark finished but did not report results. Confirm that Microsoft MPI is installed and available to mpiexec.")
            self.events.put(("complete", rows))
        except Exception as error:  # Display the actionable command failure in the GUI.
            self.events.put(("error", str(error)))

    def _poll(self) -> None:
        try:
            while True:
                event, value = self.events.get_nowait()
                self.run_button.configure(state="normal")
                if event == "error":
                    self.status.set("Benchmark failed")
                    messagebox.showerror("Benchmark failed", str(value))
                else:
                    self.results = value  # type: ignore[assignment]
                    for row in self.results:
                        self.table.insert("", END, values=(row["algorithm"].upper(), f"{row['parallel_s']:.4f}", f"{row['communication_s']:.4f}", f"{row['serial_s']:.4f}", f"{row['speedup']:.2f}x", f"{row['imbalance']:.2f}", f"{row['sent_mb']:.2f}"))
                    self.status.set("Benchmark complete. Charts compare the selected algorithms.")
                    self._draw()
        except queue.Empty:
            pass
        self.after(80, self._poll)

    def _draw(self) -> None:
        self.canvas.delete("all")
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 150 or height < 120:
            return
        if not self.results:
            self.canvas.create_text(width / 2, height / 2, text="Run a benchmark to visualize runtime, communication, and load balance.", fill="#555555", font=("Segoe UI", 12))
            return
        panels = [("Parallel time (seconds)", "parallel_s", "#3b82f6"), ("Communication (seconds)", "communication_s", "#f59e0b"), ("Load imbalance", "imbalance", "#10b981")]
        panel_width = width / len(panels)
        for index, (title, key, color) in enumerate(panels):
            x0, x1 = index * panel_width + 16, (index + 1) * panel_width - 16
            y0, y1 = 36, height - 40
            self.canvas.create_text((x0 + x1) / 2, 16, text=title, font=("Segoe UI", 11, "bold"))
            self.canvas.create_line(x0, y1, x1, y1, fill="#777777")
            maximum = max(float(row[key]) for row in self.results) or 1.0
            gap = (x1 - x0) / len(self.results)
            bar_width = gap * 0.56
            for offset, row in enumerate(self.results):
                value = float(row[key])
                bar_height = (y1 - y0) * value / maximum
                left = x0 + offset * gap + (gap - bar_width) / 2
                right = left + bar_width
                top = y1 - bar_height
                self.canvas.create_rectangle(left, top, right, y1, fill=color, outline="")
                self.canvas.create_text((left + right) / 2, top - 10, text=f"{value:.2f}", font=("Segoe UI", 9))
                self.canvas.create_text((left + right) / 2, y1 + 14, text=str(row["algorithm"]).upper(), font=("Segoe UI", 9))


if __name__ == "__main__":
    os.environ.setdefault("MPI4PY_RC_FINALIZE", "false")
    GraphDashboard().mainloop()
