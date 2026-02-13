from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable


class AppGUI:
    def __init__(self, on_toggle: Callable[[bool], None], on_save_keys: Callable[[dict], None]) -> None:
        self.on_toggle = on_toggle
        self.on_save_keys = on_save_keys
        self.running = False
        self.status_queue: "queue.Queue[str]" = queue.Queue()

        self.root = tk.Tk()
        self.root.title("Local Game Autopilot")
        self.root.geometry("520x400")

        self.status_var = tk.StringVar(value="OFF")
        self.toggle_btn = ttk.Button(self.root, text="ВКЛ", command=self._toggle)
        self.toggle_btn.pack(pady=10)

        ttk.Label(self.root, text="Состояние:").pack()
        ttk.Label(self.root, textvariable=self.status_var).pack(pady=5)

        ttk.Label(self.root, text="Горячие клавиши (действие:клавиша)").pack(pady=5)
        self.keys_entry = ttk.Entry(self.root, width=60)
        self.keys_entry.insert(0, "forward:w,backward:s,left:a,right:d,jump:space,interact:e,idle:")
        self.keys_entry.pack()
        ttk.Button(self.root, text="Сохранить клавиши", command=self._save_keys).pack(pady=5)

        self.log_text = tk.Text(self.root, height=12)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self.root.after(150, self._drain_status)

    def _toggle(self) -> None:
        self.running = not self.running
        self.toggle_btn.configure(text="ВЫКЛ" if self.running else "ВКЛ")
        self.status_var.set("ON" if self.running else "OFF")
        self.on_toggle(self.running)

    def _save_keys(self) -> None:
        text = self.keys_entry.get().strip()
        mapping = {}
        for token in text.split(","):
            if ":" not in token:
                continue
            action, key = token.split(":", 1)
            mapping[action.strip()] = key.strip()
        self.on_save_keys(mapping)
        self.status_queue.put("Key mapping updated")

    def push_log(self, msg: str) -> None:
        self.status_queue.put(msg)

    def _drain_status(self) -> None:
        while True:
            try:
                msg = self.status_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
        self.root.after(150, self._drain_status)

    def run(self) -> None:
        self.root.mainloop()


class WorkerThread:
    def __init__(self, target: Callable[[], None]) -> None:
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=target, daemon=True)

    def start(self) -> None:
        if not self.thread.is_alive():
            self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
