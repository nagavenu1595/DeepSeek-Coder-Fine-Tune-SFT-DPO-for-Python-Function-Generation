import ast
import textwrap
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

torch.set_grad_enabled(False)
MODEL_ID = "deepseek-ai/DeepSeek-Coder-1.3B-instruct"
LORA_PATH = "./dpo-final-model-v2"
USE_CUDA = torch.cuda.is_available()
DTYPE = torch.float16 if USE_CUDA else torch.float32


class StopOnClosingFence(StoppingCriteria):
    def __init__(self, tokenizer):
        self.stop_ids = set(tokenizer.encode("```", add_special_tokens=False))

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0][-1].item() in self.stop_ids


class CodeGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("")
        self.root.configure(bg="#eaf2fb")
        self.root.minsize(1050, 680)

        self.task_queue = queue.Queue()
        self.generating = False
        self._build_style()
        self._build_ui()
        self._load_models()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Return>", self._on_generate_hotkey)

    def _build_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#eaf2fb")
        style.configure("TLabel", background="#eaf2fb", foreground="#113355", font=("Segoe UI", 10))
        style.configure(
            "Title.TLabel",
            background="#eaf2fb",
            foreground="#123b63",
            font=("Segoe UI", 12, "bold"),
        )
        style.configure(
            "TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(12, 7),
            background="#cfe4ff",
            foreground="#0f3558",
        )
        style.map(
            "TButton",
            background=[("active", "#b7d6fb"), ("disabled", "#e3eefb")],
        )

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 12))

        ttk.Label(header, text="Task", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Type a task, press Enter or Generate, and compare both outputs side by side. Type quit to close.",
            wraplength=980,
        ).pack(anchor="w", pady=(2, 10))

        task_row = ttk.Frame(outer)
        task_row.pack(fill="x", pady=(0, 14))

        self.task_var = tk.StringVar()
        self.task_entry = ttk.Entry(task_row, textvariable=self.task_var, font=("Segoe UI", 11))
        self.task_entry.pack(side="left", fill="x", expand=True, ipady=7)
        self.task_entry.focus_set()
        self.task_entry.bind("<Return>", self._on_generate_hotkey)

        self.generate_btn = ttk.Button(task_row, text="Generate", command=self._submit_task)
        self.generate_btn.pack(side="left", padx=(10, 0))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(outer, textvariable=self.status_var).pack(anchor="w", pady=(0, 8))

        outputs = ttk.Frame(outer)
        outputs.pack(fill="both", expand=True)
        outputs.columnconfigure(0, weight=1)
        outputs.columnconfigure(1, weight=1)
        outputs.rowconfigure(1, weight=1)

        left_panel = ttk.Frame(outputs, padding=(0, 0, 8, 0))
        left_panel.grid(row=0, column=0, rowspan=2, sticky="nsew")
        right_panel = ttk.Frame(outputs, padding=(8, 0, 0, 0))
        right_panel.grid(row=0, column=1, rowspan=2, sticky="nsew")

        ttk.Label(left_panel, text="Your model").pack(anchor="w", pady=(0, 6))
        ttk.Label(right_panel, text="Base model").pack(anchor="w", pady=(0, 6))

        self.left_text = self._make_output_box(left_panel)
        self.right_text = self._make_output_box(right_panel)

        self._set_output(self.left_text, "Waiting for a task...\n")
        self._set_output(self.right_text, "Waiting for a task...\n")

    def _make_output_box(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        box = ScrolledText(
            frame,
            wrap="word",
            height=24,
            font=("Consolas", 10),
            bg="#f8fbff",
            fg="#10263d",
            insertbackground="#10263d",
            relief="solid",
            borderwidth=1,
        )
        box.pack(fill="both", expand=True)
        box.configure(state="disabled")
        return box

    def _load_models(self):
        self.status_var.set("Loading tokenizer and models...")
        self.root.update_idletasks()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            device_map="cpu",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(self.base_model, LORA_PATH, device_map="cpu")
        self.base_model.eval()
        self.model.eval()
        self.stop_criteria = StoppingCriteriaList([StopOnClosingFence(self.tokenizer)])
        self.status_var.set("Ready")

    def _set_output(self, widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.configure(state="disabled")

    def _append_status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def _on_generate_hotkey(self, event=None):
        self._submit_task()
        return "break"

    def _submit_task(self):
        if self.generating:
            return

        task = self.task_var.get().strip()
        if not task:
            return

        if task.lower() == "quit":
            self._on_close()
            return

        self.generating = True
        self.generate_btn.configure(state="disabled")
        self.task_entry.configure(state="disabled")
        self._append_status("Generating...")
        self._set_output(self.left_text, "Generating...\n")
        self._set_output(self.right_text, "Generating...\n")

        thread = threading.Thread(target=self._worker_generate, args=(task,), daemon=True)
        thread.start()

    def _build_prompt(self, task):
        return f"""Task: {task}
Output only a single Python function.
No comments.
No explanation.
Do not output anything before or after the function.
```python
"""

    def _extract_code(self, text):
        if "```python" in text:
            text = text.split("```python")[1]
        if "```" in text:
            text = text.split("```")[0]

        code = text.strip()

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    lines = code.split("\n")
                    return "\n".join(lines[node.lineno - 1:node.end_lineno]).strip()
        except:
            pass

        return code

    def _generate_with_model(self, model_obj, task, is_base=False):
        prompt = self._build_prompt(task)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(model_obj.device)

        with torch.no_grad():
            if is_base:
                out = model_obj.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            else:
                out = model_obj.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.3,
                    stopping_criteria=self.stop_criteria,
                )

        decoded = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return self._extract_code(decoded)

    def _worker_generate(self, task):
        try:
            your_code = self._generate_with_model(self.model, task, is_base=False)
            base_code = self._generate_with_model(self.base_model, task, is_base=True)
            self.root.after(0, self._show_result, task, your_code, base_code)
        except Exception as exc:
            self.root.after(0, self._show_error, exc)

    def _show_result(self, task, your_code, base_code):
        self._set_output(self.left_text, f"Task: {task}\n\n```python\n{your_code}\n```\n")
        self._set_output(self.right_text, f"Task: {task}\n\n```python\n{base_code}\n```\n")
        self.status_var.set("Ready")
        self.task_entry.configure(state="normal")
        self.generate_btn.configure(state="normal")
        self.task_var.set("")
        self.task_entry.focus_set()
        self.generating = False

    def _show_error(self, exc):
        msg = f"Generation failed: {exc}"
        self._set_output(self.left_text, msg)
        self._set_output(self.right_text, msg)
        self.status_var.set("Ready")
        self.task_entry.configure(state="normal")
        self.generate_btn.configure(state="normal")
        self.generating = False
        messagebox.showerror("Error", msg)

    def _on_close(self):
        self.root.destroy()


def main():
    root = tk.Tk()
    app = CodeGeneratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()