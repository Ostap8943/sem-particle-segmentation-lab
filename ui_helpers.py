import tkinter as tk
from tkinter import ttk


def make_button(parent, text, command, color):
    return tk.Button(
        parent,
        text=text,
        command=command,
        bg=color,
        fg="white",
        activebackground=color,
        activeforeground="white",
        font=("Arial", 10, "bold"),
        padx=12,
        pady=6,
        relief=tk.FLAT,
    )


def make_menu_button(parent, text, commands):
    button = tk.Menubutton(
        parent,
        text=f"{text} ▼",
        bg="#607d8b",
        fg="white",
        activebackground="#78909c",
        activeforeground="white",
        font=("Arial", 10, "bold"),
        padx=12,
        pady=6,
        relief=tk.FLAT,
    )
    menu = tk.Menu(button, tearoff=0, bg="#eceff1", fg="#102027", activebackground="#cfd8dc")
    for label, command in commands:
        menu.add_command(label=label, command=command)
    button.configure(menu=menu)
    return button


def add_combo(parent, label, variable, values):
    row = tk.Frame(parent, bg="#263238")
    row.pack(fill=tk.X, padx=8, pady=4)
    tk.Label(row, text=label, bg="#263238", fg="white", width=18, anchor="w").pack(side=tk.LEFT)
    combo = ttk.Combobox(row, textvariable=variable, values=values, state="readonly", width=22)
    combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
    return row


def add_slider(parent, label, variable, from_, to, resolution):
    row = tk.Frame(parent, bg="#263238")
    row.pack(fill=tk.X, padx=8, pady=2)
    value_label = tk.Label(row, text="", bg="#263238", fg="#cfd8dc", width=8, anchor="e")
    value_label.pack(side=tk.RIGHT)
    tk.Label(row, text=label, bg="#263238", fg="white", width=18, anchor="w").pack(side=tk.LEFT)

    def update_label(*_):
        value = variable.get()
        if isinstance(value, float):
            value_label.configure(text=f"{value:.4f}" if value < 0.1 else f"{value:.2f}")
        else:
            value_label.configure(text=str(value))

    slider = tk.Scale(
        row,
        variable=variable,
        from_=from_,
        to=to,
        orient=tk.HORIZONTAL,
        resolution=resolution,
        showvalue=False,
        bg="#263238",
        fg="white",
        troughcolor="#90a4ae",
        highlightthickness=0,
        length=170,
        command=lambda _: update_label(),
    )
    slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
    update_label()
    return row


def add_checkbox(parent, label, variable, command=None):
    row = tk.Frame(parent, bg="#263238")
    row.pack(fill=tk.X, padx=8, pady=3)
    checkbox = tk.Checkbutton(
        row,
        text=label,
        variable=variable,
        command=command,
        bg="#263238",
        fg="white",
        activebackground="#263238",
        activeforeground="white",
        selectcolor="#455a64",
        anchor="w",
    )
    checkbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
    return row
