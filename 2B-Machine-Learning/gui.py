"""
gui.py
------
Tkinter GUI for the Traffic-Based Route Guidance System (TBRGS).

Tabs:
  1. Route Finder  — enter origin/destination SCATS numbers, pick a model,
                     get up to 5 routes shown as text and on a map graph.
  2. Model Viz     — browse training-loss and prediction plots per model.
  3. Comparison    — side-by-side MAE / RMSE / R² bar chart across all models.

Run:
    conda activate intro-ai
    cd 2B-Machine-Learning
    python gui.py
"""

import sys
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------------------------------------------------------------------
# Path setup — routing/ modules import each other by name, so routing/ must
# be on sys.path before we import anything from it.
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent
ROUTING_DIR = BASE_DIR / "routing"
sys.path.insert(0, str(ROUTING_DIR))
sys.path.insert(0, str(BASE_DIR))

import config
from travel_time import load_model, load_scalers, load_sites
from graph_builder import build_graph
from yen import k_shortest_paths, format_paths

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS = ["lstm", "gru", "saes"]
MODEL_LABELS = {"lstm": "LSTM", "gru": "GRU", "saes": "SAEs"}
METRICS = ["mae", "rmse", "r2"]
METRIC_LABELS = ["MAE", "RMSE", "R²"]

WIN_W, WIN_H = 1150, 760
PATH_COLORS  = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------
class TBRGSApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TBRGS — Traffic-Based Route Guidance System")
        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self.root.resizable(True, True)

        # Shared data — loaded once at startup so every tab can use them
        self._status_init("Loading site data…")
        self.scalers = load_scalers()
        self.sites   = load_sites()

        # Model cache — avoid reloading a model that is already in memory
        self._model_cache: dict = {}

        # Stored after search so listbox selection can redraw
        self._last_graph = None
        self._last_paths = []

        # Build the notebook and its three tabs
        nb = ttk.Notebook(root)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.notebook = nb

        self._build_route_tab()
        self._build_viz_tab()
        self._build_compare_tab()

        nb.bind("<<NotebookTabChanged>>", self._on_tab_change)

    # -----------------------------------------------------------------------
    # Startup helper
    # -----------------------------------------------------------------------
    def _status_init(self, msg: str):
        """Show a temporary label while startup data is loading."""
        lbl = tk.Label(self.root, text=msg, font=("Helvetica", 12))
        lbl.place(relx=0.5, rely=0.5, anchor="center")
        self.root.update()
        lbl.destroy()

    # =========================================================================
    # TAB 1 — Route Finder
    # =========================================================================
    def _build_route_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Route Finder  ")

        # ---- Left panel: controls + results ---------------------------------
        left = ttk.Frame(tab, width=370)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 4), pady=10)
        left.pack_propagate(False)

        ttk.Label(left, text="Route Finder",
                  font=("Helvetica", 14, "bold")).pack(anchor="w", pady=(0, 12))

        # Inputs
        for label, var_name, default in [
            ("Origin SCATS site number:",      "var_origin", "2000"),
            ("Destination SCATS site number:", "var_dest",   "3002"),
        ]:
            ttk.Label(left, text=label).pack(anchor="w")
            var = tk.StringVar(value=default)
            setattr(self, var_name, var)
            ttk.Entry(left, textvariable=var, width=22).pack(anchor="w", pady=(2, 8))

        ttk.Label(left, text="ML Model:").pack(anchor="w")
        available = self._available_models()
        self.var_model = tk.StringVar(value=available[0] if available else "")
        ttk.Combobox(left, textvariable=self.var_model,
                     values=available, state="readonly",
                     width=20).pack(anchor="w", pady=(2, 14))

        self.btn_find = ttk.Button(left, text="Find Routes",
                                   command=self._on_find_routes)
        self.btn_find.pack(anchor="w")

        self.var_status = tk.StringVar(value="Ready.")
        ttk.Label(left, textvariable=self.var_status,
                  foreground="grey", wraplength=340,
                  justify=tk.LEFT).pack(anchor="w", pady=(6, 0))

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=10)

        # Route selector — click to highlight a route on the graph
        ttk.Label(left, text="Routes  (click to highlight):",
                  font=("Helvetica", 11, "bold")).pack(anchor="w")

        listbox_frame = ttk.Frame(left)
        listbox_frame.pack(fill=tk.X, pady=(4, 6))
        self.route_listbox = tk.Listbox(
            listbox_frame, height=6, font=("Helvetica", 10),
            selectmode=tk.SINGLE, relief=tk.FLAT,
            bg="#ffffff", fg="black", selectbackground="#4C72B0",
            selectforeground="white", activestyle="none",
        )
        lb_scroll = ttk.Scrollbar(listbox_frame, command=self.route_listbox.yview)
        self.route_listbox.configure(yscrollcommand=lb_scroll.set)
        lb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.route_listbox.pack(fill=tk.X, expand=True)
        self.route_listbox.bind("<<ListboxSelect>>", self._on_route_select)

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=(0, 6))

        ttk.Label(left, text="Route detail:",
                  font=("Helvetica", 10, "bold")).pack(anchor="w")

        self.results_text = tk.Text(
            left, height=7, wrap=tk.WORD, state=tk.DISABLED,
            font=("Courier", 9), relief=tk.FLAT, bg="#ffffff", fg="black",
        )
        scroll = ttk.Scrollbar(left, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        # ---- Right panel: embedded map graph --------------------------------
        right = ttk.Frame(tab)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                   padx=(4, 10), pady=10)

        self.route_fig, self.route_ax = plt.subplots(figsize=(6, 5.5))
        self.route_fig.patch.set_facecolor("#f9f9f9")
        self._reset_graph_canvas()

        self.route_canvas = FigureCanvasTkAgg(self.route_fig, master=right)
        self.route_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.route_canvas.draw()

    def _reset_graph_canvas(self):
        self.route_ax.clear()
        self.route_ax.set_facecolor("#f9f9f9")
        self.route_ax.axis("off")
        self.route_ax.set_title("Graph appears here after search", color="#aaaaaa")

    # -----------------------------------------------------------------------
    # Find Routes — entry point
    # -----------------------------------------------------------------------
    def _on_find_routes(self):
        try:
            origin = int(self.var_origin.get().strip())
            dest   = int(self.var_dest.get().strip())
        except ValueError:
            messagebox.showerror("Input error",
                                 "Origin and Destination must be integer SCATS site numbers.")
            return

        model_name = self.var_model.get()
        if not model_name:
            messagebox.showerror("No model",
                                 "No trained model available. Run a model script first.")
            return

        if origin == dest:
            messagebox.showerror("Input error", "Origin and Destination must be different.")
            return

        # Disable button and show loading status
        self.btn_find.config(state=tk.DISABLED)
        self.var_status.set("Loading model and building graph… (~30 s)")
        self._set_results("")
        self.route_listbox.delete(0, tk.END)
        self._last_graph = None
        self._last_paths = []
        self._reset_graph_canvas()
        self.route_canvas.draw()

        threading.Thread(
            target=self._find_routes_thread,
            args=(origin, dest, model_name),
            daemon=True,
        ).start()

    def _find_routes_thread(self, origin: int, dest: int, model_name: str):
        try:
            model = self._get_model(model_name)
            self.root.after(0, self.var_status.set, "Building graph…")

            graph = build_graph(
                model, model_name, self.scalers, self.sites,
                origin=origin, destinations=[dest],
            )

            self.root.after(0, self.var_status.set, "Finding top-5 routes…")
            paths = k_shortest_paths(graph, k=5)

            self.root.after(0, self._display_results, paths, graph)

        except ValueError as exc:
            self.root.after(0, messagebox.showerror, "Route error", str(exc))
            self.root.after(0, self.var_status.set, "Error — check SCATS site numbers.")
        except Exception as exc:
            self.root.after(0, messagebox.showerror, "Unexpected error", str(exc))
            self.root.after(0, self.var_status.set, "An error occurred.")
        finally:
            self.root.after(0, self.btn_find.config, {"state": tk.NORMAL})

    def _display_results(self, paths, graph):
        self.route_listbox.delete(0, tk.END)

        if not paths:
            self.var_status.set("No routes found between those sites.")
            self._set_results("No routes found.\n\nTry different site numbers.")
            return

        self._last_graph = graph
        self._last_paths = paths
        self.var_status.set(f"Done — {len(paths)} route(s) found.")

        # Populate the listbox with one-line summaries
        for i, (path, cost) in enumerate(paths):
            mins = cost / 60
            self.route_listbox.insert(
                tk.END,
                f"  Route {i+1}:  {mins:.1f} min  ({len(path)-1} links)",
            )
            # Colour the listbox entry to match the graph line
            self.route_listbox.itemconfig(i, fg=PATH_COLORS[i % len(PATH_COLORS)])

        # Show full detail for first route
        self._set_results(format_paths(paths))
        self._draw_graph(graph, paths, highlight_idx=None)

    def _on_route_select(self, _event):
        sel = self.route_listbox.curselection()
        if not sel or not self._last_paths:
            return
        idx = sel[0]
        path, cost = self._last_paths[idx]
        mins = cost / 60
        route_str = " → ".join(str(n) for n in path)
        detail = (f"Route {idx+1}:  {mins:.1f} min  ({cost:.0f}s)  "
                  f"{len(path)-1} links\n\n{route_str}")
        self._set_results(detail)
        self._draw_graph(self._last_graph, self._last_paths, highlight_idx=idx)

    def _set_results(self, text: str):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.config(state=tk.DISABLED)

    # -----------------------------------------------------------------------
    # Graph visualisation
    # -----------------------------------------------------------------------
    def _draw_graph(self, graph, paths, highlight_idx=None):
        ax = self.route_ax
        ax.clear()
        ax.set_facecolor("#f0f4f8")
        ax.axis("off")

        nodes = graph.nodes  # {site_id: (lon, lat)}

        # Collect all nodes that appear in any found route
        all_path_nodes = {n for path, _ in paths for n in path if n in nodes}

        if not all_path_nodes:
            ax.set_title("No path nodes to display.", color="grey")
            self.route_canvas.draw()
            return

        # Compute bounding box of path nodes and zoom in with padding
        path_xs = [nodes[n][0] for n in all_path_nodes]
        path_ys = [nodes[n][1] for n in all_path_nodes]
        x_pad = max((max(path_xs) - min(path_xs)) * 0.35, 0.008)
        y_pad = max((max(path_ys) - min(path_ys)) * 0.35, 0.006)
        x_min = min(path_xs) - x_pad
        x_max = max(path_xs) + x_pad
        y_min = min(path_ys) - y_pad
        y_max = max(path_ys) + y_pad

        # Grey nodes — only those visible inside the zoomed bounding box
        # These are OTHER SCATS intersections nearby (network context)
        for site_id, (x, y) in nodes.items():
            if x_min <= x <= x_max and y_min <= y <= y_max:
                ax.scatter(x, y, s=22, color="#b0b8c8", zorder=2)

        # Grey edges — only edges between nodes inside the bounding box
        for u, neighbours in graph.edges.items():
            if u not in nodes:
                continue
            x1, y1 = nodes[u]
            if not (x_min <= x1 <= x_max and y_min <= y1 <= y_max):
                continue
            for v, _ in neighbours:
                if v not in nodes:
                    continue
                x2, y2 = nodes[v]
                ax.plot([x1, x2], [y1, y2],
                        color="#d0d8e8", linewidth=0.7, zorder=1)

        # Draw each route — highlighted route is bold, others are dimmed
        for i, (path, cost) in enumerate(paths):
            color = PATH_COLORS[i % len(PATH_COLORS)]
            px = [nodes[n][0] for n in path if n in nodes]
            py = [nodes[n][1] for n in path if n in nodes]
            mins = cost / 60

            is_highlighted = (highlight_idx is None or i == highlight_idx)
            lw    = 4.0 if i == highlight_idx else (2.5 if highlight_idx is None else 1.2)
            alpha = 0.95 if i == highlight_idx else (0.85 if highlight_idx is None else 0.25)

            ax.plot(px, py, color=color, linewidth=lw, zorder=4,
                    label=f"Route {i+1}  ({mins:.1f} min)", alpha=alpha,
                    solid_capstyle="round", solid_joinstyle="round")

            # Waypoint dots only on highlighted (or all when none selected)
            if is_highlighted and len(px) > 2:
                ax.scatter(px[1:-1], py[1:-1], s=50, color=color,
                           zorder=5, alpha=alpha, edgecolors="white", linewidths=0.8)

        # Origin (green) and Destination (red) — large markers with white border
        origin = graph.origin
        dest   = graph.destinations[0]
        for site, color, prefix in [
            (origin, "#27ae60", "O"),
            (dest,   "#c0392b", "D"),
        ]:
            if site in nodes:
                ax.scatter(*nodes[site], s=220, color=color, zorder=6,
                           edgecolors="white", linewidths=2.0)
                ax.annotate(
                    f"{prefix}: {site}", nodes[site],
                    textcoords="offset points", xytext=(9, 9),
                    fontsize=9, color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=color, alpha=0.85),
                )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.85,
                  facecolor="white", edgecolor="#cccccc")
        ax.set_title(f"Routes: {origin} → {dest}", fontsize=10, pad=8)
        self.route_canvas.draw()

    # =========================================================================
    # TAB 2 — Model Visualisation
    # =========================================================================
    def _build_viz_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Model Visualisation  ")

        # Controls row
        ctrl = ttk.Frame(tab)
        ctrl.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(ctrl, text="Model:").pack(side=tk.LEFT)
        self.var_viz_model = tk.StringVar(value=MODELS[0])
        ttk.Combobox(ctrl, textvariable=self.var_viz_model,
                     values=MODELS, state="readonly",
                     width=10).pack(side=tk.LEFT, padx=8)

        ttk.Button(ctrl, text="Loss Curve",
                   command=lambda: self._show_plot("loss")).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Predictions",
                   command=lambda: self._show_plot("predictions")).pack(side=tk.LEFT, padx=4)

        # Canvas for the image
        self.viz_fig, self.viz_ax = plt.subplots(figsize=(9.5, 5.5))
        self.viz_fig.patch.set_facecolor("#f9f9f9")
        self.viz_ax.axis("off")
        self.viz_ax.set_title("Select a model and chart type above.", color="#aaaaaa")

        self.viz_canvas = FigureCanvasTkAgg(self.viz_fig, master=tab)
        self.viz_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                             padx=10, pady=(0, 10))
        self.viz_canvas.draw()

    def _show_plot(self, plot_type: str):
        model_name = self.var_viz_model.get()
        png_path   = config.OUTPUT_DIR / model_name / f"{model_name}_{plot_type}.png"

        if not png_path.exists():
            messagebox.showwarning(
                "File not found",
                f"{png_path.name} not found.\nRun {model_name}.py first.",
            )
            return

        img = mpimg.imread(str(png_path))
        self.viz_ax.clear()
        self.viz_ax.imshow(img)
        self.viz_ax.axis("off")
        self.viz_fig.tight_layout(pad=0.5)
        self.viz_canvas.draw()

    # =========================================================================
    # TAB 3 — Model Comparison
    # =========================================================================
    def _build_compare_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="  Model Comparison  ")

        ttk.Button(tab, text="Refresh",
                   command=self._load_comparison).pack(anchor="w", padx=10, pady=10)

        self.compare_fig, self.compare_axes = plt.subplots(1, 3, figsize=(11, 4.5))
        self.compare_fig.patch.set_facecolor("#f9f9f9")

        self.compare_canvas = FigureCanvasTkAgg(self.compare_fig, master=tab)
        self.compare_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                                 padx=10, pady=(0, 10))
        self.compare_canvas.draw()

    def _on_tab_change(self, _event):
        tab_text = self.notebook.tab(self.notebook.select(), "text").strip()
        if tab_text == "Model Comparison":
            self._load_comparison()

    def _load_comparison(self):
        results = {}
        for name in MODELS:
            path = config.OUTPUT_DIR / name / f"{name}_metrics.json"
            if path.exists():
                with open(path) as f:
                    results[MODEL_LABELS[name]] = json.load(f)

        if not results:
            messagebox.showinfo("No data",
                                "No metrics found. Run the model scripts first.")
            return

        bar_colors  = ["#4C72B0", "#DD8452", "#55A868"]
        model_names = list(results.keys())

        for ax, metric, metric_label in zip(self.compare_axes, METRICS, METRIC_LABELS):
            ax.clear()
            values = [results[m][metric] for m in model_names]
            bars   = ax.bar(model_names, values,
                            color=bar_colors[:len(model_names)], width=0.5)

            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.015,
                    f"{val:.4f}",
                    ha="center", va="bottom", fontsize=9,
                )

            ax.set_title(metric_label, fontsize=11)
            ax.set_ylim(0, max(values) * 1.25)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_facecolor("#f9f9f9")

        self.compare_fig.suptitle(
            "Model Comparison — Traffic Flow Prediction", fontsize=12,
        )
        self.compare_fig.tight_layout()
        self.compare_canvas.draw()

    # =========================================================================
    # Helpers
    # =========================================================================
    def _available_models(self) -> list:
        """Return model names whose .keras file exists in output/."""
        return [
            name for name in MODELS
            if (config.OUTPUT_DIR / name / f"{name}_model.keras").exists()
        ]

    def _get_model(self, model_name: str):
        """Load and cache a Keras model, reusing it on subsequent calls."""
        if model_name not in self._model_cache:
            self._model_cache[model_name] = load_model(model_name)
        return self._model_cache[model_name]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    TBRGSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
