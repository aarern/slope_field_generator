import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Safe parser with TI-84 style translation ---
def ti84_to_numpy(expr):
    return (expr.replace("^", "**")
                .replace("sin", "np.sin")
                .replace("cos", "np.cos")
                .replace("tan", "np.tan")
                .replace("log", "np.log10")
                .replace("ln", "np.log")
                .replace("sqrt", "np.sqrt"))

# --- Safe slope function with fallback ---
def safe_slope_func(expr):
    compiled = compile(expr, "<string>", "eval")
    def func(x, y):
        try:
            x_arr = np.asarray(x)
            y_arr = np.asarray(y)
            return eval(compiled, {"np": np, "x": x_arr, "y": y_arr})
        except Exception as e:
            print(f"Eval error: {e}")
            return np.full_like(x, np.nan, dtype=np.float64)
    return func

# --- Slope field plotter ---
def generate_plot(ax, expr_raw, x_max, y_max, cmap_name, streamplot, coord_system):
    ax.clear()
    expr = ti84_to_numpy(expr_raw)
    slope_func = safe_slope_func(expr)

    try:
        step = 0.05
        if coord_system == "Cartesian":
            x_vals = np.arange(-x_max, x_max + step, step)
            y_vals = np.arange(-y_max, y_max + step, step)
            X, Y = np.meshgrid(x_vals, y_vals)
        elif coord_system == "Polar":
            r = np.arange(0.01, x_max, step)
            theta = np.arange(0, 2 * np.pi, step)
            R, T = np.meshgrid(r, theta)
            X, Y = R * np.cos(T), R * np.sin(T)
        elif coord_system == "Log-Polar":
            r = np.logspace(-1, np.log10(x_max), int(x_max / step))
            theta = np.linspace(0, 2 * np.pi, int(2 * np.pi / step))
            R, T = np.meshgrid(r, theta)
            X, Y = R * np.cos(T), R * np.sin(T)
        elif coord_system == "Log-Log":
            x_vals = np.logspace(-1, np.log10(x_max), int(x_max / step))
            y_vals = np.logspace(-1, np.log10(y_max), int(y_max / step))
            X, Y = np.meshgrid(x_vals, y_vals)
        elif coord_system == "Hexagonal":
            x_vals = np.arange(-x_max, x_max + step, 0.2)
            y_vals = np.arange(-y_max, y_max + step, 0.2 * np.sqrt(3))
            X, Y = np.meshgrid(x_vals, y_vals)
        elif coord_system == "Complex Plane":
            real = np.arange(-x_max, x_max + step, step)
            imag = np.arange(-y_max, y_max + step, step)
            X, Y = np.meshgrid(real, imag)
        else:
            raise ValueError("Unsupported coordinate system")

        V = slope_func(X, Y)
        if np.isnan(V).all():
            raise ValueError("Invalid function syntax")
        U = np.ones_like(V)
        length = np.sqrt(U**2 + V**2)
        U_norm = 0.25 * U / length
        V_norm = 0.25 * V / length
        X_start = X - U_norm / 2
        X_end = X + U_norm / 2
        Y_start = Y - V_norm / 2
        Y_end = Y + V_norm / 2

        ax.set_facecolor('black')
        ax.set_xlim(np.min(X), np.max(X))
        ax.set_ylim(np.min(Y), np.max(Y))
        ax.set_aspect('equal')
        ax.tick_params(colors='white')
        ax.set_title(f"Slope Field for dy/dx = {expr_raw}", fontsize=14, color='white')

        cmap = colormaps[cmap_name]
        norm = Normalize(vmin=np.min(V), vmax=np.max(V), clip=True)

        if hasattr(ax, "colorbar") and ax.colorbar:
            ax.colorbar.remove()
            ax.colorbar = None

        if streamplot:
            ax.streamplot(X, Y, U, V, color=V, cmap=cmap_name, linewidth=0.7, density=1.5, norm=norm)
        else:
            colors = cmap(norm(V.flatten()))
            for xs, xe, ys, ye, color in zip(X_start.flatten(), X_end.flatten(), Y_start.flatten(), Y_end.flatten(), colors):
                alpha = np.clip(abs((ye - ys) / 0.25), 0.2, 1.0)
                ax.plot([xs, xe], [ys, ye], color=color, linewidth=0.7, alpha=alpha)

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.8, pad=0.02)
        cbar.set_label("Slope dy/dx", fontsize=12, color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

        ax.colorbar = cbar

    except Exception as e:
        ax.clear()
        ax.set_facecolor('black')
        ax.text(0.5, 0.5, "Error: Invalid function\nUse TI-84 style syntax like x^2 + sin(x)", 
                transform=ax.transAxes, fontsize=12, ha='center', va='center', color='white')
        ax.set_xticks([])
        ax.set_yticks([])
        print("Invalid function input. Use TI-84 syntax: x^2, sin(x), (1/5), etc.")
        return

# --- GUI app ---
def run_gui():
    root = tk.Tk()
    root.title("Slope Field Generator")

    tk.Label(root, text="Function dy/dx =", font=('Arial', 12)).grid(row=0, column=0, sticky='w')
    eq_entry = tk.Entry(root, width=40)
    eq_entry.insert(0, "sin(x*y) + cos(x - y)")
    eq_entry.grid(row=0, column=1, sticky='ew')

    tk.Label(root, text="Max X/Y:", font=('Arial', 12)).grid(row=1, column=0, sticky='w')
    range_entry = tk.Entry(root, width=10)
    range_entry.insert(0, "10")
    range_entry.grid(row=1, column=1, sticky='w')

    tk.Label(root, text="Color Palette:", font=('Arial', 12)).grid(row=2, column=0, sticky='w')
    palette_var = ttk.Combobox(root, values=['rainbow', 'viridis', 'plasma', 'magma', 'turbo', 'coolwarm', 'cividis'])
    palette_var.set('viridis')
    palette_var.grid(row=2, column=1, sticky='w')

    tk.Label(root, text="Coordinate System:", font=('Arial', 12)).grid(row=3, column=0, sticky='w')
    coord_var = ttk.Combobox(root, values=['Cartesian', 'Polar', 'Log-Polar', 'Log-Log', 'Hexagonal', 'Complex Plane'])
    coord_var.set('Cartesian')
    coord_var.grid(row=3, column=1, sticky='w')

    stream_var = tk.BooleanVar()
    tk.Checkbutton(root, text="Use streamplot", variable=stream_var).grid(row=4, column=0, sticky='w')

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('black')
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=6, columnspan=2, sticky='nsew')

    root.columnconfigure(1, weight=1)
    root.rowconfigure(6, weight=1)

    def plot():
        expr = eq_entry.get()
        try:
            x_max = float(range_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid number for range.")
            return
        generate_plot(ax, expr, x_max, x_max, palette_var.get(), stream_var.get(), coord_var.get())
        canvas.draw()

    tk.Button(root, text="Generate Plot", command=plot).grid(row=5, columnspan=2)

    root.mainloop()

run_gui()