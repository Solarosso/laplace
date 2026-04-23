#!/usr/bin/env python3
"""
Laplace Transform 3D Visualizer
F(s) = gain · prod(s - z_i) / prod(s - p_i)

Controls live in a native tkinter panel; plots are matplotlib figures
embedded via FigureCanvasTkAgg.
"""
import os
import numpy as np

# ── mpl_toolkits path fix ─────────────────────────────────────────────────────
# System mpl_toolkits (with __init__.py) shadows the pip-installed one.
# Inject the user copy BEFORE matplotlib is imported.
import mpl_toolkits as _mpl_tb
_user_tb = os.path.expanduser('~/.local/lib/python3.10/site-packages/mpl_toolkits')
if os.path.isdir(_user_tb) and _user_tb not in list(_mpl_tb.__path__):
    _mpl_tb.__path__.insert(0, _user_tb)

import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt           # only used for rcParams
import matplotlib.gridspec as gridspec
import matplotlib.figure as mfigure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── palette ───────────────────────────────────────────────────────────────────
C_BG     = '#060614'
C_PLOT   = '#0b0b1e'
C_PANEL  = '#0e0e24'
C_CARD   = '#141430'
C_BORDER = '#28285a'
C_TEXT   = '#c6caf0'
C_DIM    = '#4a4e74'
C_ACCENT = '#4d8cff'
C_YELLOW = '#f0d020'
C_RED    = '#ff3e3e'
C_GREEN  = '#26d068'
C_ORANGE = '#ff8820'
C_PURPLE = '#9955ff'


class LaplaceVisualizer:

    _PRESETS = [
        ('LP  1st',
         np.array([-2.0+0j]),
         np.array([]),
         2.0),
        ('LP  2nd',
         np.array([-1.0+2j, -1.0-2j]),
         np.array([]),
         5.0),
        ('Bandpass',
         np.array([-0.5+6j, -0.5-6j]),
         np.array([0.0+0j]),
         12.0),
        ('Highpass',
         np.array([-1.0+2j, -1.0-2j, -3.0+0j]),
         np.array([0.0+0j, 0.0+0j]),
         4.0),
        ('Notch',
         np.array([-0.3+5j, -0.3-5j, -2.0+0j]),
         np.array([0.0+5j, 0.0-5j]),
         1.0),
        ('Unstable',
         np.array([0.5+3j, 0.5-3j, -1.0+0j]),
         np.array([]),
         2.0),
    ]

    # ── init ──────────────────────────────────────────────────────────────────
    def __init__(self):
        self.poles = np.array([-1.0+0j, -0.5+3j, -0.5-3j])
        self.zeros = np.array([0.0+0j])
        self.gain  = 3.0

        self.s_min, self.s_max = -4.0,  2.0
        self.w_min, self.w_max = -20.0, 20.0
        self.Ns, self.Nw       = 90, 220
        self.clip_pct          = 96
        self.clip_scale        = 2.5
        self.slice_sigma       = 0.0
        self._surf_mag         = None
        self._surf_cap         = 1.0

        self._build_grid()
        self._build_tk_window()
        self._embed_figures()
        self._build_controls()
        self._full_update()

    # ── grid + math ───────────────────────────────────────────────────────────
    def _build_grid(self):
        self.sigmas = np.linspace(self.s_min, self.s_max, self.Ns)
        self.omegas = np.linspace(self.w_min, self.w_max, self.Nw)
        self.SIG, self.OMG = np.meshgrid(self.sigmas, self.omegas)
        self.S_grid = self.SIG + 1j * self.OMG

    def _eval(self, s):
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.full(np.shape(s), complex(self.gain), dtype=complex)
            for z in self.zeros:
                out *= (s - z)
            for p in self.poles:
                out /= (s - p)
        return out

    def _compute_surface(self):
        mag = np.abs(self._eval(self.S_grid))
        fin = mag[np.isfinite(mag)]
        if fin.size == 0:
            return np.zeros_like(mag), 1.0
        cap = float(np.percentile(fin, self.clip_pct) * self.clip_scale)
        return np.where(np.isfinite(mag), np.clip(mag, 0, cap), cap), cap

    def _compute_slice(self, sigma):
        return self._eval(sigma + 1j * self.omegas)

    def _roc_bound(self):
        return float(np.max(self.poles.real)) if self.poles.size else -np.inf

    # ── window + figure construction ──────────────────────────────────────────
    def _build_tk_window(self):
        plt.rcParams.update({
            'figure.facecolor' : C_BG,
            'axes.facecolor'   : C_PLOT,
            'text.color'       : C_TEXT,
            'axes.labelcolor'  : C_TEXT,
            'xtick.color'      : C_DIM,
            'ytick.color'      : C_DIM,
            'axes.edgecolor'   : C_BORDER,
            'grid.color'       : '#151532',
            'grid.alpha'       : 1.0,
            'font.size'        : 9,
        })

        self.root = tk.Tk()
        self.root.title('Laplace Transform Visualizer')
        self.root.configure(bg=C_BG)
        self.root.geometry('1440x920')
        self.root.minsize(900, 620)

        # ── header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=C_BG, padx=14, pady=7)
        hdr.pack(side='top', fill='x')
        tk.Label(hdr,
                 text='LAPLACE  TRANSFORM  VISUALIZER',
                 bg=C_BG, fg=C_ACCENT,
                 font=('Courier New', 12, 'bold')).pack(side='left')
        tk.Label(hdr,
                 text="  F(s) = G · ∏(s − zᵢ) / ∏(s − pᵢ)"
                      "   |   s = σ + jω",
                 bg=C_BG, fg=C_DIM,
                 font=('Courier New', 9)).pack(side='left')

        ttk.Separator(self.root, orient='horizontal').pack(fill='x')

        # ── plots area (expands) ───────────────────────────────────────────────
        plots_frame = tk.Frame(self.root, bg=C_BG)
        plots_frame.pack(side='top', fill='both', expand=True)
        plots_frame.columnconfigure(0, weight=65)
        plots_frame.columnconfigure(1, weight=35)
        plots_frame.rowconfigure(0, weight=1)

        self.frame_3d = tk.Frame(plots_frame, bg=C_PLOT)
        self.frame_3d.grid(row=0, column=0, sticky='nsew')
        self.frame_3d.rowconfigure(0, weight=1)
        self.frame_3d.columnconfigure(0, weight=1)

        self.frame_side = tk.Frame(plots_frame, bg=C_PLOT)
        self.frame_side.grid(row=0, column=1, sticky='nsew')
        self.frame_side.rowconfigure(0, weight=1)
        self.frame_side.columnconfigure(0, weight=1)

        # ── controls panel (fixed height) ─────────────────────────────────────
        self.controls_frame = tk.Frame(self.root, bg=C_PANEL, height=235)
        self.controls_frame.pack(side='top', fill='x')
        self.controls_frame.pack_propagate(False)

        # ── matplotlib figures ────────────────────────────────────────────────
        self.fig_3d = mfigure.Figure(figsize=(13, 8))
        self.fig_3d.patch.set_facecolor(C_BG)
        self.ax3d = self.fig_3d.add_subplot(111, projection='3d')
        self._style_3d(self.ax3d)

        self.fig_side = mfigure.Figure(figsize=(7, 8))
        self.fig_side.patch.set_facecolor(C_BG)
        gs_r = gridspec.GridSpec(
            3, 1, figure=self.fig_side,
            hspace=0.72, height_ratios=[0.80, 1.15, 1.15],
            left=0.16, right=0.97, top=0.96, bottom=0.07,
        )
        self.ax_pz    = self.fig_side.add_subplot(gs_r[0])
        self.ax_mag   = self.fig_side.add_subplot(gs_r[1])
        self.ax_phase = self.fig_side.add_subplot(gs_r[2])
        for ax in (self.ax_pz, self.ax_mag, self.ax_phase):
            ax.set_facecolor(C_PLOT)
            for sp in ax.spines.values():
                sp.set_edgecolor(C_BORDER)

    def _embed_figures(self):
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=self.frame_3d)
        self.canvas_3d.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        self.canvas_side = FigureCanvasTkAgg(self.fig_side, master=self.frame_side)
        self.canvas_side.get_tk_widget().grid(row=0, column=0, sticky='nsew')

    # ── control panel ─────────────────────────────────────────────────────────
    def _build_controls(self):
        cf = self.controls_frame
        cf.columnconfigure(0, weight=75)
        cf.columnconfigure(1, weight=25)
        cf.rowconfigure(0, weight=1)

        left  = tk.Frame(cf, bg=C_PANEL, padx=12, pady=6)
        left.grid(row=0, column=0, sticky='nsew')

        right = tk.Frame(cf, bg=C_CARD,
                         highlightthickness=1,
                         highlightbackground=C_BORDER)
        right.grid(row=0, column=1, sticky='nsew', padx=(0, 8), pady=6)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # ── σ display row (grid so badge toggle is stable) ────────────────────
        sigma_row = tk.Frame(left, bg=C_PANEL)
        sigma_row.pack(fill='x')
        sigma_row.columnconfigure(2, weight=1)

        self.lbl_sigma = tk.Label(
            sigma_row, text='σ = 0.000',
            bg=C_PANEL, fg=C_YELLOW,
            font=('Courier New', 18, 'bold'),
        )
        self.lbl_sigma.grid(row=0, column=0, sticky='w')

        self.lbl_ft = tk.Label(
            sigma_row, text='[  Fourier Transform  ]',
            bg=C_PANEL, fg=C_GREEN,
            font=('Courier New', 9),
        )
        self.lbl_ft.grid(row=0, column=1, padx=14)

        self.lbl_roc = tk.Label(
            sigma_row, text='',
            bg=C_PANEL, fg=C_DIM,
            font=('Courier New', 8),
        )
        self.lbl_roc.grid(row=0, column=2, sticky='w', padx=6)

        # ── σ scale ───────────────────────────────────────────────────────────
        scale_row = tk.Frame(left, bg=C_PANEL)
        scale_row.pack(fill='x', pady=(1, 3))

        tk.Label(scale_row, text=f'{self.s_min:.0f}',
                 bg=C_PANEL, fg=C_DIM,
                 font=('Courier New', 8)).pack(side='left')

        self.scale_sigma = tk.Scale(
            scale_row,
            from_=self.s_min, to=self.s_max,
            orient='horizontal',
            resolution=0.001,
            showvalue=0,
            command=self._cb_slider,
            bg=C_PANEL, fg=C_TEXT,
            troughcolor=C_CARD,
            activebackground=C_YELLOW,
            sliderrelief='flat',
            highlightthickness=0,
            bd=0,
        )
        self.scale_sigma.set(0.0)
        self.scale_sigma.pack(side='left', fill='x', expand=True, padx=6)

        tk.Label(scale_row, text=f'{self.s_max:.0f}',
                 bg=C_PANEL, fg=C_DIM,
                 font=('Courier New', 8)).pack(side='left')

        # ── separator ─────────────────────────────────────────────────────────
        ttk.Separator(left, orient='horizontal').pack(fill='x', pady=3)

        # ── input row ─────────────────────────────────────────────────────────
        input_row = tk.Frame(left, bg=C_PANEL)
        input_row.pack(fill='x', pady=(2, 2))

        def _entry(parent, lbl_text, lbl_fg, width):
            tk.Label(parent, text=lbl_text,
                     bg=C_PANEL, fg=lbl_fg,
                     font=('Courier New', 8, 'bold')).pack(side='left', padx=(0, 3))
            e = tk.Entry(
                parent, width=width,
                bg=C_CARD, fg=C_TEXT,
                insertbackground=C_YELLOW,
                relief='flat',
                highlightthickness=1,
                highlightcolor=C_ACCENT,
                highlightbackground=C_BORDER,
                font=('Courier New', 9),
            )
            e.pack(side='left', padx=(0, 12), ipady=3)
            e.bind('<Return>', self._cb_apply)
            return e

        self.entry_poles = _entry(input_row, 'Poles', C_RED,   28)
        self.entry_zeros = _entry(input_row, 'Zeros', C_GREEN, 20)
        self.entry_gain  = _entry(input_row, 'Gain',  C_ACCENT, 8)

        def _btn(parent, text, fg, hover):
            b = tk.Button(
                parent, text=text,
                bg=C_CARD, fg=fg,
                activebackground=hover, activeforeground=fg,
                relief='flat', cursor='hand2', bd=0,
                font=('Courier New', 9),
                padx=14, pady=5,
            )
            b.pack(side='left', padx=4)
            b.bind('<Enter>', lambda _e, w=b, h=hover: w.configure(bg=h))
            b.bind('<Leave>', lambda _e, w=b:          w.configure(bg=C_CARD))
            return b

        self.btn_apply  = _btn(input_row, 'Apply',  C_ACCENT, '#1a3a68')
        self.btn_random = _btn(input_row, 'Random', C_GREEN,  '#0a3820')
        self.btn_apply.configure(command=self._cb_apply)
        self.btn_random.configure(command=self._cb_random)

        # ── separator ─────────────────────────────────────────────────────────
        ttk.Separator(left, orient='horizontal').pack(fill='x', pady=3)

        # ── presets row ───────────────────────────────────────────────────────
        preset_row = tk.Frame(left, bg=C_PANEL)
        preset_row.pack(fill='x', pady=(1, 0))
        tk.Label(preset_row, text='PRESETS',
                 bg=C_PANEL, fg=C_PURPLE,
                 font=('Courier New', 7, 'bold')).pack(side='left', padx=(0, 10))

        self._preset_btns = []
        for lbl, pls, zrs, gn in self._PRESETS:
            b = tk.Button(
                preset_row, text=lbl,
                bg=C_CARD, fg=C_TEXT,
                activebackground='#1e1e50', activeforeground=C_TEXT,
                relief='flat', cursor='hand2', bd=0,
                font=('Courier New', 8),
                padx=10, pady=4,
                command=self._make_preset(pls, zrs, gn),
            )
            b.pack(side='left', padx=3)
            b.bind('<Enter>', lambda _e, w=b: w.configure(bg='#1e1e50'))
            b.bind('<Leave>', lambda _e, w=b: w.configure(bg=C_CARD))
            self._preset_btns.append(b)

        # ── info panel ────────────────────────────────────────────────────────
        tk.Label(right, text='System Info',
                 bg=C_CARD, fg=C_PURPLE,
                 font=('Courier New', 9, 'bold'),
                 pady=4).grid(row=0, column=0, sticky='w', padx=8)

        self.info_text = tk.Text(
            right,
            bg=C_CARD, fg=C_TEXT,
            font=('Courier New', 9),
            relief='flat',
            state='disabled',
            wrap='none',
            highlightthickness=0,
            bd=0,
            padx=8, pady=4,
            cursor='arrow',
        )
        self.info_text.grid(row=1, column=0, sticky='nsew', padx=4, pady=(0, 4))

        self.info_text.tag_configure('gain',     foreground=C_ACCENT)
        self.info_text.tag_configure('pole_hdr', foreground=C_RED)
        self.info_text.tag_configure('pole_val', foreground=C_RED)
        self.info_text.tag_configure('zero_hdr', foreground=C_GREEN)
        self.info_text.tag_configure('zero_val', foreground=C_GREEN)
        self.info_text.tag_configure('stable',   foreground=C_GREEN)
        self.info_text.tag_configure('unstable', foreground=C_RED)
        self.info_text.tag_configure('roc_ok',   foreground=C_GREEN)
        self.info_text.tag_configure('roc_out',  foreground=C_RED)
        self.info_text.tag_configure('dim',      foreground=C_DIM)

        # Populate entries with defaults
        self._sync_textboxes()

    # ── 3D axis styling ───────────────────────────────────────────────────────
    @staticmethod
    def _style_3d(ax):
        ax.set_facecolor(C_PLOT)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor('#16163a')

    # ── draw: 3D surface ──────────────────────────────────────────────────────
    def _draw_3d(self, mag, cap):
        ax = self.ax3d
        ax.clear()
        self._style_3d(ax)

        ax.plot_surface(
            self.SIG, self.OMG, mag,
            cmap='plasma', alpha=0.88, linewidth=0,
            antialiased=True, rcount=65, ccount=65,
        )

        # Slice curtain
        wp = np.array([self.w_min, self.w_max])
        Yp, Zp = np.meshgrid(wp, [0.0, cap])
        Xp = np.full_like(Yp, self.slice_sigma)
        ax.plot_surface(Xp, Yp, Zp, color=C_YELLOW, alpha=0.16, shade=False)
        ax.plot(
            [self.slice_sigma, self.slice_sigma],
            [self.w_min, self.w_max], [0, 0],
            color=C_YELLOW, lw=2.0, alpha=0.95,
        )

        for p in self.poles:
            ax.scatter([p.real], [p.imag], [0],
                       color=C_RED, s=90, marker='x',
                       linewidths=2.5, depthshade=False)
            ax.plot([p.real, p.real], [p.imag, p.imag], [0, cap * 0.4],
                    color=C_RED, lw=0.8, alpha=0.22)

        for z in self.zeros:
            ax.scatter([z.real], [z.imag], [0],
                       color=C_GREEN, s=70, marker='o',
                       alpha=0.85, depthshade=False)

        ax.set_xlabel('σ  (real)',  color=C_DIM, fontsize=9, labelpad=5)
        ax.set_ylabel('ω  (imag)', color=C_DIM, fontsize=9, labelpad=5)
        ax.set_zlabel('|F(s)|',    color=C_DIM, fontsize=9, labelpad=5)
        ax.set_title('|F(σ + jω)|  over the Complex Plane',
                     color=C_TEXT, fontsize=10, pad=6)
        ax.tick_params(colors=C_DIM, labelsize=7)
        ax.set_xlim(self.s_min, self.s_max)
        ax.set_ylim(self.w_min, self.w_max)
        ax.set_zlim(0, cap * 1.05)

    # ── draw: pole-zero diagram ───────────────────────────────────────────────
    def _draw_pz(self):
        ax = self.ax_pz
        ax.clear()
        ax.set_facecolor(C_PLOT)
        for sp in ax.spines.values():
            sp.set_edgecolor(C_BORDER)

        ax.axvspan(self.s_min, 0, alpha=0.08, color=C_GREEN)
        ax.axvline(0, color=C_TEXT, lw=0.8, alpha=0.25)
        ax.axhline(0, color=C_TEXT, lw=0.8, alpha=0.25)
        ax.axvline(self.slice_sigma, color=C_YELLOW, lw=1.5, alpha=0.85, ls='--')

        if self.poles.size:
            ax.scatter(self.poles.real, self.poles.imag,
                       color=C_RED, s=120, marker='x', linewidths=2.5, zorder=5)
        if self.zeros.size:
            ax.scatter(self.zeros.real, self.zeros.imag,
                       facecolors='none', edgecolors=C_GREEN,
                       s=100, linewidths=2.0, zorder=5)

        ax.set_xlim(self.s_min, self.s_max)
        ax.set_ylim(self.w_min * 0.55, self.w_max * 0.55)
        ax.set_xlabel('σ', fontsize=8)
        ax.set_ylabel('jω', fontsize=8)
        ax.set_title('Pole-Zero Diagram  (s-plane)',
                     color=C_TEXT, fontsize=8.5, pad=3)
        ax.grid(True)
        ax.tick_params(labelsize=6.5)

    # ── draw: slice plots ─────────────────────────────────────────────────────
    def _draw_slice(self):
        F        = self._compute_slice(self.slice_sigma)
        mag_sl   = np.abs(F)
        phase_sl = np.angle(F, deg=True)
        w        = self.omegas

        # magnitude
        ax = self.ax_mag
        ax.clear()
        ax.set_facecolor(C_PLOT)
        for sp in ax.spines.values():
            sp.set_edgecolor(C_BORDER)

        fin  = mag_sl[np.isfinite(mag_sl)]
        vmax = fin.max() if fin.size else 1.0
        vmin = fin[fin > 0].min() if (fin > 0).any() else 1e-9
        use_log = vmax > 0 and (vmax / max(vmin, 1e-12)) > 300

        if use_log:
            safe = np.where(np.isfinite(mag_sl) & (mag_sl > 0), mag_sl, 1e-9)
            ax.semilogy(w, safe, color=C_ACCENT, lw=1.5)
        else:
            ax.fill_between(w, 0, mag_sl, alpha=0.14, color=C_ACCENT)
            ax.plot(w, mag_sl, color=C_ACCENT, lw=1.5)

        ax.axvline(0, color=C_DIM, lw=0.7, ls='--')
        title = ('Fourier Transform  (σ = 0)'
                 if abs(self.slice_sigma) < 1e-9
                 else f'Slice  σ = {self.slice_sigma:.3f}')
        ax.set_title(title, color=C_YELLOW, fontsize=9, pad=3)
        ax.set_xlabel('ω', fontsize=8)
        ax.set_ylabel('|F(σ+jω)|', fontsize=8)
        ax.grid(True)
        ax.tick_params(labelsize=7)

        # phase
        ax = self.ax_phase
        ax.clear()
        ax.set_facecolor(C_PLOT)
        for sp in ax.spines.values():
            sp.set_edgecolor(C_BORDER)

        ax.fill_between(w, 0, phase_sl, alpha=0.12, color=C_ORANGE)
        ax.plot(w, phase_sl, color=C_ORANGE, lw=1.5)
        ax.axvline(0, color=C_DIM, lw=0.7, ls='--')
        ax.axhline(0, color=C_DIM, lw=0.7, ls='--')
        ax.set_ylim(-185, 185)
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.set_title('Phase Response', color=C_TEXT, fontsize=9, pad=3)
        ax.set_xlabel('ω', fontsize=8)
        ax.set_ylabel('∠F  [°]', fontsize=8)
        ax.grid(True)
        ax.tick_params(labelsize=7)

    # ── update labels + info panel ────────────────────────────────────────────
    def _update_labels(self):
        roc_b  = self._roc_bound()
        stable = roc_b < 0
        in_roc = self.slice_sigma > roc_b

        self.lbl_sigma.configure(text=f'σ = {self.slice_sigma:+.3f}')
        if abs(self.slice_sigma) < 1e-9:
            self.lbl_ft.grid()
        else:
            self.lbl_ft.grid_remove()

        status = 'in ROC' if in_roc else 'outside ROC'
        stab   = 'STABLE' if stable else 'UNSTABLE'
        self.lbl_roc.configure(
            text=f'ROC: σ > {roc_b:.3f}  |  {status}  |  {stab}',
            fg=C_GREEN if in_roc else C_RED,
        )

        it = self.info_text
        it.configure(state='normal')
        it.delete('1.0', 'end')

        it.insert('end', f'Gain  = {self.gain:.5g}\n', 'gain')

        it.insert('end', '\nPoles (×):\n', 'pole_hdr')
        for p in self.poles:
            it.insert('end', '  ' + _cpx(p) + '\n', 'pole_val')

        it.insert('end', '\nZeros (○):\n', 'zero_hdr')
        if self.zeros.size:
            for z in self.zeros:
                it.insert('end', '  ' + _cpx(z) + '\n', 'zero_val')
        else:
            it.insert('end', '  none\n', 'dim')

        roc_tag = 'roc_ok' if in_roc else 'roc_out'
        it.insert('end', f'\nROC:   σ > {roc_b:.3f}\n', roc_tag)
        it.insert('end',
                  f'State: {"Stable" if stable else "UNSTABLE"}\n',
                  'stable' if stable else 'unstable')
        it.insert('end',
                  f'σ in ROC: {"Yes" if in_roc else "No"}\n',
                  roc_tag)

        it.configure(state='disabled')

    # ── canvas refresh ────────────────────────────────────────────────────────
    def _draw_idle(self):
        self.canvas_3d.draw_idle()
        self.canvas_side.draw_idle()

    # ── update flow ───────────────────────────────────────────────────────────
    def _full_update(self):
        self._surf_mag, self._surf_cap = self._compute_surface()
        self._draw_3d(self._surf_mag, self._surf_cap)
        self._draw_pz()
        self._draw_slice()
        self._update_labels()
        self._draw_idle()

    def _slice_update(self):
        self._draw_3d(self._surf_mag, self._surf_cap)
        self._draw_pz()
        self._draw_slice()
        self._update_labels()
        self._draw_idle()

    # ── callbacks ─────────────────────────────────────────────────────────────
    def _cb_slider(self, val):
        self.slice_sigma = float(val)
        self._slice_update()

    def _cb_apply(self, _=None):
        try:
            self.poles = _parse(self.entry_poles.get())
            self.zeros = _parse(self.entry_zeros.get())
            self.gain  = float(self.entry_gain.get())
        except Exception as exc:
            it = self.info_text
            it.configure(state='normal')
            it.delete('1.0', 'end')
            it.insert('end', f'ERROR:\n{exc}', 'unstable')
            it.configure(state='disabled')
            return
        self._full_update()

    def _cb_random(self):
        self._randomize()
        self._sync_textboxes()
        self._full_update()

    def _make_preset(self, poles, zeros, gain):
        def handler():
            self.poles, self.zeros, self.gain = poles.copy(), zeros.copy(), gain
            self._sync_textboxes()
            self._full_update()
        return handler

    # ── randomizer ────────────────────────────────────────────────────────────
    def _randomize(self):
        rng = np.random
        n_cp, n_rp = rng.randint(1, 4), rng.randint(0, 3)
        poles: list = []
        for _ in range(n_cp):
            s = -rng.uniform(0.2, 3.0)
            w =  rng.uniform(1.0, 10.0)
            poles += [s + 1j*w, s - 1j*w]
        for _ in range(n_rp):
            poles.append(-rng.uniform(0.2, 5.0) + 0j)

        n_cz, n_rz = rng.randint(0, n_cp + 1), rng.randint(0, 3)
        zeros: list = []
        for _ in range(n_cz):
            s = rng.uniform(-2.0, 1.0)
            w = rng.uniform(1.0, 8.0)
            zeros += [s + 1j*w, s - 1j*w]
        for _ in range(n_rz):
            zeros.append(rng.uniform(-3.0, 1.0) + 0j)

        self.poles = np.array(poles)
        self.zeros = np.array(zeros)
        self.gain  = float(rng.uniform(0.5, 8.0))

    def _sync_textboxes(self):
        for entry, value in (
            (self.entry_poles, _fmt(self.poles)),
            (self.entry_zeros, _fmt(self.zeros)),
            (self.entry_gain,  f'{self.gain:.4g}'),
        ):
            entry.delete(0, 'end')
            entry.insert(0, value)

    def show(self):
        self.root.mainloop()


# ── module-level helpers ──────────────────────────────────────────────────────

def _cpx(v: complex) -> str:
    if abs(v.imag) < 1e-10:
        return f'{v.real:.4g}'
    sign = '+' if v.imag >= 0 else ''
    return f'{v.real:.4g}{sign}{v.imag:.4g}j'


def _fmt(arr: np.ndarray) -> str:
    return ', '.join(_cpx(v) for v in arr) if arr.size else ''


def _parse(text: str) -> np.ndarray:
    if not text.strip():
        return np.array([], dtype=complex)
    results = []
    for tok in text.split(','):
        tok = tok.strip().replace(' ', '')
        if not tok:
            continue
        if tok in ('j', '+j'):
            tok = '1j'
        elif tok == '-j':
            tok = '-1j'
        try:
            results.append(complex(tok))
        except ValueError:
            results.append(complex(eval(tok, {'__builtins__': {}, 'j': 1j})))
    return np.array(results, dtype=complex)


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app = LaplaceVisualizer()
    app.show()
