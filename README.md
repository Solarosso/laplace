# Laplace Transform Visualizer

An interactive 3D visualization tool for exploring Laplace transforms in the complex plane.

![Python 3](https://img.shields.io/badge/python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

## What it does

Given a transfer function defined by poles, zeros, and a gain:

```
F(s) = G · ∏(s − zᵢ) / ∏(s − pᵢ)     where s = σ + jω
```

The tool renders four live views that update together as you interact:

| Panel | What you see |
|---|---|
| **3D surface** | `\|F(σ + jω)\|` over the entire complex plane, colored with a plasma gradient |
| **Pole-zero diagram** | s-plane plot with poles (×) and zeros (○), shaded ROC |
| **Magnitude slice** | `\|F(σ + jω)\|` along a vertical slice at the chosen σ — automatically switches to log scale for high dynamic range |
| **Phase slice** | `∠F` in degrees along the same slice |

A yellow slider sweeps σ across the surface in real time. When σ = 0 the slice is the classical **Fourier Transform** and the label says so.

## Requirements

```
python >= 3.10
numpy
matplotlib
tkinter   # usually ships with Python; on Ubuntu: sudo apt install python3-tk
```

Install Python dependencies:

```bash
pip install numpy matplotlib
```

## Running

```bash
python laplace_visualizer.py
```

## Controls

### Slider
Drag the **σ** slider to sweep the vertical slice across the complex plane. The magnitude and phase plots update live, and the 3D curtain moves with it.

### Poles / Zeros / Gain fields
Enter comma-separated complex numbers, then press **Enter** or click **Apply**.

```
# Examples
Poles:  -1+2j, -1-2j, -3
Zeros:  0
Gain:   5
```

Both `a+bj` and pure real/imaginary values are accepted.

### Presets
Six built-in filter archetypes to get started quickly:

| Preset | Description |
|---|---|
| LP 1st | 1st-order low-pass |
| LP 2nd | 2nd-order low-pass (complex poles) |
| Bandpass | Bandpass filter |
| Highpass | High-pass with two complex poles |
| Notch | Notch filter (zeros on jω axis) |
| Unstable | System with poles in the right half-plane |

### Random
Generates a random stable or unstable system with conjugate pole/zero pairs.

### System Info panel
Displays the current gain, pole list, zero list, ROC boundary (`σ > max(Re(pᵢ))`), stability status, and whether the current σ is inside the ROC.

## Stability and ROC

- The **Region of Convergence (ROC)** is `σ > max(Re(pᵢ))` for a right-sided signal.
- A system is **stable** when all poles lie in the left half-plane (ROC includes the jω axis).
- The pole-zero diagram shades the left half-plane green to make this immediately visible.

## Project structure

```
laplace_visualizer.py   # single-file application, no external assets needed
```
