import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def random_window(arr, test_len):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr
    if arr.size <= test_len:
        return arr
    start = np.random.randint(0, arr.size - test_len + 1)
    return arr[start : start + test_len]


def plot_kde(ax, sample, title, color):
    if sample.size < 2:
        ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    kde = gaussian_kde(sample, bw_method='silverman')
    x_min, x_max = float(np.min(sample)), float(np.max(sample))
    if abs(x_max - x_min) < 1e-12:
        x_min -= 1.0
        x_max += 1.0

    x_grid = np.linspace(x_min, x_max, 300)
    y_kde = kde(x_grid)

    ax.plot(x_grid, y_kde, color=color, linewidth=2)
    ax.fill_between(x_grid, y_kde, color=color, alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')

def plot_vrp(Agents):
    a = Agents
    test_len = a.long_term_window
    vrp_sample = random_window(a._vrp_history, test_len)
    longterm_sample = random_window(a._longterm_spread_history, test_len)
    ewma_sample = random_window(a._ewma_spread_history, test_len)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_kde(axes[0], vrp_sample, f'VRP KDE (random window={test_len})', 'skyblue')
    plot_kde(axes[1], longterm_sample, f'Longterm Spread KDE (random window={test_len})', 'salmon')
    plot_kde(axes[2], ewma_sample, f'EWMA Spread KDE (random window={test_len})', 'lightgreen')

    plt.tight_layout()
    plt.show()