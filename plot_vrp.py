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
    test_len = a.test_length
    rv22_sample = random_window(a.memory.VRP_rv22, test_len)
    lt_sample = random_window(a.memory.VRP_lt, test_len)
    garch_sample = random_window(a.memory.VRP_garch, test_len)
    vvix_vix_sample = random_window(a.memory.vvix_vix, test_len)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    plot_kde(axes[0], rv22_sample, f'VRP_rv22 KDE (random window={test_len})', 'skyblue')
    plot_kde(axes[1], lt_sample, f'VRP_lt KDE (random window={test_len})', 'salmon')
    plot_kde(axes[2], garch_sample, f'VRP_garch KDE (random window={test_len})', 'lightgreen')
    plot_kde(axes[3], vvix_vix_sample, f'VVIX_VIX KDE (random window={test_len})', 'plum')

    plt.tight_layout()
    plt.show()