import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_equity_curves(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, data in results.items():
        ax.plot(data["dates"], data["values"], label=name, linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.set_title("Equity Curves \u2014 Test Period")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "equity_curves.png"), dpi=150)
    plt.close(fig)


def plot_drawdowns(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, data in results.items():
        vals = np.asarray(data["values"], dtype=np.float64)
        cummax = np.maximum.accumulate(vals)
        dd = (vals - cummax) / cummax
        ax.plot(data["dates"], dd, label=name, linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown Curves \u2014 Test Period")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "drawdown_curves.png"), dpi=150)
    plt.close(fig)
