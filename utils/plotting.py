import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


# Global plotting style
sns.set_theme(style="whitegrid", context="paper")
sns.set_palette("colorblind")

plt.rcParams.update({
    "figure.figsize": (10, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "lines.linewidth": 2,
})


def plot_results(args, preds, trues, zoom_to=300):

    plot_path = os.path.join("./plots", args.d_setting)
    os.makedirs(plot_path, exist_ok=True)

    preds = preds[:, -1]
    trues = trues[:, -1]

    t_range = np.arange(len(preds))

    fig, ax = plt.subplots()

    ax.plot(t_range, trues, label="Ground Truth", color="black", alpha=0.8)
    ax.plot(t_range, preds, label="Prediction", color="tab:red", alpha=0.9)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")

    ax.legend(frameon=False)
    sns.despine()

    fig.tight_layout()

    save_path = os.path.join(plot_path, "forecast.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

    # -----------------------------
    # Zoomed forecast (first zoom_to=300)
    # -----------------------------
    zoom_n = min(zoom_to, len(preds))

    fig, ax = plt.subplots()

    ax.plot(t_range[:zoom_n], trues[:zoom_n],
            label="Ground Truth", color="black", alpha=0.8)

    ax.plot(t_range[:zoom_n], preds[:zoom_n],
            label="Prediction", color="tab:red", alpha=0.9)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend(frameon=False)

    sns.despine()
    fig.tight_layout()

    fig.savefig(os.path.join(plot_path, f"forecast_zoom_{zoom_to}.png"),
                dpi=300, bbox_inches="tight")

    plt.close(fig)