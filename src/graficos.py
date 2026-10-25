"""Gráficas reutilizables."""
import matplotlib.pyplot as plt
import numpy as np


def matrices_confusion(matrices, ruta=None):
    fig, axes = plt.subplots(1, len(matrices), figsize=(4 * len(matrices), 3.6))
    for ax, (nombre, cm) in zip(axes, matrices.items()):
        ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set(title=f"{nombre}\nerrores = {cm.sum() - np.trace(cm)}", xlabel="Predicho", ylabel="Real",
               xticks=[0, 1, 2], yticks=[0, 1, 2])
    plt.tight_layout()
    if ruta:
        plt.savefig(ruta, dpi=120)
    plt.show()


def barras_comparacion(df, metrica, ruta=None, etiquetas=("Paper", "Réplica")):
    """df con columnas = etiquetas e índice = modelos."""
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, col in enumerate(etiquetas):
        b = ax.bar(x + (i - 0.5) * 0.38, df[col], 0.38, label=col, color=["#9aa5b1", "#2b6cb0"][i])
        ax.bar_label(b, fmt="%.3f", fontsize=8)
    ax.set(xticks=x, xticklabels=df.index, ylabel=metrica, ylim=(min(0.8, df.min().min() - 0.05), 1.02))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    plt.tight_layout()
    if ruta:
        plt.savefig(ruta, dpi=120)
    plt.show()
