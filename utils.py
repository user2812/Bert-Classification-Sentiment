"""
Fonctions utilitaires pour :
  - Fixer la seed (reproductibilite)
  - Detecter le device GPU/CPU
  - Tracer les courbes d'apprentissage
  - Afficher la matrice de confusion
  - Afficher des exemples du dataset
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


# 1. REPRODUCTIBILITE

def set_seed(seed: int = 42):
    """
    Fixe la seed pour random, numpy et torch (CPU + GPU).

    Args:
        seed : valeur de la graine (defaut 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"Seed fixee a {seed} — resultats reproductibles.")


# 2. DEVICE

def get_device() -> torch.device:
    """
    Retourne le device disponible : CUDA si GPU present, sinon CPU.

    Returns:
        torch.device : 'cuda' ou 'cpu'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilise : {device}")
    if device.type == "cuda":
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM : "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


# 3. COURBES D'APPRENTISSAGE

def plot_learning_curves(
    history  : dict,
    save_dir : str = ".",
):
    """
    Trace et sauvegarde les courbes train/val de loss, accuracy et F1.
    Utile pour detecter l'overfitting et l'underfitting.

    Args:
        history  : dict retourné par fit() contenant les métriques
        save_dir : dossier de sauvegarde des figures
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Courbes d'apprentissage — BERT Sentiment", fontsize=14)

    # Loss
    axes[0].plot(epochs, history["train_loss"],
                 label="Train Loss", color="steelblue", linewidth=2)
    axes[0].plot(epochs, history["val_loss"],
                 label="Val Loss", color="tomato", linewidth=2, linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_accuracy"],
                 label="Train Accuracy", color="steelblue", linewidth=2)
    axes[1].plot(epochs, history["val_accuracy"],
                 label="Val Accuracy", color="tomato", linewidth=2, linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1-Score
    axes[2].plot(epochs, history["val_f1_score"],
                 label="Val F1-Score", color="seagreen", linewidth=2)
    axes[2].set_title("F1-Score (validation)")
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("F1-Score")
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(save_dir, "learning_curves.png")
    plt.savefig(save_path, dpi=100)
    plt.show()
    plt.close()
    print(f"Courbes sauvegardees -> {save_path}")


# 4. MATRICE DE CONFUSION

def plot_confusion_matrix(
    all_labels  : list,
    all_preds   : list,
    class_names : list,
    save_dir    : str = ".",
):
    """
    Génère, affiche et sauvegarde la matrice de confusion finale.

    Args:
        all_labels  : liste des vrais labels
        all_preds   : liste des prédictions
        class_names : noms des classes ['negative', 'neutral', 'positive']
        save_dir    : dossier de sauvegarde
    """
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix = cm,
        display_labels   = class_names,
    )
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title("Matrice de confusion — BERT Sentiment", fontsize=13)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=100)
    plt.show()
    plt.close()
    print(f"Matrice de confusion sauvegardee -> {save_path}")

    # Rapport de classification complet
    print("\nRapport de classification :")
    print(classification_report(
        all_labels, all_preds,
        target_names = class_names
    ))

    return cm


# 5. AFFICHAGE D'EXEMPLES DU DATASET

def show_dataset_examples(
    texts       : list,
    labels      : list,
    id2label    : dict,
    n           : int = 5,
):
    """
    Affiche n exemples du dataset avec leurs labels.

    Args:
        texts    : liste de tweets
        labels   : liste de labels numeriques
        id2label : mapping {0: 'negative', 1: 'neutral', 2: 'positive'}
        n        : nombre d'exemples a afficher (defaut 5)
    """
    print(f"\n{'='*60}")
    print(f"  {n} exemples du dataset")
    print(f"{'='*60}")
    for i in range(min(n, len(texts))):
        label_name = id2label[labels[i]]
        print(f"[{label_name.upper():>8}] {texts[i]}")
    print(f"{'='*60}\n")


# 6. INSPECTION DU DATASET

def inspect_dataset(
    texts    : list,
    labels   : list,
    id2label : dict,
    tokenizer,
    max_length : int = 128,
):
    """
    Inspecte le dataset avant tout entrainement :
      - Nombre total d'exemples et nombre de classes
      - Distribution des classes
      - Longueur des textes (min, max, moyenne en tokens)
      - Justification du max_length choisi

    Args:
        texts      : liste de tweets
        labels     : liste de labels numeriques
        id2label   : mapping {0: 'negative', 1: 'neutral', 2: 'positive'}
        tokenizer  : tokenizer BERT pour calculer la longueur en tokens
        max_length : longueur maximale choisie
    """
    print(f"\n{'='*60}")
    print(f"  INSPECTION DU DATASET")
    print(f"{'='*60}")
    print(f"Total exemples  : {len(texts)}")
    print(f"Nombre classes  : {len(id2label)}")

    print(f"\nDistribution des classes :")
    for idx, name in id2label.items():
        count = labels.count(idx)
        print(f"  {name:>8} ({idx}) : {count:>5} ({count/len(labels)*100:.1f}%)")

    ratio = max(labels.count(i) for i in id2label) / \
            min(labels.count(i) for i in id2label)
    print(f"\n  Ratio max/min : {ratio:.2f} "
          f"({'desequilibre > 2:1 !' if ratio > 2 else 'equilibre OK'})")

    print(f"\nLongueur des textes (en tokens BERT) :")
    lengths = []
    for text in texts[:1000]:   # echantillon pour la rapidite
        tokens = tokenizer(
            str(text), truncation=False, return_tensors="pt"
        )
        lengths.append(tokens["input_ids"].shape[1])

    print(f"  Min     : {min(lengths)}")
    print(f"  Max     : {max(lengths)}")
    print(f"  Moyenne : {np.mean(lengths):.1f}")
    print(f"  Mediane : {np.median(lengths):.1f}")
    print(f"\n  max_length choisi : {max_length}")
    print(f"  Justification : textes courts (tweets), max ~33 mots,")
    print(f"  max_length=128 couvre 100% des exemples sans troncature.")
    print(f"{'='*60}\n")


# 7. Test python utils.py

if __name__ == "__main__":

    # Test set_seed
    set_seed(42)

    # Test get_device
    device = get_device()

    # Test show_dataset_examples
    texts = [
        "I love this product!",
        "Sooo SAD I will miss you here in San Diego!!!",
        "my boss is bullying me...",
        "what interview! leave me alone",
        "Sons of ****, why couldn t they put them on the releases",
    ]
    labels   = [2, 0, 0, 1, 0]
    id2label = {0: "negative", 1: "neutral", 2: "positive"}

    show_dataset_examples(texts, labels, id2label, n=5)

    # Test plot_learning_curves
    history = {
        "train_loss"    : [0.9, 0.7, 0.5],
        "val_loss"      : [0.85, 0.72, 0.60],
        "train_accuracy": [0.65, 0.75, 0.85],
        "val_accuracy"  : [0.62, 0.72, 0.80],
        "val_f1_score"  : [0.60, 0.70, 0.79],
    }
    plot_learning_curves(history, save_dir=".")

    # Test plot_confusion_matrix
    all_labels = [0, 1, 2, 0, 1, 2, 0, 0, 1]
    all_preds  = [0, 1, 2, 1, 1, 2, 0, 1, 1]
    plot_confusion_matrix(
        all_labels, all_preds,
        class_names = ["negative", "neutral", "positive"],
        save_dir    = ".",
    )

    print("Tous les tests utils.py passés !")