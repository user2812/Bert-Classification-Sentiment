"""
Boucles d'entrainement et d'évaluation pour le fine-tuning de BERT.
Implemente :
  - train_epoch() : une epoch d'entrainement complète
  - eval_epoch()  : une epoch d'évaluation complète
  - fit()         : boucle complète avec early stopping et sauvegarde

Métriques collectées à chaque epoch :
  train_loss, train_accuracy, val_loss, val_accuracy, val_f1_score, learning_rate

Aucun Trainer Hugging Face n'est utilisé.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm


# 1. TRAIN EPOCH

def train_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    optimizer : torch.optim.Optimizer,
    device    : torch.device,
    scheduler = None,
) -> tuple:
    """
    Effectue une epoch complète d'entrainement.

    Args:
        model     : BertSentimentClassifier a entrainer
        loader    : DataLoader du split train
        criterion : fonction de loss (CrossEntropyLoss)
        optimizer : optimiseur (AdamW)
        device    : 'cuda' ou 'cpu'
        scheduler : scheduler de learning rate (optionnel)

    Returns:
        (train_loss, train_accuracy) moyennes sur l'epoch
    """
    model.train()   # mode entrainement — active Dropout

    running_loss = 0.0
    correct      = 0
    total        = 0

    progress = tqdm(loader, desc="  Train", leave=False)

    for batch in progress:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)   # torch.long requis

        # Forward
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)   # [B, num_classes]
        loss   = criterion(logits, labels)

        # Backward
        loss.backward()

        # Gradient clipping — important pour BERT (évite l'explosion)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Métriques batch
        running_loss += loss.item() * input_ids.size(0)
        preds         = logits.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += input_ids.size(0)

        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc" : f"{correct / total:.4f}",
        })

    epoch_loss = running_loss / total
    epoch_acc  = correct / total

    return epoch_loss, epoch_acc


# 2. EVAL EPOCH

def eval_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    device    : torch.device,
) -> tuple:
    """
    Effectue une epoch complète d'evaluation (validation).

    Args:
        model     : BertSentimentClassifier a evaluer
        loader    : DataLoader du split valid
        criterion : fonction de loss (CrossEntropyLoss)
        device    : 'cuda' ou 'cpu'

    Returns:
        (val_loss, val_accuracy, val_f1_score) moyens sur l'epoch
    """
    model.eval()   # mode évaluation — désactive Dropout

    running_loss = 0.0
    correct      = 0
    total        = 0
    all_preds    = []
    all_labels   = []

    with torch.no_grad():   # pas de calcul de gradient en évaluation
        progress = tqdm(loader, desc="  Valid", leave=False)

        for batch in progress:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            # Forward
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)

            # Métriques batch
            running_loss += loss.item() * input_ids.size(0)
            preds         = logits.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += input_ids.size(0)

            # Collecte CPU pour F1-score global
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc" : f"{correct / total:.4f}",
            })

    epoch_loss = running_loss / total
    epoch_acc  = correct / total

    # F1-score weighted (adapté aux 3 classes)
    epoch_f1 = f1_score(all_labels, all_preds, average="weighted")

    return epoch_loss, epoch_acc, epoch_f1


# 3. BOUCLE COMPLETE : FIT

def fit(
    model          : nn.Module,
    train_loader   : DataLoader,
    valid_loader   : DataLoader,
    criterion      : nn.Module,
    optimizer      : torch.optim.Optimizer,
    device         : torch.device,
    epochs         : int,
    checkpoint_dir : str = "checkpoints",
    scheduler      = None,
    patience       : int = 3,
    save_every     : int = 1,
) -> dict:
    """
    Boucle d'entrainement complète avec :
      - Logging des métriques à chaque epoch
      - Sauvegarde du meilleur modèle (best val_loss)
      - Sauvegarde regulière (save_every epochs)
      - Early stopping si val_loss ne s'améliore plus

    Args:
        model          : BertSentimentClassifier
        train_loader   : DataLoader train
        valid_loader   : DataLoader valid
        criterion      : CrossEntropyLoss
        optimizer      : AdamW
        device         : 'cuda' ou 'cpu'
        epochs         : nombre maximum d'epochs
        checkpoint_dir : dossier de sauvegarde
        scheduler      : learning rate scheduler (optionnel)
        patience       : epochs sans amelioration avant early stopping
        save_every     : intervalle de sauvegarde reguliere

    Returns:
        history (dict) : historique complet des metriques
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss  = float("inf")
    patience_count = 0
    best_path      = os.path.join(checkpoint_dir, "bert_best.pth")

    history = {
        "train_loss"    : [],
        "train_accuracy": [],
        "val_loss"      : [],
        "val_accuracy"  : [],
        "val_f1_score"  : [],
        "learning_rate" : [],
    }

    print(f"\n{'='*55}")
    print(f"  Entrainement BERT Sentiment Classifier")
    print(f"  Epochs : {epochs} | Device : {device}")
    print(f"  Sauvegarde reguliere toutes les {save_every} epoch(s)")
    print(f"{'='*55}")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch [{epoch:02d}/{epochs:02d}]")

        # Entrainement
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )

        # Evaluation
        val_loss, val_acc, val_f1 = eval_epoch(
            model, valid_loader, criterion, device
        )

        # Learning rate courant
        current_lr = optimizer.param_groups[0]["lr"]

        # Historique
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_f1_score"].append(val_f1)
        history["learning_rate"].append(current_lr)

        # Affichage console
        print(
            f"  train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f} | lr={current_lr:.2e}"
        )

        # Sauvegarde regulière
        if epoch % save_every == 0:
            periodic_path = os.path.join(
                checkpoint_dir, f"bert_epoch{epoch:02d}.pth"
            )
            torch.save({
                "epoch"               : epoch,
                "model_state_dict"    : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss"            : val_loss,
                "val_acc"             : val_acc,
                "val_f1"              : val_f1,
            }, periodic_path)
            print(f"  Checkpoint periodique -> {periodic_path}")

        # Sauvegarde meilleur modèle (best val_loss)
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({
                "epoch"               : epoch,
                "model_state_dict"    : model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss"            : val_loss,
                "val_acc"             : val_acc,
                "val_f1"              : val_f1,
            }, best_path)
            print(f"  Meilleur modèle sauvegardé -> {best_path}")
        else:
            patience_count += 1
            print(f"  Pas d'amélioration ({patience_count}/{patience})")

        # Early stopping
        if patience_count >= patience:
            print(f"\n  Early stopping déclenché à l'epoch {epoch}.")
            break

    print(f"\n{'='*55}")
    print(f"  Entrainement terminé")
    print(f"  Meilleur val_loss : {best_val_loss:.4f}")
    print(f"  Checkpoint        : {best_path}")
    print(f"{'='*55}\n")

    return history


# 4. PIPELINE PRINCIPAL

if __name__ == "__main__":
    from transformers import BertTokenizer
    from model import BertSentimentClassifier, get_tokenizer
    from dataset import get_dataloaders, load_data, ID2LABEL
    from utils import set_seed, get_device, plot_learning_curves, \
                      plot_confusion_matrix, show_dataset_examples, \
                      inspect_dataset

    # Seed au tout début
    set_seed(42)
    device = get_device()

    # Hyperparamètres
    CSV_PATH       = "data/train-3.csv"
    MODEL_NAME     = "bert-base-uncased"
    MAX_LENGTH     = 128
    BATCH_SIZE     = 32
    EPOCHS         = 5
    LR             = 1e-5
    WEIGHT_DECAY   = 0.05
    WARMUP_RATIO   = 0.15
    PATIENCE       = 3
    CHECKPOINT_DIR = "checkpoints"

    # Chargement tokenizer
    tokenizer = get_tokenizer(MODEL_NAME)

    # Inspection du dataset
    texts, labels = load_data(CSV_PATH)
    inspect_dataset(texts, labels, ID2LABEL, tokenizer, MAX_LENGTH)
    show_dataset_examples(texts, labels, ID2LABEL, n=5)

    # DataLoaders
    train_loader, valid_loader = get_dataloaders(
        csv_path   = CSV_PATH,
        tokenizer  = tokenizer,
        max_length = MAX_LENGTH,
        batch_size = BATCH_SIZE,
        seed       = 42,
    )

    # Modèle
    model = BertSentimentClassifier(
        model_name  = MODEL_NAME,
        num_classes = 3,
        dropout     = 0.3,
    ).to(device)

    print(f"Parametres entrainables : {model.count_parameters():,}")

    # Loss + Optimiseur
    criterion = nn.CrossEntropyLoss()
    # Séparation des paramètres pour n'appliquer le Weight Decay que là où c'est nécessaire
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LR)

    # Scheduler linéaire avec warmup (recommandé pour BERT)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    # Entrainement
    history = fit(
        model          = model,
        train_loader   = train_loader,
        valid_loader   = valid_loader,
        criterion      = criterion,
        optimizer      = optimizer,
        device         = device,
        epochs         = EPOCHS,
        checkpoint_dir = CHECKPOINT_DIR,
        scheduler      = scheduler,
        patience       = PATIENCE,
        save_every     = 1,
    )

    # Courbes d'apprentissage
    plot_learning_curves(history, save_dir=CHECKPOINT_DIR)

    # Matrice de confusion sur valid
    from model import load_best_model

    best_model = load_best_model(
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "bert_best.pth"),
        model_name      = MODEL_NAME,
        num_classes     = 3,
        device          = device,
    )

    # Collecte des prédictions sur valid
    best_model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(valid_loader, desc="Evaluation finale"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch   = batch["label"].to(device)
            logits         = best_model(input_ids, attention_mask)
            preds          = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names = ["negative", "neutral", "positive"],
        save_dir    = CHECKPOINT_DIR,
    )

    print("\nPipeline terminé avec succès !")