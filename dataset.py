"""
Classe TextClassificationDataset pour le chargement et la tokenization
des tweets pour la classification de sentiment (negative/neutral/positive).
Dataset : Twitter Sentiment (27 481 exemples, 3 classes)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split


# 1. MAPPING DES LABELS

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


# 2. CLASSE DATASET

class TextClassificationDataset(Dataset):
    """
    Dataset PyTorch personnalisé pour la classification de sentiment sur tweets.

    Tokenise les textes avec BERT et retourne :
        - input_ids      : tokens numerisés
        - attention_mask : masque d'attention (1=vrai token, 0=padding)
        - label          : classe (0=negative, 1=neutral, 2=positive)

    Args:
        texts      : liste de tweets (strings)
        labels     : liste de labels numeriques (ints)
        tokenizer  : tokenizer BERT Hugging Face
        max_length : longueur maximale de tokenization (defaut 128)
    """

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Retourne le nombre total d'exemples."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Tokenise et retourne un exemple a l'index idx.

        Returns:
            dict avec input_ids, attention_mask (Tensors) et label (long)
        """
        text = str(self.texts[idx])

        # Tokenization avec padding et truncation
        encoding = self.tokenizer(
            text,
            max_length      = self.max_length,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt",
        )

        return {
            "input_ids"      : encoding["input_ids"].squeeze(0),       # [max_length]
            "attention_mask" : encoding["attention_mask"].squeeze(0),   # [max_length]
            "label"          : torch.tensor(self.labels[idx], dtype=torch.long),
        }


# 3. CHARGEMENT ET PREPARATION DU DATASET

def load_data(csv_path, text_col="text", label_col="sentiment"):
    """
    Charge le CSV, nettoie les valeurs nulles et encode les labels.

    Args:
        csv_path  : chemin vers le fichier CSV
        text_col  : nom de la colonne texte
        label_col : nom de la colonne label

    Returns:
        texts  : liste de tweets nettoyee
        labels : liste de labels numeriques
    """
    df = pd.read_csv(csv_path, encoding="latin-1")

    # Suppression de la ligne avec valeur nulle dans text
    df = df.dropna(subset=[text_col, label_col])

    # Encodage des labels
    df[label_col] = df[label_col].map(LABEL2ID)

    texts  = df[text_col].tolist()
    labels = df[label_col].tolist()

    print(f"Dataset charge : {len(texts)} exemples")
    print(f"Distribution des classes :")
    for name, idx in LABEL2ID.items():
        count = labels.count(idx)
        print(f"  {name} ({idx}) : {count} ({count/len(labels)*100:.1f}%)")

    return texts, labels


# 4. FACTORY : DataLoaders prêts a l'emploi

def get_dataloaders(
    csv_path,
    tokenizer,
    max_length  = 128,
    batch_size  = 16,
    test_size   = 0.2,
    seed        = 42,
    num_workers = 2,
):
    """
    Charge le dataset, effectue un split 80/20 stratifie et retourne
    les DataLoaders train et valid prets a l'emploi.

    Args:
        csv_path    : chemin vers le fichier CSV
        tokenizer   : tokenizer BERT Hugging Face
        max_length  : longueur maximale de tokenization
        batch_size  : taille des batchs
        test_size   : proportion du split validation (defaut 0.2)
        seed        : graine de reproductibilite
        num_workers : workers pour le chargement parallele

    Returns:
        (train_loader, valid_loader)
    """
    texts, labels = load_data(csv_path)

    # Split 80/20 stratifie
    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        texts, labels,
        test_size    = test_size,
        stratify     = labels,
        random_state = seed,
    )

    print(f"\nSplit : train={len(train_texts)} | valid={len(valid_texts)}")

    # Creation des datasets
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, max_length
    )
    valid_dataset = TextClassificationDataset(
        valid_texts, valid_labels, tokenizer, max_length
    )

    # Generateur pour la reproductibilite
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        generator   = g,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    return train_loader, valid_loader


# 5. Test python dataset.py

if __name__ == "__main__":
    from transformers import BertTokenizer

    CSV_PATH  = "data/train.csv"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_loader, valid_loader = get_dataloaders(
        csv_path  = CSV_PATH,
        tokenizer = tokenizer,
        max_length = 128,
        batch_size = 16,
    )

    # Verification d'un batch
    batch = next(iter(train_loader))
    print(f"\nBatch input_ids shape      : {batch['input_ids'].shape}")
    print(f"Batch attention_mask shape : {batch['attention_mask'].shape}")
    print(f"Batch labels shape         : {batch['label'].shape}")
    print(f"Labels dtype               : {batch['label'].dtype}")

    # Affichage de 5 exemples
    print("\n=== 5 exemples du dataset ===")
    texts, labels = load_data(CSV_PATH)
    for i in range(5):
        label_name = ID2LABEL[labels[i]]
        print(f"[{label_name}] {texts[i]}")