"""
Chargement et définition du modele BERT pour la classification de sentiment.
Modèle : bert-base-uncased (tweets en anglais)
Tâches : classification 3 classes (negative / neutral / positive)

Aucun Trainer Hugging Face n'est utilisé.
La boucle d'entrainement est implementée manuellement dans train.py.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


# 1. MODELE BERT POUR LA CLASSIFICATION

class BertSentimentClassifier(nn.Module):
    """
    Modele BERT pré-entrainé avec une tete de classification linéaire.

    Architecture :
        BERT backbone (bert-base-uncased) -> [CLS] token (768 dim)
        Dropout(p=0.3)
        Linear(768 -> num_classes)

    Le token [CLS] encode la représentation globale de la séquence
    et est utilisé pour la classification.

    Args:
        model_name  : nom du modèle BERT Hugging Face
                      (défaut : 'bert-base-uncased')
        num_classes : nombre de classes en sortie (3 pour notre tache)
        dropout     : probabilité de dropout (défaut 0.3)
    """

    def __init__(
        self,
        model_name  : str = "bert-base-uncased",
        num_classes : int = 3,
        dropout     : float = 0.3,
    ):
        super(BertSentimentClassifier, self).__init__()

        # Chargement du backbone BERT pre-entraine (Hugging Face)
        self.bert = BertModel.from_pretrained(model_name)

        # Dropout pour la régularisation
        self.dropout = nn.Dropout(p=dropout)

        # Tête de classification linéaire
        # BERT hidden size = 768 pour bert-base
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        Propagation avant du modele.

        Args:
            input_ids      : Tensor [batch, max_length] — tokens numerises
            attention_mask : Tensor [batch, max_length] — masque d'attention
                             (1 = vrai token, 0 = padding)

        Returns:
            logits : Tensor [batch, num_classes] — logits bruts
        """
        # Passage dans BERT
        # outputs.last_hidden_state : [batch, max_length, hidden_size]
        # outputs.pooler_output     : [batch, hidden_size] — representation [CLS]
        outputs = self.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask,
        )

        # Extraction du token [CLS] (position 0)
        # C'est la representation globale de la sequence
        cls_output = outputs.pooler_output   # [batch, 768]

        # Dropout + classification
        cls_output = self.dropout(cls_output)
        logits     = self.classifier(cls_output)   # [batch, num_classes]

        return logits

    def count_parameters(self):
        """
        Retourne le nombre total de parametres entrainables.

        Returns:
            int : nombre de parametres
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 2. CHARGEMENT DU TOKENIZER

def get_tokenizer(model_name: str = "bert-base-uncased"):
    """
    Charge et retourne le tokenizer BERT depuis Hugging Face.

    Args:
        model_name : nom du modele BERT (defaut : 'bert-base-uncased')

    Returns:
        BertTokenizer
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    print(f"Tokenizer charge : {model_name}")
    print(f"Vocabulaire : {tokenizer.vocab_size} tokens")
    return tokenizer


# 3. CHARGEMENT DU MEILLEUR MODELE

def load_best_model(
    checkpoint_path : str,
    model_name      : str = "bert-base-uncased",
    num_classes     : int = 3,
    device          : torch.device = torch.device("cpu"),
):
    """
    Charge les poids du meilleur checkpoint sauvegarde par train.py.

    Args:
        checkpoint_path : chemin vers le fichier .pth
        model_name      : nom du modele BERT
        num_classes     : nombre de classes
        device          : device cible (cpu ou cuda)

    Returns:
        model : BertSentimentClassifier avec les meilleurs poids charges
    """
    model = BertSentimentClassifier(
        model_name  = model_name,
        num_classes = num_classes,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Modele charge depuis : {checkpoint_path}")
    print(f"  Epoch      : {checkpoint.get('epoch', 'N/A')}")
    print(f"  val_loss   : {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  val_acc    : {checkpoint.get('val_acc', 'N/A'):.4f}")
    print(f"  val_f1     : {checkpoint.get('val_f1', 'N/A'):.4f}")

    return model


# 4. Test python model.py

if __name__ == "__main__":

    print("=" * 55)
    print("  TEST BertSentimentClassifier")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Chargement tokenizer
    tokenizer = get_tokenizer("bert-base-uncased")

    # Chargement modèle
    model = BertSentimentClassifier(
        model_name  = "bert-base-uncased",
        num_classes = 3,
        dropout     = 0.3,
    ).to(device)

    print(f"\nParametres entrainables : {model.count_parameters():,}")
    print(f"Architecture tete       : {model.classifier}")

    # Test forward avec un batch fictif
    batch_size = 4
    max_length = 128

    dummy_input_ids      = torch.randint(0, 1000, (batch_size, max_length)).to(device)
    dummy_attention_mask = torch.ones(batch_size, max_length, dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(dummy_input_ids, dummy_attention_mask)

    print(f"\nInput  shape : {dummy_input_ids.shape}")
    print(f"Output shape : {logits.shape}")   # attendu : [4, 3]

    assert logits.shape == (batch_size, 3), "Erreur : shape de sortie incorrecte !"
    print("\nTest passe — BertSentimentClassifier operationnel.")

    # Test tokenization réelle
    print("\n=== Test tokenization ===")
    texts = [
        "I love this product!",
        "This is the worst experience ever.",
        "The weather is okay today.",
    ]
    for text in texts:
        enc = tokenizer(text, max_length=128, padding="max_length",
                        truncation=True, return_tensors="pt")
        print(f"  '{text}'")
        print(f"    input_ids shape      : {enc['input_ids'].shape}")
        print(f"    attention_mask shape : {enc['attention_mask'].shape}")
        print(f"    tokens non-padding  : {enc['attention_mask'].sum().item()}")