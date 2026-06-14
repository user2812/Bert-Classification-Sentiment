"""
Interface de démonstration interactive avec Gradio pour le modèle
BERT de classification de sentiment sur tweets.

Usage :
    python demo.py

L'interface permet :
    - De saisir un texte libre
    - D'afficher la classe prédite (negative / neutral / positive)
    - D'afficher les probabilités de chaque classe
"""

import torch
import torch.nn.functional as F
import gradio as gr
from transformers import BertTokenizer

from model import BertSentimentClassifier, load_best_model
from utils import get_device


# 1. CONFIGURATION

MODEL_NAME      = "bert-base-uncased"
CHECKPOINT_PATH = "checkpoints/bert_best.pth"
MAX_LENGTH      = 128
NUM_CLASSES     = 3

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# Emojis pour rendre la démo plus visuelle
LABEL_EMOJI = {
    "negative": "Negative",
    "neutral" : "Neutral",
    "positive": "Positive",
}


# 2. CHARGEMENT DU MODELE ET DU TOKENIZER

def load_model_and_tokenizer():
    """
    Charge le meilleur modèle BERT sauvegardé et le tokenizer.

    Returns:
        model     : BertSentimentClassifier en mode eval
        tokenizer : BertTokenizer
        device    : torch.device
    """
    device    = get_device()
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model     = load_best_model(
        checkpoint_path = CHECKPOINT_PATH,
        model_name      = MODEL_NAME,
        num_classes     = NUM_CLASSES,
        device          = device,
    )
    model.eval()
    return model, tokenizer, device


# Chargement une seule fois au démarrage
model, tokenizer, device = load_model_and_tokenizer()


# 3. FONCTION DE PREDICTION

def predict(text: str) -> dict:
    """
    Effectue la prédiction de sentiment sur un texte saisi.

    Args:
        text : tweet ou texte saisi par l'utilisateur

    Returns:
        dict : probabilités par classe pour Gradio Label component
    """
    if not text or not text.strip():
        return {label: 0.0 for label in LABEL_EMOJI.values()}

    # Tokenization
    encoding = tokenizer(
        text.strip(),
        max_length     = MAX_LENGTH,
        padding        = "max_length",
        truncation     = True,
        return_tensors = "pt",
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Inférence
    with torch.no_grad():
        logits = model(input_ids, attention_mask)   # [1, 3]

    # Probabilités via softmax
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # Format attendu par gr.Label : {label: probabilité}
    results = {
        LABEL_EMOJI[ID2LABEL[i]]: float(probs[i])
        for i in range(NUM_CLASSES)
    }

    return results


# 4. EXEMPLES PRE-REMPLIS

EXAMPLES = [
    ["I love this product, it works perfectly and made my day!"],
    ["This is the worst experience I have ever had, totally disappointed."],
    ["The weather is okay today, nothing special."],
    ["Sooo SAD I will miss you here in San Diego!!!"],
    ["Just finished my work, feeling neutral about it."],
]


# 5. INTERFACE GRADIO

def build_interface():
    """
    Construit et retourne l'interface Gradio de démonstration.

    Returns:
        gr.Interface : interface prête à être lancée
    """
    interface = gr.Interface(
        fn          = predict,
        inputs      = gr.Textbox(
            lines       = 3,
            placeholder = "Saisissez un tweet ou un texte ici...",
            label       = "Texte à analyser",
        ),
        outputs     = gr.Label(
            num_top_classes = 3,
            label           = "Sentiment prédit",
        ),
        title       = "Analyse de Sentiment avec BERT",
        description = (
            "Ce modèle fine-tune BERT (bert-base-uncased) pour classifier "
            "le sentiment d'un tweet en 3 catégories : "
            "Negative, Neutral ou Positive.\n\n"
            "Dataset : Twitter Sentiment (27 481 tweets)\n"
            "Modèle  : bert-base-uncased fine-tune avec PyTorch"
        ),
        examples    = EXAMPLES,
        theme       = gr.themes.Soft(),
        allow_flagging = "never",
    )
    return interface


# 6. Lancement python démo.py

if __name__ == "__main__":
    print("Lancement de la démo Gradio...")
    print(f"Modele    : {MODEL_NAME}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Classes   : {list(ID2LABEL.values())}")

    interface = build_interface()
    interface.launch(
        share       = True,    # genere un lien public partageble
        server_name = "0.0.0.0",
        server_port = 7860,
    )