# Bert-Classification-Sentiment
Fine-tuning de BERT pour la classification de sentiment sur tweets 


**Binôme :**
- Fatoumata DIOP
- Fatoumata GASSAMA

---

## Dataset choisi

**Twitter Sentiment Dataset**
- Source : Google Drive (fourni par le professeur)
- Tâche : Classifier la colonne `text` suivant la colonne `sentiment`
- Total : 27 481 tweets
- Langue : Anglais
- Classes : 3 (negative, neutral, positive)

### Distribution des classes

| Classe   | Exemples | Pourcentage |
|----------|----------|-------------|
| neutral  | 11 118   | 40.5%       |
| positive | 8 582    | 31.2%       |
| negative | 7 781    | 28.3%       |

Ratio max/min = 1.43 < 2:1 → déséquilibre modéré, aucune stratégie particulière requise.

### Longueur des textes

| Statistique | Valeur |
|-------------|--------|
| Min         | 1 mot  |
| Max         | 33 mots|
| Moyenne     | 12.9 mots |
| Médiane     | 12 mots |

**Justification max_length=128 :** les tweets sont très courts (max 33 mots).
128 tokens couvre 100% des exemples sans troncature.

### 5 exemples du dataset

| Texte | Sentiment |
|-------|-----------|
| I'd have responded, if I were going | neutral |
| Sooo SAD I will miss you here in San Diego!!! | negative |
| my boss is bullying me... | negative |
| what interview! leave me alone | neutral |
| Sons of ****, why couldn't they put them on the releases | negative |

---

## Structure du projet

```
Bert-Classification-Sentiment/
├── data/
│   └── train.csv              ← dataset Twitter Sentiment
├── checkpoints/               ← meilleurs modèles sauvegardés
├── dataset.py                 ← TextClassificationDataset + get_dataloaders
├── model.py                   ← BertSentimentClassifier + load_best_model
├── train.py                   ← train_epoch / eval_epoch / fit + pipeline
├── demo.py                    ← interface Gradio
├── utils.py                   ← seed, device, courbes, matrice confusion
├── requirements.txt
└── README.md
```

---

## Description du modèle et choix techniques

### Modèle : bert-base-uncased

- **Backbone** : BERT pré-entraîné sur Wikipedia + BooksCorpus (110M paramètres)
- **Tokenizer** : BertTokenizer (WordPiece, vocab 30 522 tokens)
- **Tête de classification** : Dropout(0.3) + Linear(768 → 3)
- **Token utilisé** : [CLS] — représentation globale de la séquence
- **max_length** : 128 tokens (justifié par la longueur des tweets)

### Choix techniques

| Paramètre     | Valeur  | Justification |
|---------------|---------|---------------|
| Learning rate | 2e-5    | Typique fine-tuning BERT, évite le catastrophic forgetting |
| Batch size    | 16      | Compatible VRAM Google Colab T4 |
| Epochs        | 5       | BERT converge vite en fine-tuning |
| Optimiseur    | AdamW   | Recommandé pour BERT (weight_decay=0.01) |
| Scheduler     | Linéaire avec warmup (10%) | Stabilise l'entraînement en début |
| Gradient clip | 1.0     | Évite l'explosion des gradients |
| Loss          | CrossEntropyLoss | Classification multi-classes |
| Seed          | 42      | Reproductibilité |
| Split         | 80/20 stratifié | Préserve la distribution des classes |

### Pourquoi bert-base-uncased ?

Les tweets sont en anglais et en minuscules pour la plupart.
`bert-base-uncased` convertit tout en minuscules avant tokenization,
ce qui est adapté au style informel des tweets.

---

## Installation

```bash
# Cloner le repo
git clone https://github.com/user2812/Bert-Classification-Sentiment.git
cd Bert-Classification-Sentiment

# Installer les dépendances
pip install -r requirements.txt

# Placer le dataset
cp train.csv data/train-3.csv
```

---

## Exécution

### 1. Tester chaque module indépendamment

```bash
python dataset.py
python model.py
python utils.py
```

### 2. Lancer l'entraînement

```bash
python train.py
```

Les checkpoints sont sauvegardés dans `checkpoints/` :
- `bert_best.pth` → meilleur modèle (best val_loss)
- `bert_epoch{N}.pth` → checkpoint périodique

### 3. Lancer la démo Gradio

```bash
python demo.py
```

L'interface est accessible sur `http://localhost:7860`
Un lien public :`https://47d40ac7b6057a95b7.gradio.live` 

---

## Résultats

### Métriques finales (validation — 5 496 exemples)

| Classe    | Précision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| negative  | 0.79      | 0.80   | 0.80     | 1 556   |
| neutral   | 0.77      | 0.74   | 0.75     | 2 223   |
| positive  | 0.82      | 0.85   | 0.83     | 1 717   |
| **accuracy**  |       |        | **0.79** | 5 496   |
| macro avg | 0.79      | 0.80   | 0.79     | 5 496   |
| weighted avg | 0.79   | 0.79   | 0.79     | 5 496   |

### Courbes d'apprentissage

*Ref : dossier résultat*

### Matrice de confusion

*Ref : dossier résultat*

### Démo Gradio

*Ref : dossier résultat*

---

## Étapes de réalisation

1. Inspection du dataset (distribution, longueur des textes, exemples)
2. Implémentation de `dataset.py` — tokenization BERT + split 80/20 stratifié
3. Implémentation de `model.py` — BertSentimentClassifier + tête de classification
4. Implémentation de `utils.py` — seed, courbes, matrice de confusion
5. Implémentation de `train.py` — boucle PyTorch manuelle (sans Trainer HuggingFace)
6. Entraînement sur Google Colab (Tesla T4)
7. Implémentation de `demo.py` — interface Gradio interactive
8. Tests et débogage de chaque module indépendamment

---

## Difficultés rencontrées

- **Encodage du dataset** : le CSV utilisait l'encodage latin-1 (pas UTF-8)
  → résolu avec `pd.read_csv(..., encoding='latin-1')`
- **Attention mask** : essentiel pour ignorer les tokens de padding lors
  de l'attention — oubli fréquent qui dégrade les résultats
- **Learning rate** : un LR trop élevé (>5e-5) cause le catastrophic
  forgetting — BERT oublie son pré-entraînement

---

## Répartition du travail

| Fichier         | Responsable      |
|-----------------|------------------|
| `dataset.py`    | Fatoumata GASSAMA|
| `model.py`      | Fatoumata GASSAMA|
| `utils.py`      | Fatoumata GASSAMA|
| `train.py`      | Fatoumata DIOP   |
| `demo.py`       | Fatoumata DIOP   |
| `README.md`     | Les deux membres |

---

## Dépendances

Voir `requirements.txt`