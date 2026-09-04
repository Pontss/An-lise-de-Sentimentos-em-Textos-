# -*- coding: utf-8 -*-
"""
sentiment_core.py
------------------
Núcleo de Machine Learning do projeto de Análise de Sentimentos.

Este módulo concentra toda a parte "IA de verdade": carregamento de dados
reais, pré-processamento, treinamento (com otimização de hiperparâmetros
via GridSearchCV), validação cruzada (k-fold), matriz de confusão,
comparação com um baseline de mercado (VADER) e persistência do modelo
treinado em disco (joblib).

A interface gráfica (app.py) só chama funções/objetos daqui — nenhuma
lógica de ML fica misturada com código de Tkinter.
"""

import os
import re
import json
import random
import warnings

import numpy as np
import joblib
import nltk

warnings.filterwarnings("ignore")

from nltk.corpus import stopwords, movie_reviews, twitter_samples
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
from sklearn.pipeline import Pipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Downloads NLTK (silenciosos, só baixa se ainda não existir) ───────────────
for pacote in ("stopwords", "movie_reviews", "twitter_samples", "vader_lexicon"):
    try:
        nltk.data.find(f"corpora/{pacote}")
    except LookupError:
        nltk.download(pacote, quiet=True)

# ── Caminhos de persistência ───────────────────────────────────────────────────
PASTA_BASE = os.path.dirname(os.path.abspath(__file__))
DB_BANCO = os.path.join(PASTA_BASE, "banco_frases.json")
DB_PLACAR = os.path.join(PASTA_BASE, "placar_duelo.json")
MODELO_PKL = os.path.join(PASTA_BASE, "modelo_treinado.joblib")
PASTA_GRAFICOS = os.path.join(PASTA_BASE, "graficos")
os.makedirs(PASTA_GRAFICOS, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# DATASET REAL E GENERALISTA
# ═══════════════════════════════════════════════════════════════════════════
# Diferente de um dataset sintético (frases geradas por template), aqui
# combinamos DUAS fontes de dados reais e de domínios diferentes:
#
#   1) movie_reviews (NLTK/Pang & Lee) — 2.000 críticas de filmes reais,
#      texto longo e mais formal.
#   2) twitter_samples (NLTK)          — tweets reais, texto curto,
#      informal, com gírias, emojis e hashtags.
#
# Misturar os dois domínios é o que faz o modelo aprender sentimento de
# forma mais genérica, e não "aprender a reconhecer resenha de filme".
def carregar_dataset_real(n_tweets_por_classe=2000, seed=42):
    random.seed(seed)

    # 1) Críticas de filmes (reais, do corpus clássico de Pang & Lee, 2004)
    textos_filmes, labels_filmes = [], []
    for cat, label in (("pos", 1), ("neg", 0)):
        for fid in movie_reviews.fileids(cat):
            textos_filmes.append(movie_reviews.raw(fid))
            labels_filmes.append(label)

    # 2) Tweets reais (domínio genérico: dia a dia, opiniões, humor, etc.)
    tweets_pos = twitter_samples.strings("positive_tweets.json")
    tweets_neg = twitter_samples.strings("negative_tweets.json")
    random.shuffle(tweets_pos)
    random.shuffle(tweets_neg)
    tweets_pos = tweets_pos[:n_tweets_por_classe]
    tweets_neg = tweets_neg[:n_tweets_por_classe]

    textos = textos_filmes + tweets_pos + tweets_neg
    labels = labels_filmes + [1] * len(tweets_pos) + [0] * len(tweets_neg)

    info = {
        "filmes": len(textos_filmes),
        "tweets": len(tweets_pos) + len(tweets_neg),
        "total": len(textos),
    }
    return textos, labels, info


# ═══════════════════════════════════════════════════════════════════════════
# PRÉ-PROCESSAMENTO
# ═══════════════════════════════════════════════════════════════════════════
_base_stop = set(stopwords.words("english"))
# Palavras que carregam carga de sentimento/negação não podem ser removidas,
# mesmo sendo stopwords "clássicas" — senão "not good" vira igual a "good".
_manter = {"not", "no", "nor", "never", "very", "really", "so", "too", "quite",
           "rather", "extremely", "incredibly", "absolutely", "completely",
           "highly", "deeply", "truly", "utterly", "totally", "somewhat",
           "fairly", "pretty", "more", "most", "good", "bad", "great", "poor",
           "nice", "well", "best", "worst", "better", "worse", "n't"}
STOPWORDS = _base_stop - _manter

_re_url = re.compile(r"https?://\S+|www\.\S+")
_re_mencao = re.compile(r"@\w+")
_re_naochar = re.compile(r"[^a-z\s']")


def preprocessar(texto):
    """Limpa e normaliza um texto (funciona tanto para tweets quanto reviews)."""
    t = texto.lower()
    t = _re_url.sub(" ", t)
    t = _re_mencao.sub(" ", t)
    t = _re_naochar.sub(" ", t)
    palavras = [w for w in t.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(palavras)


# ═══════════════════════════════════════════════════════════════════════════
# MODELOS
# ═══════════════════════════════════════════════════════════════════════════
class Modelos:
    """
    Encapsula todo o pipeline de ML: vetorização TF-IDF + dois classificadores
    (Naive Bayes Multinomial e Regressão Logística), com suporte a:
      - treino/teste com métricas completas + matriz de confusão
      - otimização de hiperparâmetros via GridSearchCV
      - validação cruzada (k-fold) para uma estimativa mais robusta
      - comparação com o baseline VADER (léxico, sem aprendizado)
      - persistência em disco (joblib)
    """

    def __init__(self):
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=8000,
                                      sublinear_tf=True, min_df=2)
        self.nb = MultinomialNB(alpha=1.0)
        self.lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.melhores_params = {"nb_alpha": 1.0, "lr_C": 1.0}
        self.metricas = {}
        self.matrizes_confusao = {}
        self.cv_resultados = {}
        self.baseline_vader = None
        self.n = 0
        self._vader = SentimentIntensityAnalyzer()

    # ── Treino principal (rápido, usado após cada novo feedback) ──────────
    def treinar(self, textos, labels, progresso=None):
        self.n = len(textos)
        textos_limpos = [preprocessar(t) for t in textos]

        Xtr, Xte, ytr, yte = train_test_split(
            textos_limpos, labels, test_size=0.15, random_state=42, stratify=labels
        )
        if progresso: progresso("Vetorizando textos (TF-IDF)...", 0.2)
        Xtr_v = self.tfidf.fit_transform(Xtr)
        Xte_v = self.tfidf.transform(Xte)

        if progresso: progresso("Treinando Naive Bayes...", 0.5)
        self.nb = MultinomialNB(alpha=self.melhores_params["nb_alpha"])
        self.nb.fit(Xtr_v, ytr)

        if progresso: progresso("Treinando Regressão Logística...", 0.75)
        self.lr = LogisticRegression(C=self.melhores_params["lr_C"],
                                      max_iter=1000, random_state=42)
        self.lr.fit(Xtr_v, ytr)

        if progresso: progresso("Calculando métricas...", 0.9)
        self.reavaliar(textos, labels)
        if progresso: progresso("Concluído!", 1.0)

    def reavaliar(self, textos, labels):
        """
        Recalcula métricas de teste, matriz de confusão e guarda o conjunto
        de teste "cru" (para o baseline VADER) usando os modelos JÁ treinados
        (self.tfidf/self.nb/self.lr). Não refaz o fit — serve tanto logo
        após treinar() quanto logo após carregar() um modelo do cache, que
        não vem com essas informações derivadas salvas no arquivo .joblib.
        """
        textos_limpos = [preprocessar(t) for t in textos]
        Xtr, Xte, ytr, yte = train_test_split(
            textos_limpos, labels, test_size=0.15, random_state=42, stratify=labels
        )
        Xte_v = self.tfidf.transform(Xte)
        self._avaliar(Xte_v, yte)
        _, Xte_raw, _, _ = train_test_split(
            textos, labels, test_size=0.15, random_state=42, stratify=labels
        )
        self._teste_raw = Xte_raw
        self._teste_labels = yte

    def _avaliar(self, Xte_v, yte):
        for nome, modelo in (("nb", self.nb), ("lr", self.lr)):
            yp = modelo.predict(Xte_v)
            self.metricas[nome] = {
                "acc": accuracy_score(yte, yp),
                "pre": precision_score(yte, yp),
                "rec": recall_score(yte, yp),
                "f1": f1_score(yte, yp),
            }
            self.matrizes_confusao[nome] = confusion_matrix(yte, yp)

    # ── Otimização de hiperparâmetros (GridSearchCV) ───────────────────────
    def otimizar_hiperparametros(self, textos, labels, progresso=None):
        """
        Busca os melhores hiperparâmetros (alpha do NB e C da Regressão
        Logística) via validação cruzada de 3 folds. Roda uma vez só
        (é mais custoso); o resultado fica salvo para os próximos treinos.
        """
        textos_limpos = [preprocessar(t) for t in textos]
        Xtr, _, ytr, _ = train_test_split(
            textos_limpos, labels, test_size=0.15, random_state=42, stratify=labels
        )
        X = self.tfidf.fit_transform(Xtr)

        if progresso: progresso("Buscando melhor alpha (Naive Bayes)...", 0.2)
        grid_nb = GridSearchCV(MultinomialNB(), {"alpha": [0.1, 0.3, 0.5, 1.0, 2.0]},
                                cv=3, scoring="f1", n_jobs=-1)
        grid_nb.fit(X, ytr)

        if progresso: progresso("Buscando melhor C (Regressão Logística)...", 0.6)
        grid_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                                {"C": [0.1, 0.5, 1.0, 2.0, 5.0]},
                                cv=3, scoring="f1", n_jobs=-1)
        grid_lr.fit(X, ytr)

        self.melhores_params = {
            "nb_alpha": grid_nb.best_params_["alpha"],
            "lr_C": grid_lr.best_params_["C"],
        }
        if progresso: progresso("Hiperparâmetros otimizados!", 1.0)
        return self.melhores_params

    # ── Validação cruzada (k-fold) ──────────────────────────────────────────
    def validacao_cruzada(self, textos, labels, cv=5, progresso=None):
        textos_limpos = [preprocessar(t) for t in textos]
        scoring = ["accuracy", "precision", "recall", "f1"]
        resultado = {}

        pipes = {
            "nb": Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=8000,
                                           sublinear_tf=True, min_df=2)),
                ("clf", MultinomialNB(alpha=self.melhores_params["nb_alpha"])),
            ]),
            "lr": Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=8000,
                                           sublinear_tf=True, min_df=2)),
                ("clf", LogisticRegression(C=self.melhores_params["lr_C"],
                                            max_iter=1000, random_state=42)),
            ]),
        }
        for i, (nome, pipe) in enumerate(pipes.items()):
            if progresso:
                progresso(f"Validação cruzada ({cv}-fold) — "
                          f"{'Naive Bayes' if nome=='nb' else 'Reg. Logística'}...",
                          0.2 + i * 0.4)
            r = cross_validate(pipe, textos_limpos, labels, cv=cv, scoring=scoring,
                                n_jobs=-1)
            resultado[nome] = {
                m: (r[f"test_{m}"].mean(), r[f"test_{m}"].std()) for m in scoring
            }
        self.cv_resultados = resultado
        if progresso: progresso("Validação cruzada concluída!", 1.0)
        return resultado

    # ── Baseline: VADER (léxico pronto de mercado, sem aprendizado) ───────
    def comparar_baseline_vader(self):
        if not hasattr(self, "_teste_raw"):
            raise RuntimeError("Treine o modelo antes de comparar com o baseline.")
        preds = []
        for texto in self._teste_raw:
            score = self._vader.polarity_scores(texto)["compound"]
            preds.append(1 if score >= 0 else 0)
        yte = self._teste_labels
        self.baseline_vader = {
            "acc": accuracy_score(yte, preds),
            "pre": precision_score(yte, preds),
            "rec": recall_score(yte, preds),
            "f1": f1_score(yte, preds),
        }
        return self.baseline_vader

    # ── Predição de uma frase nova ──────────────────────────────────────────
    def prever(self, frase):
        limpa = preprocessar(frase)
        v = self.tfidf.transform([limpa])
        np_ = self.nb.predict_proba(v)[0]
        lp_ = self.lr.predict_proba(v)[0]
        vader_score = self._vader.polarity_scores(frase)["compound"]
        return {
            "nb": (int(np.argmax(np_)), float(max(np_))),
            "lr": (int(np.argmax(lp_)), float(max(lp_))),
            "vader": (1 if vader_score >= 0 else 0, vader_score),
            "limpa": limpa,
        }

    # ── Palavras mais influentes (interpretabilidade) ──────────────────────
    def top_palavras(self, n=8):
        c = self.lr.coef_[0]
        nomes = self.tfidf.get_feature_names_out()
        pos = [(nomes[i], c[i]) for i in np.argsort(c)[::-1][:n]]
        neg = [(nomes[i], c[i]) for i in np.argsort(c)[:n]]
        return pos, neg

    # ── Exportar matriz de confusão como imagem ─────────────────────────────
    def exportar_matriz_confusao(self, nome_modelo):
        if nome_modelo not in self.matrizes_confusao:
            return None
        cm = self.matrizes_confusao[nome_modelo]
        titulo = "Naive Bayes" if nome_modelo == "nb" else "Regressão Logística"

        fig, ax = plt.subplots(figsize=(3.7, 3.2), dpi=105)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Matriz de Confusão — {titulo}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Negativo", "Positivo"])
        ax.set_yticklabels(["Negativo", "Positivo"])
        for i in range(2):
            for j in range(2):
                cor = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color=cor, fontsize=13, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        caminho = os.path.join(PASTA_GRAFICOS, f"matriz_confusao_{nome_modelo}.png")
        fig.savefig(caminho); plt.close(fig)
        return caminho

    def exportar_grafico_comparativo(self):
        """Gráfico de barras comparando NB, LR e baseline VADER (se disponível)."""
        metricas = ["acc", "pre", "rec", "f1"]
        labels_pt = ["Acurácia", "Precisão", "Recall", "F1-Score"]
        nb_vals = [self.metricas["nb"][m] for m in metricas]
        lr_vals = [self.metricas["lr"][m] for m in metricas]

        x = np.arange(len(metricas))
        largura = 0.25 if self.baseline_vader else 0.35

        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        ax.bar(x - largura, nb_vals, largura, label="Naive Bayes", color="#4C72B0")
        ax.bar(x, lr_vals, largura, label="Reg. Logística", color="#DD8452")
        if self.baseline_vader:
            vader_vals = [self.baseline_vader[m] for m in metricas]
            ax.bar(x + largura, vader_vals, largura, label="VADER (baseline)",
                   color="#999999")
        ax.set_xticks(x); ax.set_xticklabels(labels_pt)
        ax.set_ylim(0, 1)
        ax.set_title("Comparação de Desempenho", fontsize=12, fontweight="bold")
        ax.legend()
        fig.tight_layout()
        caminho = os.path.join(PASTA_GRAFICOS, "comparativo_modelos.png")
        fig.savefig(caminho); plt.close(fig)
        return caminho

    # ── Persistência ────────────────────────────────────────────────────────
    def salvar(self, caminho=MODELO_PKL):
        joblib.dump({
            "tfidf": self.tfidf, "nb": self.nb, "lr": self.lr,
            "melhores_params": self.melhores_params,
            "metricas": self.metricas, "n": self.n,
        }, caminho)

    def carregar(self, caminho=MODELO_PKL):
        if not os.path.exists(caminho):
            return False
        dados = joblib.load(caminho)
        self.tfidf = dados["tfidf"]; self.nb = dados["nb"]; self.lr = dados["lr"]
        self.melhores_params = dados["melhores_params"]
        self.metricas = dados["metricas"]; self.n = dados["n"]
        return True


# ═══════════════════════════════════════════════════════════════════════════
# BANCO DE FRASES COLETADAS (feedback do usuário)
# ═══════════════════════════════════════════════════════════════════════════
def carregar_banco():
    if os.path.exists(DB_BANCO):
        with open(DB_BANCO, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"frases": [], "labels": []}


def salvar_banco(banco):
    with open(DB_BANCO, "w", encoding="utf-8") as f:
        json.dump(banco, f, ensure_ascii=False, indent=2)


def adicionar_ao_banco(banco, frase, label):
    banco["frases"].append(frase)
    banco["labels"].append(label)
    salvar_banco(banco)
    return banco


# ═══════════════════════════════════════════════════════════════════════════
# PLACAR DO DUELO DOS MODELOS
# ═══════════════════════════════════════════════════════════════════════════
def carregar_placar():
    if os.path.exists(DB_PLACAR):
        with open(DB_PLACAR, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"nb": 0, "lr": 0, "empates": 0, "duelos_jogados": 0}


def salvar_placar(placar):
    with open(DB_PLACAR, "w", encoding="utf-8") as f:
        json.dump(placar, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# FRASES DO DUELO (casos difíceis: negação, ironia, sentimento misto)
# ═══════════════════════════════════════════════════════════════════════════
def frases_duelo():
    return [
        ("Not the worst movie I have ever seen", 1),
        ("I wouldn't call this bad, but it's not good either", 0),
        ("The acting was great, but the plot ruined everything", 0),
        ("I expected to hate this, but I didn't", 1),
        ("This is so bad it's actually kind of fun", 1),
        ("A masterpiece, said no one who actually watched it", 0),
        ("The effects were stunning, everything else was a mess", 0),
        ("I can't say I loved it, but I didn't hate it either", 1),
        ("This had potential and wasted every bit of it", 0),
        ("Surprisingly, this turned out to be great", 1),
        ("Not exactly a masterpiece, but far from terrible", 1),
        ("The trailer promised something great, it did not deliver", 0),
        ("It's not perfect, yet it's still quite enjoyable", 1),
        ("I laughed, but only because it was unintentionally bad", 0),
        ("This is the kind of thing you forget about instantly", 0),
        ("Against all odds, this dull idea became something great", 1),
        ("It wasn't boring, which says a lot for this genre", 1),
        ("Great start, shame about literally everything after", 0),
        ("I didn't expect much and somehow got even less", 0),
        ("Not amazing, but definitely worth it", 1),
        ("I'm not even mad, just disappointed", 0),
        ("Can't believe how good this turned out to be", 1),
    ]
