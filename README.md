# 🎬 Análise de Sentimentos em Textos

Sistema de **Inteligência Artificial** para classificação de sentimentos em avaliações de filmes em inglês, comparando dois algoritmos clássicos de aprendizado de máquina: **Naive Bayes Multinomial** e **Regressão Logística**.

> Projeto acadêmico desenvolvido para a disciplina de Engenharia da Computação — FHO  

---

## 📋 Sobre o Projeto

O sistema recebe uma frase em inglês sobre um filme e classifica automaticamente se a opinião é **positiva** ou **negativa**. Ele não usa regras manuais — aprende sozinho a partir de exemplos e melhora continuamente com o feedback do usuário.

---

## ✨ Funcionalidades

- 🔍 **Análise em tempo real** — classifica qualquer frase digitada pelo usuário
- 🤖 **Dois modelos simultâneos** — Naive Bayes e Regressão Logística respondem ao mesmo tempo
- 📊 **Indicador de confiança** — mostra a porcentagem de certeza de cada modelo
- 📈 **Métricas completas** — Acurácia, Precisão, Recall e F1-Score lado a lado
- 💾 **Aprendizado contínuo** — frases corrigidas pelo usuário são salvas e usadas no retreino
- 🔤 **Palavras mais influentes** — exibe as palavras que mais pesam na decisão da Regressão Logística

---

## 🧠 Como funciona

```
Frase digitada
      │
      ▼
 Pré-processamento
 (minúsculas, sem pontuação, sem stopwords)
      │
      ▼
   TF-IDF
 (converte texto em vetores numéricos com bigramas)
      │
      ▼
 ┌────────────┐     ┌──────────────────────┐
 │ Naive Bayes│     │ Regressão Logística   │
 └────────────┘     └──────────────────────┘
      │                       │
      └──────────┬────────────┘
                 ▼
         Veredicto final
         (POSITIVO ou NEGATIVO)
```

---

## 🗂️ Estrutura do Projeto

```
📦 repositório
 ┣ 📄 Analise_sentimentos_textos.py   # Código principal
 ┣ 📄 banco_frases.json               # Frases coletadas pelo usuário (gerado automaticamente)
 ┗ 📄 README.md
```

---

## ⚙️ Requisitos

- Python 3.10 ou superior
- Bibliotecas:

```bash
pip install scikit-learn nltk numpy
```

Na primeira execução, o NLTK baixará automaticamente a lista de stopwords em inglês.

---

## ▶️ Como executar

```bash
python Analise_sentimentos_textos.py
```

Ao iniciar, o sistema treina os modelos automaticamente e exibe o menu principal:

```
[1]  Analisar uma frase em tempo real
[2]  Ver desempenho e métricas dos modelos
[3]  Ver frases coletadas no banco de dados
[0]  Encerrar
```

---

## 💡 Como usar corretamente

Ao analisar uma frase, o sistema pergunta se o sentimento está correto. **Você deve responder com base no sentimento real da frase**, não na resposta da IA:

| Situação | O que pressionar |
|---|---|
| A frase é positiva (a IA acertou ou errou) | `[P]` |
| A frase é negativa (a IA acertou ou errou) | `[N]` |
| Não tem certeza / quer pular | `[S]` |

> ⚠️ **Importante:** `[P]` e `[N]` indicam o sentimento correto da frase — não se a IA acertou.  
> Cada resposta é salva no `banco_frases.json` e usada no próximo retreino, então respostas erradas confundem o modelo.

---

## 📊 Dataset

O sistema usa 1.154 frases criadas diretamente no código, sem necessidade de download externo:

| Grupo | Descrição | Quantidade |
|---|---|---|
| Grupo 1 | Frases com palavras de sentimento forte (fantastic, terrible...) | 1.000 |
| Grupo 2 | Frases simples e diretas (good, bad) | 100 |
| Grupo 3 | Frases com negação (not bad, not good) | 54 |

As frases do usuário salvas no `banco_frases.json` são somadas a esse dataset a cada retreino.

---

## 🔬 Algoritmos

### Naive Bayes Multinomial
Calcula a probabilidade de cada palavra aparecer em textos positivos ou negativos. Muito rápido e eficiente, mesmo com poucos dados.

### Regressão Logística
Aprende um peso numérico para cada palavra. Palavras positivas recebem peso positivo; negativas, peso negativo. Geralmente mais precisa e permite interpretar quais palavras foram decisivas.

---

## 📚 Referências

- Pang, Lee e Vaithyanathan (2002) — *Thumbs up? Sentiment Classification using Machine Learning Techniques*
- Salton e Buckley (1988) — *Term-Weighting Approaches in Automatic Text Retrieval*
- McCallum e Nigam (1998) — *A Comparison of Event Models for Naive Bayes Text Classification*
- Pedregosa et al. (2011) — *Scikit-learn: Machine Learning in Python*
- Liu (2012) — *Sentiment Analysis and Opinion Mining*
