# Análise de Sentimentos em Textos

Sistema de classificação de sentimentos em avaliações de filmes em inglês, comparando dois algoritmos de aprendizado de máquina: **Naive Bayes Multinomial** e **Regressão Logística** com vetorização via **TF-IDF**.

> Projeto acadêmico — Engenharia da Computação

---

## Sobre o Projeto

O sistema recebe uma frase em inglês sobre um filme e classifica automaticamente se a opinião expressa é positiva ou negativa. Em vez de regras manuais, o modelo aprende padrões a partir de exemplos e melhora continuamente com o feedback do usuário — o que o caracteriza como um sistema de aprendizado de máquina.

---

## Funcionalidades

- Classificação de frases em tempo real via terminal
- Dois modelos rodando simultaneamente com indicador de confiança para cada um
- Métricas de avaliação completas: Acurácia, Precisão, Recall e F1-Score
- Aprendizado contínuo: frases corrigidas pelo usuário são salvas e incorporadas ao próximo treino
- Visualização das palavras com maior peso na decisão da Regressão Logística

---

## Como Funciona

```
Frase digitada
      │
      ▼
 Pré-processamento
 (minúsculas · remoção de pontuação · remoção de stopwords)
      │
      ▼
   TF-IDF
 (vetorização com unigramas e bigramas)
      │
      ▼
 Naive Bayes Multinomial     Regressão Logística
      │                               │
      └───────────────┬───────────────┘
                      ▼
              Veredicto final
```

---

## Estrutura do Projeto

```
├── Analise_sentimentos_textos.py   # Código principal
├── banco_frases.json               # Frases coletadas pelo usuário (gerado automaticamente)
└── README.md
```

---

## Requisitos

- Python 3.10 ou superior

Instale as dependências com:

```bash
pip install scikit-learn nltk numpy
```

Na primeira execução, o NLTK fará o download automático da lista de stopwords em inglês.

---

## Execução

```bash
python Analise_sentimentos_textos.py
```

O sistema treinará os modelos automaticamente ao iniciar e exibirá o menu principal:

```
[1]  Analisar uma frase em tempo real
[2]  Ver desempenho e métricas dos modelos
[3]  Ver frases coletadas no banco de dados
[0]  Encerrar
```

---

## Uso Correto do Feedback

Após cada análise, o sistema pergunta se o sentimento está correto. A resposta deve refletir o **sentimento real da frase**, independentemente do que o modelo respondeu:

| Entrada | Significado |
|---|---|
| `P` | A frase é positiva |
| `N` | A frase é negativa |
| `S` | Pular sem salvar |

> Cada resposta é adicionada ao `banco_frases.json` e usada no retreino imediato. Respostas incorretas introduzem ruído no modelo e reduzem sua acurácia ao longo do tempo.

---

## Dataset

O sistema parte de 1.154 frases geradas diretamente no código, sem dependência de arquivos externos:

| Grupo | Descrição | Quantidade |
|---|---|---|
| Grupo 1 | Frases com palavras de sentimento forte (fantastic, terrible...) | 1.000 |
| Grupo 2 | Frases simples e diretas | 100 |
| Grupo 3 | Frases com negação (not bad at all, not good at all...) | 54 |

As frases salvas pelo usuário no `banco_frases.json` são somadas a esse conjunto base a cada retreino.

---

## Algoritmos

**Naive Bayes Multinomial**  
Classifica textos com base na probabilidade de cada palavra pertencer a uma classe. É rápido, funciona bem com poucos dados e serve como boa linha de base para comparação.

**Regressão Logística**  
Aprende um peso numérico para cada palavra do vocabulário. Palavras associadas a textos positivos recebem pesos positivos; negativas, pesos negativos. Em geral apresenta desempenho ligeiramente superior e permite interpretar quais termos foram mais decisivos na classificação.

---

## Referências

- B. Pang, L. Lee, S. Vaithyanathan — *Thumbs up? Sentiment Classification using Machine Learning Techniques*, EMNLP 2002
- G. Salton, C. Buckley — *Term-Weighting Approaches in Automatic Text Retrieval*, 1988
- A. McCallum, K. Nigam — *A Comparison of Event Models for Naive Bayes Text Classification*, 1998
- F. Pedregosa et al. — *Scikit-learn: Machine Learning in Python*, JMLR 2011
- B. Liu — *Sentiment Analysis and Opinion Mining*, 2012
