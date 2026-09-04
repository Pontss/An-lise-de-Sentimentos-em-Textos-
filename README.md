[README.md](https://github.com/user-attachments/files/31834441/README.md)
# Análise de Sentimentos em Textos

Aplicação desktop que classifica o sentimento (positivo ou negativo) de frases em inglês usando modelos de Machine Learning supervisionado, treinados com dados reais.

## Motivação

A maioria dos projetos introdutórios de análise de sentimentos usa datasets sintéticos ou restritos a um único domínio (geralmente resenhas de filmes), o que limita a capacidade de generalização do modelo. Este projeto combina duas fontes de dados reais e de domínios diferentes — críticas de filmes e tweets — para treinar modelos capazes de classificar frases sobre qualquer assunto, não apenas cinema.

## Funcionalidades

- Classificação de sentimento em tempo real com dois modelos (Naive Bayes Multinomial e Regressão Logística) rodando em paralelo, cada um com seu próprio nível de confiança.
- Comparação automática com o VADER, um baseline de mercado baseado em léxico (sem aprendizado).
- Otimização de hiperparâmetros via GridSearchCV (alpha do Naive Bayes e C da Regressão Logística).
- Validação cruzada (k-fold, k=5) para uma estimativa de desempenho mais robusta do que um único split treino/teste.
- Matriz de confusão exportada como imagem para cada modelo.
- Aprendizado contínuo: o usuário pode corrigir uma classificação, e o modelo é retreinado automaticamente com esse novo dado.
- Persistência do modelo treinado em disco, evitando retreinar do zero a cada execução.
- Interpretabilidade: exibição das palavras que mais influenciam a decisão do modelo em cada direção (positiva ou negativa).
- Modo "Duelo dos Modelos": Naive Bayes e Regressão Logística são testados contra um conjunto de frases difíceis (negação, ironia, sentimento misto) com gabarito conhecido, com placar acumulado entre sessões.

## Dataset

O modelo é treinado com 6.000 textos reais, combinando dois domínios:

| Fonte | Quantidade | Características |
|---|---|---|
| `movie_reviews` (NLTK / corpus Pang & Lee, 2004) | 2.000 | Críticas de filmes, texto longo e mais formal |
| `twitter_samples` (NLTK) | 4.000 | Tweets reais, texto curto e informal, com gírias e hashtags |

Os dois corpora são baixados automaticamente pelo NLTK na primeira execução.

## Tecnologias utilizadas

- Python 3
- scikit-learn (TF-IDF, Naive Bayes, Regressão Logística, GridSearchCV, validação cruzada, métricas)
- NLTK (corpora de treino, stopwords, VADER)
- Tkinter (interface gráfica)
- Matplotlib (matriz de confusão e gráficos comparativos)
- joblib (persistência do modelo treinado)

## Estrutura do projeto

```
.
├── app.py                  # Interface gráfica (Tkinter)
├── sentiment_core.py        # Núcleo de Machine Learning (treino, avaliação, persistência)
├── requirements.txt
└── .gitignore
```

Toda a lógica de Machine Learning fica isolada em `sentiment_core.py`; `app.py` cuida apenas de janelas, botões e visualização. Essa separação permite reutilizar o núcleo de ML em outro contexto (por exemplo, uma API ou um script de linha de comando) sem depender da interface gráfica.

## Instalação

Pré-requisito: Python 3.9 ou superior.

```bash
git clone <url-do-repositorio>
cd <pasta-do-repositorio>
pip install -r requirements.txt
```

## Execução

```bash
python app.py
```

Na primeira execução, o programa baixa os corpora do NLTK, otimiza os hiperparâmetros e treina os modelos — esse processo leva cerca de 15 a 20 segundos, com uma barra de progresso indicando cada etapa. Nas execuções seguintes, o modelo treinado é carregado a partir do cache em disco, tornando a inicialização quase instantânea.

## Como o modelo é avaliado

- O conjunto de dados é dividido em 85% para treino e 15% para teste (hold-out), de forma estratificada.
- As métricas reportadas são acurácia, precisão, recall e F1-score.
- A validação cruzada (k-fold) pode ser executada a qualquer momento pela interface, fornecendo média e desvio padrão de cada métrica.
- A matriz de confusão de cada modelo é gerada e exibida como imagem.
- O desempenho dos modelos treinados é comparado com o do VADER, permitindo avaliar se o aprendizado supervisionado está de fato superando uma ferramenta pronta baseada em léxico.

## Limitações conhecidas

- O modelo funciona apenas com frases em inglês.
- A abordagem é bag-of-words (TF-IDF com uni e bigramas); não captura contexto de forma profunda, o que explica falhas em casos de ironia ou sarcasmo mais sutil, evidenciados no modo Duelo dos Modelos.
- O baseline VADER, por ser um léxico ajustado especificamente para textos informais, tende a ter vantagem em frases curtas e coloquiais.

## Possíveis extensões futuras

- Suporte a múltiplos idiomas.
- Uso de embeddings pré-treinados (Word2Vec, GloVe) ou modelos baseados em transformers.
- Classificação de intensidade do sentimento, além da polaridade binária.

## Autor

Desenvolvido como projeto de estudo de Machine Learning e Processamento de Linguagem Natural.
