# -*- coding: utf-8 -*-
"""
app.py
------
Interface gráfica (Tkinter) do projeto de Análise de Sentimentos.

Toda a lógica de Machine Learning vive em sentiment_core.py — este arquivo
cuida apenas de janelas, botões e visualização.
"""

import threading
import queue
import random
import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image, ImageTk

import sentiment_core as core

# ── Paleta de cores ─────────────────────────────────────────────────────────
COR_FUNDO = "#1e2028"
COR_PAINEL = "#262933"
COR_PAINEL_2 = "#2d313d"
COR_TEXTO = "#e6e6e6"
COR_TEXTO_FRACO = "#9aa0ac"
COR_AZUL = "#5b8def"
COR_LARANJA = "#e8935c"
COR_VERDE = "#4caf7d"
COR_VERMELHO = "#e5636b"
COR_CINZA = "#565b68"
FONTE = "Segoe UI"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Análise de Sentimentos — IA (NLP)")
        self.geometry("1040x740")
        self.minsize(940, 640)
        self.configure(bg=COR_FUNDO)

        self._fila = queue.Queue()
        self.modelo = core.Modelos()
        self.banco = core.carregar_banco()
        self.placar = core.carregar_placar()
        self.dataset_info = {}
        self._textos_treino = []
        self._labels_treino = []

        self._montar_estilo()
        self._montar_tela_carregamento()
        self.after(200, self._iniciar_carregamento_inicial)

    # ── Estilo visual (ttk) ─────────────────────────────────────────────────
    def _montar_estilo(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".", background=COR_FUNDO, foreground=COR_TEXTO,
                         font=(FONTE, 10))
        style.configure("TFrame", background=COR_FUNDO)
        style.configure("Painel.TFrame", background=COR_PAINEL)
        style.configure("Painel2.TFrame", background=COR_PAINEL_2)

        style.configure("TLabel", background=COR_FUNDO, foreground=COR_TEXTO,
                         font=(FONTE, 10))
        style.configure("Painel.TLabel", background=COR_PAINEL, foreground=COR_TEXTO)
        style.configure("Painel2.TLabel", background=COR_PAINEL_2, foreground=COR_TEXTO)
        style.configure("Titulo.TLabel", background=COR_FUNDO, foreground=COR_TEXTO,
                         font=(FONTE, 16, "bold"))
        style.configure("Subtitulo.TLabel", background=COR_FUNDO,
                         foreground=COR_TEXTO_FRACO, font=(FONTE, 10))
        style.configure("Fraco.TLabel", background=COR_PAINEL,
                         foreground=COR_TEXTO_FRACO, font=(FONTE, 9))

        style.configure("TNotebook", background=COR_FUNDO, borderwidth=0)
        style.configure("TNotebook.Tab", background=COR_PAINEL, foreground=COR_TEXTO_FRACO,
                         padding=(16, 10), font=(FONTE, 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", COR_AZUL)],
                  foreground=[("selected", "#ffffff")])

        style.configure("TButton", background=COR_AZUL, foreground="#ffffff",
                         font=(FONTE, 10, "bold"), padding=(14, 8), borderwidth=0)
        style.map("TButton", background=[("active", "#4a7bd6"), ("disabled", COR_CINZA)])

        style.configure("Secundario.TButton", background=COR_PAINEL_2,
                         foreground=COR_TEXTO, font=(FONTE, 10), padding=(14, 8))
        style.map("Secundario.TButton", background=[("active", "#3a3f4d")])

        style.configure("Positivo.TButton", background=COR_VERDE, foreground="#ffffff",
                         font=(FONTE, 10, "bold"), padding=(14, 8))
        style.map("Positivo.TButton", background=[("active", "#3d9468")])

        style.configure("Negativo.TButton", background=COR_VERMELHO, foreground="#ffffff",
                         font=(FONTE, 10, "bold"), padding=(14, 8))
        style.map("Negativo.TButton", background=[("active", "#c8535a")])

        style.configure("TProgressbar", background=COR_AZUL, troughcolor=COR_PAINEL_2,
                         borderwidth=0, thickness=10)

        style.configure("Treeview", background=COR_PAINEL_2, fieldbackground=COR_PAINEL_2,
                         foreground=COR_TEXTO, rowheight=26, borderwidth=0,
                         font=(FONTE, 9))
        style.configure("Treeview.Heading", background=COR_PAINEL,
                         foreground=COR_TEXTO, font=(FONTE, 9, "bold"))
        style.map("Treeview", background=[("selected", COR_AZUL)])

    # ── Tela de carregamento inicial ────────────────────────────────────────
    def _montar_tela_carregamento(self):
        self.tela_load = ttk.Frame(self, style="TFrame")
        self.tela_load.place(relx=0.5, rely=0.5, anchor="center")

        ttk.Label(self.tela_load, text="ANÁLISE DE SENTIMENTOS",
                  style="Titulo.TLabel").pack(pady=(0, 4))
        ttk.Label(self.tela_load,
                  text="Naive Bayes  ·  Regressão Logística  ·  TF-IDF  ·  NLTK",
                  style="Subtitulo.TLabel").pack(pady=(0, 30))

        self.status_load = ttk.Label(self.tela_load, text="Iniciando...",
                                      style="Subtitulo.TLabel")
        self.status_load.pack(pady=(0, 10))

        self.barra_load = ttk.Progressbar(self.tela_load, style="TProgressbar",
                                           length=420, mode="determinate", maximum=1.0)
        self.barra_load.pack()

    def _iniciar_carregamento_inicial(self):
        t = threading.Thread(target=self._worker_carregamento_inicial, daemon=True)
        t.start()
        self.after(100, self._checar_fila)

    def _worker_carregamento_inicial(self):
        def prog(msg, p):
            self._fila.put(("status", msg, p))

        prog("Carregando dataset real (críticas de filmes + tweets)...", 0.05)
        textos, labels, info = core.carregar_dataset_real(n_tweets_por_classe=2000)
        self.dataset_info = info
        self._textos_treino = textos + self.banco["frases"]
        self._labels_treino = labels + self.banco["labels"]

        carregou = self.modelo.carregar()
        if carregou:
            prog("Modelo salvo encontrado, carregando...", 0.9)
            self.modelo.reavaliar(self._textos_treino, self._labels_treino)
        else:
            self.modelo.otimizar_hiperparametros(
                self._textos_treino, self._labels_treino,
                progresso=lambda m, p: prog(m, 0.1 + p * 0.3))
            self.modelo.treinar(
                self._textos_treino, self._labels_treino,
                progresso=lambda m, p: prog(m, 0.4 + p * 0.5))
            self.modelo.salvar()

        self._fila.put(("pronto", None, None))

    def _checar_fila(self):
        try:
            while True:
                tipo, msg, p = self._fila.get_nowait()
                if tipo == "status":
                    self.status_load.config(text=msg)
                    self.barra_load["value"] = p
                elif tipo == "pronto":
                    self._finalizar_carregamento()
                    return
        except queue.Empty:
            pass
        self.after(100, self._checar_fila)

    def _finalizar_carregamento(self):
        self.tela_load.destroy()
        self._montar_interface_principal()

    # ── Interface principal ─────────────────────────────────────────────────
    def _montar_interface_principal(self):
        header = ttk.Frame(self, style="TFrame")
        header.pack(fill="x", padx=24, pady=(20, 8))
        ttk.Label(header, text="Análise de Sentimentos em Textos",
                  style="Titulo.TLabel").pack(anchor="w")
        info = self.dataset_info
        ttk.Label(header,
                  text=(f"Treinado com {info.get('total', self.modelo.n)} textos reais  ·  "
                        f"{info.get('filmes', 0)} críticas de filmes + "
                        f"{info.get('tweets', 0)} tweets  ·  "
                        f"{len(self.banco['frases'])} frases coletadas por você"),
                  style="Subtitulo.TLabel").pack(anchor="w", pady=(2, 0))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=(10, 20))

        self.aba_analise = ttk.Frame(self.notebook, style="TFrame")
        self.aba_desempenho = ttk.Frame(self.notebook, style="TFrame")
        self.aba_banco = ttk.Frame(self.notebook, style="TFrame")
        self.aba_duelo = ttk.Frame(self.notebook, style="TFrame")

        self.notebook.add(self.aba_analise, text="  Analisar Frase  ")
        self.notebook.add(self.aba_desempenho, text="  Desempenho & IA  ")
        self.notebook.add(self.aba_banco, text="  Banco de Dados  ")
        self.notebook.add(self.aba_duelo, text="  ⚔ Duelo dos Modelos  ")

        self._montar_aba_analise()
        self._montar_aba_desempenho()
        self._montar_aba_banco()
        self._montar_aba_duelo()

    # ── ABA 1: Analisar Frase ───────────────────────────────────────────────
    def _montar_aba_analise(self):
        aba = self.aba_analise
        topo = ttk.Frame(aba, style="TFrame")
        topo.pack(fill="x", padx=4, pady=(16, 8))

        ttk.Label(topo, text="Digite uma frase em inglês (qualquer assunto, não só filmes):",
                  style="TLabel").pack(anchor="w", pady=(0, 8))

        linha_input = ttk.Frame(topo, style="TFrame")
        linha_input.pack(fill="x")
        self.entrada_frase = tk.Entry(linha_input, font=(FONTE, 12), bg=COR_PAINEL_2,
                                       fg=COR_TEXTO, insertbackground=COR_TEXTO,
                                       relief="flat")
        self.entrada_frase.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 10))
        self.entrada_frase.bind("<Return>", lambda e: self._analisar_frase())
        ttk.Button(linha_input, text="Analisar", command=self._analisar_frase
                    ).pack(side="left")

        exemplos = ttk.Frame(topo, style="TFrame")
        exemplos.pack(fill="x", pady=(8, 0))
        ttk.Label(exemplos, text="Exemplos:", style="Fraco.TLabel",
                  background=COR_FUNDO).pack(side="left", padx=(0, 6))
        for ex in ["Not bad for a Monday morning",
                   "I absolutely love this",
                   "This is the worst service I've ever had"]:
            b = tk.Label(exemplos, text=ex, fg=COR_AZUL, bg=COR_FUNDO,
                         font=(FONTE, 9, "underline"), cursor="hand2")
            b.pack(side="left", padx=6)
            b.bind("<Button-1>", lambda e, t=ex: self._preencher_exemplo(t))

        # Painel de resultado (dentro de um canvas rolável — o conteúdo pode
        # crescer mais do que a altura da janela)
        canvas = tk.Canvas(aba, bg=COR_FUNDO, highlightthickness=0)
        scroll = ttk.Scrollbar(aba, orient="vertical", command=canvas.yview)
        self.painel_resultado = ttk.Frame(canvas, style="Painel.TFrame")
        self._id_janela_resultado = canvas.create_window(
            (0, 0), window=self.painel_resultado, anchor="nw")
        self.painel_resultado.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(self._id_janela_resultado, width=e.width))
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=16)
        scroll.pack(side="right", fill="y", pady=16)

        def _scroll_mouse(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _scroll_linux(event):
            canvas.yview_scroll(-1 if event.num == 4 else 1, "units")
        def _ligar_scroll(e):
            canvas.bind_all("<MouseWheel>", _scroll_mouse)
            canvas.bind_all("<Button-4>", _scroll_linux)
            canvas.bind_all("<Button-5>", _scroll_linux)
        def _desligar_scroll(e):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")
        canvas.bind("<Enter>", _ligar_scroll)
        canvas.bind("<Leave>", _desligar_scroll)

        self._resultado_vazio()

    def _preencher_exemplo(self, texto):
        self.entrada_frase.delete(0, "end")
        self.entrada_frase.insert(0, texto)
        self._analisar_frase()

    def _resultado_vazio(self):
        for w in self.painel_resultado.winfo_children():
            w.destroy()
        ttk.Label(self.painel_resultado,
                  text="Digite uma frase acima e clique em Analisar para ver o "
                       "que cada modelo de IA entende dela.",
                  style="Painel.TLabel", foreground=COR_TEXTO_FRACO
                  ).pack(expand=True, pady=60)

    def _analisar_frase(self):
        frase = self.entrada_frase.get().strip()
        if not frase:
            return
        if not core.preprocessar(frase):
            messagebox.showwarning("Frase vazia",
                                    "Não sobrou nenhuma palavra útil após o "
                                    "pré-processamento. Tente outra frase.")
            return

        resultado = self.modelo.prever(frase)
        self._mostrar_resultado(frase, resultado)

    def _mostrar_resultado(self, frase, resultado):
        for w in self.painel_resultado.winfo_children():
            w.destroy()

        wrap = ttk.Frame(self.painel_resultado, style="Painel.TFrame")
        wrap.pack(fill="both", expand=True, padx=24, pady=20)

        ttk.Label(wrap, text="Frase analisada", style="Fraco.TLabel").pack(anchor="w")
        ttk.Label(wrap, text=f'"{frase}"', style="Painel.TLabel",
                  font=(FONTE, 12, "italic"), wraplength=860, justify="left"
                  ).pack(anchor="w", pady=(2, 4))
        ttk.Label(wrap, text=f"tokens usados: [{resultado['limpa']}]",
                  style="Fraco.TLabel", wraplength=860).pack(anchor="w", pady=(0, 16))

        cards = ttk.Frame(wrap, style="Painel.TFrame")
        cards.pack(fill="x", pady=(0, 16))

        self._card_modelo(cards, "Naive Bayes Multinomial", resultado["nb"], 0)
        self._card_modelo(cards, "Regressão Logística", resultado["lr"], 1)
        self._card_vader(cards, resultado["vader"], 2)

        nb_pred = resultado["nb"][0]; lr_pred = resultado["lr"][0]
        linha_final = ttk.Frame(wrap, style="Painel.TFrame")
        linha_final.pack(fill="x", pady=(4, 16))
        if nb_pred == lr_pred:
            veredicto = "POSITIVO" if nb_pred else "NEGATIVO"
            cor = COR_VERDE if nb_pred else COR_VERMELHO
            tk.Label(linha_final, text=f"Veredicto (modelos treinados): {veredicto}",
                     bg=COR_PAINEL, fg=cor, font=(FONTE, 12, "bold")).pack(anchor="w")
        else:
            tk.Label(linha_final, text="⚠ Os modelos discordaram — caso interessante!",
                     bg=COR_PAINEL, fg=COR_LARANJA, font=(FONTE, 12, "bold")
                     ).pack(anchor="w")

        # Feedback / coleta de dado
        feedback = ttk.Frame(wrap, style="Painel.TFrame")
        feedback.pack(fill="x", pady=(8, 0))
        ttk.Label(feedback, text="O sentimento real dessa frase é:",
                  style="Painel.TLabel").pack(anchor="w", pady=(0, 8))
        botoes = ttk.Frame(feedback, style="Painel.TFrame")
        botoes.pack(anchor="w")
        ttk.Button(botoes, text="Positivo", style="Positivo.TButton",
                   command=lambda: self._registrar_feedback(frase, 1)).pack(side="left", padx=(0, 8))
        ttk.Button(botoes, text="Negativo", style="Negativo.TButton",
                   command=lambda: self._registrar_feedback(frase, 0)).pack(side="left")
        self.status_feedback = ttk.Label(feedback, text="", style="Painel.TLabel")
        self.status_feedback.pack(anchor="w", pady=(10, 0))

    def _card_modelo(self, parent, nome, pred_conf, coluna):
        pred, conf = pred_conf
        cor = COR_VERDE if pred else COR_VERMELHO
        texto = "POSITIVO" if pred else "NEGATIVO"

        card = tk.Frame(parent, bg=COR_PAINEL_2, padx=16, pady=14)
        card.grid(row=0, column=coluna, padx=6, sticky="nsew")
        parent.grid_columnconfigure(coluna, weight=1)

        tk.Label(card, text=nome, bg=COR_PAINEL_2, fg=COR_TEXTO_FRACO,
                 font=(FONTE, 9, "bold")).pack(anchor="w")
        tk.Label(card, text=texto, bg=COR_PAINEL_2, fg=cor,
                 font=(FONTE, 15, "bold")).pack(anchor="w", pady=(4, 8))

        canvas = tk.Canvas(card, width=180, height=10, bg=COR_CINZA,
                            highlightthickness=0)
        canvas.pack(anchor="w")
        canvas.create_rectangle(0, 0, 180 * conf, 10, fill=cor, width=0)
        tk.Label(card, text=f"confiança: {conf:.0%}", bg=COR_PAINEL_2,
                 fg=COR_TEXTO_FRACO, font=(FONTE, 9)).pack(anchor="w", pady=(4, 0))

    def _card_vader(self, parent, pred_score, coluna):
        pred, score = pred_score
        cor = COR_VERDE if pred else COR_VERMELHO
        texto = "POSITIVO" if pred else "NEGATIVO"

        card = tk.Frame(parent, bg=COR_PAINEL_2, padx=16, pady=14)
        card.grid(row=0, column=coluna, padx=6, sticky="nsew")
        parent.grid_columnconfigure(coluna, weight=1)

        tk.Label(card, text="VADER (baseline de mercado)", bg=COR_PAINEL_2,
                 fg=COR_TEXTO_FRACO, font=(FONTE, 9, "bold")).pack(anchor="w")
        tk.Label(card, text=texto, bg=COR_PAINEL_2, fg=cor,
                 font=(FONTE, 15, "bold")).pack(anchor="w", pady=(4, 8))
        tk.Label(card, text=f"compound score: {score:+.2f}", bg=COR_PAINEL_2,
                 fg=COR_TEXTO_FRACO, font=(FONTE, 9)).pack(anchor="w", pady=(4, 0))
        tk.Label(card, text="(léxico pronto, não aprende com feedback)",
                 bg=COR_PAINEL_2, fg=COR_TEXTO_FRACO, font=(FONTE, 8)
                 ).pack(anchor="w", pady=(4, 0))

    def _registrar_feedback(self, frase, label):
        self.banco = core.adicionar_ao_banco(self.banco, frase, label)
        self.status_feedback.config(
            text=f"✓ Frase salva! Total coletado: {len(self.banco['frases'])}. Retreinando...",
            foreground=COR_VERDE)
        self._textos_treino = self._textos_treino + [frase]
        self._labels_treino = self._labels_treino + [label]
        self.update_idletasks()

        def worker():
            self.modelo.treinar(self._textos_treino, self._labels_treino)
            self.modelo.salvar()
            self._fila.put(("retreino_ok", None, None))
        threading.Thread(target=worker, daemon=True).start()
        self.after(100, self._checar_fila_feedback)

    def _checar_fila_feedback(self):
        try:
            tipo, _, _ = self._fila.get_nowait()
            if tipo == "retreino_ok":
                self.status_feedback.config(
                    text=f"✓ Modelo retreinado com {self.modelo.n} frases no total "
                         f"(acurácia NB {self.modelo.metricas['nb']['acc']:.1%} · "
                         f"LR {self.modelo.metricas['lr']['acc']:.1%})",
                    foreground=COR_VERDE)
                self._atualizar_aba_banco()
                self._atualizar_aba_desempenho()
                return
        except queue.Empty:
            pass
        self.after(100, self._checar_fila_feedback)

    # ── ABA 2: Desempenho & IA ──────────────────────────────────────────────
    def _montar_aba_desempenho(self):
        aba = self.aba_desempenho
        # Canvas com scroll (o conteúdo pode crescer bastante)
        canvas = tk.Canvas(aba, bg=COR_FUNDO, highlightthickness=0)
        scroll = ttk.Scrollbar(aba, orient="vertical", command=canvas.yview)
        self.frame_desempenho = ttk.Frame(canvas, style="TFrame")
        id_janela = canvas.create_window((0, 0), window=self.frame_desempenho, anchor="nw")
        self.frame_desempenho.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(id_janela, width=e.width))
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=8)
        scroll.pack(side="right", fill="y")

        def _scroll_mouse(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _scroll_linux(event):
            canvas.yview_scroll(-1 if event.num == 4 else 1, "units")
        def _ligar_scroll(e):
            canvas.bind_all("<MouseWheel>", _scroll_mouse)
            canvas.bind_all("<Button-4>", _scroll_linux)
            canvas.bind_all("<Button-5>", _scroll_linux)
        def _desligar_scroll(e):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")
        canvas.bind("<Enter>", _ligar_scroll)
        canvas.bind("<Leave>", _desligar_scroll)

        self._atualizar_aba_desempenho()

    def _atualizar_aba_desempenho(self):
        for w in self.frame_desempenho.winfo_children():
            w.destroy()
        f = self.frame_desempenho

        # Botões de ações avançadas
        acoes = ttk.Frame(f, style="TFrame")
        acoes.pack(fill="x", pady=(0, 16))
        ttk.Button(acoes, text="⚙ Otimizar Hiperparâmetros (GridSearchCV)",
                   style="Secundario.TButton",
                   command=self._acao_otimizar).pack(side="left", padx=(0, 8))
        ttk.Button(acoes, text="📊 Validação Cruzada (k-fold)",
                   style="Secundario.TButton",
                   command=self._acao_kfold).pack(side="left", padx=(0, 8))
        ttk.Button(acoes, text="⚖ Comparar com baseline (VADER)",
                   style="Secundario.TButton",
                   command=self._acao_baseline).pack(side="left")
        self.status_acao = ttk.Label(f, text="", style="TLabel", foreground=COR_TEXTO_FRACO)
        self.status_acao.pack(anchor="w", pady=(0, 12))

        # Tabela de métricas
        painel_metricas = tk.Frame(f, bg=COR_PAINEL, padx=20, pady=16)
        painel_metricas.pack(fill="x", pady=(0, 16))
        tk.Label(painel_metricas, text="Desempenho no conjunto de teste (hold-out 15%)",
                 bg=COR_PAINEL, fg=COR_TEXTO, font=(FONTE, 11, "bold")
                 ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))

        cabecalhos = ["Métrica", "Naive Bayes", "Reg. Logística"]
        if self.modelo.baseline_vader:
            cabecalhos.append("VADER (baseline)")
        for c, txt in enumerate(cabecalhos):
            tk.Label(painel_metricas, text=txt, bg=COR_PAINEL, fg=COR_TEXTO_FRACO,
                     font=(FONTE, 9, "bold")).grid(row=1, column=c, sticky="w",
                                                    padx=(0, 30), pady=4)
        nomes_metricas = [("acc", "Acurácia"), ("pre", "Precisão"),
                           ("rec", "Recall"), ("f1", "F1-Score")]
        for i, (chave, label) in enumerate(nomes_metricas, start=2):
            tk.Label(painel_metricas, text=label, bg=COR_PAINEL, fg=COR_TEXTO,
                     font=(FONTE, 10)).grid(row=i, column=0, sticky="w", pady=3)
            tk.Label(painel_metricas, text=f"{self.modelo.metricas['nb'][chave]:.1%}",
                     bg=COR_PAINEL, fg=COR_AZUL, font=(FONTE, 10, "bold")
                     ).grid(row=i, column=1, sticky="w", pady=3)
            tk.Label(painel_metricas, text=f"{self.modelo.metricas['lr'][chave]:.1%}",
                     bg=COR_PAINEL, fg=COR_LARANJA, font=(FONTE, 10, "bold")
                     ).grid(row=i, column=2, sticky="w", pady=3)
            if self.modelo.baseline_vader:
                tk.Label(painel_metricas, text=f"{self.modelo.baseline_vader[chave]:.1%}",
                         bg=COR_PAINEL, fg=COR_TEXTO_FRACO, font=(FONTE, 10, "bold")
                         ).grid(row=i, column=3, sticky="w", pady=3)
        tk.Label(painel_metricas,
                 text=f"Hiperparâmetros usados: alpha (NB) = "
                      f"{self.modelo.melhores_params['nb_alpha']}   ·   "
                      f"C (Reg. Logística) = {self.modelo.melhores_params['lr_C']}"
                      f"   ·   treinado com {self.modelo.n} textos",
                 bg=COR_PAINEL, fg=COR_TEXTO_FRACO, font=(FONTE, 9)
                 ).grid(row=len(nomes_metricas) + 2, column=0, columnspan=4,
                        sticky="w", pady=(12, 0))

        # Validação cruzada (se já rodou)
        if self.modelo.cv_resultados:
            painel_cv = tk.Frame(f, bg=COR_PAINEL, padx=20, pady=16)
            painel_cv.pack(fill="x", pady=(0, 16))
            tk.Label(painel_cv, text="Validação Cruzada (k-fold, k=5) — média ± desvio padrão",
                     bg=COR_PAINEL, fg=COR_TEXTO, font=(FONTE, 11, "bold")
                     ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
            for c, txt in enumerate(["Métrica", "Naive Bayes", "Reg. Logística"]):
                tk.Label(painel_cv, text=txt, bg=COR_PAINEL, fg=COR_TEXTO_FRACO,
                         font=(FONTE, 9, "bold")).grid(row=1, column=c, sticky="w",
                                                        padx=(0, 30), pady=4)
            for i, (chave, label) in enumerate(
                    [("accuracy", "Acurácia"), ("precision", "Precisão"),
                     ("recall", "Recall"), ("f1", "F1-Score")], start=2):
                m_nb, s_nb = self.modelo.cv_resultados["nb"][chave]
                m_lr, s_lr = self.modelo.cv_resultados["lr"][chave]
                tk.Label(painel_cv, text=label, bg=COR_PAINEL, fg=COR_TEXTO,
                         font=(FONTE, 10)).grid(row=i, column=0, sticky="w", pady=3)
                tk.Label(painel_cv, text=f"{m_nb:.1%} ± {s_nb:.1%}", bg=COR_PAINEL,
                         fg=COR_AZUL, font=(FONTE, 10, "bold")
                         ).grid(row=i, column=1, sticky="w", pady=3)
                tk.Label(painel_cv, text=f"{m_lr:.1%} ± {s_lr:.1%}", bg=COR_PAINEL,
                         fg=COR_LARANJA, font=(FONTE, 10, "bold")
                         ).grid(row=i, column=2, sticky="w", pady=3)

        # Matrizes de confusão (imagens)
        painel_cm = tk.Frame(f, bg=COR_FUNDO)
        painel_cm.pack(fill="x", pady=(0, 16))
        self._imagens_cm = []  # evita garbage collection das imagens
        for i, nome in enumerate(("nb", "lr")):
            caminho = self.modelo.exportar_matriz_confusao(nome)
            if not caminho:
                continue
            img = Image.open(caminho)
            img_tk = ImageTk.PhotoImage(img)
            self._imagens_cm.append(img_tk)
            lbl = tk.Label(painel_cm, image=img_tk, bg=COR_FUNDO)
            lbl.grid(row=0, column=i, padx=(0, 16))

        # Palavras mais influentes
        painel_palavras = tk.Frame(f, bg=COR_PAINEL, padx=20, pady=16)
        painel_palavras.pack(fill="x", pady=(0, 16))
        tk.Label(painel_palavras, text="Palavras mais influentes (Regressão Logística)",
                 bg=COR_PAINEL, fg=COR_TEXTO, font=(FONTE, 11, "bold")
                 ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        pos, neg = self.modelo.top_palavras(8)
        tk.Label(painel_palavras, text="Empurram para POSITIVO", bg=COR_PAINEL,
                 fg=COR_VERDE, font=(FONTE, 9, "bold")).grid(row=1, column=0, sticky="w")
        tk.Label(painel_palavras, text="Empurram para NEGATIVO", bg=COR_PAINEL,
                 fg=COR_VERMELHO, font=(FONTE, 9, "bold")).grid(row=1, column=1, sticky="w",
                                                                 padx=(40, 0))
        for i, ((wp, vp), (wn, vn)) in enumerate(zip(pos, neg), start=2):
            tk.Label(painel_palavras, text=f"{wp}  ({vp:+.2f})", bg=COR_PAINEL,
                     fg=COR_TEXTO, font=(FONTE, 9)).grid(row=i, column=0, sticky="w", pady=1)
            tk.Label(painel_palavras, text=f"{wn}  ({vn:+.2f})", bg=COR_PAINEL,
                     fg=COR_TEXTO, font=(FONTE, 9)).grid(row=i, column=1, sticky="w",
                                                          padx=(40, 0), pady=1)

    def _acao_otimizar(self):
        self.status_acao.config(text="Otimizando hiperparâmetros (GridSearchCV)... isso pode levar alguns segundos.")
        def worker():
            self.modelo.otimizar_hiperparametros(self._textos_treino, self._labels_treino)
            self.modelo.treinar(self._textos_treino, self._labels_treino)
            self.modelo.salvar()
            self._fila.put(("otimizado", None, None))
        threading.Thread(target=worker, daemon=True).start()
        self._poll_generico("otimizado", "Hiperparâmetros otimizados e modelo retreinado!")

    def _acao_kfold(self):
        self.status_acao.config(text="Rodando validação cruzada (k-fold)... isso pode levar até 1 minuto.")
        def worker():
            self.modelo.validacao_cruzada(self._textos_treino, self._labels_treino, cv=5)
            self._fila.put(("kfold_ok", None, None))
        threading.Thread(target=worker, daemon=True).start()
        self._poll_generico("kfold_ok", "Validação cruzada concluída!")

    def _acao_baseline(self):
        self.status_acao.config(text="Comparando com baseline VADER...")
        def worker():
            self.modelo.comparar_baseline_vader()
            self._fila.put(("baseline_ok", None, None))
        threading.Thread(target=worker, daemon=True).start()
        self._poll_generico("baseline_ok", "Comparação com baseline concluída!")

    def _poll_generico(self, tipo_esperado, msg_sucesso):
        def checar():
            try:
                tipo, _, _ = self._fila.get_nowait()
                if tipo == tipo_esperado:
                    self.status_acao.config(text=f"✓ {msg_sucesso}", foreground=COR_VERDE)
                    self._atualizar_aba_desempenho()
                    return
            except queue.Empty:
                pass
            self.after(150, checar)
        self.after(150, checar)

    # ── ABA 3: Banco de Dados ───────────────────────────────────────────────
    def _montar_aba_banco(self):
        aba = self.aba_banco
        ttk.Label(aba, text="Frases que você classificou manualmente durante o uso "
                             "(usadas para retreinar o modelo):",
                  style="TLabel").pack(anchor="w", padx=4, pady=(16, 10))

        colunas = ("sentimento", "frase")
        self.tree_banco = ttk.Treeview(aba, columns=colunas, show="headings", height=18)
        self.tree_banco.heading("sentimento", text="Sentimento")
        self.tree_banco.heading("frase", text="Frase")
        self.tree_banco.column("sentimento", width=110, anchor="w")
        self.tree_banco.column("frase", width=760, anchor="w")
        self.tree_banco.pack(fill="both", expand=True, padx=4, pady=(0, 12))
        self.tree_banco.tag_configure("pos", foreground=COR_VERDE)
        self.tree_banco.tag_configure("neg", foreground=COR_VERMELHO)

        self._atualizar_aba_banco()

    def _atualizar_aba_banco(self):
        for item in self.tree_banco.get_children():
            self.tree_banco.delete(item)
        for frase, label in zip(self.banco["frases"], self.banco["labels"]):
            tag = "pos" if label else "neg"
            texto = "POSITIVO" if label else "NEGATIVO"
            self.tree_banco.insert("", "end", values=(texto, frase), tags=(tag,))

    # ── ABA 4: Duelo dos Modelos ────────────────────────────────────────────
    def _montar_aba_duelo(self):
        aba = self.aba_duelo
        topo = ttk.Frame(aba, style="TFrame")
        topo.pack(fill="x", padx=4, pady=(16, 10))
        ttk.Label(topo, text="Naive Bayes × Regressão Logística competem classificando "
                             "frases difíceis (negação, ironia, sentimento misto) com "
                             "gabarito conhecido.", style="TLabel", wraplength=880
                  ).pack(anchor="w")

        self.placar_label = ttk.Label(topo, text="", style="Subtitulo.TLabel")
        self.placar_label.pack(anchor="w", pady=(6, 0))
        self._atualizar_placar_label()

        ttk.Button(topo, text="▶ Iniciar novo duelo (8 rodadas)",
                   command=self._iniciar_duelo).pack(anchor="w", pady=(10, 0))

        self.painel_duelo = ttk.Frame(aba, style="Painel.TFrame")
        self.painel_duelo.pack(fill="both", expand=True, padx=4, pady=16)
        ttk.Label(self.painel_duelo, text="Clique em \"Iniciar novo duelo\" para começar.",
                  style="Painel.TLabel", foreground=COR_TEXTO_FRACO).pack(expand=True, pady=60)

    def _atualizar_placar_label(self):
        p = self.placar
        self.placar_label.config(
            text=f"Placar histórico  ·  Naive Bayes: {p['nb']}   ·   "
                 f"Reg. Logística: {p['lr']}   ·   Empates: {p['empates']}   ·   "
                 f"Rodadas jogadas: {p['duelos_jogados']}")

    def _iniciar_duelo(self):
        pool = core.frases_duelo()
        self._duelo_rodada = random.sample(pool, min(8, len(pool)))
        self._duelo_idx = 0
        self._duelo_pts_nb = 0
        self._duelo_pts_lr = 0
        self._proxima_rodada_duelo()

    def _proxima_rodada_duelo(self):
        for w in self.painel_duelo.winfo_children():
            w.destroy()

        if self._duelo_idx >= len(self._duelo_rodada):
            self._finalizar_duelo()
            return

        frase, gabarito = self._duelo_rodada[self._duelo_idx]
        resultado = self.modelo.prever(frase)
        nb_pred, nb_conf = resultado["nb"]
        lr_pred, lr_conf = resultado["lr"]
        nb_ok = nb_pred == gabarito
        lr_ok = lr_pred == gabarito
        if nb_ok: self._duelo_pts_nb += 1
        if lr_ok: self._duelo_pts_lr += 1

        wrap = ttk.Frame(self.painel_duelo, style="Painel.TFrame")
        wrap.pack(fill="both", expand=True, padx=24, pady=20)

        ttk.Label(wrap, text=f"Rodada {self._duelo_idx + 1} de {len(self._duelo_rodada)}",
                  style="Fraco.TLabel").pack(anchor="w")
        ttk.Label(wrap, text=f'"{frase}"', style="Painel.TLabel",
                  font=(FONTE, 13, "italic"), wraplength=860).pack(anchor="w", pady=(4, 16))

        linhas = [
            ("Naive Bayes", nb_pred, nb_conf, nb_ok),
            ("Reg. Logística", lr_pred, lr_conf, lr_ok),
        ]
        for nome, pred, conf, ok in linhas:
            cor_resultado = COR_VERDE if ok else COR_VERMELHO
            txt_resultado = "✓ acertou" if ok else "✗ errou"
            txt_pred = "POSITIVO" if pred else "NEGATIVO"
            linha = tk.Frame(wrap, bg=COR_PAINEL)
            linha.pack(fill="x", pady=4)
            tk.Label(linha, text=f"{nome}:", bg=COR_PAINEL, fg=COR_TEXTO,
                     font=(FONTE, 10, "bold"), width=16, anchor="w").pack(side="left")
            tk.Label(linha, text=f"{txt_pred} ({conf:.0%})", bg=COR_PAINEL,
                     fg=COR_TEXTO, font=(FONTE, 10), width=22, anchor="w").pack(side="left")
            tk.Label(linha, text=txt_resultado, bg=COR_PAINEL, fg=cor_resultado,
                     font=(FONTE, 10, "bold")).pack(side="left")

        gab_txt = "POSITIVO" if gabarito else "NEGATIVO"
        tk.Label(wrap, text=f"Gabarito real: {gab_txt}", bg=COR_PAINEL,
                 fg=COR_TEXTO_FRACO, font=(FONTE, 10, "italic")).pack(anchor="w", pady=(12, 0))

        placar_rodada = ttk.Frame(wrap, style="Painel.TFrame")
        placar_rodada.pack(fill="x", pady=(20, 0))
        tk.Label(placar_rodada, text=f"Placar da rodada atual — Naive Bayes: "
                                      f"{self._duelo_pts_nb}   ·   "
                                      f"Reg. Logística: {self._duelo_pts_lr}",
                 bg=COR_PAINEL, fg=COR_TEXTO_FRACO, font=(FONTE, 9)).pack(anchor="w")

        texto_botao = ("Próxima frase →" if self._duelo_idx < len(self._duelo_rodada) - 1
                        else "Ver resultado final")
        ttk.Button(wrap, text=texto_botao, command=self._avancar_duelo
                   ).pack(anchor="w", pady=(16, 0))

    def _avancar_duelo(self):
        self._duelo_idx += 1
        self._proxima_rodada_duelo()

    def _finalizar_duelo(self):
        for w in self.painel_duelo.winfo_children():
            w.destroy()

        if self._duelo_pts_nb > self._duelo_pts_lr:
            vencedor = "Naive Bayes venceu esta rodada!"
            self.placar["nb"] += 1
        elif self._duelo_pts_lr > self._duelo_pts_nb:
            vencedor = "Regressão Logística venceu esta rodada!"
            self.placar["lr"] += 1
        else:
            vencedor = "Empate nesta rodada!"
            self.placar["empates"] += 1
        self.placar["duelos_jogados"] += 1
        core.salvar_placar(self.placar)
        self._atualizar_placar_label()

        wrap = ttk.Frame(self.painel_duelo, style="Painel.TFrame")
        wrap.pack(expand=True, pady=60)
        tk.Label(wrap, text="🏆 " + vencedor, bg=COR_PAINEL, fg=COR_TEXTO,
                 font=(FONTE, 16, "bold")).pack(pady=(0, 10))
        tk.Label(wrap, text=f"Naive Bayes: {self._duelo_pts_nb} pontos   ·   "
                             f"Reg. Logística: {self._duelo_pts_lr} pontos",
                 bg=COR_PAINEL, fg=COR_TEXTO_FRACO, font=(FONTE, 10)).pack()
        ttk.Button(wrap, text="Jogar novamente", command=self._iniciar_duelo
                   ).pack(pady=(20, 0))


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
