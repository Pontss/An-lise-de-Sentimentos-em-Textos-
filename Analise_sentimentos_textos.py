import re, os, json, time, warnings, numpy as np, nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")
nltk.download("stopwords", quiet=True)

# ── Cores ────────────────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"
C = "\033[96m"; W = "\033[97m"; GR = "\033[90m"; BO = "\033[1m"; RS = "\033[0m"

def clr():  os.system("cls" if os.name == "nt" else "clear")
def sep(c="═", n=62, cor=C): print(f"{cor}{c*n}{RS}")
def linha(c="─", n=62, cor=GR): print(f"{cor}{c*n}{RS}")
def titulo(t): sep(); print(f"{BO}{C}  {t}{RS}"); sep()
def ok(t):    print(f"{G}  ✓  {t}{RS}")
def info(t):  print(f"{B}  ●  {t}{RS}")
def warn(t):  print(f"{Y}  ⚠  {t}{RS}")

# ── Banco de dados ────────────────────────────────────────────────────────────
DB = "banco_frases.json"

def carregar():
    if os.path.exists(DB):
        with open(DB, "r", encoding="utf-8") as f: return json.load(f)
    return {"frases": [], "labels": []}

def salvar(banco):
    with open(DB, "w", encoding="utf-8") as f:
        json.dump(banco, f, ensure_ascii=False, indent=2)

def adicionar(banco, frase, label):
    banco["frases"].append(frase); banco["labels"].append(label)
    salvar(banco); return banco

# ── Dataset ───────────────────────────────────────────────────────────────────
def dataset():
    pw = ["fantastic","brilliant","wonderful","superb","amazing","excellent",
          "outstanding","magnificent","extraordinary","marvelous","spectacular",
          "incredible","exceptional","perfect","stunning","breathtaking",
          "masterful","captivating","delightful","phenomenal","remarkable",
          "impressive","glorious","splendid","terrific","beautiful","engaging",
          "touching","inspiring","heartwarming","refreshing","uplifting",
          "moving","compelling","fascinating","riveting","gripping",
          "unforgettable","powerful","flawless","genius","visionary","thrilling",
          "electrifying","enchanting","mesmerizing","enthralling","absorbing",
          "rewarding","dazzling"]
    nw = ["terrible","awful","horrible","dreadful","dull","atrocious","appalling",
          "abysmal","catastrophic","dismal","disappointing","pathetic","mediocre",
          "laughable","unbearable","boring","tedious","forgettable","unwatchable",
          "wretched","pitiful","deplorable","miserable","weak","ugly","disengaging",
          "uninspiring","depressing","stale","demoralizing","painful","unconvincing",
          "ridiculous","pointless","agonizing","powerless","flawed","senseless",
          "unimaginative","flat","lifeless","confusing","disjointed","unpleasant",
          "nonsensical","insipid","vapid","pretentious","tiresome","infuriating"]
    tp = ["This movie was absolutely {w}","I thought this film was completely {w}",
          "The acting in this movie was {w}","The story was {w} from beginning to end",
          "I found this film to be truly {w}","Every scene in this film was {w}",
          "The plot of this movie was {w} throughout",
          "The performances in this film were {w}",
          "The director created something {w} with this movie",
          "This movie is {w} in every single way"]
    p1 = [t.format(w=w) for w in pw for t in tp]
    n1 = [t.format(w=w) for w in nw for t in tp]
    p2 = ["This movie is good","This film is very good","I liked this movie",
          "This film was nice","A good movie overall","Pretty good film",
          "This was a decent movie","I enjoyed this film","Good story overall",
          "I liked the film a lot","This movie is really good","Nice film",
          "A solid movie","This was a fun film","Really enjoyable movie",
          "I had a great time watching this","This film was quite good",
          "A very enjoyable film","Good acting and good story",
          "I would recommend this film","Worth watching this movie",
          "This film exceeded my expectations","A satisfying movie experience",
          "The film was better than expected","I really liked this movie",
          "Good movie with great performances","This film made me smile",
          "A feel good movie","Genuinely good film","This movie was fun",
          "I loved this film","A great movie experience","So good",
          "Really liked it","Loved every minute of this film",
          "This is a must watch film","Highly recommend this movie",
          "This film was heartfelt and good","A very good watch",
          "This movie left me happy","A wonderfully made film",
          "This movie is top notch","Thoroughly enjoyed this film",
          "This was a pleasant movie","A lovely film",
          "This movie was entertaining","Great film to watch",
          "Very well made movie","I loved this movie",
          "Such a good film","This movie was brilliant and good"]
    n2 = ["This movie is bad","This film is very bad","I did not like this movie",
          "This film was not nice","A bad movie overall","Pretty bad film",
          "This was a poor movie","I did not enjoy this film","Bad story overall",
          "I did not like the film","This movie is really bad","Not a nice film",
          "A weak movie","This was a dull film","Really boring movie",
          "I had a bad time watching this","This film was quite bad",
          "A very unenjoyable film","Bad acting and bad story",
          "I would not recommend this film","Not worth watching this movie",
          "This film did not meet my expectations","A disappointing movie experience",
          "The film was worse than expected","I really did not like this movie",
          "Bad movie with poor performances","This film made me unhappy",
          "A bad movie","Genuinely bad film","This movie was not fun",
          "I hated this film","A terrible movie experience","So bad",
          "Really hated it","Hated every minute of this film",
          "This is not worth watching","Do not recommend this movie",
          "This film was cold and bad","A very bad watch",
          "This movie left me disappointed","A poorly made film",
          "This movie is not good at all","Did not enjoy this film at all",
          "This was an unpleasant movie","A bad film",
          "This movie was not entertaining","Bad film to watch",
          "Very poorly made movie","I hated this movie",
          "Such a bad film","This movie was awful and bad"]
    p3 = ["This movie is very good","This film is really good","Not bad at all",
          "This movie was not bad","Not a bad film","This film is not bad at all",
          "Very good acting in this film","Really good story in this movie",
          "This film turned out to be very good","Very fun and very good movie",
          "Not a single bad scene","Very nice film indeed","Not a bad movie at all",
          "Really enjoyable and good movie","So very good","Not boring at all",
          "Really not bad","Very heartfelt and good film","Really good and touching movie",
          "This film is really not bad at all","Very satisfying and good movie",
          "This movie really surprised me in a good way","Really well done film",
          "Not as bad as I expected","Very captivating and good"]
    n3 = ["This movie is very bad","This film is really bad","Not good at all",
          "This movie was not good","Not a good film","This film is not good at all",
          "Very bad acting in this film","Really bad story in this movie",
          "This film turned out to be very bad","Very dull and very bad movie",
          "Not a single good scene","Very bad film indeed","Not a good movie at all",
          "Really unenjoyable and bad movie","So very bad","Not interesting at all",
          "Really very bad","Very cold and bad film","Really bad and painful movie",
          "This film is really not good at all","Very disappointing and bad movie",
          "This movie really let me down in a bad way","Really poorly done film",
          "Not as good as I expected","Very boring and bad"]
    tx = p1+n1+p2+n2+p3+n3
    lb = [1]*len(p1)+[0]*len(n1)+[1]*len(p2)+[0]*len(n2)+[1]*len(p3)+[0]*len(n3)
    assert len(tx)==len(lb)
    return tx, lb

# ── Pré-processamento ─────────────────────────────────────────────────────────
_base = set(stopwords.words("english"))
_keep = {"not","no","nor","never","very","really","so","too","quite","rather",
         "extremely","incredibly","absolutely","completely","highly","deeply",
         "truly","utterly","totally","somewhat","fairly","pretty","more","most",
         "good","bad","great","poor","nice","well","best","worst","better","worse"}
SW = _base - _keep

def prep(t):
    t = t.lower(); t = re.sub(r"[^a-z\s]","",t)
    return " ".join(w for w in t.split() if w not in SW and len(w)>1)

# ── Modelos ───────────────────────────────────────────────────────────────────
class Modelos:
    def __init__(self):
        self.tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=5000, sublinear_tf=True)
        self.nb    = MultinomialNB(alpha=1.0)
        self.lr    = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        self.metricas = {}; self.n = 0

    def treinar(self, textos, labels):
        self.n = len(textos)
        tl = [prep(t) for t in textos]
        Xtr,Xte,ytr,yte = train_test_split(tl,labels,test_size=0.15,
                                            random_state=42,stratify=labels)
        Xtr = self.tfidf.fit_transform(Xtr); Xte = self.tfidf.transform(Xte)
        self.nb.fit(Xtr,ytr); self.lr.fit(Xtr,ytr)
        for nome, m, yp in [("nb",self.nb,self.nb.predict(Xte)),
                             ("lr",self.lr,self.lr.predict(Xte))]:
            self.metricas[nome] = {
                "acc": accuracy_score(yte,yp),
                "pre": precision_score(yte,yp),
                "rec": recall_score(yte,yp),
                "f1" : f1_score(yte,yp),
            }

    def prever(self, frase):
        v = self.tfidf.transform([prep(frase)])
        np_ = self.nb.predict_proba(v)[0]; lp_ = self.lr.predict_proba(v)[0]
        return {"nb":(int(np.argmax(np_)), float(max(np_))),
                "lr":(int(np.argmax(lp_)), float(max(lp_))),
                "limpa": prep(frase)}

    def top_palavras(self, n=8):
        c = self.lr.coef_[0]; nm = self.tfidf.get_feature_names_out()
        return ([(nm[i],c[i]) for i in np.argsort(c)[::-1][:n]],
                [(nm[i],c[i]) for i in np.argsort(c)[:n]])

# ── Barra de progresso ────────────────────────────────────────────────────────
def barra(val, cor, w=28):
    f = int(val*w); return f"{cor}{'█'*f}{'░'*(w-f)}{RS} {BO}{val:.1%}{RS}"

def barra_treino(msg, dur=1.2):
    print(f"\n  {GR}{msg}{RS}  ", end="", flush=True)
    steps = 25
    for i in range(steps+1):
        p = i/steps; f = int(p*20)
        bar = f"\r  {GR}{msg}{RS}  {C}[{'█'*f}{'░'*(20-f)}]{RS} {BO}{p:.0%}{RS}  "
        print(bar, end="", flush=True)
        time.sleep(dur/steps)
    print()

# ── Tela de abertura ──────────────────────────────────────────────────────────
def tela_abertura():
    clr()
    print(f"\n{C}{'═'*62}{RS}")
    print(f"{C}║{RS}{'':^60}{C}║{RS}")
    print(f"{C}║{RS}{BO}{W}{'ANÁLISE DE SENTIMENTOS EM TEXTOS':^60}{RS}{C}║{RS}")
    print(f"{C}║{RS}{GR}{'Naive Bayes  ×  Regressão Logística  ×  TF-IDF':^60}{RS}{C}║{RS}")
    print(f"{C}║{RS}{'':^60}{C}║{RS}")
    print(f"{C}║{RS}{GR}{'Engenharia da Computação — FHO':^60}{RS}{C}║{RS}")
    print(f"{C}║{RS}{'':^60}{C}║{RS}")
    print(f"{C}{'═'*62}{RS}")
    print(f"\n  {GR}Gustavo Oliveira  ·  Matheus Pontes  ·  Mauricio Guilherme{RS}\n")

# ── Tela de resultado ─────────────────────────────────────────────────────────
def tela_resultado(res, frase):
    nb_pred, nb_conf = res["nb"]; lr_pred, lr_conf = res["lr"]
    nb_txt = "POSITIVO ✓" if nb_pred else "NEGATIVO ✗"
    lr_txt = "POSITIVO ✓" if lr_pred else "NEGATIVO ✗"
    nb_cor = G if nb_pred else R; lr_cor = G if lr_pred else R
    concordam = nb_pred == lr_pred

    print()
    linha()
    print(f"  {BO}{W}Frase analisada:{RS}")
    print(f"  {C}» {frase}{RS}")
    print(f"  {GR}  tokens: [{res['limpa']}]{RS}")
    linha()

    # Naive Bayes
    print(f"\n  {BO}{B}[ NAIVE BAYES MULTINOMIAL ]{RS}")
    print(f"    Sentimento  :  {nb_cor}{BO}{nb_txt}{RS}")
    print(f"    Confiança   :  {barra(nb_conf, nb_cor)}")

    # Regressão Logística
    print(f"\n  {BO}{Y}[ REGRESSÃO LOGÍSTICA ]{RS}")
    print(f"    Sentimento  :  {lr_cor}{BO}{lr_txt}{RS}")
    print(f"    Confiança   :  {barra(lr_conf, lr_cor)}")

    print()
    if concordam:
        veredicto = "POSITIVO ✓" if nb_pred else "NEGATIVO ✗"
        vcor = G if nb_pred else R
        linha()
        print(f"  {BO}Veredicto final  →  {vcor}{veredicto}{RS}  {GR}(ambos concordam){RS}")
    else:
        linha()
        print(f"  {Y}{BO}⚠  Os modelos discordaram — caso interessante para análise!{RS}")
    linha()

# ── Tela de estatísticas ──────────────────────────────────────────────────────
def tela_estatisticas(m, banco):
    titulo("DESEMPENHO DOS MODELOS")
    mn = m.metricas
    print(f"\n  {BO}{'Métrica':<14}{'Naive Bayes':>16}{'Reg. Logística':>18}{RS}")
    linha()
    for k, label in [("acc","Acurácia"),("pre","Precisão"),
                     ("rec","Recall"),("f1","F1-Score")]:
        vn = mn["nb"][k]; vl = mn["lr"][k]
        seta = f"{G}◄ LR{RS}" if vl>vn else (f"{B}NB ►{RS}" if vn>vl else "  =  ")
        print(f"  {label:<14}{vn:>14.1%}{vl:>18.1%}   {seta}")

    print(f"\n  {BO}Banco de dados:{RS}")
    info(f"Frases base        : 1.154")
    info(f"Frases coletadas   : {len(banco['frases'])}")
    info(f"Total no treino    : {m.n}")

    print(f"\n  {BO}Palavras mais influentes (Regressão Logística):{RS}")
    tp, tn = m.top_palavras(6)
    print(f"  {G}  Positivas{RS}")
    for w,v in tp:
        b = "█"*min(int(abs(v)*4),20)
        print(f"    {G}{w:<22}{RS} {GR}{b}{RS} {v:+.3f}")
    print(f"  {R}  Negativas{RS}")
    for w,v in tn:
        b = "█"*min(int(abs(v)*4),20)
        print(f"    {R}{w:<22}{RS} {GR}{b}{RS} {v:+.3f}")

# ── Tela do banco de dados ────────────────────────────────────────────────────
def tela_banco(banco):
    titulo("FRASES COLETADAS DURANTE A SESSÃO")
    if not banco["frases"]:
        warn("Nenhuma frase adicionada ainda."); return
    print(f"\n  {BO}{'#':>3}  {'Sentimento':<12}  Frase{RS}")
    linha()
    for i,(f,l) in enumerate(zip(banco["frases"],banco["labels"]),1):
        cor = G if l else R; sent = "POSITIVO" if l else "NEGATIVO"
        print(f"  {GR}{i:>3}{RS}  {cor}{sent:<12}{RS}  {f}")

# ── Loop principal ────────────────────────────────────────────────────────────
def main():
    tela_abertura()

    # Carrega banco e dataset
    banco = carregar()
    txb, lbb = dataset()
    n_col = len(banco["frases"])

    # Treino inicial
    print(f"  {GR}Dataset base: {len(txb)} frases  |  Coletadas: {n_col} frases{RS}\n")
    barra_treino("Treinando os modelos", 1.4)

    modelo = Modelos()
    modelo.treinar(txb + banco["frases"], lbb + banco["labels"])

    ok(f"Naive Bayes      →  acurácia {modelo.metricas['nb']['acc']:.1%}")
    ok(f"Reg. Logística   →  acurácia {modelo.metricas['lr']['acc']:.1%}")
    print()

    # Menu
    while True:
        sep()
        print(f"  {BO}MENU PRINCIPAL{RS}\n")
        print(f"  {G}[1]{RS}  Analisar uma frase em tempo real")
        print(f"  {B}[2]{RS}  Ver desempenho e métricas dos modelos")
        print(f"  {Y}[3]{RS}  Ver frases coletadas no banco de dados")
        print(f"  {GR}[0]{RS}  Encerrar")
        sep()
        op = input(f"\n  {BO}Opção → {RS}").strip()

        # ── OPÇÃO 1: análise ──────────────────────────────────────────────
        if op == "1":
            titulo("ANÁLISE EM TEMPO REAL")
            print(f"  {GR}Digite uma frase em inglês sobre um filme.")
            print(f"  Ex: \"This movie is very good\" / \"Not good at all\"{RS}\n")

            frase = input(f"  {BO}» {RS}").strip()
            if not frase: warn("Frase vazia."); continue
            if not prep(frase): warn("Nenhuma palavra útil após pré-processamento."); continue

            print(f"\n  {GR}Processando...{RS}", end="", flush=True)
            time.sleep(0.6); print()

            res = modelo.prever(frase)
            tela_resultado(res, frase)

            # Confirmação
            print(f"\n  {BO}O sentimento está correto?{RS}")
            print(f"  {G}[P]{RS} Positivo   {R}[N]{RS} Negativo   {GR}[S]{RS} Pular\n")
            r = input(f"  {BO}→ {RS}").strip().upper()

            if r == "S": info("Frase não adicionada ao banco."); continue
            if r not in ("P","N"): warn("Opção inválida."); continue

            lab = 1 if r=="P" else 0
            banco = adicionar(banco, frase, lab)
            ok(f"Frase salva! Total no banco: {len(banco['frases'])} frases.")

            barra_treino("Retreinando com novo dado", 0.9)
            modelo.treinar(txb+banco["frases"], lbb+banco["labels"])
            ok(f"Retreino concluído  →  {modelo.n} frases no total")
            ok(f"Naive Bayes     {modelo.metricas['nb']['acc']:.1%}  |  "
               f"Reg. Logística  {modelo.metricas['lr']['acc']:.1%}")

        # ── OPÇÃO 2: métricas ─────────────────────────────────────────────
        elif op == "2":
            tela_estatisticas(modelo, banco)

        # ── OPÇÃO 3: banco ────────────────────────────────────────────────
        elif op == "3":
            tela_banco(banco)

        # ── OPÇÃO 0: sair ─────────────────────────────────────────────────
        elif op == "0":
            sep()
            print(f"\n  {G}{BO}Sistema encerrado.{RS}")
            info(f"Banco salvo com {len(banco['frases'])} frase(s) coletada(s).")
            print()
            break

        else:
            warn("Opção inválida.")

if __name__ == "__main__":
    main()
