                                  
                                       
                                                
                                                                                       
 
                                      
                                                                                
                                          
                                 

import csv
import math
import random
import re
import urllib.request
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


                                                                             
              

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "uci_students"
CSV_PATH = DATA_DIR / "data.csv"
ZIP_PATH = PROJECT_DIR / "data" / "uci_students.zip"
DATA_URL = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"

LABEL_NAMES = ["Dropout", "Enrolled", "Graduate"]

FEATURE_NAMES = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Previous qualification (grade)",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

NUMERIC_FEATURES = {
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
}

BUCKET_NAMES = ["very_low", "low", "mid", "high", "very_high"]

                                          
DIM=64
HEAD=4
WIDE=128
LAYER=2
DROP=0.1
RATE=0.002
EPOCH=15
SIZE=96
DEV="cpu"


def ensure_dataset():
                                                  
    if CSV_PATH.exists():
        return CSV_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("downloading UCI dataset...")
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        ZIP_PATH.write_bytes(response.read())

    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(DATA_DIR)

    if not CSV_PATH.exists():
        raise FileNotFoundError("UCI archive did not contain data.csv")
    return CSV_PATH


def load_uci_records():
    path = ensure_dataset()
    records = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for raw in reader:
                                                       
            row = {key.strip(): value.strip() for key, value in raw.items()}
            records.append(row)
    return records


def make_dataset(seed=8):
    data = load_uci_records()
    random.Random(seed).shuffle(data)
    return data


def split_dataset(data, ratio=0.8, seed=8):
                                             
    rng = random.Random(seed)
    groups = {name: [] for name in LABEL_NAMES}
    for row in data:
        groups[row["Target"]].append(row)

    train = []
    validation = []
    for label in LABEL_NAMES:
        rows = groups[label]
        rng.shuffle(rows)
        cut = int(len(rows) * ratio)
        train.extend(rows[:cut])
        validation.extend(rows[cut:])

    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def class_counts(records):
    counts = {name: 0 for name in LABEL_NAMES}
    for row in records:
        counts[row["Target"]] += 1
    return counts


def parse_float(value):
    return float(value.replace(",", "."))


def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    mix = position - low
    return ordered[low] * (1.0 - mix) + ordered[high] * mix


def make_numeric_bins(records):
                                           
                               
    bins = {}
    for feature in NUMERIC_FEATURES:
        values = [parse_float(row[feature]) for row in records]
        bins[feature] = [
            quantile(values, 0.2),
            quantile(values, 0.4),
            quantile(values, 0.6),
            quantile(values, 0.8),
        ]
    return bins


def clean_token_piece(text):
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "blank"


def bucket_numeric(value, edges):
    for index, edge in enumerate(edges):
        if value <= edge:
            return BUCKET_NAMES[index]
    return BUCKET_NAMES[-1]


def record_to_tokens(row, numeric_bins):
                                                  
    tokens = []
    for feature in FEATURE_NAMES:
        name = clean_token_piece(feature)
        if feature in NUMERIC_FEATURES:
            bucket = bucket_numeric(parse_float(row[feature]), numeric_bins[feature])
            tokens.append(name + "_" + bucket)
        else:
            value = clean_token_piece(row[feature])
            tokens.append(name + "_" + value)
    return tokens


def make_vocabulary(records, numeric_bins):
    words = ["__pad__", "__unk__"]
    seen = set(words)
    for row in records:
        for token in record_to_tokens(row, numeric_bins):
            if token not in seen:
                seen.add(token)
                words.append(token)
    word_to_id = {word: index for index, word in enumerate(words)}
    return word_to_id, words


def make_label_map():
    label_to_id = {name: index for index, name in enumerate(LABEL_NAMES)}
    id_to_label = {index: name for name, index in label_to_id.items()}
    return label_to_id, id_to_label


def encode_tokens(tokens, word_to_id):
    unk = word_to_id["__unk__"]
    return [word_to_id.get(token, unk) for token in tokens]


def make_batch(records, word_to_id, label_to_id, numeric_bins, device="cpu"):
    token_rows = [
        encode_tokens(record_to_tokens(row, numeric_bins), word_to_id)
        for row in records
    ]
    max_len = max(len(row) for row in token_rows)
    pad = word_to_id["__pad__"]
    x = torch.full((len(records), max_len), pad, dtype=torch.long)
    y = torch.empty(len(records), dtype=torch.long)
    for index, row in enumerate(token_rows):
        x[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        y[index] = label_to_id[records[index]["Target"]]
    return x.to(device), y.to(device)


                                                                             
                     

def makeattn(dim=DIM):
    p={}
    s=1.0/math.sqrt(dim)
                                           
    p["wq"]=nn.Parameter(torch.randn(dim, dim)*s)
    p["wk"]=nn.Parameter(torch.randn(dim, dim)*s)
    p["wv"]=nn.Parameter(torch.randn(dim, dim)*s)
    p["wo"]=nn.Parameter(torch.randn(dim, dim)*s)
    return p


def dotattn(q, k, v, m=None):
                                  
    d=q.size(-1)
    a=torch.matmul(q, k.transpose(-2, -1))/math.sqrt(d)
    if m is not None:
        a=a.masked_fill(m, -1e9)
    w=torch.softmax(a, dim=-1)
    y=torch.matmul(w, v)
    return y, w


def manyattn(x1, x2, x3, p, head=HEAD, m=None):
                                                          
    b=x1.size(0)
    n1=x1.size(1)
    n2=x2.size(1)
    dim=x1.size(2)
    d=dim//head

    q=torch.matmul(x1, p["wq"])
    k=torch.matmul(x2, p["wk"])
    v=torch.matmul(x3, p["wv"])

                                   
    q=q.view(b, n1, head, d).transpose(1, 2)
    k=k.view(b, n2, head, d).transpose(1, 2)
    v=v.view(b, n2, head, d).transpose(1, 2)

    if m is not None and m.dim()==2:
        m=m[:, None, None, :]
    y, w=dotattn(q, k, v, m)
    y=y.transpose(1, 2).contiguous()
    y=y.view(b, n1, dim)
    return torch.matmul(y, p["wo"]), w


                                                                             
                                   

def pos(n, dim=DIM, dev="cpu"):
                        
    t=torch.zeros(n, dim, device=dev)
    for i in range(n):
        for j in range(0, dim, 2):
            a=i/(10000 ** (j/dim))
            t[i, j]=math.sin(a)
            if j+1<dim:
                t[i, j+1]=math.cos(a)
    return t


def makeblock(dim=DIM, wide=WIDE):
    p={}
    p["attn"]=makeattn(dim)
                                           
    p["ff1w"]=nn.Parameter(torch.randn(dim, wide)/math.sqrt(dim))
    p["ff1b"]=nn.Parameter(torch.zeros(wide))
    p["ff2w"]=nn.Parameter(torch.randn(wide, dim)/math.sqrt(wide))
    p["ff2b"]=nn.Parameter(torch.zeros(dim))
    return p


def block(x, p, head=HEAD, m=None, drop=DROP, train=False):
    if m is not None:
        m=m[:, None, None, :]

    y, w=manyattn(x, x, x, p["attn"], head, m)
    y=F.dropout(y, drop, training=train)
    x=F.layer_norm(x+y, (x.size(-1),))

    z=torch.matmul(x, p["ff1w"])+p["ff1b"]
    z=F.relu(z)
    y=torch.matmul(z, p["ff2w"])+p["ff2b"]
    y=F.dropout(y, drop, training=train)
    return F.layer_norm(x+y, (x.size(-1),)), w


                                                                             
                                           

def makemodel(vocab, classes, dim=DIM, head=HEAD, wide=WIDE, layer=LAYER):
    if dim%head!=0:
        raise ValueError("dim must be divisible by head")
    p={}
    p["emb"]=nn.Parameter(torch.randn(vocab, dim)*0.02)
    p["blocks"]=[makeblock(dim, wide) for _ in range(layer)]
    p["cw"]=nn.Parameter(torch.randn(dim, classes)/math.sqrt(dim))
    p["cb"]=nn.Parameter(torch.zeros(classes))
    c={"dim": dim, "head": head, "drop": DROP, "pad": 0}
    return p, c


def plist(x):
    r=[]
    if isinstance(x, dict):
        xs=x.values()
    elif isinstance(x, list):
        xs=x
    else:
        return [x]
    for x1 in xs:
        if isinstance(x1, dict) or isinstance(x1, list):
            r.extend(plist(x1))
        else:
            r.append(x1)
    return r


def movep(p, dev):
    if dev=="cuda":
        for x in plist(p):
            x.data=x.data.cuda()


def copyp(p):
    return [x.detach().cpu().clone() for x in plist(p)]


def loadp(p, old):
    for x, y in zip(plist(p), old):
        x.data.copy_(y.to(x.device))


def go(tok, p, c, train=False, ret=False):
    dim=c["dim"]
    x=F.embedding(tok, p["emb"])*math.sqrt(dim)
    x=x+pos(tok.size(1), dim, tok.device)
    m=tok.eq(c["pad"])

    ws=[]
    for p1 in p["blocks"]:
        x, w=block(x, p1, c["head"], m, c["drop"], train)
        ws.append(w)

                                  
    ok=m.logical_not().float().unsqueeze(-1)
    x=(x*ok).sum(dim=1)/ok.sum(dim=1).clamp_min(1.0)
    y=torch.matmul(x, p["cw"])+p["cb"]
    if ret:
        return y, ws
    return y


def batch(rows, size, seed):
    rows=list(rows)
    random.Random(seed).shuffle(rows)
    for i in range(0, len(rows), size):
        yield rows[i : i+size]


def acc(rows, wid, lid, bins, p, c, size=128, dev="cpu"):
    good=0
    all1=0
    with torch.no_grad():
        for rows1 in batch(rows, size, 0):
            x, y=make_batch(rows1, wid, lid, bins, dev)
            y1=go(x, p, c).argmax(dim=-1)
            good+=y1.eq(y).sum().item()
            all1+=y.numel()
    return good/all1


def table(rows, wid, lid, bins, p, c, dev="cpu"):
    t=torch.zeros(len(LABEL_NAMES), len(LABEL_NAMES), dtype=torch.long)
    with torch.no_grad():
        for rows1 in batch(rows, 128, 0):
            x, y=make_batch(rows1, wid, lid, bins, dev)
            y1=go(x, p, c).argmax(dim=-1)
            for a, b in zip(y.cpu(), y1.cpu()):
                t[int(a), int(b)]+=1
    return t


def printtable(t):
    w=11
    print("".ljust(w), end="")
    for name in LABEL_NAMES:
        print(name[:9].rjust(w), end="")
    print()
    for i, row in enumerate(t):
        print(LABEL_NAMES[i][:9].ljust(w), end="")
        for x in row:
            print(str(int(x)).rjust(w), end="")
        print()


def show(rows, wid, lid, names, bins, p, c, dev="cpu"):
    with torch.no_grad():
        for row in rows:
            x, y=make_batch([row], wid, lid, bins, dev)
            y1=go(x, p, c).argmax(dim=-1).item()
            tok=record_to_tokens(row, bins)
            print("tokens:", " ".join(tok[:8])+" ...")
            print("true: ", names[y.item()])
            print("pred: ", names[y1])
            print()


                                                                             
                                

def save_training_curve(history, output_dir):
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "uci_01_training_curve.png"
    epochs = [item["epoch"] for item in history]
    losses = [item["loss"] for item in history]
    accuracies = [item["val_acc"] for item in history]

    fig, ax_loss = plt.subplots(figsize=(8, 4.8))
    ax_acc = ax_loss.twinx()
    ax_loss.plot(epochs, losses, color="#2f6f9f", marker="o", markersize=3, label="train loss")
    ax_acc.plot(epochs, accuracies, color="#c4492d", marker="s", markersize=3, label="validation accuracy")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross entropy loss")
    ax_acc.set_ylabel("validation accuracy")
    ax_loss.set_title("UCI student outcome training curve")
    ax_loss.grid(alpha=0.25)
    lines_1, labels_1 = ax_loss.get_legend_handles_labels()
    lines_2, labels_2 = ax_acc.get_legend_handles_labels()
    ax_loss.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_confusion_plot(matrix, output_dir):
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "uci_02_confusion_matrix.png"
    data = matrix.float()

    fig, ax = plt.subplots(figsize=(6, 5.4))
    image = ax.imshow(data, cmap="Blues")
    ax.set_title("Validation confusion matrix")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks(range(len(LABEL_NAMES)), LABEL_NAMES, rotation=25, ha="right")
    ax.set_yticks(range(len(LABEL_NAMES)), LABEL_NAMES)
    for row in range(data.size(0)):
        for col in range(data.size(1)):
            value = int(data[row, col].item())
            color = "white" if value > data.max().item() * 0.55 else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def shorten_token(token):
    parts = token.split("_")
    if len(parts) <= 3:
        return token[:18]
    return ("_".join(parts[:2]) + "_" + parts[-1])[:18]


def save_attention_plot(record, word_to_id, label_to_id, numeric_bins, p, settings, output_dir, device="cpu"):
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "uci_03_attention_features.png"
    tokens = record_to_tokens(record, numeric_bins)
    x, _ = make_batch([record], word_to_id, label_to_id, numeric_bins, device)
    with torch.no_grad():
        _, attentions = go(x, p, settings, ret=True)
    weight = attentions[-1][0].mean(dim=0).cpu()
    labels = [shorten_token(token) for token in tokens]

    fig, ax = plt.subplots(figsize=(10, 8.5))
    image = ax.imshow(weight, cmap="magma")
    ax.set_title("Learned attention on one validation student")
    ax.set_xlabel("key feature token")
    ax.set_ylabel("query feature token")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_plots(history, matrix, validation, word_to_id, label_to_id, numeric_bins, p, settings, device="cpu"):
    output_dir = PROJECT_DIR / "result_pictures"
    paths = [
        save_training_curve(history, output_dir),
        save_confusion_plot(matrix, output_dir),
    ]
    if validation:
        paths.append(
            save_attention_plot(
                validation[0],
                word_to_id,
                label_to_id,
                numeric_bins,
                p,
                settings,
                output_dir,
                device,
            )
        )
    return paths


                                                                             
              

def trainit(epochs=EPOCH, size=SIZE, dev=DEV, pics=True):
    torch.manual_seed(8)
    random.seed(8)
    rows=make_dataset(seed=8)
    train, val=split_dataset(rows, ratio=0.8, seed=8)
    bins=make_numeric_bins(train)
    wid, words=make_vocabulary(train, bins)
    lid, names=make_label_map()
    p, c=makemodel(len(wid), len(lid))
    movep(p, dev)
    opt=torch.optim.Adam(plist(p), lr=RATE)

    print("dataset:", len(rows), "rows")
    print("train counts:", class_counts(train))
    print("validation counts:", class_counts(val))
    print("vocab size:", len(wid))

    his=[]
    beste=0
    besta=-1.0
    bestp=None
    for e in range(1, epochs+1):
        loss1=0.0
        n=0
        for rows1 in batch(train, size, e):
            x, y=make_batch(rows1, wid, lid, bins, dev)
            opt.zero_grad()
            y1=go(x, p, c, train=True)
            loss=F.cross_entropy(y1, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(plist(p), 1.0)
            opt.step()
            loss1+=loss.item()
            n+=1
        a=acc(val, wid, lid, bins, p, c, size, dev)
        his.append({"epoch": e, "loss": loss1/n, "val_acc": a})
        if a>besta:
            beste=e
            besta=a
            bestp=copyp(p)
        if e==1 or e%5==0 or e==epochs:
            print(e, round(loss1/n, 4), round(a, 4))

    if bestp is not None:
        loadp(p, bestp)
        print("best epoch:", beste, "best validation accuracy:", round(besta, 4))

    print("\nconfusion matrix: row=true, col=pred")
    t=table(val, wid, lid, bins, p, c, dev)
    printtable(t)
    print("\nexamples:")
    show(val[:5], wid, lid, names, bins, p, c, dev)

    paths=[]
    if pics:
        paths=save_plots(his, t, val, wid, lid, bins, p, c, dev)
        print("saved plots:")
        for path in paths:
            print(path)

    return p, c, wid, words, lid, names, bins, his, val, paths


def eyeattn(dim):
    p=makeattn(dim)
    with torch.no_grad():
        e=torch.eye(dim)
        p["wq"].copy_(e)
        p["wk"].copy_(e)
        p["wv"].copy_(e)
        p["wo"].copy_(e)
    return p


def showattn(tok, w):
    wide=max(len(x) for x in tok)+2
    print("".ljust(wide), end="")
    for x in tok:
        print(x[:8].rjust(9), end="")
    print()
    for i, x in enumerate(tok):
        print(x[:wide-1].ljust(wide), end="")
        for n in w[i]:
            print(("%.3f"%float(n)).rjust(9), end="")
        print()


def probe():
    tok=["approved_high", "grade_high", "debtor_0", "tuition_paid_1"]
    x=torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    ).unsqueeze(0)
    p=eyeattn(4)
    _, w=manyattn(x, x, x, p, head=1)
    print("attention weight table (row=query, column=key)")
    showattn(tok, w[0, 0].detach())


def check():
    x=torch.randn(2, 4, 8)
    p=makeattn(8)
    y, w=manyattn(x, x, x, p, head=2)
    assert y.shape==(2, 4, 8)
    assert w.shape==(2, 2, 4, 4)
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, 2, 4), atol=1e-5)

    t=pos(6, 8)
    assert t.shape==(6, 8)

    rows=make_dataset(seed=3)
    train, val=split_dataset(rows[:300], ratio=0.8, seed=3)
    bins=make_numeric_bins(train)
    wid, _=make_vocabulary(train, bins)
    lid, _=make_label_map()
    x, y=make_batch(train[:8], wid, lid, bins)
    p, c=makemodel(len(wid), len(lid), dim=24, head=4, wide=48, layer=2)
    y1, ws=go(x, p, c, train=True, ret=True)
    assert y1.shape==(8, len(lid))
    assert len(ws)==2
    assert y.shape==(8,)
    y1.mean().backward()
    assert all(x.grad is not None for x in plist(p))
    assert len(record_to_tokens(train[0], bins))==36
    assert len(val)>0
    print("all UCI dataset-transformer checks passed")


def demo():
    rows=make_dataset(seed=4)
    train, _=split_dataset(rows, ratio=0.8, seed=4)
    bins=make_numeric_bins(train)
    wid, _=make_vocabulary(train, bins)
    lid, names=make_label_map()
    p, c=makemodel(len(wid), len(lid))
    x, y=make_batch(train[:3], wid, lid, bins)
    y1=go(x, p, c)
    print("sample tokens:")
    for row in train[:3]:
        print(" ".join(record_to_tokens(row, bins)[:8])+" ...", "=>", row["Target"])
    print("logits shape:", tuple(y1.shape))
    print("classes:", ", ".join(names[i] for i in range(len(names))))
    print("parameter count:", sum(x.numel() for x in plist(p)))


def main():
                                       
    mode="train"
                 
                  
                  

    if mode=="demo":
        demo()
    elif mode=="check":
        check()
    elif mode=="probe":
        probe()
    else:
        trainit()


if __name__=="__main__":
    main()
