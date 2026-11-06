"""Plots for the Lugbara MT notebooks: data, grid search, training, test scores, comparisons.

Every function draws one figure, saves it as a PNG (150 dpi) and returns the path, so notebooks
can show it inline and the model repos can carry it. Static PNGs, light surface.
Palette = validated categorical order (blue, orange, aqua, yellow); sequential = one blue ramp.
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # fixed order, never cycled
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
LANG_COLOR = {"english": SERIES[0], "lugbara": SERIES[1], "luganda": SERIES[2]}
DS_COLOR = {"salt": SERIES[0], "local": SERIES[1]}              # training corpus, in the 2 x 2 charts
DS_NAME = {"salt": "SALT", "local": "local corpus"}
THESIS_INK = INK2


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 1, "axes.labelcolor": INK2,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.labelsize": 10, "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1,
        "grid.linestyle": "-", "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK2, "font.size": 10,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def _save(fig, path, note=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if note:
        fig.text(0.01, 0.005, note, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _bar_labels(ax, bars, fmt="{:.1f}", horizontal=False):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if horizontal:
            ax.annotate(fmt.format(v), (v, b.get_y() + b.get_height() / 2), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK2)
        else:
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=INK2)


def _pretty(d):
    return d.replace("-", " → ").title()


# ---------------------------------------------------------------- data / Lugbara analysis
def data_figures(analysis, out_dir):
    """Figures from lugbara_analysis.json (written by lugbara_mt.py analyze)."""
    _style()
    out_dir = Path(out_dir)
    paths = []
    hist = analysis.get("length_hist")
    corpus = analysis.get("corpus", {})
    langs = [l for l in ("english", "lugbara", "luganda") if l in corpus]
    local = analysis.get("dataset") == "local"
    if hist:                                     # words per sentence, the same sentences in each language
        fig, ax = plt.subplots(figsize=(8, 4))
        for lang in langs:
            h = hist[lang]
            xs = [int(k) for k in h]
            ys = [h[k] for k in h]
            ax.plot(xs, ys, color=LANG_COLOR[lang], label=lang.title(), drawstyle="steps-mid")
            mean = analysis["corpus"][lang]["mean_len"]
            ax.axvline(mean, color=LANG_COLOR[lang], linewidth=1, alpha=0.6)
            ax.annotate(f"mean {mean:.1f}", (mean, max(ys)), xytext=(3, 0), textcoords="offset points",
                        fontsize=8, color=INK2, rotation=90, va="top")
        n_sent = corpus["lugbara"]["sentences"]
        n_lang = {2: "two", 3: "three"}.get(len(langs), str(len(langs)))
        ax.set(title=f"Sentence length: the same {n_sent:,} sentences in {n_lang} languages",
               xlabel="Words per sentence", ylabel="Sentences")
        ax.legend(loc="upper right")
        paths.append(_save(fig, out_dir / "data_sentence_length.png"))

    top = analysis.get("top_words", [])[:20]
    if top:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        words, counts = [w for w, _ in top][::-1], [c for _, c in top][::-1]
        bars = ax.barh(words, counts, color=SERIES[1], height=0.6, edgecolor=SURFACE, linewidth=2)
        _bar_labels(ax, bars, "{:,.0f}", horizontal=True)
        ax.set(title="Most frequent Lugbara words", xlabel="Occurrences in the " + ("corpus" if local else "sample"))
        ax.grid(axis="y", visible=False)
        paths.append(_save(fig, out_dir / "data_top_lugbara_words.png"))

    ap = analysis.get("apostrophe_followed_by", [])[:8]
    if ap:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        labels, counts = [f"'{k}" for k, _ in ap][::-1], [c for _, c in ap][::-1]
        bars = ax.barh(labels, counts, color=SERIES[1], height=0.6, edgecolor=SURFACE, linewidth=2)
        _bar_labels(ax, bars, "{:,.0f}", horizontal=True)
        share = analysis.get("apostrophe_word_share")
        ax.set(title="Lugbara glottal apostrophe: which letter follows it",
               xlabel="Occurrences")
        ax.grid(axis="y", visible=False)
        paths.append(_save(fig, out_dir / "data_apostrophe_letters.png",
                           note=f"{100 * share:.1f}% of Lugbara word tokens contain an apostrophe." if share else None))

    tok = analysis.get("tokenizers") or {}
    if tok:
        fig, ax = plt.subplots(figsize=(7, 4))
        names = list(tok)
        w = 0.36
        for i, name in enumerate(names):
            vals = [tok[name]["pieces_per_word"].get(l, 0) for l in langs]
            xs = [j + (i - (len(names) - 1) / 2) * w for j in range(len(langs))]
            bars = ax.bar(xs, vals, width=w, color=SERIES[i], edgecolor=SURFACE, linewidth=2,
                          label="XLM-R" if name == "xlmr" else "mT5")
            _bar_labels(ax, bars, "{:.2f}")
        ax.set_xticks(range(len(langs)), [l.title() for l in langs])
        ax.set(title="How finely each thesis tokenizer cuts words", ylabel="Sub-word pieces per word")
        ax.legend(loc="upper left")
        ax.grid(axis="x", visible=False)
        shares = {("XLM-R" if k == "xlmr" else "mT5"): v["apostrophe_cut_off_share"] for k, v in tok.items()}
        note = ("Both tokenizers split the Lugbara apostrophe off as its own piece in every apostrophe word."
                if all(v >= 0.999 for v in shares.values()) else
                "Apostrophe words whose apostrophe becomes its own piece: "
                + ", ".join(f"{k} {100 * v:.0f}%" for k, v in shares.items()) + ".")
        paths.append(_save(fig, out_dir / "data_tokenizer_pieces.png", note=note))

    vs = analysis.get("vs_salt")
    if vs:                                       # local corpus vs SALT sample: test words unseen in training
        u = vs["unseen_test_words"]
        fig, ax = plt.subplots(figsize=(7.5, 4))
        w = 0.36
        for i, tr in enumerate(("salt", "local")):
            xs = [j + (i - 0.5) * w for j in range(2)]
            bars = ax.bar(xs, [100 * u["lugbara"][f"{tr}>{te}"] for te in ("salt", "local")], width=w,
                          color=DS_COLOR[tr], label=f"Training split of {DS_NAME[tr]}", edgecolor=SURFACE, linewidth=2)
            _bar_labels(ax, bars, "{:.1f}%")
        ax.set_xticks(range(2), ["SALT test split", "Local test split"])
        top = max(100 * v for v in u["lugbara"].values())
        ax.set_ylim(0, max(1, top) * 1.3)
        ax.set(title="Lugbara test words never seen in training (lower is better)",
               ylabel="% of Lugbara word tokens")
        ax.legend(loc="upper left")
        ax.grid(axis="x", visible=False)
        eng = u["english"]
        paths.append(_save(fig, out_dir / "data_unseen_words.png",
                           note="A model cannot write a word it never saw. English, same order: "
                                + ", ".join(f"{100 * eng[k]:.1f}%" for k in ("salt>salt", "local>salt", "salt>local", "local>local"))
                                + "."))
    return paths


# ---------------------------------------------------------------- grid search
def grid_figure(grid, out_path, model_name):
    _style()
    res = grid["results"]
    lrs = sorted({r["lr"] for r in res})
    bss = sorted({r["batch_size"] for r in res})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key, title, fmt in ((axes[0], "val_loss", "Validation loss (lower is better)", "{:.2f}"),
                                (axes[1], "val_bleu", "Validation BLEU (higher is better)", "{:.2f}")):
        grid_v = [[next((r[key] for r in res if r["lr"] == lr and r["batch_size"] == bs), math.nan)
                   for lr in lrs] for bs in bss]
        flat = [v for row in grid_v for v in row if not math.isnan(v)]
        lo, hi = min(flat), max(flat)
        im = ax.imshow(grid_v, cmap=BLUES if key == "val_bleu" else BLUES.reversed(), aspect="auto",
                       vmin=lo, vmax=hi if hi > lo else lo + 1e-9)
        for i, row in enumerate(grid_v):
            for j, v in enumerate(row):
                if math.isnan(v):
                    continue
                t = (v - lo) / (hi - lo) if hi > lo else 0.5
                dark = t > 0.55 if key == "val_bleu" else t < 0.45
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9,
                        color="#ffffff" if dark else INK)
        best = grid["best"]
        bj, bi = lrs.index(best["lr"]), bss.index(best["batch_size"])
        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=2))
        ax.set_xticks(range(len(lrs)), [f"{lr:g}" for lr in lrs])
        ax.set_yticks(range(len(bss)), [str(b) for b in bss])
        ax.set(xlabel="Learning rate", ylabel="Batch size", title=title)
        ax.grid(False)
    fig.suptitle(f"{model_name}: learning rate × batch size grid search", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    return _save(fig, out_path, note=f"Black box = chosen setting (lr {best['lr']:g}, batch {best['batch_size']}). "
                                     "Each cell = one short training run scored on the validation set.")


# ---------------------------------------------------------------- training curves
def training_figure(train_log, out_path, model_name):
    _style()
    hist = train_log["history"]
    tr = [(h["epoch"], h["loss"]) for h in hist if "loss" in h and "eval_loss" not in h]
    ev = [(h["epoch"], h["eval_loss"]) for h in hist if "eval_loss" in h]
    fig, ax = plt.subplots(figsize=(8, 4))
    if tr:
        ax.plot([e for e, _ in tr], [l for _, l in tr], color=SERIES[0], label="Training loss", linewidth=1.5)
    if ev:
        ax.plot([e for e, _ in ev], [l for _, l in ev], color=SERIES[1], label="Validation loss",
                marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
        be, bl = min(ev, key=lambda t: t[1])
        ax.annotate(f"best {bl:.2f} (pass {be:.0f})", (be, bl), xytext=(0, -16), textcoords="offset points",
                    ha="center", fontsize=8, color=INK2)
    if tr:   # the first logged steps are very high; keep the useful range readable
        ys = [l for _, l in tr] + [l for _, l in ev]
        ax.set_ylim(min(ys) * 0.9, min(max(ys), sorted(ys)[int(len(ys) * 0.97)] * 1.15))
    ax.set(title=f"{model_name}: training", xlabel="Pass through the training data (epoch)", ylabel="Loss")
    ax.legend(loc="upper right")
    return _save(fig, out_path, note=f"Early stopping on validation loss; {train_log.get('global_step', 0):,} steps, "
                                     f"{train_log.get('minutes', '?')} min on a Kaggle T4.")


# ---------------------------------------------------------------- test scores
def scores_figure(scores, out_path, model_name, thesis=None, note=None):
    """BLEU and chrF per direction (0-100, same axis), thesis BLEU as a reference tick; TER separately."""
    _style()
    dirs = list(scores)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    w = 0.34
    for i, (key, label) in enumerate((("bleu", "BLEU"), ("chrf", "chrF"))):
        xs = [j + (i - 0.5) * w for j in range(len(dirs))]
        bars = ax.bar(xs, [scores[d][key] for d in dirs], width=w, color=SERIES[i], label=label,
                      edgecolor=SURFACE, linewidth=2)
        _bar_labels(ax, bars)
    if thesis:
        first = True
        for j, d in enumerate(dirs):
            ref = thesis.get(d)
            if not ref:
                continue
            name, v = max(ref.items(), key=lambda kv: kv[1])
            ax.plot([j - 0.6, j - 0.05], [v, v], color=THESIS_INK, linewidth=2, linestyle=(0, (3, 2)),
                    label="Thesis BLEU (own corpus)" if first else None)
            ax.annotate(f"thesis {v:.1f}", (j - 0.6, v), xytext=(0, 3), textcoords="offset points",
                        fontsize=8, color=INK2)
            first = False
    tops = [scores[d][k] for d in dirs for k in ("bleu", "chrf")]
    if thesis:
        tops += [max(thesis[d].values()) for d in dirs if thesis.get(d)]
    ax.set_ylim(0, max(tops + [1]) * 1.3)
    ax.set_xticks(range(len(dirs)), [_pretty(d) for d in dirs])
    ax.set(title=f"{model_name}: test scores (higher is better)", ylabel="Score (0–100)")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    ax2 = axes[1]
    bars = ax2.bar(range(len(dirs)), [scores[d]["ter"] for d in dirs], width=0.5, color=SERIES[0],
                   edgecolor=SURFACE, linewidth=2)
    _bar_labels(ax2, bars)
    ax2.set_xticks(range(len(dirs)), [_pretty(d) for d in dirs], rotation=15)
    ax2.set(title="TER (lower is better)", ylabel="Edits per 100 reference words")
    ax2.grid(axis="x", visible=False)
    n = sum(s["n"] for s in scores.values())
    return _save(fig, out_path, note=f"{n:,} " + (note or "test sentence pairs from SALT. Thesis scores come from a "
                                                   "different corpus and are a reference point, not a like-for-like comparison."))


def length_figure(outputs_by_dir, out_path, model_name):
    """Sentence-level chrF by source length (from test_outputs.*.jsonl)."""
    import sacrebleu
    _style()
    bins = [(1, 5), (6, 8), (9, 11), (12, 15), (16, 20), (21, 200)]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (d, rows) in enumerate(outputs_by_dir.items()):
        ys, xs = [], []
        for k, (a, b) in enumerate(bins):
            sel = [r for r in rows if a <= len(r["src"].split(" ", 1)[-1].split()) <= b]
            if len(sel) < 10:
                continue
            ys.append(sum(sacrebleu.sentence_chrf(r["hyp"], [r["ref"]]).score for r in sel) / len(sel))
            xs.append(k)
        ax.plot(xs, ys, color=SERIES[i % len(SERIES)], marker="o", markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=2, label=_pretty(d))
        if ys:
            ax.annotate(_pretty(d), (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        fontsize=8, color=INK2, va="center")
    ax.set_xticks(range(len(bins)), [f"{a}–{b}" if b < 200 else f"{a}+" for a, b in bins])
    ax.set(title=f"{model_name}: quality by sentence length", xlabel="Source sentence length (words)",
           ylabel="Average sentence chrF")
    ax.legend(loc="upper right")
    return _save(fig, out_path, note="Length bins with fewer than 10 test sentences are left out.")


def compare_figure(score_sets, out_path, metric="bleu", title=None, note=None):
    """One metric per direction across models (≤ 3 models: the validated all-pairs slots)."""
    _style()
    names = list(score_sets)[:3]
    dirs = sorted({d for s in score_sets.values() for d in s},
                  key=lambda d: (d.split("-")[0] != "english", d))
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.8 / max(1, len(names))
    for i, name in enumerate(names):
        xs, vs = [], []
        for j, d in enumerate(dirs):
            if d in score_sets[name]:
                xs.append(j + (i - (len(names) - 1) / 2) * w)
                vs.append(score_sets[name][d][metric])
        bars = ax.bar(xs, vs, width=w, color=SERIES[i], label=name, edgecolor=SURFACE, linewidth=2)
        _bar_labels(ax, bars)
    ax.set_xticks(range(len(dirs)), [_pretty(d) for d in dirs])
    ax.set(title=title or f"Test {metric.upper()} by model and direction", ylabel=metric.upper())
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    return _save(fig, out_path, note=note or "Same SALT test split for every model.")


def dataset_figure(comparison, out_path, model_name, subtitle=None):
    """Cross-test (compare.json): bars = training corpus, groups = test split; BLEU and chrF per direction."""
    _style()
    table = comparison["table"]
    dirs = list(table)
    fig, axes = plt.subplots(len(dirs), 2, figsize=(10, 3.9 * len(dirs)), squeeze=False)
    w = 0.36
    for r, d in enumerate(dirs):
        cells = table[d]
        for c, (key, label) in enumerate((("bleu", "BLEU"), ("chrf", "chrF"))):
            ax = axes[r][c]
            for i, tr in enumerate(("salt", "local")):
                xs, vs = [], []
                for j, te in enumerate(("salt", "local")):
                    if f"{tr}>{te}" in cells:
                        xs.append(j + (i - 0.5) * w)
                        vs.append(cells[f"{tr}>{te}"][key])
                bars = ax.bar(xs, vs, width=w, color=DS_COLOR[tr], label=f"Trained on {DS_NAME[tr]}",
                              edgecolor=SURFACE, linewidth=2)
                _bar_labels(ax, bars)
            ax.set_xticks(range(2), ["SALT test split", "Local test split"])
            ax.set_ylim(0, max([v[key] for v in cells.values()] + [1]) * 1.35)
            ax.set(title=f"{_pretty(d)}: {label}", ylabel=f"{label} (higher is better)")
            ax.grid(axis="x", visible=False)
            if r == 0 and c == 0:
                ax.legend(loc="upper left")
    fig.suptitle(f"{model_name}: trained on SALT vs trained on the local corpus"
                 + (f" ({subtitle})" if subtitle else ""), x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    return _save(fig, out_path, note="Same model and settings, trained once on each corpus; each bar is one model "
                                     "scored on one test split.")


# ---------------------------------------------------------------- notebook helper
def show(paths):
    """Display PNGs inline in a notebook."""
    from IPython.display import Image, display
    for p in paths:
        if Path(p).exists():
            display(Image(filename=str(p)))


def model_figures(model_dir_out, model_name, thesis=None, test_note=None):
    """All per-model figures from a model's out/ folder (grid.json, train_log.json, scores.json, outputs)."""
    d = Path(model_dir_out)
    figs = d / "figures"
    paths = []
    if (d / "grid.json").exists():
        paths.append(grid_figure(json.loads((d / "grid.json").read_text()), figs / "grid_search.png", model_name))
    if (d / "train_log.json").exists():
        paths.append(training_figure(json.loads((d / "train_log.json").read_text()), figs / "training.png", model_name))
    if (d / "scores.json").exists():
        paths.append(scores_figure(json.loads((d / "scores.json").read_text()), figs / "test_scores.png",
                                   model_name, thesis, note=test_note))
    outs = {p.name[len("test_outputs."):-len(".jsonl")]: [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
            for p in sorted(d.glob("test_outputs.*.jsonl"))}
    if outs:
        paths.append(length_figure(outs, figs / "quality_by_length.png", model_name))
    return paths


# ---------------------------------------------------------------- report across all published models
REPORT_REPOS = {
    "XLM-R": "ckuma/lugbara-multilingual-translator",
    "mT5": "ckuma/lugbara-english-translator",
    "XLM-R + mT5 fusion": "ckuma/lugbara-ensemble-translator",
}


def report(out_dir="report", repos=None, thesis=None):
    """Download the published results of every model and draw per-model and comparison figures."""
    from huggingface_hub import HfApi, hf_hub_download
    repos = repos or REPORT_REPOS
    out_dir = Path(out_dir)
    api, paths, scores, analysis_done = HfApi(), [], {}, False
    for name, repo in repos.items():
        try:
            files = set(api.list_repo_files(repo))
        except Exception as e:
            print(f"{name}: {repo} not available yet ({type(e).__name__})")
            continue
        local = out_dir / repo.replace("/", "__")
        local.mkdir(parents=True, exist_ok=True)
        for f in ("scores.json", "train_log.json", "grid.json", "lugbara_analysis.json"):
            if f in files:
                Path(hf_hub_download(repo, f, local_dir=str(local)))
        for f in files:
            if f.startswith("test_outputs/") and f.endswith(".jsonl"):
                p = Path(hf_hub_download(repo, f, local_dir=str(local)))
                (local / p.name).write_bytes(p.read_bytes())
        if (local / "lugbara_analysis.json").exists() and not analysis_done:
            paths += data_figures(json.loads((local / "lugbara_analysis.json").read_text(encoding="utf-8")),
                                  out_dir / "figures")
            analysis_done = True
        if (local / "scores.json").exists():
            scores[name] = json.loads((local / "scores.json").read_text())
        (out_dir / "figures").mkdir(parents=True, exist_ok=True)
        prefix = name.split(" ")[0].lower().replace("-", "")
        for p in model_figures(local, name, thesis):
            dest = out_dir / "figures" / f"{prefix}_{p.name}"
            p.replace(dest)
            paths.append(dest)
    if len(scores) > 1:
        for metric in ("bleu", "chrf", "ter"):
            paths.append(compare_figure(scores, out_dir / "figures" / f"compare_{metric}.png", metric,
                                        f"Test {metric.upper() if metric != 'chrf' else 'chrF'} by model and direction"
                                        + (" (lower is better)" if metric == "ter" else "")))
    return paths
