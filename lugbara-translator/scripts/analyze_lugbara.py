"""Lugbara language analysis of the SALT corpus and of the translation model.

Writes analysis/lugbara_analysis.json and LUGBARA_ANALYSIS.md (uploaded with the model).
Everything here is measured from the data; the short background section is cited prose.
"""
import json
import re
import unicodedata
from collections import Counter
from statistics import mean, median

from transformers import AutoTokenizer

from common import ANALYSIS, DATA, LANGS, OUTPUT_MODEL, ROOT, SOURCE_MODEL, tag

WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*|'[^\W\d_]+", re.UNICODE)
VOWELS = set("aeiouAEIOU")


def words(t):
    return WORD.findall(t.lower())


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def table(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def corpus_stats(rows, key):
    toks = [w for r in rows if r.get(key) for w in words(r[key])]
    lens = [len(words(r[key])) for r in rows if r.get(key)]
    c = Counter(toks)
    return {
        "sentences": len(lens), "tokens": len(toks), "types": len(c),
        "ttr": round(len(c) / max(1, len(toks)), 4),
        "hapax_share": round(sum(1 for v in c.values() if v == 1) / max(1, len(c)), 3),
        "mean_len": round(mean(lens), 2), "median_len": median(lens),
        "vowel_final_share": round(sum(v for w, v in c.items() if w[-1] in VOWELS) / max(1, len(toks)), 3),
        "counter": c,
    }


def main():
    rows = load_jsonl(DATA / "salt_lugbara_all.jsonl")
    stats = json.loads((DATA / "stats.json").read_text())
    res = {"coverage": stats.get("coverage")}

    # --- corpus statistics, same sentences across languages --------------------------
    per = {"english": corpus_stats(rows, "eng"), "lugbara": corpus_stats(rows, "lgg")}
    for lang in LANGS:
        if lang != "lugbara":
            per[lang] = corpus_stats(rows, lang)
    lg = per["lugbara"]
    res["corpus"] = {k: {kk: vv for kk, vv in v.items() if kk != "counter"} for k, v in per.items()}
    res["top_words"] = lg["counter"].most_common(40)

    # --- orthography ------------------------------------------------------------------
    raw = "".join(r["raw"] or "" for r in rows)
    norm = "".join(r["lgg"] for r in rows)
    letters = Counter(ch for ch in norm.lower() if ch.isalpha())
    res["letters"] = letters.most_common()
    res["non_ascii_letters"] = {ch: n for ch, n in letters.items() if ord(ch) > 127}
    apos_raw = Counter(ch for ch in raw if ch in "'’‘ʼʻ`´′")
    res["apostrophe_variants_raw"] = {f"U+{ord(k):04X} {unicodedata.name(k, '?')}": v for k, v in apos_raw.items()}
    after_apos = Counter(m.group(1) for m in re.finditer(r"'([a-z])", norm.lower()))
    res["apostrophe_followed_by"] = after_apos.most_common()
    lw = lg["counter"]
    res["words_with_apostrophe_share"] = round(sum(v for w, v in lw.items() if "'" in w) / max(1, lg["tokens"]), 3)
    onsets = Counter()
    for w, v in lw.items():
        m = re.match(r"'?([^aeiou']+)", w)
        if m and len(m.group(1)) >= 2:
            onsets[m.group(1)] += v
    res["word_initial_clusters"] = onsets.most_common(20)

    # --- relatedness: shared vocabulary with the other languages -----------------------
    top_lg = {w for w, _ in lw.most_common(2000)}
    overlap = {}
    for lang, s in per.items():
        if lang in ("lugbara",):
            continue
        top = {w for w, _ in s["counter"].most_common(2000)}
        overlap[lang] = round(len(top_lg & top) / 2000, 3)
    bantu = ["rutooro", "luganda", "runyankore"]
    for a in bantu:
        for b in bantu:
            if a < b:
                ta = {w for w, _ in per[a]["counter"].most_common(2000)}
                tb = {w for w, _ in per[b]["counter"].most_common(2000)}
                overlap[f"{a}~{b}"] = round(len(ta & tb) / 2000, 3)
    res["top2000_overlap_with_lugbara"] = overlap

    # --- tokenizer fit ------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(SOURCE_MODEL)
    fert = {}
    sample = rows[:3000]
    for lang, key in [("english", "eng"), ("lugbara", "lgg")] + [(l, l) for l in LANGS if l != "lugbara"]:
        texts = [r[key] for r in sample if r.get(key)]
        if not texts:
            continue
        enc = tok(text_target=texts, add_special_tokens=False)["input_ids"]
        n_pieces = sum(len(e) for e in enc)
        n_words = sum(len(t.split()) for t in texts)
        n_unk = sum(e.count(tok.unk_token_id) for e in enc)
        fert[lang] = {"pieces_per_word": round(n_pieces / n_words, 2), "unk_per_1000": round(1000 * n_unk / n_pieces, 2)}
    res["tokenizer"] = fert

    # --- language-tag check in the source model ----------------------------------------
    ids = {l: tok(f"{tag(l)} The market opens early.")["input_ids"] for l in LANGS}
    res["source_tag_ids"] = {l: v[:2] for l, v in ids.items()}
    res["source_tags_all_unk"] = all(v[0] == tok.unk_token_id for v in ids.values())

    # --- model scores ---------------------------------------------------------------------
    for name in ("source", "lugbara"):
        p = ANALYSIS / f"scores.{name}.json"
        if p.exists():
            res[f"scores_{name}"] = json.loads(p.read_text())
    samples = []
    po, pn = ANALYSIS / "outputs.source.lugbara.jsonl", ANALYSIS / "outputs.lugbara.lugbara.jsonl"
    if po.exists() and pn.exists():
        for a, b in list(zip(load_jsonl(po), load_jsonl(pn)))[:10]:
            samples.append({"en": a["src"], "ref": a["ref"], "before": a["hyp"], "after": b["hyp"]})
    res["samples"] = samples

    (ANALYSIS / "lugbara_analysis.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    (ROOT / "LUGBARA_ANALYSIS.md").write_text(render(res), encoding="utf-8")
    print("wrote LUGBARA_ANALYSIS.md")


def render(r):
    c = r["corpus"]
    L = ["# Lugbara language analysis", "",
         "Measured on the Lugbara side of the Sunbird SALT corpus (`text-all`, all splits) and on this "
         "model's outputs for the SALT test split. Figures are computed by `scripts/analyze_lugbara.py`.", "",
         "## Background", "",
         "Lugbara (ISO 639-3 `lgg`) is a Central Sudanic language of the West Nile region of Uganda "
         "(around Arua, Maracha and Terego) and neighbouring Ituri in the DR Congo. It is not "
         "related to the Bantu languages (Rutooro, Luganda, Runyankore) or to Nilotic Acholi that the "
         "Rutooro model was built on, and it is not a target language of the base model "
         "`Helsinki-NLP/opus-mt-en-mul`. Lugbara is tonal, but tone is not marked in everyday "
         "spelling, and an apostrophe marks implosive or glottalised consonants (`'b`, `'d`, `'y`).", ""]
    if r.get("coverage"):
        L += ["## How much Lugbara SALT contains", "",
              table([[s, v["rows"]] + [f"{v[l]} ({100 * v[l] / max(1, v['rows']):.1f}%)" for l in LANGS]
                     for s, v in r["coverage"].items()], ["Split", "Rows"] + [l.title() for l in LANGS]), ""]
    L += ["## Corpus statistics (same sentences in every language)", "",
          table([[k.title(), v["sentences"], v["tokens"], v["types"], v["ttr"], v["mean_len"],
                  v["hapax_share"], v["vowel_final_share"]] for k, v in c.items()],
                ["Language", "Sentences", "Words", "Distinct words", "Type/token", "Words per sentence",
                 "Share of words seen once", "Word tokens ending in a vowel"]), "",
          f"Lugbara sentences average **{c['lugbara']['mean_len']} words** against "
          f"{c['english']['mean_len']} for the same English sentences "
          f"(ratio {c['lugbara']['mean_len'] / c['english']['mean_len']:.2f}). Its distinct-word ratio "
          f"({c['lugbara']['ttr']}) is lower than the Bantu languages' because Lugbara is mostly "
          "analytic: short, separate words where Bantu languages build long words from prefixes.", "",
          "### Most frequent Lugbara words", "",
          ", ".join(f"`{w}` ({n})" for w, n in r["top_words"][:40]), "",
          "## Spelling and characters", "",
          f"* Share of word tokens containing an apostrophe: **{100 * r['words_with_apostrophe_share']:.1f}%**",
          "* Apostrophe followed by: " + ", ".join(f"`'{k}` {v}" for k, v in r["apostrophe_followed_by"][:8]),
          "* Apostrophe characters in the raw text (all folded to `'` before training): "
          + (", ".join(f"{k}: {v}" for k, v in r["apostrophe_variants_raw"].items()) or "none"),
          "* Letters outside ASCII: " + (", ".join(f"`{k}` {v}" for k, v in r["non_ascii_letters"].items()) or "none"),
          "* Most common word-initial consonant clusters: "
          + ", ".join(f"`{k}` {v}" for k, v in r["word_initial_clusters"][:15]), "",
          "## Relationship to the model's other languages", "",
          "Share of the 2,000 most frequent words that two languages have in common (spelling-identical):", "",
          table([[k.replace("~", " vs ").title() if "~" in k else f"Lugbara vs {k.title()}", f"{100 * v:.1f}%"]
                 for k, v in r["top2000_overlap_with_lugbara"].items()], ["Pair", "Shared top-2000 words"]), "",
          "Lugbara shares almost no vocabulary with the Bantu languages, so nothing the Rutooro model "
          "learned about word forms transfers directly; it has to be taught Lugbara from parallel data.", "",
          "## Tokenizer fit", "",
          table([[k.title(), v["pieces_per_word"], v["unk_per_1000"]] for k, v in r["tokenizer"].items()],
                ["Language", "Sub-word pieces per word", "Unknown pieces per 1,000"]), "",
          "The Marian SentencePiece vocabulary covers Lugbara's Latin letters and apostrophe without "
          "unknown pieces, so the vocabulary was kept; only the language tags were added.", "",
          "## Language tags", "",
          f"In `{'mmuniversity/rutooro-multilingual-translator'}` the tags `>>rutooro<<`, `>>luganda<<`, "
          "`>>acholi<<` and `>>runyankore<<` are not in the vocabulary, so the tokenizer maps every one "
          f"of them to `<unk>` (all tags unknown: **{r['source_tags_all_unk']}**). The model therefore "
          "receives the same input whichever tag is used. This extension registers all five tags "
          "(including `>>lugbara<<`) as real tokens and trains with them, so the tag now selects the "
          "output language.", ""]
    if r.get("scores_source") or r.get("scores_lugbara"):
        s0, s1 = r.get("scores_source", {}), r.get("scores_lugbara", {})
        L += ["## Translation quality (SALT test split)", "",
              table([[l.title(), s1.get(l, s0.get(l, {})).get("n", ""),
                      s0.get(l, {}).get("bleu", "–"), s1.get(l, {}).get("bleu", "–"),
                      s0.get(l, {}).get("chrf", "–"), s1.get(l, {}).get("chrf", "–")] for l in LANGS],
                    ["Target", "Sentences", "BLEU before", "BLEU after", "chrF before", "chrF after"]), "",
              "*Before* is the Rutooro model as published; *after* is this model. Lugbara is scored on "
              "all 500 test sentences, the other languages on the first 200. Beam search 4.", ""]
    if r.get("samples"):
        L += ["## Sample Lugbara translations", "",
              table([[s["en"].replace(">>lugbara<< ", ""), s["ref"], s["before"], s["after"]] for s in r["samples"]],
                    ["English", "Reference (SALT)", "Rutooro model", "This model"]), ""]
    L += ["## Limits", "",
          "* SALT sentences are short and focused on agriculture, health and community life; other "
          "domains will be weaker.",
          "* Tone is not written, so homographs that differ only in tone stay ambiguous.",
          "* Scores use one reference translation per sentence; human review by Lugbara speakers is "
          "still needed before relying on the output.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
