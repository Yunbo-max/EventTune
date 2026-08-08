#!/usr/bin/env python3
"""Generate experiment table + figure for the ICASSP paper from disk metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BASE = Path("runs/all_datasets")
OUT = Path("paper")

EVENTS = [
    "turkey-earthquake",
    "la_palma-volcano",
    "beirut-explosion",
    "congo-volcano",
    "haiti-earthquake",
    "libya-flood",
    "marshall-wildfire",
    "bata-explosion",
    "noto-earthquake",
    "morocco-earthquake",
]


def agg(d):
    m = json.load(open(d))
    return m.get("aggregate", m)


def collect():
    rows = []
    for ev in EVENTS:
        se = BASE / ev / "source_eval" / "metrics.json"
        ke = BASE / ev / "kv_eval" / "metrics.json"
        if not (se.exists() and ke.exists()):
            continue
        s, k = agg(se), agg(ke)
        rows.append(
            {
                "event": ev,
                "s_f1": s["macro_f1"],
                "k_f1": k["macro_f1"],
                "s_nll": s["nll"],
                "k_nll": k["nll"],
                "s_bac": s.get("balanced_accuracy", float("nan")),
                "k_bac": k.get("balanced_accuracy", float("nan")),
                "s_mae": s.get("ordinal_mae", float("nan")),
                "k_mae": k.get("ordinal_mae", float("nan")),
                "count": k.get("count", 300),
            }
        )
    return rows


def fmt(x, sign=False):
    pre = "+" if sign and x > 0 else ""
    return f"{pre}{x:.3f}"


def write_tables(rows):
    OUT.mkdir(exist_ok=True)
    lines = []
    for r in rows:
        short = r["event"].split("-")[0]
        lines.append(
            f"{short} & {r['s_f1']:.3f} & {r['k_f1']:.3f} & {fmt(r['k_f1']-r['s_f1'],True)} "
            f"& {r['s_nll']:.2f} & {r['k_nll']:.2f} & {fmt(r['k_nll']-r['s_nll'],True)} \\\\"
        )
    if rows:
        df1 = np.mean([r["k_f1"] - r["s_f1"] for r in rows])
        dnll = np.mean([r["k_nll"] - r["s_nll"] for r in rows])
        s_f1 = np.mean([r["s_f1"] for r in rows])
        k_f1 = np.mean([r["k_f1"] for r in rows])
        n = len(rows)
        lines.append("\\midrule")
        lines.append(
            f"\\textit{{mean ({n} events)}} & {s_f1:.3f} & {k_f1:.3f} & {fmt(df1,True)} "
            f"& -- & -- & {fmt(dnll,True)} \\\\"
        )
        lines.append(
            f"\\textit{{wins (KV $>$ S)}} & \\multicolumn{{3}}{{c}}{{{sum(1 for r in rows if r['k_f1']>r['s_f1'])}/{n} (F1)}} "
            f"& \\multicolumn{{3}}{{c}}{{{sum(1 for r in rows if r['k_nll']<r['s_nll'])}/{n} (NLL)}} \\\\"
        )
    (OUT / "tables" / "main_body.tex").write_text("\n".join(lines) + "\n")


def plot(rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    short = [r["event"].split("-")[0] for r in rows]
    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.bar(x - w / 2, [r["s_f1"] for r in rows], w, label="Source (LoRA)")
    ax.bar(x + w / 2, [r["k_f1"] for r in rows], w, label="KV-TTT (proposed)")
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8, frameon=False)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "per_event_f1.pdf")
    fig.savefig(OUT / "figures" / "per_event_f1.png", dpi=200)


if __name__ == "__main__":
    rows = collect()
    for r in rows:
        r["short"] = r["event"].split("-")[0]
    write_tables(rows)
    plot(rows)
    print(f"wrote tables for {len(rows)} events")
    for r in rows:
        print(r["short"], f"F1 {r['s_f1']:.3f}->{r['k_f1']:.3f}", f"NLL {r['s_nll']:.2f}->{r['k_nll']:.2f}")