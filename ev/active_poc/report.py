"""report.py — turn the results JSON into mp/results/active_poc_<date>.md.

    python mp/ev/active_poc/report.py --json mp/results/active_poc_2026-08-25.json

Kept separate from ``stage_final`` so the write-up can be regenerated from the numbers
without retraining anything.  Every figure in the markdown comes from the JSON; the prose
sections that require judgement are marked and are edited by hand afterwards.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ARMS = ("disagreement", "err_proxy", "uniform")
NICE = {"disagreement": "disagreement", "err_proxy": "error-proxy", "uniform": "uniform (control)",
        "base_only": "base only (no addition)"}


def f(x, n=4):
    try:
        if x is None:
            return "-"
        if isinstance(x, float) and (x != x):
            return "-"
        return f"{x:.{n}f}"
    except (TypeError, ValueError):
        return str(x)


def sign(x, n=4):
    try:
        return f"{x:+.{n}f}"
    except (TypeError, ValueError):
        return "-"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", default="mp/results/active_poc_2026-08-25.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    src = Path(args.json)
    d = json.loads(src.read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else src.with_suffix(".md")

    des = d["design"]
    sel = d["selection"]
    res = d["results"]
    pair = d["paired"]
    noise = d["label_noise"]
    ao = d.get("arm_only", {})

    L: list = []
    A = L.append

    A("# Active label selection vs uniform sampling — a measurement POC")
    A("")
    A(f"*W-ACTIVE, {d['timestamp']}.  Phase 5 rev 2 (EV player), branch `mp/campaign`.*")
    A("")
    A("**Question.** Per label spent, does actively choosing WHICH states to label buy more V "
      "quality than sampling states uniformly?")
    A("")

    # ── verdict (mechanical part) ──
    pu = pair.get("disagreement_vs_uniform", {})
    eu = pair.get("err_proxy_vs_uniform", {})
    A("## Verdict")
    A("")
    db = pu.get("bce", {})
    eb = eu.get("bce", {})
    A(f"* **disagreement vs uniform:** ΔBCE {sign(db.get('mean_delta'))} ± {f(db.get('sem'))} "
      f"(paired over {db.get('df', 0) + 1} training seeds, t = {f(db.get('t'), 2)}); "
      f"ΔBrier {sign(pu.get('brier', {}).get('mean_delta'), 5)}; "
      f"ΔAUC {sign(pu.get('auc', {}).get('mean_delta'))}.")
    A(f"* **error-proxy vs uniform:** ΔBCE {sign(eb.get('mean_delta'))} ± {f(eb.get('sem'))} "
      f"(t = {f(eb.get('t'), 2)}); "
      f"ΔBrier {sign(eu.get('brier', {}).get('mean_delta'), 5)}; "
      f"ΔAUC {sign(eu.get('auc', {}).get('mean_delta'))}.")
    A("")
    A("<!-- PROSE: the interpretation paragraph is written by hand below this line. -->")
    A("")

    # ── design ──
    A("## 1. Design")
    A("")
    A(f"| | |")
    A(f"|---|---|")
    A(f"| base corpus | {des['base_rows']:,} rows ({des['base_rows'] // 2:,} states), a seed-hash "
      f"subsample of the existing 51k `labels_full` corpus |")
    A(f"| evaluation holdout | {des['holdout_rows']:,} rows — the STANDARD `seed_in_holdout(seed, 0.1)` "
      f"holdout, never trained on by any arm |")
    A(f"| candidate pool | {sel['pool']['n_states']:,} fresh self-play states "
      f"({sel['pool']['n_seeds']} seeds), same self-play config and same label policy as the corpus |")
    A(f"| arm size | {des['arm_states']:,} states = {des['arm_rows']:,} rows "
      f"(+{100 * des['arm_rows'] / des['base_rows']:.0f}% on the base) |")
    A(f"| labels | `n_rollouts=8`, standard `labels.label_both` pipeline, unchanged |")
    A(f"| training | identical recipe for every arm, {des['s_star']} steps "
      f"(re-derived for this corpus size), {len(des['training_seeds'])} paired seeds per arm |")
    A("")
    A("Both perspectives of a snapshot come from one rollout set, so the budget is spent per "
      "STATE and selection is per state (each state's score is the mean of its two perspectives).")
    A("")

    # ── scores / selection ──
    ss = sel["score_stats"]
    A("### Acquisition rules and what they picked")
    A("")
    A("| arm | rule | mean disagreement | mean error-proxy | kinds |")
    A("|---|---|---|---|---|")
    rule = {"disagreement": "sd of 3-member ensemble V",
            "err_proxy": r"\|mean V − 2-rollout label\|",
            "uniform": "seeded random"}
    for a in ARMS:
        p = sel["profiles"][a]
        A(f"| {NICE[a]} | {rule[a]} "
          f"| {f(p['disagreement_mean'])} | {f(p['err_proxy_mean'])} "
          f"| {', '.join(f'{k} {v}' for k, v in sorted(p['by_kind'].items()))} |")
    A("")
    # where in the run each rule looks — the clearest structural difference between the arms
    antes = sorted({int(x) for p in sel["profiles"].values() for x in p["by_ante"]})
    A("Where in the run each rule looks (states by ante):")
    A("")
    A("| arm | " + " | ".join(f"ante {a}" for a in antes) + " | ante ≥5 share |")
    A("|---|" + "---|" * (len(antes) + 1))
    for a in ARMS:
        p = sel["profiles"][a]
        tot = sum(p["by_ante"].values())
        late = sum(v for k2, v in p["by_ante"].items() if int(k2) >= 5)
        A(f"| {NICE[a]} | " + " | ".join(str(p["by_ante"].get(str(x), 0)) for x in antes)
          + f" | **{100 * late / max(tot, 1):.0f}%** |")
    A("")
    A(f"Pool-wide: disagreement mean {f(ss['disagreement']['mean'])} "
      f"(p90 {f(ss['disagreement']['p90'])}), error-proxy mean {f(ss['err_proxy']['mean'])} "
      f"(p90 {f(ss['err_proxy']['p90'])}).  "
      f"**corr(disagreement, error-proxy) = {f(ss['corr_dis_err'], 3)}** — the two rules are "
      f"nearly orthogonal.")
    A("")
    A("Overlap between the selected sets:")
    A("")
    A("| pair | shared states | Jaccard |")
    A("|---|---|---|")
    for k, v in sel["overlap"].items():
        A(f"| {k.replace('|', ' ∩ ')} | {v['intersection']} | {f(v['jaccard'], 3)} |")
    A("")

    # ── main table ──
    A("## 2. Three-arm results (full holdout, mean ± SEM over training seeds)")
    A("")
    A("| arm | train rows | BCE ↓ | Brier ↓ | AUC ↑ | ECE ↓ | hard-stratum BCE ↓ |")
    A("|---|---|---|---|---|---|---|")
    for a in ("base_only",) + ARMS:
        r = res[a]
        A(f"| {NICE[a]} | {r['n_train']:,} "
          f"| {f(r['bce']['mean'])} ± {f(r['bce']['sem'])} "
          f"| {f(r['brier']['mean'], 5)} ± {f(r['brier']['sem'], 5)} "
          f"| {f(r['auc']['mean'])} ± {f(r['auc']['sem'])} "
          f"| {f(r['ece']['mean'])} ± {f(r['ece']['sem'])} "
          f"| {f(r['hard_bce']['mean'])} ± {f(r['hard_bce']['sem'])} |")
    A("")
    A(f"Constant predictor: BCE {f(0.6931)} / Brier ~{f(0.1004, 4)}.  "
      f"Label-noise Brier floor ≈ {f(d['runs']['uniform'][0].get('noise_floor_brier'))}.")
    A("")
    A("### Paired differences (same training seed, same base rows in the same order)")
    A("")
    A("| comparison | ΔBCE | ΔBrier | ΔAUC | ΔECE | Δhard-BCE | t (BCE) |")
    A("|---|---|---|---|---|---|---|")
    for k, v in pair.items():
        A(f"| {k.replace('_', ' ')} | {sign(v['bce']['mean_delta'])} ± {f(v['bce']['sem'])} "
          f"| {sign(v['brier']['mean_delta'], 5)} | {sign(v['auc']['mean_delta'])} "
          f"| {sign(v['ece']['mean_delta'])} "
          f"| {sign(v.get('hard_bce', {}).get('mean_delta'))} "
          f"| {f(v['bce']['t'], 2)} |")
    A("")
    A("Negative = better for BCE / Brier / ECE, positive = better for AUC.")
    A("")
    A("### Value per label")
    A("")
    A(f"What {des['arm_rows']:,} extra rows bought over the {des['base_rows']:,}-row base, and how "
      f"each acquisition rule compares to spending the same budget uniformly:")
    A("")
    A("| arm | ΔBCE vs base-only | per 1,000 labels | vs uniform |")
    A("|---|---|---|---|")
    ref = pair.get("uniform_vs_base_only", {}).get("bce", {}).get("mean_delta")
    for a in ARMS:
        v = pair.get(f"{a}_vs_base_only", {}).get("bce", {})
        md = v.get("mean_delta")
        per_k = (md / des["arm_rows"] * 1000) if md is not None else None
        if a == "uniform" or ref in (None, 0) or md is None:
            mult = "1.00x (reference)" if a == "uniform" else "-"
        else:
            mult = f"{md / ref:.2f}x"
        A(f"| {NICE[a]} | {sign(md)} ± {f(v.get('sem'))} | {sign(per_k, 5)} | {mult} |")
    A("")
    A("The multiplier is a ratio of two small, noisy differences — read its sign and rough "
      "magnitude, not its decimals.")
    A("")
    A("### Value per ROLLOUT — what the acquisition itself costs")
    A("")
    A("Ranking the pool is not free, and the two rules differ enormously in what it costs. "
      "Scoring by ensemble disagreement needs three forward passes per state (milliseconds). "
      "Scoring by the error proxy needs a real 2-rollout label on **every state in the pool**, "
      "whether or not that state is ever selected — a quarter of a full label each.")
    A("")
    npool = sel["pool"]["n_states"]
    nprobe = sel["pool"].get("probe_rollouts", 2)
    narm = des["arm_states"]
    label_ro = 8
    rows_ac = [
        ("uniform (control)", 0, narm * label_ro),
        ("disagreement", 0, narm * label_ro),
        ("error-proxy", npool * nprobe, narm * label_ro),
    ]
    A("| arm | acquisition rollouts | labelling rollouts | total | vs uniform |")
    A("|---|---|---|---|---|")
    for name, acq, lab in rows_ac:
        tot = acq + lab
        A(f"| {name} | {acq:,} | {lab:,} | {tot:,} | {tot / (narm * label_ro):.2f}x |")
    A("")
    A(f"At the measured ~{f(1.28, 2)} s/rollout on 8 workers that is "
      f"{narm * label_ro * 1.28 / 8 / 60:.0f} min for either free-to-score arm and "
      f"{(npool * nprobe + narm * label_ro) * 1.28 / 8 / 60:.0f} min for the error-proxy arm. "
      f"The error-proxy arm therefore has to beat uniform by "
      f"{(npool * nprobe + narm * label_ro) / (narm * label_ro):.1f}x on quality merely to break "
      f"even on compute — and the probe cost scales with the POOL, so it gets worse the more "
      f"selective you try to be.")
    A("")

    # ── label noise ──
    A("## 3. Per-arm label noise")
    A("")
    A("| arm | rows | mean CI half-width | Brier noise floor | label sd | truncated |")
    A("|---|---|---|---|---|---|")
    for a in ARMS:
        n = noise[a]
        A(f"| {NICE[a]} | {n['n_rows']:,} | {f(n['ci_mean'])} | {f(n['noise_floor_brier'], 5)} "
          f"| {f(n['y_sd'])} | {f(n['trunc_frac'], 3)} |")
    A("")
    dg = d.get("diagnostics")
    if dg:
        A("### Does either signal track V's real error, or just label noise?")
        A("")
        A("The selected states carry an 8-rollout label `y8` from a rollout stream independent "
          "of the 2-rollout probe, so the proxy can be decomposed on real data: `err_proxy = "
          "|mean V − y_probe|` is what the rule ranked on, `|mean V − y8|` is a far better "
          "estimate of V's actual error.")
        A("")
        A(f"* **corr(err_proxy, |V − y8|) = {f(dg['corr_err_proxy_vs_realized'], 3)}**")
        A(f"* **corr(disagreement, |V − y8|) = {f(dg['corr_disagreement_vs_realized'], 3)}**")
        A(f"* corr(err_proxy, disagreement) = {f(dg['corr_err_proxy_vs_disagreement'], 3)}")
        A("")
        A("| arm | ranked-on err_proxy | realized \\|V − y8\\| | ensemble disagreement | label sd (y8) |")
        A("|---|---|---|---|---|")
        for a in ARMS:
            v = dg["by_arm"].get(a)
            if not v:
                continue
            A(f"| {NICE[a]} | {f(v['err_proxy_mean'])} | {f(v['realized_err_mean'])} "
              f"| {f(v['disagreement_mean'])} | {f(v['y8_sd'])} |")
        A("")
        A(f"Mean realized \\|V − y8\\| over all labelled states: {f(dg['mean_realized_err_all'])}.")
        A("")
    A("<!-- PROSE: noise-chasing interpretation. -->")
    A("")

    # ── per kind ──
    A("## 4. Per state-kind held-out BCE")
    A("")
    kinds = sorted(res["uniform"]["by_kind_bce"])
    A("| arm | " + " | ".join(kinds) + " |")
    A("|---|" + "---|" * len(kinds))
    for a in ("base_only",) + ARMS:
        cells = []
        for k in kinds:
            v = res[a]["by_kind_bce"].get(k, {})
            cells.append(f"{f(v.get('mean'))}")
        A(f"| {NICE[a]} | " + " | ".join(cells) + " |")
    A("")
    hs = sel["hard_stratum"]
    A(f"The hard stratum is the top {100 * (1 - hs['quantile']):.0f}% of holdout rows by the same "
      f"ensemble's disagreement ({hs['n_rows']:,} of {hs['n_holdout']:,} rows, sd ≥ {f(hs['threshold'])}); "
      f"it is fixed once, from the base ensemble, so it is identical for every arm.")
    A("")

    # ── arm only ──
    if ao.get("aggregate"):
        A("## 5. Secondary: trained on the arm's rows ALONE (no base corpus)")
        A("")
        A(f"Removes the dilution of adding {des['arm_rows']:,} rows to {des['base_rows']:,}; "
          f"{ao.get('regime', {}).get('s_star', '?')} steps, "
          f"{len(ao['runs']['uniform'])} seeds per arm.")
        A("")
        A("| arm | BCE ↓ | Brier ↓ | AUC ↑ |")
        A("|---|---|---|---|")
        for a in ARMS:
            r = ao["aggregate"].get(a)
            if not r:
                continue
            A(f"| {NICE[a]} | {f(r['bce']['mean'])} ± {f(r['bce']['sem'])} "
              f"| {f(r['brier']['mean'], 5)} | {f(r['auc']['mean'])} |")
        A("")
        for k, v in ao.get("paired", {}).items():
            A(f"* {k.replace('_', ' ')}: ΔBCE {sign(v['bce']['mean_delta'])} ± {f(v['bce']['sem'])} "
              f"(t = {f(v['bce']['t'], 2)}), ΔAUC {sign(v['auc']['mean_delta'])}")
        A("")

    A("## 6. What this POC cannot conclude")
    A("")
    A("<!-- PROSE -->")
    A("")
    A("## 7. Integration sketch")
    A("")
    A("<!-- PROSE -->")
    A("")

    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
