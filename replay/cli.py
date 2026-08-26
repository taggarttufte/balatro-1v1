"""
cli.py — command-line entry points for replay.

    python -m replay.cli show <file> <idx>                  # narrate one line
    python -m replay.cli verify <file>                       # replay every line, report mismatches
    python -m replay.cli filter <file> --tag X [--min-interest F]
    python -m replay.cli stats <file>                        # tag counts, skip rate, ante histogram
    python -m replay.cli tag <file>                          # tag_file() in place
    python -m replay.cli export-viz <file> <idx> <out.json> [--player 0|1]

Run from the repo root (it need not be a package -- see REPLAY_NOTES.md "how tests /
CLI resolve the engine"); ``python -m replay.cli`` works because Python adds the repo
root to sys.path for ``-m`` invocations and ``_bootstrap.py`` takes it from there.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

from .export_viz import export_viz_to_file
from .replay import ReplayMismatch, load_line, load_lines, narrate, replay_line, verify_file
from .tags import interest_score, interest_score_match, tag_episode, tag_file, tag_match


def _cmd_show(args) -> int:
    line = load_line(args.file, args.idx)
    print(narrate(line))
    return 0


def _cmd_verify(args) -> int:
    result = verify_file(args.file)
    print(f"{result['ok']}/{result['total']} lines replay clean")
    for idx, err in result["mismatches"]:
        print(f"  line {idx}: {err}")
    return 0 if not result["mismatches"] else 1


def _line_tags(line: dict) -> list:
    """Flat tag list for a line regardless of kind (match lines: union of both sides,
    each prefixed 'p0:'/'p1:' so --tag win still finds a match line via 'p0:win')."""
    if line.get("kind") == "match":
        tags = line.get("tags")
        if isinstance(tags, dict):
            out = []
            for side, t in tags.items():
                out.extend(f"p{side}:{x}" for x in t)
            return out
        # not yet tagged: compute on the fly (no archetype_novel without tag_file())
        out = []
        for p in (0, 1):
            out.extend(f"p{p}:{x}" for x in tag_match(line, p))
        return out
    tags = line.get("tags")
    if isinstance(tags, list) and tags:
        return tags
    return tag_episode(line)


def _line_interest(line: dict) -> float:
    if line.get("kind") == "match":
        stored = line.get("interest_score")
        if isinstance(stored, dict):
            return max(float(v) for v in stored.values())
        return max(interest_score_match(line, 0), interest_score_match(line, 1))
    stored = line.get("interest_score")
    if isinstance(stored, (int, float)):
        return float(stored)
    return interest_score(line)


def _cmd_filter(args) -> int:
    lines = load_lines(args.file)
    matched = 0
    for idx, line in enumerate(lines):
        tags = _line_tags(line)
        if args.tag is not None and args.tag not in tags and f"p0:{args.tag}" not in tags \
           and f"p1:{args.tag}" not in tags:
            continue
        score = _line_interest(line)
        if args.min_interest is not None and score < args.min_interest:
            continue
        matched += 1
        kind = line.get("kind", "episode")
        print(f"[{idx}] kind={kind} seed={line.get('seed')} interest={score:.2f} "
              f"tags={tags}")
    print(f"-- {matched}/{len(lines)} lines matched --")
    return 0


def _cmd_stats(args) -> int:
    lines = load_lines(args.file)
    tag_counts: Counter = Counter()
    ante_hist: Counter = Counter()
    skip_rates = []
    for line in lines:
        for t in _line_tags(line):
            tag_counts[t] += 1
        if line.get("kind") == "match":
            for p in (0, 1):
                view_steps = [s["players"][p] for s in line.get("steps", [])]
                if view_steps:
                    ante_hist[max(s["ante"] for s in view_steps)] += 1
        else:
            steps = line.get("steps") or []
            if steps:
                ante_hist[max(s["ante"] for s in steps)] += 1
            actions = line.get("actions") or []
            plays = sum(1 for a in actions if a.get("type") == "play_blind")
            skips = sum(1 for a in actions if a.get("type") == "skip_blind")
            if plays + skips:
                skip_rates.append(skips / (plays + skips))

    print(f"{len(lines)} lines")
    print("tag counts:")
    for tag, n in tag_counts.most_common():
        print(f"  {tag}: {n}")
    print("ante histogram (final ante -> count):")
    for ante, n in sorted(ante_hist.items()):
        print(f"  {ante}: {n}")
    if skip_rates:
        print(f"mean skip rate (episode lines): {sum(skip_rates) / len(skip_rates):.3f}")
    return 0


def _cmd_tag(args) -> int:
    result = tag_file(args.file)
    print(f"retagged {result['retagged']}/{result['total']} lines in {args.file}")
    return 0


def _cmd_export_viz(args) -> int:
    line = load_line(args.file, args.idx)
    doc = export_viz_to_file(line, args.out, episode_id=args.idx, player=args.player)
    print(f"wrote {args.out} ({len(doc['trajectory'])} trajectory entries)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m replay.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_show = sub.add_parser("show", help="narrate one logged line")
    p_show.add_argument("file")
    p_show.add_argument("idx", type=int)
    p_show.set_defaults(func=_cmd_show)

    p_verify = sub.add_parser("verify", help="replay every line, report signature mismatches")
    p_verify.add_argument("file")
    p_verify.set_defaults(func=_cmd_verify)

    p_filter = sub.add_parser("filter", help="list lines matching a tag / interest threshold")
    p_filter.add_argument("file")
    p_filter.add_argument("--tag", default=None)
    p_filter.add_argument("--min-interest", type=float, default=None)
    p_filter.set_defaults(func=_cmd_filter)

    p_stats = sub.add_parser("stats", help="tag counts, ante histogram, skip rate")
    p_stats.add_argument("file")
    p_stats.set_defaults(func=_cmd_stats)

    p_tag = sub.add_parser("tag", help="tag_file(): retag a log in place")
    p_tag.add_argument("file")
    p_tag.set_defaults(func=_cmd_tag)

    p_ev = sub.add_parser("export-viz", help="export one line to a V7 viz/ trajectory.json")
    p_ev.add_argument("file")
    p_ev.add_argument("idx", type=int)
    p_ev.add_argument("out")
    p_ev.add_argument("--player", type=int, default=None, help="match lines only: 0 or 1")
    p_ev.set_defaults(func=_cmd_export_viz)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ReplayMismatch as e:
        print(f"replay mismatch: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
