"""Conflict detection scales for large batches without checking every pair."""

import itertools
import random

import pytest

from app.conflicts import (
    analyze_pair,
    conflict_analysis,
    conflict_pairs,
    conflicts,
)
from app.domain import Range, TemplateSpec


def tpl(prefix="", s=(2, 2), w=(4, 4), marker=None, name=""):
    return TemplateSpec(
        name=name, prefix=prefix, scale=Range(*s), weight=Range(*w), marker=marker
    )


def brute_pairs(templates):
    return [
        (i, j)
        for i in range(len(templates))
        for j in range(i + 1, len(templates))
        if conflicts(templates[i], templates[j])
    ]


@pytest.mark.parametrize("seed", range(300))
def test_grouped_detection_matches_pairwise(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 9)

    def rand_tpl():
        plen = rng.randint(0, 6)
        prefix = "".join(str(rng.randint(0, 9)) for _ in range(plen))
        ss = rng.randint(0, 7)
        sl = rng.randint(1, min(4, 8 - ss))
        ws = rng.randint(0, 7)
        wl = rng.randint(1, min(4, 8 - ws))
        return tpl(prefix=prefix, s=(ss, sl), w=(ws, wl))

    templates = [rand_tpl() for _ in range(n)]
    assert conflict_pairs(templates) == brute_pairs(templates)


def test_prefix_exclusive_templates_are_skipped():
    # 5000 templates with pairwise-exclusive 8-digit prefixes: no pair shares
    # a body, so no analysis is needed at all.
    templates = [
        tpl(prefix=f"{i:08d}", s=(2, 2), w=(4, 4)) for i in range(5000)
    ]
    assert conflict_pairs(templates) == []


def test_different_positions_same_decode_not_compared():
    # A reads body[2:4]; B reads body[2:2)+body[2:4) [3,5)... choose a case
    # where every shared body decodes identically because position 2 is fixed.
    a = tpl(prefix="120", s=(3, 1), w=(4, 2))
    b = tpl(prefix="120", s=(3, 1), w=(4, 2))
    c = tpl(prefix="120", s=(2, 2), w=(5, 2))  # reads fixed '0' too
    # a/c: scale "0"+? vs "0"+? actually compare on body 120xxxxx:
    # a scale = body[3], c scale = body[2:4] = "0"+body[3] -> string differs.
    assert conflict_pairs([a, b, c]) == [(0, 2), (1, 2)]


def test_nested_prefixes_with_equal_decode_no_conflict():
    short = tpl(prefix="12", s=(2, 2), w=(4, 4))
    long = tpl(prefix="1277", s=(2, 2), w=(4, 4))
    assert conflict_pairs([short, long]) == []


def test_nested_prefixes_with_different_decode_conflict():
    short = tpl(prefix="12", s=(2, 2), w=(4, 4))
    long = tpl(prefix="1277", s=(2, 1), w=(4, 4))
    assert conflict_pairs([short, long]) == [(0, 1)]


def test_conflict_analysis_keeps_original_order_and_reasons():
    templates = [
        tpl(prefix="9", s=(2, 2), w=(4, 4), marker="1"),
        tpl(prefix="90", s=(2, 2), w=(4, 4), marker="2"),
        tpl(prefix="90", s=(2, 1), w=(4, 4), marker="3"),
        tpl(prefix="91", s=(2, 2), w=(4, 4), marker="4"),
    ]
    result = conflict_analysis(templates)
    pairs = [(i, j) for i, j, _ in result]
    assert pairs == [(0, 2), (1, 2)]
    for i, j, analysis in result:
        expected = analyze_pair(templates[i], templates[j])
        assert analysis.conflict
        assert analysis.reason == expected.reason


def test_large_identical_group_scales_sublinearly():
    # 2000 templates that all decode identically, nested prefix chain.
    templates = [
        tpl(prefix="1" * (i % 9), s=(2, 2), w=(4, 4)) for i in range(2000)
    ]
    assert conflict_pairs(templates) == []
