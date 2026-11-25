"""Exact compatibility / conflict analysis between templates.

Two templates *conflict* when there EXISTS an 8-digit body which:

1. matches both literal prefixes, and
2. makes the two templates decode to different ``(scale_no, net_weight_g)``.

Only conflicting pairs are forced to use different marker digits.  If every
shared body decodes identically under both templates, they may share a marker.

The analysis is exact (no brute force over 10**8 bodies):

* A common body exists iff the two prefixes agree on their overlap.
* Scale numbers are compared as strings (leading zeros are significant).
  For equal-length ranges the extracted characters can be forced equal by
  prefix-fixed digits; otherwise a free digit can always be chosen to make
  them differ.
* Net weight is an integer number of grams.  Its value is an affine form
  ``C + sum(c_j * digit_j)`` over body positions; it is equal for every
  common completion iff ``C == 0`` and every non-zero coefficient lands on a
  prefix-fixed digit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .domain import BODY_LENGTH, TemplateSpec


def prefixes_compatible(p1: str, p2: str) -> bool:
    """True iff some body can start with both prefixes."""
    n = min(len(p1), len(p2))
    return p1[:n] == p2[:n]


def fixed_digits(t: TemplateSpec) -> dict[int, int]:
    """Body positions fixed by the template prefix."""
    return {i: int(ch) for i, ch in enumerate(t.prefix)}


def common_fixed_digits(a: TemplateSpec, b: TemplateSpec) -> Optional[dict[int, int]]:
    """Merged fixed digits for a common body, or None if prefixes clash."""
    if not prefixes_compatible(a.prefix, b.prefix):
        return None
    fixed = fixed_digits(a)
    for pos, digit in fixed_digits(b).items():
        if pos in fixed and fixed[pos] != digit:
            return None
        fixed[pos] = digit
    return fixed


def scale_always_equal(a: TemplateSpec, b: TemplateSpec, fixed: dict[int, int]) -> bool:
    """Whether scale strings are equal for EVERY body consistent with ``fixed``."""
    ra, rb = a.scale, b.scale
    if ra.length != rb.length:
        # Different lengths -> strings can never be equal.
        return False
    for k in range(ra.length):
        pa, pb = ra.start + k, rb.start + k
        if pa == pb:
            continue
        da = fixed.get(pa)
        db = fixed.get(pb)
        if da is None or db is None:
            # At least one digit is free: it can be chosen to differ.
            return False
        if da != db:
            # Fixed to different digits: always differ.
            return False
    return True


def weight_always_equal(a: TemplateSpec, b: TemplateSpec, fixed: dict[int, int]) -> bool:
    """Whether integer gram values are equal for EVERY consistent body."""
    coeff = [0] * BODY_LENGTH

    la = a.weight.length
    for k in range(la):
        pos = a.weight.start + k
        coeff[pos] += 10 ** (la - 1 - k)

    lb = b.weight.length
    for k in range(lb):
        pos = b.weight.start + k
        coeff[pos] -= 10 ** (lb - 1 - k)

    constant = 0
    for pos, c in enumerate(coeff):
        if c == 0:
            continue
        if pos in fixed:
            constant += c * fixed[pos]
        else:
            # Free variable with a non-zero coefficient: changing its digit
            # changes the value, so equality cannot hold for all bodies.
            return False
    return constant == 0


@dataclass
class PairAnalysis:
    compatible_body: bool
    scale_can_differ: bool
    weight_can_differ: bool
    conflict: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "compatible_body": self.compatible_body,
            "scale_can_differ": self.scale_can_differ,
            "weight_can_differ": self.weight_can_differ,
            "conflict": self.conflict,
            "reason": self.reason,
        }


def analyze_pair(a: TemplateSpec, b: TemplateSpec) -> PairAnalysis:
    fixed = common_fixed_digits(a, b)
    if fixed is None:
        return PairAnalysis(
            compatible_body=False,
            scale_can_differ=False,
            weight_can_differ=False,
            conflict=False,
            reason="prefixes never match the same body",
        )

    scale_equal = scale_always_equal(a, b, fixed)
    weight_equal = weight_always_equal(a, b, fixed)
    scale_diff = not scale_equal
    weight_diff = not weight_equal
    conflict = scale_diff or weight_diff

    if conflict:
        parts = []
        if scale_diff:
            parts.append("scale")
        if weight_diff:
            parts.append("net_weight")
        reason = "same body may decode differently in: " + ", ".join(parts)
    else:
        reason = "all shared bodies decode identically"

    return PairAnalysis(True, scale_diff, weight_diff, conflict, reason)


def conflicts(a: TemplateSpec, b: TemplateSpec) -> bool:
    return analyze_pair(a, b).conflict


def build_adjacency(templates: list[TemplateSpec]) -> list[set[int]]:
    """Conflict graph adjacency (indices into ``templates``).

    Uses :func:`conflict_pairs`, which is near-linear for large conflict-free
    batches instead of blindly analysing every pair.
    """
    n = len(templates)
    adj: list[set[int]] = [set() for _ in range(n)]
    for i, j in conflict_pairs(templates):
        adj[i].add(j)
        adj[j].add(i)
    return adj


# ---------------------------------------------------------------------------
# Scalable detection for whole template collections
# ---------------------------------------------------------------------------
#
# ``analyze_pair`` is exact but pairwise.  When thousands of templates are
# force-imported, calling it for every pair is O(n^2) even when the templates
# are all compatible.  We instead walk the prefix trie:
#
# * Templates in different trie branches have mutually exclusive prefixes and
#   are never compared ("前缀互斥的模板要跳过").
# * At every trie node P only templates whose prefix is a prefix of P are
#   visible.  For a pair (a, b) the longer prefix is the pair's common domain,
#   so its conflict is decided exactly once, at the node of the longer prefix.
# * Templates visible at a node are canonicalised into a *decode signature*
#   relative to P: the scale token sequence (fixed digit / free body position)
#   and the affine net-weight form.  Two templates conflict on domain P iff
#   their signatures differ.  Templates whose extraction positions differ but
#   which decode identically on every shared body therefore land in the same
#   group and are not compared pair by pair.
#
# All templates sharing one signature at a node remain identical after any
# further prefix digit is fixed, so a whole group is restricted once and
# carried down the trie; an entirely conflict-free batch keeps one group per
# node and costs O(templates * prefix-length), not O(pairs).

Signature = tuple  # (scale_tokens, weight_constant, weight_coefficients)


class _Group:
    """Lazily materialised set of template indices (persistent, cheap merge)."""

    __slots__ = ("parts",)

    def __init__(self, parts: tuple) -> None:
        self.parts = parts


def _group_members(group: _Group) -> list[int]:
    out: list[int] = []
    stack: list = [group]
    while stack:
        item = stack.pop()
        if isinstance(item, _Group):
            stack.extend(item.parts)
        elif isinstance(item, tuple):
            out.extend(item)
        else:
            out.append(item)
    return out


def _weight_coefficients(t: TemplateSpec) -> tuple[int, ...]:
    """Affine coefficients of the template's gram value over body digits."""
    coeff = [0] * BODY_LENGTH
    la = t.weight.length
    for k in range(la):
        coeff[t.weight.start + k] += 10 ** (la - 1 - k)
    return tuple(coeff)


def _signature(t: TemplateSpec, digits: tuple[int, ...]) -> Signature:
    """Canonical decode function of ``t`` restricted to bodies ``digits + *``.

    ``digits`` are the fixed leading body digits of the longer prefix.  Two
    templates with the same signature decode identically on that whole domain.
    """
    depth = len(digits)
    tokens = tuple(
        ("c", digits[pos]) if pos < depth else ("v", pos)
        for pos in range(t.scale.start, t.scale.end)
    )
    coeff = list(_weight_coefficients(t))
    constant = 0
    for pos in range(depth):
        c = coeff[pos]
        if c:
            constant += c * digits[pos]
            coeff[pos] = 0
    return (tokens, constant, tuple(coeff))


def _restrict(sig: Signature, pos: int, digit: int) -> Signature:
    """Signature on the sub-domain where body digit ``pos`` equals ``digit``."""
    tokens, constant, coeffs = sig
    new_tokens = tuple(
        ("c", digit) if kind == "v" and p == pos else (kind, p)
        for kind, p in tokens
    )
    coeff = list(coeffs)
    c = coeff[pos]
    if c:
        constant += c * digit
        coeff[pos] = 0
    return (new_tokens, constant, tuple(coeff))


def _add_cross_pairs(
    pairs: set[tuple[int, int]], ga: _Group, gb: _Group
) -> None:
    for a in _group_members(ga):
        for b in _group_members(gb):
            pairs.add((a, b) if a < b else (b, a))


def conflict_pairs(templates: list[TemplateSpec]) -> list[tuple[int, int]]:
    """Return every conflicting ``(i, j)`` pair with ``i < j``, sorted.

    The result is exactly the set of pairs for which :func:`analyze_pair`
    reports a conflict, but computed by prefix-trie grouping so conflict-free
    batches cost far less than the quadratic pair count.
    """
    n = len(templates)
    if n < 2:
        return []

    root: dict = {"children": {}, "exact": [], "digits": ()}
    for idx, t in enumerate(templates):
        node = root
        for ch in t.prefix:
            child = node["children"].get(ch)
            if child is None:
                child = {
                    "children": {},
                    "exact": [],
                    "digits": node["digits"] + (int(ch),),
                }
                node["children"][ch] = child
            node = child
        node["exact"].append(idx)

    pairs: set[tuple[int, int]] = set()
    # (node, ancestor signature -> group) ; root has no visible ancestors.
    stack: list[tuple[dict, dict[Signature, _Group]]] = [(root, {})]

    while stack:
        node, ancestors = stack.pop()
        digits = node["digits"]

        raw: dict[Signature, list[int]] = {}
        for idx in node["exact"]:
            sig = _signature(templates[idx], digits)
            raw.setdefault(sig, []).append(idx)
        exact_groups = {sig: _Group((tuple(idxs),)) for sig, idxs in raw.items()}

        # Exact templates conflict with each other (distinct signatures by
        # construction) and with visible ancestors of a different signature.
        # If ``ancestors`` already contains the same signature, then ancestor
        # templates decode identically with these (and recursively with each
        # other), so they are not compared again.
        items = list(exact_groups.items())
        for x, (sig_e, ge) in enumerate(items):
            for sig_o, go in items[x + 1 :]:
                _add_cross_pairs(pairs, ge, go)
            for sig_a, ga in ancestors.items():
                if sig_a != sig_e:
                    _add_cross_pairs(pairs, ge, ga)

        # Everything visible at this node is visible to its children.
        merged: dict[Signature, _Group] = dict(ancestors)
        for sig, ge in exact_groups.items():
            prev = merged.get(sig)
            merged[sig] = ge if prev is None else _Group((prev, ge))

        depth = len(digits)
        for ch, child in node["children"].items():
            digit = int(ch)
            child_groups: dict[Signature, _Group] = {}
            for sig, group in merged.items():
                restricted = _restrict(sig, depth, digit)
                prev = child_groups.get(restricted)
                child_groups[restricted] = (
                    group if prev is None else _Group((prev, group))
                )
            stack.append((child, child_groups))

    return sorted(pairs)


def conflict_analysis(
    templates: list[TemplateSpec],
) -> list[tuple[int, int, PairAnalysis]]:
    """Conflicting pairs (original order) with their per-pair analysis."""
    return [
        (i, j, analyze_pair(templates[i], templates[j]))
        for i, j in conflict_pairs(templates)
    ]
