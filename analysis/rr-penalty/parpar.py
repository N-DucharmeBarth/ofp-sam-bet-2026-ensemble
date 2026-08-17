"""Minimal MFCL .par reader for the tag reporting-rate blocks (Task 1).

Reads only what the reporting-rate penalty analysis needs and validates every
block against the shapes implied by the file itself -- nothing about nfish,
ntag or the number of reporting-rate groups is hardcoded.

Cached to a pickle keyed on (path, mtime, size) so downstream tasks re-read
nothing.
"""

from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass, field

import numpy as np

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".par-cache.pkl")

# Block headers, in file order, matched after stripping trailing whitespace.
TAG_BLOCKS = [
    "# tag flags",
    "# tagmort",
    "# tag fish rep",
    "# tag fish rep group flags",
    "# tag_fish_rep active flags",
    "# tag_fish_rep target",
    "# tag_fish_rep penalty",
    "# region control flags",  # terminator only
]


class ParError(Exception):
    pass


@dataclass
class ParTagBlocks:
    path: str
    member_id: int
    ntag: int          # number of tag release groups (rows of tag flags)
    nfish: int         # number of fisheries (cols of the rep matrices)
    nrep_rows: int     # rows of the rep matrices (ntag + 1 pooled group)
    nage: int
    tag_flags: np.ndarray        # (ntag, 10) int
    parest_flags: np.ndarray     # (n,) int
    age_flags: np.ndarray        # (n,) int
    fish_flags: np.ndarray       # (nfish, 100) int
    rep: np.ndarray              # (nrep_rows, nfish) float
    rep_group: np.ndarray        # (nrep_rows, nfish) int
    rep_active: np.ndarray       # (nrep_rows, nfish) int
    rep_target: np.ndarray       # (nrep_rows, nfish) float, PERCENT as stored
    rep_penalty: np.ndarray      # (nrep_rows, nfish) float
    notes: list = field(default_factory=list)

    @property
    def arm_flag(self) -> int:
        """tag_flags(:,2): 0 = include reporting-rate adjustment, 1 = exclude.

        The flag is per release group, not per model. Release groups with a
        zero-length mixing window carry flag 1 even in the `include` arm --
        there is no mixing removal to raise, so `include` is undefined for
        them and the design forces face-value removal. The arm label is
        therefore read off the release groups that actually have a mixing
        window, and the carve-out is checked rather than assumed.
        """
        f2 = self.tag_flags[:, 1]
        mixing = self.tag_flags[:, 0]
        live = mixing > 0
        vals = set(f2[live].tolist())
        if len(vals) != 1:
            raise ParError(
                f"{self.path}: tag_flags(:,2) not constant across release groups with "
                f"a mixing window: {sorted(vals)}"
            )
        arm = int(vals.pop())
        # Under `include`, every flag-1 group must be a zero-mixing group.
        if arm == 0 and np.any(f2[~live] == 0) and np.any(~live):
            pass  # zero-mixing groups may legitimately hold either value
        if arm == 0:
            offenders = np.where((f2 == 1) & live)[0] + 1
            if offenders.size:
                raise ParError(
                    f"{self.path}: flag-1 release groups with a mixing window: {offenders.tolist()}"
                )
        return arm

    @property
    def n_zero_mixing(self) -> int:
        return int((self.tag_flags[:, 0] == 0).sum())

    @property
    def n_flag_exclude(self) -> int:
        return int((self.tag_flags[:, 1] == 1).sum())

    @property
    def arm(self) -> str:
        return {0: "include", 1: "exclude"}[self.arm_flag]

    @property
    def mixing(self) -> np.ndarray:
        """tag_flags(:,1): mixing period in periods, per release group."""
        return self.tag_flags[:, 0]


def _numeric_rows(lines, lo, hi):
    """Rows of numbers between two line indices, skipping comments/blanks."""
    out = []
    for ln in lines[lo:hi]:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append([float(t) for t in s.split()])
    return out


def _rect(rows, what, path):
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ParError(f"{path}: block '{what}' is ragged, row widths {sorted(widths)}")
    return np.array(rows, dtype=float)


def _find_headers(lines, wanted):
    """First line index of each wanted header, in file order."""
    idx = {}
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s in wanted and s not in idx:
            idx[s] = i
    missing = [w for w in wanted if w not in idx]
    if missing:
        raise ParError(f"missing par blocks: {missing}")
    return idx


def parse_par(path: str) -> ParTagBlocks:
    with open(path) as fh:
        lines = fh.read().split("\n")

    m = re.search(r"(\d+)\s*$", os.path.basename(os.path.dirname(path)))
    if not m:
        raise ParError(f"cannot derive member id from {path}")
    member_id = int(m.group(1))

    idx = _find_headers(lines, set(TAG_BLOCKS))
    order = sorted(idx.items(), key=lambda kv: kv[1])
    if [k for k, _ in order] != TAG_BLOCKS:
        raise ParError(f"{path}: tag blocks out of expected order: {[k for k, _ in order]}")
    bounds = {}
    for j, (name, start) in enumerate(order[:-1]):
        bounds[name] = (start + 1, order[j + 1][1])

    def block(name):
        lo, hi = bounds[name]
        return _rect(_numeric_rows(lines, lo, hi), name, path)

    # --- scalar / vector preamble blocks -------------------------------------
    ix = _find_headers(
        lines,
        {
            "# The parest_flags",
            "# The number of age classes",
            "# age flags",
            "# fish flags",
            "# tag flags",
        },
    )
    pf_rows = _numeric_rows(
        lines, ix["# The parest_flags"] + 1, ix["# The number of age classes"]
    )
    parest_flags = np.array([v for r in pf_rows for v in r], dtype=int)
    nage_rows = _numeric_rows(lines, ix["# The number of age classes"] + 1, ix["# age flags"])
    nage = int(nage_rows[0][0])
    age_rows = _numeric_rows(lines, ix["# age flags"] + 1, ix["# fish flags"])
    age_flags = np.array([v for r in age_rows for v in r], dtype=int)
    fish_flags = _rect(
        _numeric_rows(lines, ix["# fish flags"] + 1, ix["# tag flags"]), "# fish flags", path
    ).astype(int)

    tag_flags = block("# tag flags").astype(int)
    tagmort_rows = _numeric_rows(*(lines,) + bounds["# tagmort"])
    tagmort = np.array([v for r in tagmort_rows for v in r], dtype=float)

    rep = block("# tag fish rep")
    rep_group = block("# tag fish rep group flags").astype(int)
    rep_active = block("# tag_fish_rep active flags").astype(int)
    rep_target = block("# tag_fish_rep target")
    rep_penalty = block("# tag_fish_rep penalty")

    ntag, ntagcol = tag_flags.shape
    nrep_rows, nfish = rep.shape

    # --- post-conditions ------------------------------------------------------
    if ntagcol != 10:
        raise ParError(f"{path}: tag flags has {ntagcol} columns, expected 10")
    if fish_flags.shape[0] != nfish:
        raise ParError(
            f"{path}: fish flags rows ({fish_flags.shape[0]}) != rep matrix cols ({nfish})"
        )
    if tagmort.size != ntag:
        raise ParError(f"{path}: tagmort length {tagmort.size} != ntag {ntag}")
    for nm, arr in [
        ("group flags", rep_group),
        ("active flags", rep_active),
        ("target", rep_target),
        ("penalty", rep_penalty),
    ]:
        if arr.shape != rep.shape:
            raise ParError(f"{path}: tag_fish_rep {nm} shape {arr.shape} != rep shape {rep.shape}")
    # The rep matrices carry one extra row: the pooled tag release group.
    if nrep_rows != ntag + 1:
        raise ParError(
            f"{path}: rep matrices have {nrep_rows} rows, expected ntag+1 = {ntag + 1}"
        )
    if not np.all((rep >= 0) & (rep <= 1)):
        raise ParError(f"{path}: tag fish rep outside [0,1]")
    if not np.all(np.isin(rep_active, (0, 1))):
        raise ParError(f"{path}: tag_fish_rep active flags not 0/1")

    return ParTagBlocks(
        path=path,
        member_id=member_id,
        ntag=ntag,
        nfish=nfish,
        nrep_rows=nrep_rows,
        nage=nage,
        tag_flags=tag_flags,
        parest_flags=parest_flags,
        age_flags=age_flags,
        fish_flags=fish_flags,
        rep=rep,
        rep_group=rep_group,
        rep_active=rep_active,
        rep_target=rep_target,
        rep_penalty=rep_penalty,
    )


def parse_all(root="final-par", use_cache=True):
    """Parse every final-par/ensemble-*/final.par, newest-mtime-aware cache."""
    paths = sorted(
        os.path.join(root, d, "final.par")
        for d in os.listdir(root)
        if d.startswith("ensemble-") and os.path.isfile(os.path.join(root, d, "final.par"))
    )
    key = {p: (os.path.getmtime(p), os.path.getsize(p)) for p in paths}
    if use_cache and os.path.exists(CACHE):
        with open(CACHE, "rb") as fh:
            cached_key, cached = pickle.load(fh)
        if cached_key == key:
            return cached
    out = [parse_par(p) for p in paths]
    with open(CACHE, "wb") as fh:
        pickle.dump((key, out), fh)
    return out
