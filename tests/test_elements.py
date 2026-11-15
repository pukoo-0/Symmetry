"""The strict check: for 16 molecules, exact and rotated with noise, count every axis by order, every plane and
the inversion centre against textbook values, and check that every operation maps the molecule onto itself,
that the operations form a closed group, and that every axis and plane has its operations in the animation list."""
import collections
import itertools
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import symmetry_core as sc  # noqa: E402

P = (1 + 5 ** .5) / 2
def load(n): return sc.parse_xyz(open(os.path.join(ROOT, "examples", n + ".xyz"), encoding="utf-8").read())[0]
def out(v, d): v = np.array(v, float); return v / np.linalg.norm(v) * (np.linalg.norm(v) + d)
mols = {}
for n in ["c60", "cubane", "benzene", "methane", "sf6", "adamantane", "ethane", "allene", "ammonia", "water"]:
    f = load(n); mols[n] = (f.symbols, f.coords)
# B12H12(2-): icosahedron
ico = sorted({c for b in [(0, 1, P)] for sg in itertools.product([1, -1], repeat=3) for v in [tuple(s * x for s, x in zip(sg, b))] for c in [v, v[1:] + v[:1], v[2:] + v[:2]]})
B = [np.array(v) * 0.85 for v in ico]; mols["B12H12"] = (["B"] * 12 + ["H"] * 12, np.array(B + [out(v, 1.19) for v in B]))
# dodecahedrane C20H20
dod = [np.array(v) for v in itertools.product([1, -1], repeat=3)]
for sg in itertools.product([1, -1], repeat=2):
    for v in [(0, sg[0] / P, sg[1] * P)]:
        dod += [np.array(v), np.array(v[1:] + v[:1]), np.array(v[2:] + v[:2])]
C = [v * 1.25 for v in dod]; mols["C20H20"] = (["C"] * 20 + ["H"] * 20, np.array(C + [out(v, 1.09) for v in C]))
# ferrocene staggered / eclipsed
def fc(stag):
    S, X = ["Fe"], [[0, 0, 0]]
    for z, off in ((1.66, 0), (-1.66, 36 if stag else 0)):
        for k in range(5):
            a = np.radians(72 * k + off)
            S += ["C", "H"]; X += [[1.216 * np.cos(a), 1.216 * np.sin(a), z], [2.29 * np.cos(a), 2.29 * np.sin(a), z]]
    return S, np.array(X)
mols["ferrocene_stag"] = fc(True); mols["ferrocene_ecl"] = fc(False)
mols["PCl5"] = (["P", "Cl", "Cl"] + ["Cl"] * 3, np.array([[0, 0, 0], [0, 0, 2.12], [0, 0, -2.12]] + [[2.02 * np.cos(a), 2.02 * np.sin(a), 0] for a in np.radians([0, 120, 240])]))
S, X = [], []
for k in range(6):
    a, z = np.radians(60 * k), 0.25 * (-1) ** k
    S += ["C", "H", "H"]; X += [[1.446 * np.cos(a), 1.446 * np.sin(a), z], [1.446 * np.cos(a), 1.446 * np.sin(a), z + 1.09 * np.sign(z)], [2.47 * np.cos(a), 2.47 * np.sin(a), z - 0.36 * np.sign(z)]]
mols["cyclohexane_chair"] = (S, np.array(X))

EXPECT = {  # point group, axes by order, number of planes, inversion centre, order h
    "c60": ("Ih", {"C5": 6, "C3": 10, "C2": 15}, 15, True, 120), "B12H12": ("Ih", {"C5": 6, "C3": 10, "C2": 15}, 15, True, 120),
    "C20H20": ("Ih", {"C5": 6, "C3": 10, "C2": 15}, 15, True, 120),
    "cubane": ("Oh", {"C4": 3, "C3": 4, "C2": 6}, 9, True, 48), "sf6": ("Oh", {"C4": 3, "C3": 4, "C2": 6}, 9, True, 48),
    "benzene": ("D6h", {"C6": 1, "C2'": 3, "C2''": 3}, 7, True, 24),
    "methane": ("Td", {"C3": 4, "C2": 3}, 6, False, 24), "adamantane": ("Td", {"C3": 4, "C2": 3}, 6, False, 24),
    "ethane": ("D3d", {"C3": 1, "C2'": 3}, 3, True, 12), "allene": ("D2d", {"C2": 1, "C2'": 2}, 2, False, 8),
    "ammonia": ("C3v", {"C3": 1}, 3, False, 6), "water": ("C2v", {"C2": 1}, 2, False, 4),
    "ferrocene_stag": ("D5d", {"C5": 1, "C2'": 5}, 5, True, 20), "ferrocene_ecl": ("D5h", {"C5": 1, "C2'": 5}, 6, False, 20),
    "PCl5": ("D3h", {"C3": 1, "C2'": 3}, 4, False, 12), "cyclohexane_chair": ("D3d", {"C3": 1, "C2'": 3}, 3, True, 12),
}


@pytest.mark.parametrize("name", list(EXPECT))
@pytest.mark.parametrize("noise,tol", [(0.0, 0.3), (0.01, 0.1)])
def test_elements_operations_and_group(name, noise, tol):
    syms, X0 = mols[name]
    rng = np.random.default_rng(abs(hash((name, noise))) % 2**32)
    R = sc.rotation(rng.normal(size=3), rng.uniform(0, 6))
    X = X0 @ R.T + rng.normal(size=3) * 2 + rng.normal(scale=noise, size=X0.shape)
    r = sc.analyse(sc.Frame("", list(syms), X), tol)
    pg, axes, nplanes, inv, h = EXPECT[name]
    assert r["pg"] == pg and r["order"] == h and r["consistent"]
    assert dict(collections.Counter(sc.plain(a["short"]) for a in r["axes"])) == axes
    assert len(r["planes"]) == nplanes and r["has_i"] == inv
    assert len(r["ops"]) == h - 1
    _, mats, mol = sc.point_group(list(syms), X, tol)
    assert max(mol.deviation(M) for M in mats) <= tol
    assert all(any(np.allclose(A @ B, M, atol=0.02) for M in mats) for A in mats for B in mats)
    rots = collections.Counter()
    for o in r["ops"]:
        if o["kind"] == "C":
            rots[int(np.argmax([abs(np.dot(a["u"], o["u"])) for a in r["axes"]]))] += 1
    assert all(rots[j] >= a["n"] - 1 for j, a in enumerate(r["axes"]))
    assert sum(o["kind"] == "s" for o in r["ops"]) == nplanes
