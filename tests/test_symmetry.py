"""Checks for symmetry_core.  Run:  pytest

Test molecules with an exactly known point group are built by applying the group's operations to a few
seed atoms (tests/groups.py), then rotated randomly and, in some tests, shaken with random noise."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "web"), os.path.join(ROOT, "tests")]
import groups  # noqa: E402
import symmetry_core as sc  # noqa: E402

RNG = np.random.default_rng(1)


def shuffled(X):
    R = sc.rotation(RNG.normal(size=3), RNG.uniform(0, 6))
    return X @ R.T + RNG.normal(size=3) * 3


@pytest.mark.parametrize("pg", groups.ALL)
def test_every_point_group_exact(pg):
    syms, X, h = groups.build(pg)
    got, mats, mol = sc.point_group(syms, shuffled(X), 0.05)
    assert got == pg and len(mats) == h and mol.complete


@pytest.mark.parametrize("pg", ["C2h", "C3h", "S4", "S6", "C4v", "D3d", "D4h", "D6h", "Td", "Oh", "Ih"])
def test_noisy_coordinates(pg):
    syms, X, h = groups.build(pg)
    Y = shuffled(X) + RNG.normal(scale=0.02, size=X.shape)
    got, mats, _ = sc.point_group(syms, Y, 0.15)
    assert got == pg and len(mats) == h


TEXTBOOK = {   # point group and classes as in standard character tables
    "water": ("C2v", "E, C2, σv, σv'"),
    "ammonia": ("C3v", "E, 2C3, 3σv"),
    "methane": ("Td", "E, 8C3, 3C2, 6S4, 6σd"),
    "ethane": ("D3d", "E, 2C3, 3C2', i, 2S6, 3σd"),
    "allene": ("D2d", "E, C2, 2C2', 2S4, 2σd"),
    "benzene": ("D6h", "E, 2C6, 2C3, C2, 3C2', 3C2'', i, 2S6, 2S3, σh, 3σv, 3σd"),
    "sf6": ("Oh", "E, 6C4, 8C3, 6C2, 3C2 (=C4^2), i, 8S6, 6S4, 3σh, 6σd"),
    "cubane": ("Oh", "E, 6C4, 8C3, 6C2, 3C2 (=C4^2), i, 8S6, 6S4, 3σh, 6σd"),
    "adamantane": ("Td", "E, 8C3, 3C2, 6S4, 6σd"),
    "c60": ("Ih", "E, 12C5, 12C5^2, 20C3, 15C2, i, 12S10, 12S10^3, 20S6, 15σ"),
    "co2": ("D*h", "E, 2C∞, ∞σv, i, 2S∞, ∞C2"),
}


@pytest.mark.parametrize("name", list(TEXTBOOK))
def test_real_molecules_match_character_tables(name):
    with open(os.path.join(ROOT, "examples", name + ".xyz"), encoding="utf-8") as fh:
        res = sc.analyse(sc.parse_xyz(fh.read())[0], 0.3)
    pg, classes = TEXTBOOK[name]
    assert res["pg"] == pg and res["consistent"]
    assert sc.plain(res["ops_html"]) == classes
    if not res["linear"]:
        assert sum(c[0] for c in res["classes"]) == res["order"]


@pytest.mark.parametrize("pg,classes", [
    ("Th", "E, 4C3, 4C3^2, 3C2, i, 4S6, 4S6^5, 3σh"),
    ("D5h", "E, 2C5, 2C5^2, 5C2', 2S5, 2S5^3, σh, 5σv"),
    ("C5h", "E, C5, C5^2, C5^3, C5^4, S5, S5^3, S5^7, S5^9, σh"),
    ("D2h", "E, C2(z), C2(y), C2(x), i, σ(xy), σ(xz), σ(yz)"),
    ("S8", "E, C4, C4^3, C2, S8, S8^3, S8^5, S8^7"),
    ("I", "E, 12C5, 12C5^2, 20C3, 15C2"),
])
def test_conjugacy_classes(pg, classes):
    syms, X, _ = groups.build(pg)
    assert sc.plain(sc.analyse(sc.Frame("", syms, X), 0.1)["ops_html"]) == classes


def test_every_operation_really_is_a_symmetry_operation():
    syms, X, _ = groups.build("D4d")
    res = sc.analyse(sc.Frame("", syms, shuffled(X)), 0.05)
    assert res["ops"] and all(o["dev"] < 1e-6 for o in res["ops"])


def test_special_cases():
    assert sc.point_group(["Ne"], np.zeros((1, 3)))[0] == "Kh"
    assert sc.point_group(["H", "C", "N"], np.array([[0, 0, -1.06], [0, 0, 0], [0, 0, 1.15]]))[0] == "C*v"
    chiral = np.array([[0, 0, 0], [0.63, 0.63, 0.63], [-0.9, -0.95, 0.9], [-0.95, 1.1, -1.05], [1.2, -1.0, -1.1]])
    assert sc.point_group(["C", "H", "F", "Cl", "Br"], chiral)[0] == "C1"


def test_tolerance_matters_for_a_distorted_molecule():
    with open(os.path.join(ROOT, "examples", "ammonia.xyz"), encoding="utf-8") as fh:
        fr = sc.parse_xyz(fh.read())[0]
    fr.coords[1] += [0.08, 0, 0]          # pull one H 0.08 Å sideways
    assert sc.point_group(fr.symbols, fr.coords, 0.02)[0] in ("Cs", "C1")
    assert sc.point_group(fr.symbols, fr.coords, 0.3)[0] == "C3v"


def test_labels():
    assert sc.sub("σv'") == "σ<sub>v</sub>'"
    assert sc.sub("C2''") == "C<sub>2</sub>''"
    assert sc.sub("D*h") == "D<sub>∞h</sub>"
    assert sc.formula(["C"] * 60) == "C60" and sc.formula(["C", "O", "O"]) == "CO2"


def test_parse_errors():
    with pytest.raises(ValueError):
        sc.parse_xyz("4\nshort\nC 0 0 0\n")


def test_bridge_round_trip():
    import bridge
    bridge.HERE = os.path.join(ROOT, "examples")
    assert json.loads(bridge.example("benzene"))["ok"]
    out = json.loads(bridge.analyse(0, 0.3))
    assert out["ok"] and out["pg"] == "D6h" and len(out["ops"]) == 23
    assert len(json.loads(bridge.scan(0))["rows"]) == 9
    files = json.loads(bridge.exports(0, 0.3, "benzene"))["files"]
    assert "Point group: D6h" in files["benzene_pointgroup.txt"]


def test_built_page_is_up_to_date():
    import build_web
    with open(os.path.join(ROOT, "docs", "index.html"), encoding="utf-8") as fh:
        assert fh.read() == build_web.build(), "run: python build_web.py"
