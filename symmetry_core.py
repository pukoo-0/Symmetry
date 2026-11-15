"""
symmetry_core.py — the chemistry behind Point Group Finder.

Finds the point group of a molecule from its .xyz coordinates, with NumPy only (no pymatgen), so it also
runs inside the browser through Pyodide.

How it works
  1. Move the centre of mass to the origin.
  2. Sort atoms into sets that could be swapped by symmetry: same element, same distance from the centre.
  3. Collect candidate axis directions (principal axes, atoms, midpoints of atom pairs, normals of atom
     triangles) and candidate mirror normals (principal axes, differences of atom pairs).
  4. Test each candidate: an operation is a symmetry operation when every atom lands within the tolerance
     (Å) of an atom of the same element, one to one.
  5. Build the whole group from the operations found (multiply them until nothing new appears).
  6. Name the group (Schoenflies symbol) from its axes, mirror planes, inversion centre and S axes.
  7. Label axes and planes (C2', C2'', σv, σd, σh ...) and describe each operation. These labelling functions
     come from George's original pymatgen-based app and are kept as they were.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from math import gcd

import numpy as np

# ---------------------------------------------------------------- element data
SYMBOLS = [
    "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
Z_OF = {s: i for i, s in enumerate(SYMBOLS) if i > 0}


# Covalent radii in Å (Cordero et al., Dalton Trans. 2008, 2832; low-spin values for Mn, Fe, Co; sp3 for C).
COVALENT_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "Ne": 0.58, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32, "Co": 1.26,
    "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54, "Tc": 1.47, "Ru": 1.46, "Rh": 1.42,
    "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01, "Pm": 1.99, "Sm": 1.98, "Eu": 1.98,
    "Gd": 1.96, "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87, "Lu": 1.87, "Hf": 1.75,
    "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Tl": 1.45,
    "Pb": 1.46, "Bi": 1.48, "Po": 1.40, "At": 1.50, "Rn": 1.50, "Fr": 2.60, "Ra": 2.21, "Ac": 2.15, "Th": 2.06,
    "Pa": 2.00, "U": 1.96, "Np": 1.90, "Pu": 1.87, "Am": 1.80, "Cm": 1.69,
}
DEFAULT_RADIUS = 1.50


# Standard atomic weights, IUPAC abridged values (as listed on Wikipedia "Standard atomic weight", 2024).
# Elements without a standard atomic weight use the mass number of their longest-lived isotope (marked *).
_MASS = """H 1.0080 He 4.0026 Li 6.94 Be 9.0122 B 10.81 C 12.011 N 14.007 O 15.999 F 18.998 Ne 20.180
Na 22.990 Mg 24.305 Al 26.982 Si 28.085 P 30.974 S 32.06 Cl 35.45 Ar 39.95 K 39.098 Ca 40.078 Sc 44.956
Ti 47.867 V 50.942 Cr 51.996 Mn 54.938 Fe 55.845 Co 58.933 Ni 58.693 Cu 63.546 Zn 65.38 Ga 69.723 Ge 72.630
As 74.922 Se 78.971 Br 79.904 Kr 83.798 Rb 85.468 Sr 87.62 Y 88.906 Zr 91.222 Nb 92.906 Mo 95.95 Tc 97*
Ru 101.07 Rh 102.91 Pd 106.42 Ag 107.87 Cd 112.41 In 114.82 Sn 118.71 Sb 121.76 Te 127.60 I 126.90 Xe 131.29
Cs 132.91 Ba 137.33 La 138.91 Ce 140.12 Pr 140.91 Nd 144.24 Pm 145* Sm 150.36 Eu 151.96 Gd 157.25 Tb 158.93
Dy 162.50 Ho 164.93 Er 167.26 Tm 168.93 Yb 173.05 Lu 174.97 Hf 178.49 Ta 180.95 W 183.84 Re 186.21 Os 190.23
Ir 192.22 Pt 195.08 Au 196.97 Hg 200.59 Tl 204.38 Pb 207.2 Bi 208.98 Po 209* At 210* Rn 222* Fr 223* Ra 226*
Ac 227* Th 232.04 Pa 231.04 U 238.03 Np 237* Pu 244* Am 243* Cm 247*"""
ATOMIC_MASS = {a: float(b.rstrip("*")) for a, b in zip(_MASS.split()[0::2], _MASS.split()[1::2])}
Z_OF = {s: i for i, s in enumerate(SYMBOLS) if i > 0}

# van der Waals radii for the space-filling view (Å, Bondi), as in the original app
VDW = {"H": 1.1, "C": 1.7, "N": 1.55, "O": 1.52, "F": 1.47, "Cl": 1.75, "S": 1.8, "P": 1.8, "Br": 1.85}


def element_symbol(token: str) -> str:
    t = token.strip()
    if t.isdigit():
        z = int(t)
        return SYMBOLS[z] if 0 < z < len(SYMBOLS) else f"Z{z}"
    letters = "".join(ch for ch in t if ch.isalpha())
    if not letters:
        raise ValueError(f"'{token}' is not an element")
    two = letters[:2].capitalize()
    if len(letters) >= 2 and two in Z_OF:
        return two
    if letters[0].upper() in Z_OF:
        return letters[0].upper()
    raise ValueError(f"'{token}' is not an element")


# ---------------------------------------------------------------- reading .xyz
@dataclass
class Frame:
    comment: str
    symbols: list
    coords: np.ndarray


def parse_xyz(text: str) -> list:
    """Every frame of an .xyz file (single structures and multi-frame trajectories)."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    frames, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            n = int(lines[i].split()[0])
        except ValueError:
            raise ValueError(f"Line {i + 1} should hold the number of atoms, but it reads “{lines[i].strip()[:40]}”.")
        if n <= 0:
            raise ValueError(f"Line {i + 1}: the number of atoms must be positive.")
        comment = lines[i + 1].strip() if i + 1 < len(lines) else ""
        syms, xyz = [], []
        for k in range(n):
            ln = i + 2 + k
            if ln >= len(lines) or not lines[ln].strip():
                raise ValueError(f"Frame {len(frames) + 1} says it has {n} atoms, but only {k} were found.")
            parts = lines[ln].split()
            if len(parts) < 4:
                raise ValueError(f"Line {ln + 1} should look like “C 0.0 0.0 0.0”.")
            try:
                syms.append(element_symbol(parts[0]))
                xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError as e:
                raise ValueError(f"Line {ln + 1}: {e}") from None
        frames.append(Frame(comment, syms, np.array(xyz, dtype=float)))
        i += 2 + n
    if not frames:
        raise ValueError("The file has no atoms in it.")
    return frames


def formula(symbols: list) -> str:
    c = Counter(symbols)
    hill = ["C", "H"] if "C" in c else []                       # Hill order: C, H, then alphabetical
    order = [e for e in hill if e in c] + sorted(k for k in c if k not in hill)
    return "".join(f"{e}{c[e] if c[e] > 1 else ''}" for e in order)


# ---------------------------------------------------------------- symmetry operations as 3×3 matrices
def unit(x):
    norm = np.linalg.norm(x)
    return x / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])


def rotation(u, angle):
    """Rotation by `angle` (radians) about unit vector u (Rodrigues' formula)."""
    u = unit(np.asarray(u, float))
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def reflection(n):
    """Mirror through the plane through the origin with normal n."""
    n = unit(np.asarray(n, float))
    return np.eye(3) - 2.0 * np.outer(n, n)


def improper(u, angle):
    """S: rotate about u, then reflect through the plane perpendicular to u."""
    return reflection(u) @ rotation(u, angle)


INVERSION = -np.eye(3)


class Molecule:
    """Centred coordinates plus the bookkeeping needed to test operations quickly."""

    def __init__(self, symbols, coords, tol):
        self.symbols = list(symbols)
        X = np.asarray(coords, float)
        m = np.array([ATOMIC_MASS.get(s, 1.0) for s in self.symbols])
        self.center = (m[:, None] * X).sum(0) / m.sum()           # centre of mass, like pymatgen
        self.X = X - self.center
        self.tol = float(tol)
        self.n = len(self.symbols)
        self.by_element = [np.array([i for i, s in enumerate(self.symbols) if s == e]) for e in sorted(set(self.symbols))]
        # inertia tensor (mass-weighted), its eigenvalues and principal axes
        Xc = self.X
        I = np.zeros((3, 3))
        for mi, x in zip(m, Xc):
            I += mi * (np.dot(x, x) * np.eye(3) - np.outer(x, x))
        self.moments, vecs = np.linalg.eigh(I)
        self.principal = [vecs[:, k] for k in range(3)]
        self.seas = self._equivalent_sets()
        self.complete = True
        self.elements = [(np.arange(self.n), 1)]          # the group as (atom permutation, det), set later

    def _equivalent_sets(self):
        """Atoms that could be swapped by symmetry: same element and same distance from the centre."""
        r = np.linalg.norm(self.X, axis=1)
        sets = []
        for idx in self.by_element:
            order = idx[np.argsort(r[idx])]
            group = [order[0]]
            for a in order[1:]:
                if r[a] - r[group[-1]] <= self.tol:
                    group.append(a)
                else:
                    sets.append(np.array(group))
                    group = [a]
            sets.append(np.array(group))
        return sorted(sets, key=len)

    def match(self, R, stop=None):
        """Where every atom goes under R. Returns (permutation, largest displacement in Å):
        perm[i] = j means atom i lands on atom j (same element). perm is None if two atoms would land
        on the same atom. Stops early once the displacement exceeds `stop`."""
        Y = self.X @ R.T
        perm = np.empty(self.n, dtype=int)
        worst = 0.0
        for idx in self.seas:                       # small sets first: they fail fastest
            d = np.linalg.norm(Y[idx][:, None, :] - self.X[idx][None, :, :], axis=2)
            j = d.argmin(axis=1)
            if len(set(j.tolist())) < len(idx):
                return None, float("inf")
            worst = max(worst, float(d[np.arange(len(idx)), j].max()))
            if stop is not None and worst > stop:
                return None, worst
            perm[idx] = idx[j]
        return perm, worst

    def deviation(self, R):
        return self.match(R)[1]

    def test(self, R):
        """Is R (approximately) a symmetry operation? A candidate built from noisy atoms can be slightly
        off, so if R nearly works, the exact operation for that atom permutation is fitted by least squares
        and tested again. Returns the permutation, or None."""
        perm, dev = self.match(R, stop=3 * self.tol)
        if perm is None:
            return None
        if dev <= self.tol:
            return perm
        det = 1 if np.linalg.det(R) > 0 else -1
        p2, dev2 = self.match(self.fit(perm, det), stop=self.tol)
        if p2 is None or dev2 > self.tol or not np.array_equal(p2, perm):
            return None
        return perm

    def is_symmetry(self, R):
        return self.test(R) is not None

    def fit(self, perm, det):
        """The orthogonal matrix (rotation if det = +1, improper if -1) that best moves every atom i onto
        atom perm[i], by least squares (Kabsch). This removes noise from the operation."""
        H = self.X.T @ self.X[perm]
        U, _, Vt = np.linalg.svd(H)
        V = Vt.T
        s = det * np.sign(np.linalg.det(V @ U.T) or 1.0)
        return V @ np.diag([1.0, 1.0, s]) @ U.T


# ---------------------------------------------------------------- candidates
def _dedupe_directions(V, cos_tol=0.9995):
    V = np.asarray([unit(v) for v in V if np.linalg.norm(v) > 1e-6])
    if not len(V):
        return []
    # make the sign canonical so u and -u look the same
    k = np.argmax(np.abs(V) > 1e-6, axis=1)
    V = V * np.sign(V[np.arange(len(V)), k])[:, None]
    _, first = np.unique(np.round(V, 3), axis=0, return_index=True)
    V = V[np.sort(first)]
    kept = []
    for v in V:
        if not kept or np.max(np.abs(np.asarray(kept) @ v)) < cos_tol:
            kept.append(v)
    return kept


def reference_atoms(mol: Molecule):
    """Up to three atoms that, with the centre, span space (two for a planar molecule). Every mirror plane
    and axis must move at least one of them onto another atom of its set, which keeps the candidates few."""
    X, tol = mol.X, mol.tol
    r = np.linalg.norm(X, axis=1)
    if r.max() <= tol:
        return []
    refs = [int(np.argmax(r))]
    a = unit(X[refs[0]])
    d_line = np.linalg.norm(X - np.outer(X @ a, a), axis=1)
    if d_line.max() > tol:
        refs.append(int(np.argmax(d_line)))
        nrm = unit(np.cross(X[refs[0]], X[refs[1]]))
        d_plane = np.abs(X @ nrm)
        if d_plane.max() > tol:
            refs.append(int(np.argmax(d_plane)))
    return refs


def _set_of(mol: Molecule, i):
    for s in mol.seas:
        if i in s:
            return s
    return np.array([i])


def candidate_axes(mol: Molecule):
    X, tol, c = mol.X, mol.tol, list(mol.principal)
    c += [x for x in X if np.linalg.norm(x) > tol]                          # through an atom
    for ref in reference_atoms(mol):
        s = _set_of(mol, ref)
        P, r0 = X[s], X[ref]
        mid = (P + r0) / 2                                                  # C2: through the middle of ref and its image
        c += list(mid[np.linalg.norm(mid, axis=1) > tol])
        d = np.linalg.norm(P - r0, axis=1)                                  # Cn: ref and its two neighbours on the ring
        for shell in _clusters(d, tol):                                     # sit at the same distance from ref
            shell = [k for k in shell if d[k] > tol]
            for ii in range(len(shell)):
                for jj in range(ii + 1, len(shell)):
                    c.append(np.cross(P[shell[ii]] - r0, P[shell[jj]] - r0))
    return _dedupe_directions(c)


def candidate_planes(mol: Molecule):
    X, c = mol.X, list(mol.principal)
    for ref in reference_atoms(mol):
        s = _set_of(mol, ref)
        c += list(X[s] - X[ref])                                            # a mirror that moves ref onto another atom
    return _dedupe_directions(c)


def _clusters(values, tol):
    """Split sorted values wherever the gap is larger than tol; returns lists of positions."""
    order = np.argsort(values)
    cuts = np.nonzero(np.diff(values[order]) > tol)[0] + 1
    return np.split(order, cuts)


def _possible_orders(mol: Molecule, u, nmax=24):
    """Orders n that an axis along u could have: off-axis atoms of each set come in rings of n
    (same height, same distance from the axis), so n divides the size of every ring."""
    h = mol.X @ u
    r = np.linalg.norm(mol.X - np.outer(h, u), axis=1)
    g = 0
    for s in mol.seas:
        off = s[r[s] > mol.tol]
        if not len(off):
            continue
        for hc in _clusters(h[off], mol.tol):
            for rc in _clusters(r[off][hc], mol.tol):
                g = gcd(g, len(rc))
                if g == 1:
                    return []
    if g == 0:                       # every atom is on the line: a linear molecule
        return []
    return [n for n in range(min(g, nmax), 1, -1) if g % n == 0]


# ---------------------------------------------------------------- finding the group
def is_linear(mol: Molecule) -> bool:
    if mol.n < 2:
        return False
    u = mol.principal[0]                                     # smallest moment: along the molecule
    d = np.linalg.norm(mol.X - np.outer(mol.X @ u, u), axis=1)
    return bool(np.all(d <= mol.tol))


def find_generators(mol: Molecule):
    """Every proper axis (highest order), S axis, mirror plane and inversion centre that passes the test.
    Returns a list of (permutation, det)."""
    gens, axes = [], []
    for u in candidate_axes(mol):
        if any(abs(np.dot(u, a)) > 0.999 for a, _ in axes):
            continue
        for n in _possible_orders(mol, u):
            perm = mol.test(rotation(u, 2 * np.pi / n))
            if perm is not None:
                exact = extract_axis(mol.fit(perm, 1))          # the fitted axis, free of noise
                axes.append((exact, n))
                gens.append((perm, 1))
                break
    for u, n in axes:                                         # S2n along a Cn axis (S4 in Td, S6 in D3d ...)
        perm = mol.test(improper(u, np.pi / n))
        if perm is not None:
            gens.append((perm, -1))
    for nrm in candidate_planes(mol):
        perm = mol.test(reflection(nrm))
        if perm is not None:
            gens.append((perm, -1))
    perm = mol.test(INVERSION)
    if perm is not None:
        gens.append((perm, -1))
    return gens


def _snap(R):
    """Nearest exact orthogonal matrix (removes rounding noise), as in the original app."""
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


def closure(generators, atol=5e-2, limit=240):
    """All products of exact matrices (used for building test molecules)."""
    group = [np.eye(3)]
    frontier = [np.eye(3)]
    while frontier and len(group) <= limit:
        new = []
        for A in frontier:
            for g in generators:
                M = _snap(g @ A)
                if not any(np.allclose(M, G, atol=atol) for G in group):
                    group.append(M)
                    new.append(M)
        frontier = new
    return group


def group_from_permutations(mol: Molecule, generators, limit=240):
    """The whole point group, built by multiplying atom permutations (exact, so noise cannot make the group
    grow without end), then one least-squares matrix per operation. Returns (matrices, complete):
    complete is False if the products went past `limit` operations, which means the tolerance is too loose."""
    gens = list(generators)
    ident = (tuple(range(mol.n)), 1)
    seen = {ident}
    frontier = [ident]
    while frontier:
        new = []
        for p, d in frontier:
            pa = np.array(p)
            for g, dg in gens:
                q = (tuple(g[pa].tolist()), d * dg)          # apply p, then g
                if q not in seen:
                    seen.add(q)
                    new.append(q)
                    if len(seen) > limit:
                        mol.elements = [(np.arange(mol.n), 1)]
                        return [np.eye(3)], False
        frontier = new
    elements = sorted(seen, key=lambda e: (e != ident, -e[1]))
    mol.elements = [(np.array(e[0]), e[1]) for e in elements]
    return [np.eye(3) if e == ident else mol.fit(np.array(e[0]), e[1]) for e in elements], True


GROUP_ORDER_FIXED = {"C1": 1, "Cs": 2, "Ci": 2, "T": 12, "Td": 24, "Th": 24, "O": 24, "Oh": 48, "I": 60, "Ih": 120,
                     "Kh": 1, "C*v": 0, "D*h": 0}


def expected_order(pg: str) -> int:
    if pg in GROUP_ORDER_FIXED:
        return GROUP_ORDER_FIXED[pg]
    m = re.match(r"^([CDS])(\d+)([hvd]?)$", pg)
    if not m:
        return 0
    letter, n, suffix = m.group(1), int(m.group(2)), m.group(3)
    if letter == "S":
        return n
    if letter == "C":
        return n * (2 if suffix else 1)
    return 2 * n * (2 if suffix else 1)


def classify(mats) -> str:
    """Schoenflies symbol from the full list of operation matrices."""
    axes = []                                           # [direction, highest order]
    mirrors, has_i, improper_ops = [], False, []
    for R in mats:
        det, tr = np.linalg.det(R), np.trace(R)
        if det > 0:
            a = rot_angle(R)
            if a < 0.05:
                continue
            n, _ = angle_to_order(a)
            u = extract_axis(R)
            for ax in axes:
                if abs(np.dot(ax[0], u)) > 0.99:
                    ax[1] = max(ax[1], n)
                    break
            else:
                axes.append([u, n])
        elif abs(tr + 3.0) < 0.08:
            has_i = True
        elif abs(tr - 1.0) < 0.08:
            mirrors.append(extract_improper_axis(R))
        else:
            u = extract_improper_axis(R)
            Q = R @ (np.eye(3) - 2.0 * np.outer(u, u))
            improper_ops.append((u, angle_to_order(rot_angle(Q))[0]))

    high = [ax for ax in axes if ax[1] >= 3]
    if len(high) > 1:                                    # several axes of order ≥ 3: cubic or icosahedral
        orders = {ax[1] for ax in axes}
        if 5 in orders:
            return "Ih" if has_i else "I"
        if 4 in orders:
            return "Oh" if has_i else "O"
        if has_i:
            return "Th"
        return "Td" if mirrors else "T"
    if not axes:
        return "Cs" if mirrors else "Ci" if has_i else "C1"

    nmax = max(ax[1] for ax in axes)
    top = [ax for ax in axes if ax[1] == nmax]
    main = top[0][0]
    for ax in top:                                       # prefer the axis that also carries S2n (D2d, S4)
        if any(abs(np.dot(u, ax[0])) > 0.99 and m == 2 * nmax for u, m in improper_ops):
            main = ax[0]
            break
    perp_c2 = [ax for ax in axes if ax[1] == 2 and abs(np.dot(ax[0], main)) < 0.1]
    sigma_h = any(abs(np.dot(m, main)) > 0.99 for m in mirrors)
    sigma_v = sum(abs(np.dot(m, main)) < 0.1 for m in mirrors)
    if len(perp_c2) >= nmax:
        return f"D{nmax}h" if sigma_h else f"D{nmax}d" if sigma_v >= nmax else f"D{nmax}"
    if sigma_h:
        return f"C{nmax}h"
    if sigma_v >= nmax:
        return f"C{nmax}v"
    if any(abs(np.dot(u, main)) > 0.99 and m == 2 * nmax for u, m in improper_ops):
        return f"S{2 * nmax}"
    return f"C{nmax}"


def linear_ops(mol: Molecule, centrosymmetric: bool):
    """A few representative operations for a linear molecule, for drawing and animating."""
    u = unit(mol.principal[0])
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    p1 = unit(a - np.dot(a, u) * u)
    p2 = np.cross(u, p1)
    ops = [np.eye(3), rotation(u, np.pi), reflection(p1), reflection(p2)]
    if centrosymmetric:
        ops += [INVERSION, reflection(u)]
    return ops


def point_group(symbols, coords, tol=0.3):
    """(Schoenflies symbol, operation matrices, Molecule). Linear groups use '*' for ∞ (C*v, D*h)."""
    mol = Molecule(symbols, coords, tol)
    if mol.n == 1:
        return "Kh", [np.eye(3)], mol
    if is_linear(mol):
        centro = mol.is_symmetry(INVERSION)
        return ("D*h" if centro else "C*v"), linear_ops(mol, centro), mol
    mats, complete = group_from_permutations(mol, find_generators(mol))
    mol.complete = complete
    return classify(mats), mats, mol


# =====================================================================================================
# Labelling and describing. The functions below come from the original app (George's app.py) and are
# kept as they were, apart from the colours, which now follow the new interface.
# =====================================================================================================
# axis colours by order: C2 cobalt, C3 brick, C4 green, C5 violet, C6 amber (0 = C∞ for linear molecules)
AXIS_COLORS = {0: "#1D2A31", 2: "#2357C2", 3: "#C2412D", 4: "#2E7D4F", 5: "#7A4FB5", 6: "#B86E00", 7: "#1B8A8A", 8: "#B03A78"}
AXIS_SUBTYPE_COLORS = {"C2": "#2357C2", "C2'": "#4F7FDB", "C2''": "#86A6E8"}
PLANE_COLORS = {"σh": "#D08A00", "σv": "#1B8A8A", "σv'": "#2A86B8", "σd": "#B03A78", "σ": "#5B5FC7"}
PLANE_ORDER = ["σh", "σv", "σv'", "σd", "σ"]


def sub(s):
    s = s.replace("*", "∞")
    if len(s) == 1:
        return s
    m = re.match(r"^([A-Zσ])(.+?)('+)?$", s)        # one base letter (was [A-Za-zσ]+, which made σv' into σv<sub>'</sub>)
    if m:
        base, sub_part, primes = m.group(1), m.group(2), m.group(3) or ""
        return f"{base}<sub>{sub_part}</sub>{primes}"
    return s[0] + "<sub>" + s[1:] + "</sub>"


def axis_name(n, subtype=None):
    if subtype:
        return subtype
    return "C∞" if n == 0 else f"C{n}"


def rot_angle(R):
    return float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))


def angle_to_order(a, max_denom=24):
    f = Fraction(float(a) / (2.0 * np.pi)).limit_denominator(max_denom)
    return f.denominator, f.numerator


def extract_axis(Q):
    a = rot_angle(Q)
    if a < 1e-3:
        return np.array([0.0, 0.0, 1.0])
    if abs(a - np.pi) > 1e-2:
        skew = np.array([Q[2, 1] - Q[1, 2], Q[0, 2] - Q[2, 0], Q[1, 0] - Q[0, 1]])
        if np.linalg.norm(skew) > 1e-5:
            return unit(skew)
    M = Q + np.eye(3)
    norms = np.linalg.norm(M, axis=0)
    return unit(M[:, int(np.argmax(norms))])


def extract_improper_axis(R):
    return extract_axis(-R)


def get_axes(matrices, pg, coords, tol):
    axes = []
    improper_orders = []
    for R in matrices:
        det = np.linalg.det(R)
        if det > 0:
            a = rot_angle(R)
            if a < 0.05:
                continue
            n, _ = angle_to_order(a)
            u = extract_axis(R)
            for ax in axes:
                if abs(np.dot(ax["u"], u)) > 0.99:
                    ax["n"] = max(ax["n"], n)
                    break
            else:
                axes.append({"u": u, "n": n, "sn": 0, "subtype": None})
        else:
            tr = np.trace(R)
            if abs(tr - 1.0) < 0.08 or abs(tr + 3.0) < 0.08:
                continue
            u = extract_improper_axis(R)
            Q = R @ (np.eye(3) - 2.0 * np.outer(u, u))
            a = rot_angle(Q)
            if a > 0.05:
                sn, _ = angle_to_order(a)
                improper_orders.append((u, sn))

    for u, sn in improper_orders:
        for ax in axes:
            if abs(np.dot(ax["u"], u)) > 0.99:
                ax["sn"] = max(ax["sn"], sn)
                break

    if not axes:
        return axes

    if pg.startswith("D") and pg not in ("D*h", "D2", "D2h"):
        nmax = max(a["n"] for a in axes)
        top = [a for a in axes if a["n"] == nmax]
        if len(top) > 1:
            top_s = [a for a in top if a["sn"] > nmax]
            main = top_s[0]["u"] if top_s else top[0]["u"]
        else:
            main = top[0]["u"]

        perp_c2 = [ax for ax in axes if ax["n"] == 2 and abs(np.dot(ax["u"], main)) < 0.1]
        if nmax % 2 == 0 and len(perp_c2) == nmax and nmax >= 4:
            on_ax = [
                int(np.sum(np.linalg.norm(coords - np.outer(coords @ ax["u"], ax["u"]), axis=1) < tol))
                for ax in perp_c2
            ]
            ref_idx = int(np.argmax(on_ax))
            ref_u = perp_c2[ref_idx]["u"]
            for ax in perp_c2:
                ang = np.arccos(np.clip(abs(np.dot(ax["u"], ref_u)), 0.0, 1.0))
                k = round(ang / (np.pi / nmax))
                ax["subtype"] = "C2'" if k % 2 == 0 else "C2''"
        elif len(perp_c2) == nmax:
            for ax in perp_c2:
                ax["subtype"] = "C2'"

    return axes


def get_planes(matrices):
    normals = []
    for R in matrices:
        if np.linalg.det(R) > 0 or abs(np.trace(R) - 1.0) > 0.08:
            continue
        n = extract_improper_axis(R)
        if all(abs(np.dot(n, m)) < 0.99 for m in normals):
            normals.append(n)
    return normals


def label_planes(normals, pg, axes, coords, tol):
    if not normals:
        return []
    if pg == "Cs":
        return ["σh"]
    if pg == "Ih" or not axes:
        return ["σ"] * len(normals)
    if pg == "Td":
        return ["σd"] * len(normals)
    if pg in ("Oh", "Th"):
        cube = [a["u"] for a in axes if a["n"] == (4 if pg == "Oh" else 2)]
        return ["σh" if any(abs(np.dot(n, u)) > 0.99 for u in cube) else "σd" for n in normals]

    nmax = max(a["n"] for a in axes)
    top = [a for a in axes if a["n"] == nmax]
    is_dnd = pg[0] == "D" and pg.endswith("d")
    if len(top) > 1 and not is_dnd:
        return ["σ"] * len(normals)
    if is_dnd:
        return ["σd"] * len(normals)
    main = top[0]["u"]

    labels = ["σh" if abs(np.dot(n, main)) > 0.99 else "σv" for n in normals]

    vert = [i for i, l in enumerate(labels) if l == "σv"]
    if nmax % 2 == 0 and len(vert) > 1:
        c2p = [a["u"] for a in axes if a.get("subtype") == "C2'"]
        on_plane = [int(np.sum(abs(coords @ normals[i]) < tol)) for i in vert]
        if max(on_plane) > min(on_plane):
            ref = normals[vert[int(np.argmax(on_plane))]]
        elif c2p:
            perp_scores = [max(abs(np.dot(normals[i], u)) for u in c2p) for i in vert]
            ref = normals[vert[int(np.argmin(perp_scores))]]
        else:
            ref = normals[vert[0]]

        other = "σv'" if nmax == 2 else "σd"
        for i in vert:
            ang = np.arccos(np.clip(abs(np.dot(normals[i], ref)), 0.0, 1.0))
            k = round(ang / (np.pi / nmax))
            labels[i] = "σv" if k % 2 == 0 else other
    return labels


def count_ops(matrices, axes, planes):
    c = Counter()
    for R in matrices:
        det = np.linalg.det(R)
        tr = np.trace(R)
        if det > 0:
            a = rot_angle(R)
            if a < 0.05:
                c["E"] += 1
            else:
                n, _ = angle_to_order(a)
                u = extract_axis(R)
                label = f"C{n}"
                if n == 2 and axes:
                    j = int(np.argmax([abs(np.dot(ax["u"], u)) for ax in axes]))
                    if axes[j].get("subtype"):
                        label = axes[j]["subtype"]
                c[label] += 1
        else:
            if abs(tr - 1.0) < 0.08:
                if planes:
                    n_vec = extract_improper_axis(R)
                    k = int(np.argmax([abs(np.dot(p["n"], n_vec)) for p in planes]))
                    c[planes[k]["type"]] += 1
                else:
                    c["σ"] += 1
            elif abs(tr + 3.0) < 0.08:
                c["i"] += 1
            else:
                u = extract_improper_axis(R)
                Q = R @ (np.eye(3) - 2.0 * np.outer(u, u))
                a = rot_angle(Q)
                n, _ = angle_to_order(a)
                c[f"S{n}"] += 1
    return c


def _ops_key(k):
    if k == "E":
        return (0, 0, 0)
    if k[0] == "C":
        m = re.match(r"^C(\d+)('*)$", k)
        n = int(m.group(1))
        primes = len(m.group(2))
        return (1, -n, primes)
    if k == "i":
        return (2, 0, 0)
    if k[0] == "S":
        return (3, -int(k[1:]), 0)
    return (4, PLANE_ORDER.index(k) if k in PLANE_ORDER else 99, 0)


def ops_string(c):
    out = []
    for k in sorted(c, key=_ops_key):
        n = c[k] if c[k] > 1 else ""
        out.append(f"{n}{sub(k) if k[0] in 'CSσ' else k}")
    return ", ".join(out)


def ops_classes(c):
    """The same as ops_string, as a list of [count, label (HTML), label (plain)] for the class table."""
    return [[int(c[k]), sub(k) if k[0] in "CSσ" else k, k] for k in sorted(c, key=_ops_key)]


# ---------------------------------------------------------------- classes (the header of a character table)
def conjugacy_classes(elements):
    """Split the group into conjugacy classes {g x g⁻¹}. Exact, because elements are atom permutations.
    Returns a list of lists of element indices."""
    key = {(p.tobytes(), d): i for i, (p, d) in enumerate(elements)}
    inv = [np.argsort(p) for p, _ in elements]
    done, out = set(), []
    for i, (x, dx) in enumerate(elements):
        if i in done:
            continue
        cls = set()
        for (g, dg), gi in zip(elements, inv):
            c = key.get((g[x[gi]].tobytes(), dx))           # apply g⁻¹, then x, then g
            if c is not None:
                cls.add(c)
        done |= cls
        out.append(sorted(cls))
    return out


def _op_kind(R):
    """('E'|'i'|'s'|'C'|'S', n, k, axis) for one operation; rotation angle 2πk/n with k ≤ n/2."""
    det, tr = np.linalg.det(R), np.trace(R)
    if det > 0:
        a = rot_angle(R)
        if a < 0.05:
            return "E", 1, 0, None
        n, k = angle_to_order(a)
        return "C", n, k, extract_axis(R)
    if abs(tr + 3.0) < 0.08:
        return "i", 2, 1, None
    if abs(tr - 1.0) < 0.08:
        return "s", 1, 1, extract_improper_axis(R)
    u = extract_improper_axis(R)
    n, k = angle_to_order(rot_angle(R @ (np.eye(3) - 2.0 * np.outer(u, u))))
    return "S", n, k, u


def class_table(matrices, elements, axes, planes, principal):
    """[[count, label HTML, label plain], ...] in the usual order, labelled with the same names as the
    elements (C2', C2'', σv, σd ...). Classes that share a label are told apart the way character tables do:
    C3 and C3², S6 and S6⁵, 3C2 (=C4²), C2(z) C2(y) C2(x), σ(xy) σ(xz) σ(yz)."""
    rows = []
    for cls in conjugacy_classes(elements):
        R = matrices[cls[0]]
        kind, n, k, u = _op_kind(R)
        if kind == "E":
            base = "E"
        elif kind == "i":
            base = "i"
        elif kind == "s":
            base = "σ"
            if planes:
                j = int(np.argmax([abs(np.dot(p["n"], u)) for p in planes]))
                base = planes[j]["type"]
        elif kind == "C":
            base = f"C{n}"
            if n == 2 and axes:
                j = int(np.argmax([abs(np.dot(ax["u"], u)) for ax in axes]))
                base = axes[j].get("subtype") or base
        else:
            base = f"S{n}"
            if n % 2 and k % 2 == 0:          # powers of S_n with n odd are odd: S5³, not S5²
                k = n - k
        rows.append({"size": len(cls), "base": base, "kind": kind, "n": n, "k": k, "u": u, "note": ""})

    # same label twice: tell them apart
    groups = {}
    for r in rows:
        groups.setdefault((r["base"], r["k"]), []).append(r)
    letters = {}
    if principal is not None:
        for name, v in zip("xyz", principal):
            letters[name] = v
    for (base, k), same in groups.items():
        if len(same) < 2:
            continue
        if same[0]["kind"] in "CS" and same[0]["n"] > 2:          # C3 / C3², S6 / S6⁵, C4 / C4³ ...
            n = same[0]["n"]
            for r in same[1:]:
                r["k"] = (2 * n - k) if (r["kind"] == "S" and n % 2) else (n - k)
            continue
        if same[0]["kind"] == "C" and same[0]["n"] == 2:
            higher = [ax for ax in axes if ax["n"] > 2 and ax["n"] % 2 == 0]
            rest = []
            for r in same:
                hit = [ax for ax in higher if abs(np.dot(ax["u"], r["u"])) > 0.99]
                if hit:
                    m = hit[0]["n"]
                    r["note"] = f" (=C<sub>{m}</sub><sup>{m // 2}</sup>)"
                    r["plain_note"] = f" (=C{m}^{m // 2})"
                else:
                    rest.append(r)
            if len(rest) > 1 and letters:
                for r in rest:
                    ax = max("xyz", key=lambda L: abs(np.dot(letters[L], r["u"])))
                    r["note"], r["plain_note"] = f"({ax})", f"({ax})"
            continue
        if same[0]["kind"] == "s" and letters:
            for r in same:
                ax = max("xyz", key=lambda L: abs(np.dot(letters[L], r["u"])))
                plane = "".join(c for c in "xyz" if c != ax)
                r["note"], r["plain_note"] = f"({plane})", f"({plane})"

    note_order = {"(z)": 0, "(y)": 1, "(x)": 2, "(xy)": 0, "(xz)": 1, "(yz)": 2}

    def key(r):
        return (_ops_key(r["base"]), r["k"], note_order.get(r["note"], 3), r["note"])

    out = []
    for r in sorted(rows, key=key):
        html = sub(r["base"]) if r["base"][0] in "CSσ" else r["base"]
        plain_lab = r["base"]
        if r["kind"] in "CS" and r["k"] > 1:
            html += f"<sup>{r['k']}</sup>"
            plain_lab += f"^{r['k']}"
        html += r["note"]
        plain_lab += r.get("plain_note", "")
        out.append([r["size"], html, plain_lab])
    return out


def linear_stuff(pg, coords):
    main = unit(np.linalg.svd(coords - coords.mean(0))[2][0])
    a = np.array([1.0, 0, 0]) if abs(main[0]) < 0.9 else np.array([0, 1.0, 0])
    p1 = unit(a - np.dot(a, main) * main)
    p2 = np.cross(main, p1)
    axes = [{"u": main, "n": 0, "sn": 0, "subtype": None}]
    normals = [p1, p2]
    types = ["σv", "σv"]
    if pg == "D*h":
        normals.insert(0, main)
        types.insert(0, "σh")
    return axes, normals, types


def rodrigues(u, a):
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def describe_ops(matrices, axes, planes):
    out = []
    for idx, R in enumerate(matrices):
        det, tr = np.linalg.det(R), np.trace(R)

        if det < 0 and abs(tr - 1.0) < 0.08:
            if not planes:
                continue
            n = extract_improper_axis(R)
            k = int(np.argmax([abs(np.dot(p["n"], n)) for p in planes]))
            p = planes[k]
            out.append({
                "kind": "s", "n": p["n"].tolist(), "name": p["name"], "group": "Reflections",
                "color": PLANE_COLORS[p["type"]], "sort": (3, k, 0, 0, 0), "m": idx,
                "desc": f"Reflect every atom through the {p['name']} plane"
            })
            continue

        if det < 0 and abs(tr + 3.0) < 0.08:
            out.append({
                "kind": "i", "name": "i", "group": "Inversion", "color": "#1D2A31",
                "sort": (1, 0, 0, 0, 0), "m": idx,
                "desc": "Send every atom straight through the centre to the opposite side"
            })
            continue

        if det > 0:
            u = extract_axis(R)
            Q = R
        else:
            u = extract_improper_axis(R)
            Q = R @ (np.eye(3) - 2.0 * np.outer(u, u))

        a = rot_angle(Q)
        if a < 0.05:
            continue
        if not np.allclose(rodrigues(u, a), Q, atol=0.05):
            a = -a

        j = int(np.argmax([abs(np.dot(ax["u"], u)) for ax in axes])) if axes else -1
        if j >= 0 and np.dot(axes[j]["u"], u) < 0:
            u, a = -u, -a
        a = float(a % (2.0 * np.pi))

        n, k = angle_to_order(a)
        letter = "C" if det > 0 else "S"

        primes = ""
        if letter == "C" and n == 2 and j >= 0 and axes[j].get("subtype"):
            primes = axes[j]["subtype"][2:]

        if k == 1:
            name = f"{letter}<sub>{n}</sub>{primes}"
        else:
            name = f"{letter}<span class='ss'><sup>{k}</sup><sub>{n}</sub></span>{primes}"
        tag = name

        if j >= 0:
            nax = axes[j]["n"]
            sub_ax = axes[j].get("subtype")
            same = [
                i for i, ax in enumerate(axes)
                if ax["n"] == nax and ax.get("subtype") == sub_ax
            ]
            num = f" {same.index(j) + 1}" if len(same) > 1 else ""
            ax_label = sub(axis_name(nax, sub_ax))
            if nax != n or letter == "S":
                name += f" (on {ax_label} axis{num})"
            elif num:
                name += f" (axis{num})"

        deg = round(np.degrees(a))
        if letter == "C":
            ax_str = sub(axis_name(axes[j]["n"], axes[j].get("subtype"))) if j >= 0 else ""
            desc = f"Rotate {deg}° about the {ax_str} axis"
            color = AXIS_SUBTYPE_COLORS.get(axes[j].get("subtype"), AXIS_COLORS.get(n, "#374151")) if j >= 0 else AXIS_COLORS.get(n, "#374151")
        else:
            desc = f"Rotate {deg}° about the axis, then reflect through the plane perpendicular to it"
            color = "#0F766E"

        out.append({
            "kind": letter, "u": u.tolist(), "a": a, "name": name, "tag": tag,
            "color": color, "desc": desc, "m": idx,
            "group": "Rotations" if letter == "C" else "Improper rotations",
            "sort": (0 if letter == "C" else 2, -n, len(primes), int(j >= 0 and axes[j]["n"] != n), j + k * 0.01)
        })

    out.sort(key=lambda o: o["sort"])
    for o in out:
        del o["sort"]
    return out


def atom_labels(symbols):
    count, labs = Counter(), []
    for el in symbols:
        count[el] += 1
        labs.append(f"{el}{count[el]}")
    return labs


# ---------------------------------------------------------------- everything the interface needs
def bonds_for_display(symbols, X, scale=1.2):
    r = np.array([COVALENT_RADII.get(s, 1.5) for s in symbols])
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    iu, ju = np.triu_indices(len(symbols), 1)
    ok = D[iu, ju] <= scale * (r[iu] + r[ju])
    return [[int(i), int(j)] for i, j in zip(iu[ok], ju[ok])]


def analyse(frame: Frame, tol: float = 0.3) -> dict:
    tol = float(tol)
    pg, matrices, mol = point_group(frame.symbols, frame.coords, tol)
    coords = mol.X
    linear = "*" in pg
    if linear:
        axes, normals, types = linear_stuff(pg, coords)
    elif pg == "Kh":
        axes, normals, types = [], [], []
    else:
        axes = get_axes(matrices, pg, coords, tol)
        normals = get_planes(matrices)
        types = label_planes(normals, pg, axes, coords, tol)

    order_idx = sorted(range(len(types)), key=lambda i: PLANE_ORDER.index(types[i]))
    planes, seen = [], Counter()
    for i in order_idx:
        t = types[i]
        seen[t] += 1
        name = sub(t) + (f" {seen[t]}" if types.count(t) > 1 else "")
        planes.append({"n": normals[i], "type": t, "name": name})

    counts = count_ops(matrices, axes, planes)
    devs = [mol.deviation(R) for R in matrices]
    ops = describe_ops(matrices, axes, planes) if pg != "Kh" else []
    for o in ops:
        o["dev"] = devs[o.pop("m")]

    if pg == "C*v":
        classes = [[1, "E", "E"], [2, "C<sub>∞</sub>", "C∞"], [0, "σ<sub>v</sub>", "σv"]]
    elif pg == "D*h":
        classes = [[1, "E", "E"], [2, "C<sub>∞</sub>", "C∞"], [0, "σ<sub>v</sub>", "σv"], [1, "i", "i"],
                   [2, "S<sub>∞</sub>", "S∞"], [0, "C<sub>2</sub>", "C2"]]
    elif pg == "Kh":
        classes = [[1, "E", "E"]]
    else:
        # the principal axes, sorted so z has the largest moment (used only to name x, y, z when needed)
        principal = [mol.principal[0], mol.principal[1], mol.principal[2]]
        classes = class_table(matrices, mol.elements, axes, planes, principal)

    # symmetry elements, with the atoms that sit on them and how well they fit
    ax_data = []
    for ax in axes:
        u = ax["u"]
        sub_ax = ax.get("subtype")
        same = [a for a in axes if a["n"] == ax["n"] and a.get("subtype") == sub_ax]
        short = sub(axis_name(ax["n"], sub_ax))
        name = short + " axis" + (f" {[id(a) for a in same].index(id(ax)) + 1}" if len(same) > 1 else "")
        dist = np.linalg.norm(coords - np.outer(coords @ u, u), axis=1)
        dev = max([o["dev"] for o in ops if o["kind"] == "C" and abs(np.dot(o["u"], u)) > 0.99] or [0.0])
        ax_data.append({"u": u.tolist(), "n": ax["n"], "sub": sub_ax, "short": short, "name": name,
                        "on": [int(i) for i in np.where(dist < tol)[0]], "dev": dev,
                        "color": AXIS_SUBTYPE_COLORS.get(sub_ax, AXIS_COLORS.get(ax["n"], "#374151"))})
    pl_data = []
    for p in planes:
        n = p["n"]
        dev = max([o["dev"] for o in ops if o["kind"] == "s" and abs(np.dot(o["n"], n)) > 0.99] or [0.0])
        pl_data.append({"n": n.tolist(), "type": p["type"], "short": sub(p["type"]), "name": p["name"],
                        "on": [int(i) for i in np.where(abs(coords @ n) < tol)[0]], "dev": dev,
                        "color": PLANE_COLORS[p["type"]]})
    has_i = bool(counts.get("i"))
    i_dev = next((o["dev"] for o in ops if o["kind"] == "i"), None)

    order = len(matrices)
    expected = expected_order(pg)
    return {
        "pg": pg, "pg_html": sub(pg), "linear": linear, "order": None if linear else order,
        "expected_order": expected, "consistent": linear or pg == "Kh" or (order == expected and mol.complete),
        "classes": classes, "ops_html": ", ".join(
            (f"{c[0]}" if c[0] > 1 else "∞" if c[0] == 0 else "") + c[1] for c in classes),
        "axes": ax_data, "planes": pl_data, "has_i": has_i, "i_dev": i_dev,
        "i_on": [int(i) for i in np.where(np.linalg.norm(coords, axis=1) < tol)[0]],
        "ops": ops, "tol": tol,
        "n": mol.n, "symbols": frame.symbols, "labels": atom_labels(frame.symbols),
        "coords": coords.tolist(), "center": mol.center.tolist(), "formula": formula(frame.symbols),
        "comment": frame.comment, "bonds": bonds_for_display(frame.symbols, coords),
        "vdw": [VDW.get(s, 1.8) for s in frame.symbols],
        "moments": mol.moments.tolist(),
    }


def tolerance_scan(frame: Frame, tols=(0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)):
    """The point group found at each tolerance: shows whether the answer depends on the setting."""
    out = []
    for t in tols:
        pg, mats, mol = point_group(frame.symbols, frame.coords, t)
        out.append({"tol": t, "pg": pg, "pg_html": sub(pg), "order": None if "*" in pg else len(mats),
                    "consistent": "*" in pg or pg == "Kh" or (len(mats) == expected_order(pg) and mol.complete)})
    return out


def plain(html: str) -> str:
    """HTML labels as plain text: C<sub>3</sub><sup>2</sup> -> C3^2."""
    html = re.sub(r"<span class='ss'><sup>(\d+)</sup><sub>(\d+)</sub></span>", r"\2^\1", html)
    html = re.sub(r"<sup>(\d+)</sup>", r"^\1", html)
    return re.sub(r"<[^>]+>", "", html)


def report_text(res: dict, name: str = "molecule.xyz") -> str:
    plane_counts = Counter(p["type"] for p in res["planes"])
    axis_counts = Counter(plain(a["short"]) for a in res["axes"])
    L = [
        "Point Group Finder report",
        f"File: {name}",
        f"Comment line: {res['comment']}",
        f"Formula: {res['formula']}   atoms: {res['n']}",
        f"Tolerance: {res['tol']} Å (largest distance an atom may be from its symmetric image)",
        "",
        f"Point group: {res['pg'].replace('*', '∞')}",
        f"Order h: {'∞' if res['linear'] else res['order']}",
        f"Operations: {plain(res['ops_html'])}",
    ]
    if not res["consistent"]:
        L.append(f"WARNING: {res['order']} operations found, but {res['pg']} has {res['expected_order']}. "
                 "Try a different tolerance.")
    L += ["", "Symmetry elements:"]
    L += [f"  {k} axis: {v}" for k, v in axis_counts.items()]
    L += [f"  {k} plane: {v if not res['linear'] or k != 'σv' else '∞'}" for k, v in plane_counts.items()]
    if res["has_i"]:
        L.append("  inversion centre: 1")
    if not res["ops"]:
        L.append("  none (only E)")
    L += ["", "Operations and how well they fit (largest atom displacement, Å):"]
    L += [f"  {plain(o['name']):<34}{o['dev']:.4f}" for o in res["ops"]]
    L += ["", "Centred coordinates (Å, centre of mass at the origin):"]
    L += [f"  {lab:<6}{x:12.6f}{y:12.6f}{z:12.6f}" for lab, (x, y, z) in zip(res["labels"], res["coords"])]
    return "\n".join(L) + "\n"
