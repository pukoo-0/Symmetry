"""Build test molecules with an exactly known point group: apply the group's operations to a few seed atoms."""
import numpy as np
import symmetry_core as sc

PHI = (1 + 5 ** 0.5) / 2
z = np.array([0, 0, 1.0]); x = np.array([1.0, 0, 0])


def gens(pg):
    import re
    if pg == "C1": return []
    if pg == "Cs": return [sc.reflection(z)]
    if pg == "Ci": return [sc.INVERSION]
    if pg in ("T", "Td", "Th"):
        g = [sc.rotation(z, np.pi), sc.rotation([1, 1, 1], 2 * np.pi / 3)]
        if pg == "Td": g.append(sc.reflection([1, -1, 0]))
        if pg == "Th": g.append(sc.INVERSION)
        return g
    if pg in ("O", "Oh"):
        return [sc.rotation(z, np.pi / 2), sc.rotation([1, 1, 1], 2 * np.pi / 3)] + ([sc.INVERSION] if pg == "Oh" else [])
    if pg in ("I", "Ih"):
        return [sc.rotation(z, np.pi), sc.rotation([1, 1, 1], 2 * np.pi / 3), sc.rotation([0, 1, PHI], 2 * np.pi / 5)] + ([sc.INVERSION] if pg == "Ih" else [])
    m = re.match(r"^([CDS])(\d+)([hvd]?)$", pg)
    L, n, s = m.group(1), int(m.group(2)), m.group(3)
    if L == "S": return [sc.improper(z, 2 * np.pi / n)]
    g = [sc.rotation(z, 2 * np.pi / n)]
    if L == "D": g.append(sc.rotation(x, np.pi))
    if s == "h": g.append(sc.reflection(z))
    if s == "v": g.append(sc.reflection([0, 1, 0]))
    if s == "d": g.append(sc.reflection([np.cos(np.pi / (2 * n) + np.pi / 2), np.sin(np.pi / (2 * n) + np.pi / 2), 0]))
    return g


SEEDS = (("C", [1.31, 0.47, 0.83]), ("N", [0.91, 1.37, -0.58]), ("O", [-0.52, 0.89, 1.43]))


ALT_SEEDS = (("C", [1.43, 0.31, 0.77]), ("N", [0.67, 1.52, -0.49]), ("O", [-0.83, 0.38, 1.36]))


def _polar(r, deg, z):
    return [r * np.cos(np.radians(deg)), r * np.sin(np.radians(deg)), z]


# angles chosen away from multiples of 15°, so no atom sits near a possible mirror plane
POLAR_SEEDS = (("C", _polar(1.4, 22, 0.83)), ("N", _polar(1.6, 52, -0.58)), ("O", _polar(1.0, 98, 1.43)))


def build(pg, seeds=None, center="Fe", scale=2.0):
    """Atoms at least 0.5 Å apart (otherwise the test molecule itself is degenerate)."""
    for s, f in [(s, f) for f in (1, 1.6, 2.5) for s in ([seeds] if seeds else [POLAR_SEEDS, SEEDS, ALT_SEEDS])]:
        syms, X, h = _build(pg, s, center, scale * f)
        D = np.linalg.norm(X[:, None] - X[None], axis=2)
        np.fill_diagonal(D, 9)
        if D.min() > 0.5:
            return syms, X, h
    rng = np.random.default_rng(7)                   # crowded groups (I, Ih): search for clear seed positions
    best = None
    for _ in range(400):
        s = tuple((el, list(rng.normal(size=3) * 1.5)) for el in ("C", "N", "O"))
        syms, X, h = _build(pg, s, center, scale)
        D = np.linalg.norm(X[:, None] - X[None], axis=2)
        np.fill_diagonal(D, 9)
        if best is None or D.min() > best[0]:
            best = (D.min(), syms, X, h)
        if D.min() > 0.5:
            return syms, X, h
    raise ValueError(f"no clean test molecule for {pg} (best spacing {best[0]:.2f} Å)")


def _build(pg, seeds, center, scale):
    group = sc.closure(gens(pg), atol=1e-6, limit=500) if gens(pg) else [np.eye(3)]
    syms, pts = [], []
    for el, p in seeds:
        p = np.array(p) * scale
        for R in group:
            q = R @ p
            if not any(np.linalg.norm(q - t) < 1e-6 for t in pts):
                syms.append(el); pts.append(q)
    if center:
        syms.append(center); pts.append(np.zeros(3))
    return syms, np.array(pts), len(group)


ALL = ["C1", "Cs", "Ci", "C2", "C3", "C4", "C5", "C6", "C2v", "C3v", "C4v", "C5v", "C6v", "C2h", "C3h", "C4h", "C6h",
       "D2", "D3", "D4", "D5", "D6", "D2h", "D3h", "D4h", "D5h", "D6h", "D8h", "D2d", "D3d", "D4d", "D5d", "D6d",
       "S4", "S6", "S8", "T", "Td", "Th", "O", "Oh", "I", "Ih"]
