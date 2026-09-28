import json
import re
from collections import Counter
from fractions import Fraction

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

# wide layout
st.set_page_config(page_title="Point group finder", page_icon="⚛️", layout="wide")

# css so it doesn't look like shit
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {background:#ffffff;}
.block-container {padding-top:2.4rem; padding-bottom:3rem; max-width:1320px;}
.stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .ui, .ui * {font-family:'Inter', system-ui, sans-serif;}
footer {visibility:hidden;}
header[data-testid="stHeader"] {background:transparent;}
[data-testid="stAppDeployButton"] {display:none;}
[data-testid="stFileUploaderDropzone"] {background:#f8fafc; border:1.5px dashed #d6dbe3; border-radius:12px;}
[data-testid="stExpander"] details {border:1px solid #eceef1; border-radius:12px;}
.stDownloadButton button {border-radius:10px; border:1px solid #e5e7eb; width:100%;}

.ui {color:#111827;}
.top {margin-bottom:20px;}
.title {font-size:30px; font-weight:700; letter-spacing:-.02em; margin:0;}
.subtitle {color:#6b7280; font-size:15px; margin:4px 0 0 0;}

.card {background:#fff; border:1px solid #eceef1; border-radius:14px; padding:18px 20px;
       box-shadow:0 1px 2px rgba(16,24,40,.04); margin-bottom:12px;}
.cap {font-size:13px; font-weight:500; color:#6b7280; margin:0;}
.pg-symbol {font-family:Georgia, 'Times New Roman', serif; font-size:68px; line-height:1.05; margin:4px 0 4px 0;}
.pg-ops {font-family:Georgia, 'Times New Roman', serif; font-size:19px; color:#374151; margin:0;}
.pg-order {font-size:13px; color:#6b7280; margin-top:12px;}

.stats {display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:12px;}
.stat {border:1px solid #eceef1; border-radius:12px; padding:12px 14px; background:#fafbfc;}
.stat .v {font-size:19px; font-weight:600; margin-top:2px;}

.row {display:flex; align-items:center; justify-content:space-between; padding:8px 0;
      border-bottom:1px solid #f3f4f6; font-size:14.5px;}
.row:last-child {border-bottom:0;}
.dot {display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:10px; vertical-align:middle;}
.cnt {color:#6b7280; font-variant-numeric:tabular-nums;}

.empty {text-align:center; padding:80px 20px; color:#6b7280; border:1px dashed #e2e5ea; border-radius:14px;}
.empty b {color:#111827; font-size:17px;}
</style>
""", unsafe_allow_html=True)

# axis colors, C2 blue C3 red C4 green etc etc (0 = C∞ for linear stuff)
AXIS_COLORS = {0: "#111827", 2: "#2563eb", 3: "#dc2626", 4: "#16a34a", 5: "#9333ea", 6: "#ea580c"}
# plane colors
PLANE_COLORS = {"σh": "#f59e0b", "σv": "#14b8a6", "σv'": "#0ea5e9", "σd": "#ec4899", "σ": "#6366f1"}
PLANE_ORDER = ["σh", "σv", "σv'", "σd", "σ"]


def sub(s):
    # turns C2v into C<sub>2v</sub> because its textbook symmetry
    s = s.replace("*", "∞")
    if len(s) == 1:
        return s
    return s[0] + "<sub>" + s[1:] + "</sub>"


def axis_name(n):
    return "C∞" if n == 0 else f"C{n}"


def unit(x):
    return x / np.linalg.norm(x)


# rotation angle from the matrix using the trace
def rot_angle(R):
    return np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))


def count_ops(ops):
    # sort the operations into E, Cn, i, Sn, sigma
    # det > 0 is a normal rotation, det < 0 means theres a mirror in it
    c = Counter()
    for op in ops:
        R = op.rotation_matrix
        if np.linalg.det(R) > 0:
            a = rot_angle(R)
            c["E" if a < 0.1 else "C" + str(round(2 * np.pi / a))] += 1
        else:
            a = np.arccos(np.clip((np.trace(R) + 1) / 2, -1, 1))
            if a < 0.1:
                c["σ"] += 1
            elif abs(a - np.pi) < 0.1:
                c["i"] += 1
            else:
                c["S" + str(round(2 * np.pi / a))] += 1
    return c


# makes the E, 8C3, 3C2 thing like the top of the character table
def ops_string(c):
    # E first, sigma last, same order as it was in symmetry
    def key(k):
        if k == "E": return (0, 0)
        if k[0] == "C": return (1, -int(k[1:]))
        if k == "i": return (2, 0)
        if k[0] == "S": return (3, -int(k[1:]))
        return (4, PLANE_ORDER.index(k))
    out = []
    for k in sorted(c, key=key):
        n = c[k] if c[k] > 1 else ""
        out.append(f"{n}{sub(k) if k[0] in 'CSσ' else k}")
    return ", ".join(out)


def get_axes(ops):
    # find the rotation axes, its the eigenvector with eigenvalue 1
    # if C4 and C2 are on the same axis just keep the C4
    axes = []
    for op in ops:
        R = op.rotation_matrix
        if np.linalg.det(R) < 0:
            continue
        a = rot_angle(R)
        if a < 0.1:
            continue
        n = round(2 * np.pi / a)
        w, v = np.linalg.eig(R)
        u = unit(np.real(v[:, np.argmin(abs(w - 1))]))
        for ax in axes:
            if abs(np.dot(ax["u"], u)) > 0.99:
                ax["n"] = max(ax["n"], n)
                break
        else:
            axes.append({"u": u, "n": n})
    return axes


def get_planes(ops):
    # mirror planes = ops with det -1 and trace 1 (only one direction gets flipped)
    # the plane normal is the eigenvector with eigenvalue -1, the flipped direction
    normals = []
    for op in ops:
        R = op.rotation_matrix
        if np.linalg.det(R) > 0 or abs(np.trace(R) - 1) > 0.1:
            continue
        w, v = np.linalg.eig(R)
        n = unit(np.real(v[:, np.argmin(abs(w + 1))]))
        if all(abs(np.dot(n, m)) < 0.99 for m in normals):
            normals.append(n)
    return normals


def label_planes(normals, pg, axes, coords, tol):
    # decide if each plane is σh, σv or σd
    # σh = perpendicular to the main axis, σv = contains the main axis,
    # σd = contains the main axis but sits in between the C2 axes
    if not normals:
        return []
    if pg == "Cs":
        return ["σh"]
    if pg == "Ih" or not axes:
        return ["σ"] * len(normals)
    if pg == "Td":
        return ["σd"] * len(normals)
    if pg in ("Oh", "Th"):
        # cube groups, planes facing a C4 (Oh) or C2 (Th) are σh, rest are σd
        cube = [a["u"] for a in axes if a["n"] == (4 if pg == "Oh" else 2)]
        return ["σh" if any(abs(np.dot(n, u)) > 0.99 for u in cube) else "σd" for n in normals]

    nmax = max(a["n"] for a in axes)
    top = [a for a in axes if a["n"] == nmax]
    is_dnd = pg[0] == "D" and pg.endswith("d")
    if len(top) > 1 and not is_dnd:
        # like D2h, no single main axis so just call them σ
        return ["σ"] * len(normals)
    if is_dnd:
        # D2d: the main C2 is the one sitting on the S4, the others are perpendicular to the planes anyway
        return ["σd"] * len(normals)
    main = top[0]["u"]

    labels = ["σh" if abs(np.dot(n, main)) > 0.99 else "σv" for n in normals]

    # C4v, D4h, D6h etc (even n) have 2 sets of vertical planes
    # the set going through more atoms is σv, the other set is σd
    # C2v is the same idea but textbook calls them σv and σv'
    vert = [i for i, l in enumerate(labels) if l == "σv"]
    if nmax % 2 == 0 and len(vert) > 1:
        on_plane = [np.sum(abs(coords @ normals[i]) < tol) for i in vert]
        ref = normals[vert[int(np.argmax(on_plane))]]
        other = "σv'" if nmax == 2 else "σd"
        for i in vert:
            ang = np.arccos(np.clip(abs(np.dot(normals[i], ref)), 0, 1))
            k = round(ang / (np.pi / nmax))
            labels[i] = "σv" if k % 2 == 0 else other
    return labels


def linear_stuff(pg, coords):
    # linear molecules have infinite planes so pymatgen doesnt list them
    # just draw the C∞ axis, 2 of the σv as an example, and σh if its D∞h
    main = unit(np.linalg.svd(coords - coords.mean(0))[2][0])
    a = np.array([1.0, 0, 0]) if abs(main[0]) < 0.9 else np.array([0, 1.0, 0])
    p1 = unit(a - np.dot(a, main) * main)
    p2 = np.cross(main, p1)
    axes = [{"u": main, "n": 0}]
    normals = [p1, p2]
    types = ["σv", "σv"]
    if pg == "D*h":
        normals.insert(0, main)
        types.insert(0, "σh")
    return axes, normals, types


def rodrigues(u, a):
    # rotation matrix for angle a around axis u
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def eigvec(R, val):
    w, v = np.linalg.eig(R)
    return unit(np.real(v[:, np.argmin(abs(w - val))]))




def describe_ops(ops, axes, planes):
    # turn every symmetry op into something the animation can play
    # C = rotate, s = reflect, i = invert, S = rotate then reflect
    out = []
    for op in ops:
        R = op.rotation_matrix
        det, tr = np.linalg.det(R), np.trace(R)

        if det < 0 and abs(tr - 1) < 0.1:
            if not planes:
                continue
            # mirror plane, find which one of our planes it is
            n = eigvec(R, -1)
            k = int(np.argmax([abs(np.dot(p["n"], n)) for p in planes]))
            p = planes[k]
            out.append({"kind": "s", "n": p["n"].tolist(), "name": p["name"], "group": "Reflections",
                        "color": PLANE_COLORS[p["type"]], "sort": (3, k, 0),
                        "desc": f"Reflect every atom through the {p['name']} plane"})
            continue
        if det < 0 and abs(tr + 3) < 0.1:
            out.append({"kind": "i", "name": "i", "group": "Inversion", "color": "#111827", "sort": (1, 0, 0),
                        "desc": "Send every atom straight through the center to the opposite side"})
            continue

        # rotation or improper rotation, both have an axis and an angle
        if det > 0:
            u = eigvec(R, 1)
            Q = R
        else:
            u = eigvec(R, -1)
            Q = R @ (np.eye(3) - 2 * np.outer(u, u))  # take the mirror part off, whats left is a rotation
        a = rot_angle(Q)
        if a < 0.1:
            continue  # this is E
        if not np.allclose(rodrigues(u, a), Q, atol=0.05):
            a = -a

        # line the axis up with one from the axes list so C3 and C3² point the same way
        j = int(np.argmax([abs(np.dot(ax["u"], u)) for ax in axes])) if axes else -1
        if j >= 0 and np.dot(axes[j]["u"], u) < 0:
            u, a = -u, -a
        a = a % (2 * np.pi)

        # angle as a fraction of a full turn gives the n and the power, e.g. 240° = 2/3 turn = C3²
        f = Fraction(a / (2 * np.pi)).limit_denominator(12)
        n, k = f.denominator, f.numerator
        letter = "C" if det > 0 else "S"
        # names are html so the subscripts come out properly, like C<sub>3</sub><sup>2</sup>
        # power on top of the subscript like in the textbook: C with 2 over 3
        name = f"{letter}<sub>{n}</sub>" if k == 1 else f"{letter}<span class='ss'><sup>{k}</sup><sub>{n}</sub></span>"
        tag = name  # short version for the label in the 3d view

        # say which axis if there is more than one of the same kind
        # like C₃² (axis 2), or C₂ (on C₄ axis 1) when it sits on a bigger axis, S always says which axis
        if j >= 0:
            nax = axes[j]["n"]
            same = [i for i, ax in enumerate(axes) if ax["n"] == nax]
            num = f" {same.index(j) + 1}" if len(same) > 1 else ""
            if nax != n or letter == "S":
                name += f" (on {sub(axis_name(nax))} axis{num})"
            elif num:
                name += f" (axis{num})"

        deg = round(np.degrees(a))
        if letter == "C":
            desc = f"Rotate {deg}° about the " + (sub(axis_name(axes[j]["n"])) if j >= 0 else "") + " axis"
            color = AXIS_COLORS.get(n, "#374151")
        else:
            desc = f"Rotate {deg}° about the axis, then reflect through the plane perpendicular to it"
            color = "#0f766e"
        out.append({"kind": letter, "u": u.tolist(), "a": a, "name": name, "tag": tag, "color": color, "desc": desc,
                    "group": "Rotations" if letter == "C" else "Improper rotations",
                    "sort": (0 if letter == "C" else 2, -n, int(j >= 0 and axes[j]["n"] != n), j, k)})

    out.sort(key=lambda o: o["sort"])
    for o in out:
        del o["sort"]
    return out


def atom_labels(mol):
    # number each element separately, C1 C2 C3 ... H1 H2 H3 ..., in the same order as the xyz file
    count, labs = Counter(), []
    for s in mol:
        el = str(s.specie)
        count[el] += 1
        labs.append(f"{el}{count[el]}")
    return labs


# define the viewer to see the molecule
def viewer(mol, axes, planes, tol, anim_ops, has_i, height=540):
    xyz = mol.to(fmt="xyz")
    coords = mol.cart_coords
    size = max(np.linalg.norm(coords, axis=1).max(), 1.0)

    # atom positions, element, and a rough size so the marker ball fits around it in spacefill too
    vdw = {"H": 1.1, "C": 1.7, "N": 1.55, "O": 1.52, "F": 1.47, "Cl": 1.75, "S": 1.8, "P": 1.8, "Br": 1.85}
    # atom colors 3Dmol uses (rasmol), so the label text can be dark on light atoms and white on dark ones
    colors = {"H": "ffffff", "B": "00ff00", "C": "c8c8c8", "N": "8f8fff", "O": "f00000", "F": "daa520",
              "Na": "0000ff", "Mg": "228b22", "Si": "daa520", "P": "ffa500", "S": "ffc832", "Cl": "00ff00",
              "Fe": "ffa500", "Cu": "a52a2a", "Zn": "a52a2a", "Br": "a52a2a", "I": "a020f0"}

    def light(el):
        h = colors.get(el, "ff1493")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return 0.299 * r + 0.587 * g + 0.114 * b > 150

    atoms = [{"p": c.tolist(), "el": str(s.specie), "lab": lab, "r": vdw.get(str(s.specie), 1.8),
              "light": light(str(s.specie))}
             for c, s, lab in zip(coords, mol, atom_labels(mol))]

    # axes, numbered per order like C₃ axis 1, C₃ axis 2 so u can pick them one by one
    # atoms on the axis = distance from the axis line less than the tolerance
    ax_data = []
    for ax in axes:
        u = ax["u"]
        same = [a for a in axes if a["n"] == ax["n"]]
        name = sub(axis_name(ax["n"])) + " axis"
        if len(same) > 1:
            name += f" {[id(a) for a in same].index(id(ax)) + 1}"
        dist = np.linalg.norm(coords - np.outer(coords @ u, u), axis=1)
        ax_data.append({"u": u.tolist(), "n": ax["n"], "short": sub(axis_name(ax["n"])), "name": name,
                        "on": [int(i) for i in np.where(dist < tol)[0]],
                        "color": AXIS_COLORS.get(ax["n"], "#374151")})

    # planes, atoms on the plane = distance to the plane less than the tolerance
    pl_data = []
    for p in planes:
        n = p["n"]
        on = [int(i) for i in np.where(abs(coords @ n) < tol)[0]]
        pl_data.append({"n": n.tolist(), "on": on, "type": p["type"], "short": sub(p["type"]), "name": p["name"],
                        "color": PLANE_COLORS[p["type"]]})

    # display styles
    styles = {
        "Ball & stick": {"stick": {"radius": 0.14}, "sphere": {"scale": 0.25}},
        "Sticks": {"stick": {"radius": 0.18}},
        "Spacefill": {"sphere": {"scale": 0.9}},
    }

    data = {"xyz": xyz, "atoms": atoms, "axes": ax_data, "planes": pl_data, "ops": anim_ops, "hasI": bool(has_i),
            "styles": styles, "R": size * 1.12, "L": size * 1.3}

    # 3Dmol does the 3d part, then make it so i can input a file into the 3Dmol
    # the buttons are inside here too so the camera doesnt reset every time u click something
    html = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
* {box-sizing:border-box; font-family:Inter, system-ui, sans-serif;}
body {margin:0; background:#fff; color:#111827;}
.wrap {border:1px solid #eceef1; border-radius:14px; overflow:hidden; background:#fff; box-shadow:0 1px 2px rgba(16,24,40,.04);}
.bar {display:flex; flex-wrap:wrap; gap:8px; align-items:center; justify-content:space-between; padding:10px 12px; border-bottom:1px solid #f1f2f4;}
.seg {display:inline-flex; background:#f3f4f6; border-radius:9px; padding:3px;}
.seg button {border:0; background:transparent; padding:5px 11px; font-size:12.5px; border-radius:7px; cursor:pointer; color:#4b5563;}
.seg button.on {background:#fff; color:#111827; box-shadow:0 1px 2px rgba(0,0,0,.1); font-weight:500;}
.group {display:inline-flex; gap:6px; flex-wrap:wrap; align-items:center;}
.chip {border:1px solid #e5e7eb; background:#fff; border-radius:999px; padding:4px 11px; font-size:12.5px; cursor:pointer;
       color:#374151; display:inline-flex; align-items:center; gap:6px;}
.chip:hover {border-color:#cbd0d8;}
button:focus-visible, select:focus-visible, input:focus-visible {outline:2px solid #2563eb; outline-offset:2px;}
.chip.off {color:#a1a7b0; background:#fafafa;}
.chip .d {width:8px; height:8px; border-radius:50%;}
.chip.off .d {opacity:.25;}
select {font-size:13px; padding:6px 8px; border:1px solid #e5e7eb; border-radius:9px; background:#fff; color:#111827;}
.small {font-size:12.5px; padding:4px 6px;}
#vwrap {position:relative;}
#v {width:100%; height:__H__px; position:relative;}
#alabs {position:absolute; inset:0; pointer-events:none; overflow:hidden; z-index:5;}
.al {position:absolute; left:0; top:0; font-size:11px; font-weight:600; line-height:13px; white-space:nowrap;
     will-change:transform; color:#111827; text-shadow:0 0 2px rgba(255,255,255,.9), 0 0 1px rgba(255,255,255,.9);}
.al.dk {color:#fff; text-shadow:0 0 2px rgba(0,0,0,.75), 0 0 1px rgba(0,0,0,.75);}
.tag {position:absolute; left:0; top:0; font-size:12px; font-weight:600; color:#fff; padding:1px 7px 2px; border-radius:6px;
      white-space:nowrap; will-change:transform; z-index:2;}
sub, sup {font-size:.72em; line-height:0;}
.ss {display:inline-flex; flex-direction:column; vertical-align:middle; font-size:.7em; line-height:1; margin-left:1px;}
.ss sup, .ss sub {font-size:1em; line-height:1; vertical-align:baseline;}

/* dropdown made by hand, the normal one cant show subscripts */
.dd {position:relative; display:inline-block;}
.dd-btn {display:inline-flex; align-items:center; justify-content:space-between; gap:12px; min-width:200px; font-size:13px;
         padding:6px 10px; border:1px solid #e5e7eb; border-radius:9px; background:#fff; color:#111827; cursor:pointer; text-align:left;}
.dd-btn:hover {border-color:#cbd0d8;}
.dd-caret {color:#9ca3af; font-size:11px;}
.dd-menu {display:none; position:absolute; left:0; top:calc(100% + 4px); z-index:20; min-width:100%; max-height:280px; overflow:auto;
          background:#fff; border:1px solid #e5e7eb; border-radius:10px; box-shadow:0 8px 24px rgba(16,24,40,.12); padding:4px;}
.dd.up .dd-menu {top:auto; bottom:calc(100% + 4px);}
.dd.open .dd-menu {display:block;}
.dd-group {font-size:11.5px; color:#6b7280; font-weight:600; padding:8px 10px 4px;}
.dd-item {display:block; width:100%; text-align:left; border:0; background:none; padding:6px 10px; font-size:13px;
          border-radius:7px; cursor:pointer; color:#111827; white-space:nowrap;}
.dd-item:hover, .dd-item:focus-visible {background:#f3f4f6;}
.dd-item.sel {background:#eef2ff; font-weight:500;}
.row {padding:10px 12px; border-top:1px solid #f1f2f4; display:flex; flex-wrap:wrap; gap:8px; align-items:center;}
.row.dim {opacity:.4; pointer-events:none;}
.lbl {font-size:12.5px; color:#6b7280; font-weight:500; margin-right:2px;}
.muted {font-size:12.5px; color:#6b7280;}
.step {border:1px solid #e5e7eb; background:#fff; border-radius:9px; width:32px; height:31px; cursor:pointer; font-size:13px; color:#374151;}
.step:hover {border-color:#cbd0d8;}
.swatch {display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:middle;}
#elSel {min-width:200px;}

.anim {padding:12px; border-top:1px solid #f1f2f4; background:#fafbfc;}
.anim .row1 {display:flex; flex-wrap:wrap; gap:8px; align-items:center;}
#opSel {min-width:190px;}
.play {border:0; background:#111827; color:#fff; border-radius:9px; padding:6px 14px; font-size:13px; font-weight:500; cursor:pointer; min-width:72px;}
.play:disabled {background:#d1d5db; cursor:default;}
.anim input[type=range] {flex:1; min-width:140px; accent-color:#111827;}
.cap {font-size:13px; color:#374151; margin-top:9px; min-height:18px;}
.cap .ok {color:#15803d; font-weight:500;}
</style>

<div class="wrap">
  <div class="bar">
    <div class="seg" id="seg"></div>
    <div class="group">
      <label class="lbl" for="atLab">Atom labels</label>
      <select id="atLab" class="small">
        <option value="off">Off</option>
        <option value="el">C, H</option>
        <option value="num">C1, C2 … H1, H2</option>
      </select>
      <button class="chip" id="tLab">Element labels</button>
      <button class="chip" id="tFill">Plane fill</button>
      <button class="chip" id="tMark"><span class="d" style="background:#f59e0b"></span>Mark atoms on element</button>
      <button class="chip" id="tReset">Reset view</button>
    </div>
  </div>
  <div id="vwrap"><div id="v"></div><div id="alabs"></div></div>
  <div class="row" id="elRow">
    <span class="lbl">Show</span>
    <div class="dd" id="elSel"></div>
    <button class="step" id="prev" aria-label="Previous element">◀</button>
    <button class="step" id="next" aria-label="Next element">▶</button>
    <span class="muted" id="elInfo"></span>
  </div>
  <div class="anim">
    <div class="row1">
      <span class="lbl">Animate</span>
      <div class="dd" id="opSel"></div>
      <button class="play" id="play">Play</button>
      <input type="range" id="scrub" min="0" max="1000" value="0" aria-label="Animation progress">
      <button class="chip" id="tLoop">Loop</button>
      <button class="chip" id="tGhost">Start positions</button>
    </div>
    <div class="cap" id="cap"></div>
  </div>
</div>

<script>
const D = __DATA__;
const S = {style: Object.keys(D.styles)[0], labels: true, fill: true, mark: true, atomLab: "off",
           el: "all", op: -1, t: 0, playing: false, loop: true, ghost: true};
const v = $3Dmol.createViewer("v", {backgroundColor: "white"});
const $ = id => document.getElementById(id);
const model = v.addModel(D.xyz, "xyz");
const atoms = model.selectedAtoms({});
const P0 = D.atoms.map(a => a.p);
// second copy of the molecule that never moves, drawn faint grey = the start positions
const ghost = v.addModel(D.xyz, "xyz");
function ghostStyle() {
  const g = {color: "#9ca3af", opacity: 0.3};
  if (S.style === "Spacefill") return {sphere: {...g, scale: 0.92}};
  if (S.style === "Sticks") return {sphere: {...g, radius: 0.24}};
  return {sphere: {...g, scale: 0.3}};
}

// every symmetry element in one list: axes, then planes, then the inversion center
const ELS = [];
// axes sorted biggest order first (C∞ counts as the biggest)
[...D.axes].sort((a, b) => (b.n || 99) - (a.n || 99)).forEach(a => ELS.push({kind: "axis", name: a.name, color: a.color, u: a.u, short: a.short, on: a.on}));
D.planes.forEach(p => ELS.push({kind: "plane", name: p.name, color: p.color, n: p.n, short: p.short, on: p.on}));
if (D.hasI) ELS.push({kind: "i", name: "Inversion center", color: "#111827", on: P0.map((p, i) => i).filter(i => Math.hypot(...P0[i]) < 0.3)});

// small vector math
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const add = (a, b, s = 1) => [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
function pt(p) { return {x: p[0], y: p[1], z: p[2]}; }
function norm(a) { const l = Math.hypot(...a); return mul(a, 1 / l); }
function basis(n) {
  const a = Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const e1 = norm(add(a, n, -dot(a, n)));
  return {n: n, e1: e1, e2: cross(n, e1)};
}
// rotate p around axis u by angle a (rodrigues)
function rotate(p, u, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return add(add(mul(p, c), cross(u, p), s), u, dot(u, p) * (1 - c));
}
// move p towards its mirror image, s = 0 start, s = 1 fully reflected
function reflect(p, n, s) { return add(p, n, -2 * s * dot(p, n)); }
const ease = x => x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2;

// where an atom is at time t of the operation
function pos(p, op, t) {
  if (op.kind === "C") return rotate(p, op.u, op.a * ease(t));
  if (op.kind === "s") return reflect(p, op.n, ease(t));
  if (op.kind === "i") return mul(p, 1 - 2 * ease(t));
  // S = rotate first half, reflect through the perpendicular plane second half
  if (t < 0.5) return rotate(p, op.u, op.a * ease(2 * t));
  return reflect(rotate(p, op.u, op.a), op.u, ease(2 * t - 1));
}

// disc (or a ring if r0 > 0) for a plane, both faces so u can see it from any angle
function disc(b, r0, r1, color, alpha) {
  const N = 72, V = [], Nm = [], F = [];
  for (const s of [1, -1]) {
    const off = V.length;
    for (let i = 0; i < N; i++) {
      const t = 2 * Math.PI * i / N, c = Math.cos(t), sn = Math.sin(t);
      for (const r of [r0, r1]) {
        V.push(pt([0, 1, 2].map(k => r * (c * b.e1[k] + sn * b.e2[k]))));
        Nm.push(pt(mul(b.n, s)));
      }
    }
    for (let i = 0; i < N; i++) {
      const a = off + 2 * i, bb = a + 1, c2 = off + 2 * ((i + 1) % N), d = c2 + 1;
      if (s > 0) F.push(a, bb, d, a, d, c2); else F.push(a, d, bb, a, c2, d);
    }
  }
  v.addCustom({vertexArr: V, normalArr: Nm, faceArr: F, color: color, alpha: alpha});
}
// strong = only this plane is on screen, so it gets a darker fill and grid lines
// the grid lines make it easy to tell where the plane is from any angle, even face on
function plane(n, color, text, strong) {
  const b = basis(n);
  if (S.fill) {
    disc(b, 0, D.R, color, strong ? 0.2 : 0.08);
    if (strong) {
      const k = 8;  // grid lines every 1/8 of the radius
      for (let i = -k + 1; i < k; i++) {
        const d = D.R * i / k, h = Math.sqrt(D.R * D.R - d * d);
        for (const [e, f] of [[b.e1, b.e2], [b.e2, b.e1]]) {
          v.addCylinder({start: pt(add(mul(e, d), f, -h)), end: pt(add(mul(e, d), f, h)),
                         radius: 0.015, color: color, alpha: 0.45, fromCap: 1, toCap: 1});
        }
      }
    }
  }
  disc(b, D.R * 0.96, D.R, color, 1);
  if (S.labels && text) label(text, pt([0, 1, 2].map(k => D.R * (Math.cos(0.7) * b.e1[k] + Math.sin(0.7) * b.e2[k]))), color);
}
function axisLine(u, color, text) {
  const p = mul(u, D.L);
  v.addCylinder({start: pt(mul(p, -1)), end: pt(p), radius: 0.07, color: color, fromCap: 1, toCap: 1});
  if (S.labels && text) label(text, pt(p), color);
}
function center(color) {
  v.addSphere({center: {x: 0, y: 0, z: 0}, radius: 0.18, color: color});
  if (S.labels) label("i", {x: 0, y: 0, z: 0}, color);
}
// element labels are html too (like the atom labels) so C<sub>3</sub> and σ<sub>h</sub> get real subscripts
let tags = [];
function label(html, pos, color) { tags.push({html: html, pos: pos, color: color}); }
function atomR(a) { return S.style === "Spacefill" ? a.r * 0.9 : S.style === "Sticks" ? 0.18 : a.r * 0.25; }
function drawEl(e) {
  if (e.kind === "axis") axisLine(e.u, e.color, e.short);
  if (e.kind === "plane") plane(e.n, e.color, e.short, typeof S.el === "number");
  if (e.kind === "i") center(e.color);
}
// see-through ball around the atoms that sit on an element, first element wins if an atom is on more than one
function markAtoms(list) {
  const done = new Set();
  list.forEach(e => e.on.forEach(j => {
    if (done.has(j)) return;
    done.add(j);
    v.addSphere({center: pt(P0[j]), radius: atomR(D.atoms[j]) + 0.22, color: e.color, alpha: 0.45});
  }));
}

// draw() redoes everything, only used when something changes (new element, new operation, a toggle)
// during the animation only frame() runs, which just moves the atoms, way cheaper
function draw() {
  const op = S.op >= 0 ? D.ops[S.op] : null;
  v.removeAllShapes();
  v.removeAllLabels();
  tags = [];
  ghost.setStyle({}, op && S.ghost ? ghostStyle() : {});

  if (op) {
    // animation mode, only show the element that belongs to this operation
    if (op.kind === "C") axisLine(op.u, op.color, op.tag);
    if (op.kind === "S") { axisLine(op.u, op.color, op.tag); plane(op.u, "#64748b", "", true); }
    if (op.kind === "s") plane(op.n, op.color, op.name, true);
    if (op.kind === "i") center(op.color);
  } else {
    // which elements to show: everything, only axes, only planes, or just one
    let list;
    if (S.el === "all") list = ELS;
    else if (S.el === "axes") list = ELS.filter(e => e.kind === "axis");
    else if (S.el === "planes") list = ELS.filter(e => e.kind === "plane");
    else list = [ELS[S.el]];
    list.forEach(drawEl);
    // in the all views only mark atoms on planes, otherwise the whole molecule lights up
    if (S.mark) markAtoms(typeof S.el === "number" ? list : list.filter(e => e.kind === "plane"));
  }
  moveAtoms();
  model.setStyle({}, D.styles[S.style]);  // full rebuild of the molecule
  buildLabels();
  v.render();
  ui();
}

// put every atom where it should be at time t
function moveAtoms() {
  const op = S.op >= 0 ? D.ops[S.op] : null;
  atoms.forEach((a, i) => {
    const q = op ? pos(P0[i], op, S.t) : P0[i];
    a.x = q[0]; a.y = q[1]; a.z = q[2];
  });
}

// one animation frame: move atoms, then either update the existing 3d shapes in place
// (newer 3Dmol has syncAtomPositions for this) or rebuild just the molecule
function frame() {
  moveAtoms();
  if (!(model.syncAtomPositions && model.syncAtomPositions())) model.setStyle({}, D.styles[S.style]);
  v.render();
  uiAnim();
}

// atom labels are normal html on top of the 3d view, moving them is basically free
// compared to making 120 new 3Dmol labels every frame (that was what made it laggy)
const lbox = $("alabs");
let lspans = [], tspans = [];
function buildLabels() {
  lbox.innerHTML = "";
  lspans = [];
  tspans = [];
  if (S.atomLab !== "off") D.atoms.forEach(a => {
    const s = document.createElement("span");
    s.className = a.light ? "al" : "al dk";
    s.textContent = S.atomLab === "el" ? a.el : a.lab;
    lbox.appendChild(s);
    lspans.push(s);
  });
  tags.forEach(t => {
    const s = document.createElement("span");
    s.className = "tag";
    s.style.background = t.color;
    s.innerHTML = t.html;
    lbox.appendChild(s);
    tspans.push(s);
  });
  placeLabels();
}
function placeLabels() {
  if (!lspans.length && !tspans.length) return;
  const cv = $("v").querySelector("canvas");
  if (!cv) return;
  const r = cv.getBoundingClientRect();
  const left = r.left + window.pageXOffset - document.documentElement.clientLeft;
  const top = r.top + window.pageYOffset - document.documentElement.clientTop;
  const pts = (lspans.length ? atoms.map(a => ({x: a.x, y: a.y, z: a.z})) : []).concat(tags.map(t => t.pos));
  const sc = v.modelToScreen(pts);
  const spans = lspans.concat(tspans);
  sc.forEach((p, i) => {
    spans[i].style.transform = "translate(" + (p.x - left) + "px," + (p.y - top) + "px) translate(-50%,-50%)";
  });
}
// 3Dmol calls this after every render, so labels follow when u rotate/zoom too
v.setViewChangeCallback(placeLabels);

// toolbar
const seg = $("seg");
Object.keys(D.styles).forEach(k => {
  const b = document.createElement("button");
  b.textContent = k;
  b.onclick = () => { S.style = k; draw(); };
  seg.appendChild(b);
});
$("atLab").onchange = () => { S.atomLab = $("atLab").value; draw(); };
$("tLab").onclick = () => { S.labels = !S.labels; draw(); };
$("tFill").onclick = () => { S.fill = !S.fill; draw(); };
$("tMark").onclick = () => { S.mark = !S.mark; draw(); };
$("tReset").onclick = () => { v.zoomTo(); v.render(); };

// dropdown that can show html (subscripts), groups = [{title, items: [{value, html}]}]
function makeDD(root, groups, onPick) {
  root.innerHTML = '<button type="button" class="dd-btn" aria-haspopup="listbox"><span class="dd-val"></span>' +
                   '<span class="dd-caret">▾</span></button><div class="dd-menu" role="listbox"></div>';
  const btn = root.querySelector(".dd-btn"), menu = root.querySelector(".dd-menu"), val = root.querySelector(".dd-val");
  const byVal = {};
  groups.forEach(g => {
    if (g.title) {
      const h = document.createElement("div");
      h.className = "dd-group"; h.textContent = g.title;
      menu.appendChild(h);
    }
    g.items.forEach(it => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "dd-item"; b.innerHTML = it.html;
      b.onclick = () => { root.classList.remove("open"); onPick(it.value); };
      menu.appendChild(b);
      byVal[String(it.value)] = {it: it, b: b};
    });
  });
  btn.onclick = ev => {
    ev.stopPropagation();
    const opening = !root.classList.contains("open");
    closeDDs();
    if (!opening) return;
    // open upwards if there is no room below (the viewer is above, so there always is room there)
    const r = btn.getBoundingClientRect();
    root.classList.toggle("up", window.innerHeight - r.bottom < 300);
    root.classList.add("open");
    const cur = menu.querySelector(".sel");
    if (cur && cur.scrollIntoView) cur.scrollIntoView({block: "nearest"});
  };
  return {set(v) {
    const x = byVal[String(v)];
    val.innerHTML = x ? x.it.html : "";
    Object.values(byVal).forEach(o => o.b.classList.toggle("sel", o === x));
  }};
}
function closeDDs() { document.querySelectorAll(".dd.open").forEach(d => d.classList.remove("open")); }
document.addEventListener("click", closeDDs);
document.addEventListener("keydown", e => { if (e.key === "Escape") closeDDs(); });

// element picker, the list plus ◀ ▶ to go through them one at a time
const elGroups = [];
if (ELS.length === 0) {
  elGroups.push({title: null, items: [{value: "all", html: "No symmetry elements"}]});
} else {
  const top = [{value: "all", html: "All elements"}];
  if (D.axes.length) top.push({value: "axes", html: "All axes"});
  if (D.planes.length) top.push({value: "planes", html: "All mirror planes"});
  elGroups.push({title: null, items: top});
  for (const [kind, title] of [["axis", "Axes"], ["plane", "Mirror planes"], ["i", "Other"]]) {
    const idx = ELS.map((e, i) => i).filter(i => ELS[i].kind === kind);
    if (idx.length) elGroups.push({title: title, items: idx.map(i => ({value: i, html: ELS[i].name}))});
  }
}
function pick(val) {
  S.el = (val === "all" || val === "axes" || val === "planes") ? val : +val;
  draw();
}
const elDD = makeDD($("elSel"), elGroups, pick);
function stepEl(d) {
  if (!ELS.length) return;
  let i = typeof S.el === "number" ? S.el + d : (d > 0 ? 0 : ELS.length - 1);
  pick((i + ELS.length) % ELS.length);
}
$("prev").onclick = () => stepEl(-1);
$("next").onclick = () => stepEl(1);

// animation controls, the list is grouped like rotations / inversion / reflections
const playBtn = $("play"), scrub = $("scrub"), cap = $("cap");
const opGroups = [{title: null, items: [{value: -1, html: D.ops.length ? "Pick an operation" : "Only E, nothing to animate"}]}];
D.ops.forEach((o, i) => {
  let g = opGroups[opGroups.length - 1];
  if (g.title !== o.group) { g = {title: o.group, items: []}; opGroups.push(g); }
  g.items.push({value: i, html: o.name});
});
function pickOp(i) {
  S.op = +i; S.t = 0;
  S.playing = S.op >= 0;
  draw();
  if (S.playing) start();
}
const opDD = makeDD($("opSel"), opGroups, pickOp);
playBtn.onclick = () => {
  if (S.op < 0) return;
  if (S.playing) { S.playing = false; ui(); return; }
  if (S.t >= 1) S.t = 0;
  S.playing = true; start();
};
scrub.oninput = () => { if (S.op < 0) return; S.playing = false; S.t = scrub.value / 1000; frame(); ui(); };
$("tLoop").onclick = () => { S.loop = !S.loop; ui(); };
$("tGhost").onclick = () => { S.ghost = !S.ghost; draw(); };

// the actual animation loop, holds a bit at the end so u can see it matches, then goes again
let last = null, hold = 0;
function start() { last = null; hold = 0; requestAnimationFrame(tick); }
function tick(ts) {
  if (!S.playing || S.op < 0) return;
  if (last === null) last = ts;
  const dt = Math.min((ts - last) / 1000, 0.1);
  last = ts;
  const dur = D.ops[S.op].kind === "S" ? 3.4 : 2.2;
  if (hold > 0) {
    hold -= dt;
    if (hold <= 0) {
      if (S.loop) S.t = 0;
      else { S.playing = false; ui(); return; }
    }
  } else {
    S.t = Math.min(1, S.t + dt / dur);
    if (S.t >= 1) hold = 1.0;
  }
  frame();
  requestAnimationFrame(tick);
}

function ui() {
  [...seg.children].forEach(b => b.classList.toggle("on", b.textContent === S.style));
  $("tLab").classList.toggle("off", !S.labels);
  $("tFill").classList.toggle("off", !S.fill);
  $("tMark").classList.toggle("off", !S.mark);
  $("tLoop").classList.toggle("off", !S.loop);
  $("tGhost").classList.toggle("off", !S.ghost);

  // element row
  $("elRow").classList.toggle("dim", S.op >= 0);
  elDD.set(S.el);
  const info = $("elInfo");
  if (S.op >= 0) info.textContent = 'Choose "Pick an operation" below to go back';
  else if (typeof S.el === "number") {
    const e = ELS[S.el], n = e.on.length;
    const where = e.kind === "axis" ? "on this axis" : e.kind === "plane" ? "on this plane" : "at the center";
    info.innerHTML = '<span class="swatch" style="background:' + e.color + '"></span>' +
      (S.el + 1) + " of " + ELS.length + ", " + n + (n === 1 ? " atom " : " atoms ") + where;
  } else info.textContent = ELS.length ? ELS.length + " elements, use ◀ ▶ to see them one at a time" : "";

  // animation row
  playBtn.disabled = S.op < 0;
  playBtn.textContent = S.playing ? "Pause" : "Play";
  opDD.set(S.op);
  uiAnim();
}

// the bits that change every frame: slider and caption
let lastCap = "";
function uiAnim() {
  scrub.value = Math.round(S.t * 1000);
  let txt = "";
  if (S.op < 0) {
    txt = D.ops.length ? "Pick a symmetry operation to watch the molecule do it." : "";
  } else {
    const o = D.ops[S.op];
    txt = o.desc + ".";
    if (o.kind === "S") txt += S.t < 0.5 ? " <b>Step 1: rotating</b>" : " <b>Step 2: reflecting</b>";
    if (S.t >= 1) txt += ' <span class="ok">Looks the same as before, so it is a symmetry operation.</span>';
  }
  if (txt !== lastCap) { cap.innerHTML = txt; lastCap = txt; }
}
v.zoomTo();
draw();
</script>
"""
    html = html.replace("__H__", str(height)).replace("__DATA__", json.dumps(data))
    components.html(html, height=height + 250)


# title
st.markdown("""
<div class="ui top">
  <p class="title">Point group finder</p>
  <p class="subtitle">Upload an .xyz file to get its point group, rotation axes and mirror planes.</p>
</div>
""", unsafe_allow_html=True)

# file on the left, tolerance on the right
c1, c2 = st.columns([3, 1.3], gap="large")
with c1:
    up = st.file_uploader("XYZ file", type=["xyz"], label_visibility="collapsed")
with c2:
    # tolerance
    tol = st.slider("Tolerance (Å)", 0.05, 1.0, 0.3, 0.05,
                    help="How far atoms can be off and still count as symmetric. "
                         "Raise it if a symmetric molecule comes out as C1.")

# no file no point group
if up is None:
    st.markdown("""
<div class="ui empty"><b>No molecule yet</b><br>Drop an .xyz file above to get started.</div>
""", unsafe_allow_html=True)
    st.stop()

# read the xyz, if it errors the file is probably cooked
try:
    mol = Molecule.from_str(up.getvalue().decode("utf-8", errors="ignore"), fmt="xyz")
except Exception as e:
    st.error(f"Couldn't read {up.name}. Check the file is a normal xyz "
             f"(atom count, comment line, then element x y z). Error: {e}")
    st.stop()

# the actual pymatgen part, the rest is just ui tbh
with st.spinner("Finding symmetry..."):
    pga = PointGroupAnalyzer(mol, tolerance=tol)
    pg = pga.sch_symbol
    ops = pga.get_symmetry_operations()
    cmol = pga.centered_mol  # centered at center of mass, the ops are based on this

# count the ops, find the axes and the planes
counts = count_ops(ops)
if "*" in pg:
    axes, normals, types = linear_stuff(pg, cmol.cart_coords)
else:
    axes = get_axes(ops)
    normals = get_planes(ops)
    types = label_planes(normals, pg, axes, cmol.cart_coords, tol)

# sort planes σh first then σv σd, and name them σv 1, σv 2 etc so u can tell them apart
order = sorted(range(len(types)), key=lambda i: PLANE_ORDER.index(types[i]))
planes, seen = [], Counter()
for i in order:
    t = types[i]
    seen[t] += 1
    name = sub(t) + (f" {seen[t]}" if types.count(t) > 1 else "")
    planes.append({"n": normals[i], "type": t, "name": name})

# swap the plain σ count for σh / σv / σd (not for linear, infinite planes there)
if "*" not in pg:
    counts.pop("σ", None)
    counts.update(types)

# molecule on the left, results on the right
left, right = st.columns([3, 1.3], gap="large")

with left:
    viewer(cmol, axes, planes, tol, describe_ops(ops, axes, planes), counts.get("i"))

with right:
    # the main thing, the point group
    order_txt = "∞ (linear, only a few planes drawn)" if "*" in pg else str(len(ops))
    ops_txt = ops_string(counts)
    if pg == "C*v":
        ops_txt = "E, 2C<sub>∞</sub>, ∞σ<sub>v</sub>"
    elif pg == "D*h":
        ops_txt = "E, 2C<sub>∞</sub>, ∞σ<sub>v</sub>, i, 2S<sub>∞</sub>, ∞C<sub>2</sub>"
    st.markdown(f"""
<div class="ui card">
  <p class="cap">Point group</p>
  <p class="pg-symbol">{sub(pg)}</p>
  <p class="pg-ops">{ops_txt}</p>
  <p class="pg-order">Order h = {order_txt}</p>
</div>
""", unsafe_allow_html=True)

    # extra info
    formula = re.sub(r"(\d+)", r"<sub>\1</sub>", mol.composition.formula.replace(" ", ""))
    st.markdown(f"""
<div class="ui stats">
  <div class="stat"><p class="cap">Formula</p><div class="v">{formula}</div></div>
  <div class="stat"><p class="cap">Atoms</p><div class="v">{len(mol)}</div></div>
  <div class="stat"><p class="cap">Mirror planes</p><div class="v">{len(planes) if "*" not in pg else "∞"}</div></div>
  <div class="stat"><p class="cap">Tolerance</p><div class="v">{tol} Å</div></div>
</div>
""", unsafe_allow_html=True)

    # list of the symmetry elements and how many of each
    rows = ""
    for n in sorted({a["n"] for a in axes}, key=lambda n: -99 if n == 0 else -n):
        k = "1" if n == 0 else str(sum(a["n"] == n for a in axes))
        rows += (f'<div class="row"><span><span class="dot" style="background:{AXIS_COLORS.get(n, "#374151")}"></span>'
                 f'{sub(axis_name(n))} axis</span><span class="cnt">{k}</span></div>')
    for t in PLANE_ORDER:
        k = types.count(t)
        if k:
            k = "∞" if ("*" in pg and t == "σv") else k
            rows += (f'<div class="row"><span><span class="dot" style="background:{PLANE_COLORS[t]}"></span>'
                     f'{sub(t)} plane</span><span class="cnt">{k}</span></div>')
    if counts.get("i"):
        rows += ('<div class="row"><span><span class="dot" style="background:#111827"></span>'
                 'Inversion center</span><span class="cnt">1</span></div>')
    if not rows:
        rows = '<div class="row"><span>Only E, no symmetry at all</span></div>'
    st.markdown(f'<div class="ui card"><p class="cap" style="margin-bottom:6px">Symmetry elements</p>{rows}</div>',
                unsafe_allow_html=True)

    # coordinates in a dropdown so it doesnt take the whole page
    with st.expander("Coordinates (centered)"):
        st.dataframe(
            {"atom": atom_labels(cmol),
             "x": cmol.cart_coords[:, 0].round(4),
             "y": cmol.cart_coords[:, 1].round(4),
             "z": cmol.cart_coords[:, 2].round(4)},
            height=300, hide_index=True)

    # download the result as txt
    plane_txt = ", ".join(f"{types.count(t)} {t}" for t in PLANE_ORDER if types.count(t)) or "none"
    if "*" in pg:
        plane_txt = "infinite (linear)"
    st.download_button("Download result",
                       f"file: {up.name}\nformula: {mol.composition.formula}\n"
                       f"atoms: {len(mol)}\ntolerance: {tol} A\n"
                       f"point group: {pg}\nsym ops: {len(ops)}\n"
                       f"mirror planes: {plane_txt}\n",
                       file_name=up.name.rsplit(".", 1)[0] + "_pointgroup.txt")