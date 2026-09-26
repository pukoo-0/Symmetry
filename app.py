import json
import re
from collections import Counter

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


# define the viewer to see the molecule
def viewer(mol, axes, planes, tol, height=540):
    xyz = mol.to(fmt="xyz")
    coords = mol.cart_coords
    size = max(np.linalg.norm(coords, axis=1).max(), 1.0)

    # atom positions + a rough size for each atom so the marker ring fits around it in spacefill too
    vdw = {"H": 1.1, "C": 1.7, "N": 1.55, "O": 1.52, "F": 1.47, "Cl": 1.75, "S": 1.8, "P": 1.8, "Br": 1.85}
    atoms = [{"p": c.tolist(), "r": vdw.get(str(s.specie), 1.8)} for c, s in zip(coords, mol)]

    # make the axes a bit longer than the molecule so u can see them
    ax_data = []
    for ax in axes:
        p = (ax["u"] * size * 1.3).tolist()
        ax_data.append({"p": p, "name": axis_name(ax["n"]), "color": AXIS_COLORS.get(ax["n"], "#374151")})

    # planes get drawn as a see-through disc, need 2 vectors lying inside the plane to draw it
    pl_data = []
    for p in planes:
        n = p["n"]
        a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = unit(a - np.dot(a, n) * n)
        e2 = np.cross(n, e1)
        # atoms sitting on the plane = distance to the plane less than the tolerance
        on = [int(i) for i in np.where(abs(coords @ n) < tol)[0]]
        pl_data.append({"n": n.tolist(), "e1": e1.tolist(), "e2": e2.tolist(), "on": on,
                        "type": p["type"], "name": p["name"], "color": PLANE_COLORS[p["type"]]})

    # display styles
    styles = {
        "Ball & stick": {"stick": {"radius": 0.14}, "sphere": {"scale": 0.25}},
        "Sticks": {"stick": {"radius": 0.18}},
        "Spacefill": {"sphere": {"scale": 0.9}},
    }

    data = {"xyz": xyz, "atoms": atoms, "axes": ax_data, "planes": pl_data, "styles": styles, "R": size * 1.12}

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
.chip:focus-visible, .seg button:focus-visible, .link:focus-visible {outline:2px solid #2563eb; outline-offset:2px;}
.chip.off {color:#a1a7b0; background:#fafafa;}
.chip .d {width:8px; height:8px; border-radius:50%;}
.chip.off .d {opacity:.25;}
.chip sub {font-size:9px;}
.chip .k {color:#9ca3af; font-size:11.5px;}
#v {width:100%; height:__H__px; position:relative;}
.planes {padding:10px 12px; border-top:1px solid #f1f2f4; display:flex; flex-wrap:wrap; gap:6px; align-items:center; max-height:96px; overflow:auto;}
.lbl {font-size:12.5px; color:#6b7280; font-weight:500; margin-right:2px;}
.muted {font-size:12.5px; color:#9ca3af;}
.link {border:0; background:none; color:#2563eb; font-size:12.5px; cursor:pointer; padding:2px 4px;}
</style>

<div class="wrap">
  <div class="bar">
    <div class="seg" id="seg"></div>
    <div class="group">
      <button class="chip" id="tAxes"><span class="d" style="background:#2563eb"></span>Rotation axes</button>
      <button class="chip" id="tLab">Labels</button>
      <button class="chip" id="tFill">Plane fill</button>
      <button class="chip" id="tMark"><span class="d" style="background:#f59e0b"></span>Mark atoms on planes</button>
      <button class="chip" id="tReset">Reset view</button>
    </div>
  </div>
  <div id="v"></div>
  <div class="planes" id="planes"></div>
</div>

<script>
const D = __DATA__;
const S = {style: Object.keys(D.styles)[0], axes: true, labels: true, fill: true, mark: true,
           planes: D.planes.map(() => true)};
const v = $3Dmol.createViewer("v", {backgroundColor: "white"});
v.addModel(D.xyz, "xyz");

function pt(p) { return {x: p[0], y: p[1], z: p[2]}; }

// disc (or a ring if r0 > 0) for the mirror plane, both faces so u can see it from any angle
function disc(p, r0, r1, color, alpha) {
  const N = 72, V = [], Nm = [], F = [];
  for (const s of [1, -1]) {
    const off = V.length;
    for (let i = 0; i < N; i++) {
      const t = 2 * Math.PI * i / N, c = Math.cos(t), sn = Math.sin(t);
      for (const r of [r0, r1]) {
        V.push(pt([0, 1, 2].map(k => r * (c * p.e1[k] + sn * p.e2[k]))));
        Nm.push(pt(p.n.map(x => s * x)));
      }
    }
    for (let i = 0; i < N; i++) {
      const a = off + 2 * i, b = a + 1, c2 = off + 2 * ((i + 1) % N), d = c2 + 1;
      if (s > 0) F.push(a, b, d, a, d, c2); else F.push(a, d, b, a, c2, d);
    }
  }
  v.addCustom({vertexArr: V, normalArr: Nm, faceArr: F, color: color, alpha: alpha});
}

function label(text, pos, color) {
  v.addLabel(text, {position: pos, fontSize: 12, fontColor: "white", backgroundColor: color,
                    backgroundOpacity: 0.95, inFront: true});
}

function draw() {
  v.removeAllShapes();
  v.removeAllLabels();
  v.setStyle({}, D.styles[S.style]);
  if (S.axes) {
    for (const a of D.axes) {
      v.addCylinder({start: pt(a.p.map(x => -x)), end: pt(a.p), radius: 0.07, color: a.color, fromCap: 1, toCap: 1});
      if (S.labels) label(a.name, pt(a.p), a.color);
    }
  }
  // planes: very light fill so the atoms behind are still easy to see, the outline shows where the plane is
  D.planes.forEach((p, i) => {
    if (!S.planes[i]) return;
    if (S.fill) disc(p, 0, D.R, p.color, 0.07);
    disc(p, D.R * 0.975, D.R, p.color, 0.9);
    if (S.labels) {
      const t = 0.7;
      label(p.type, pt([0, 1, 2].map(k => D.R * (Math.cos(t) * p.e1[k] + Math.sin(t) * p.e2[k]))), p.color);
    }
  });
  // mark the atoms that lie on a shown plane with a see-through ball around them
  // if an atom is on more than one plane it gets the color of the first one
  if (S.mark) {
    const done = new Set();
    D.planes.forEach((p, i) => {
      if (!S.planes[i]) return;
      for (const j of p.on) {
        if (done.has(j)) continue;
        done.add(j);
        const a = D.atoms[j];
        const r = S.style === "Spacefill" ? a.r * 0.9 : S.style === "Sticks" ? 0.18 : a.r * 0.25;
        v.addSphere({center: pt(a.p), radius: r + 0.22, color: p.color, alpha: 0.45});
      }
    });
  }
  v.render();
  ui();
}

// buttons
const seg = document.getElementById("seg");
Object.keys(D.styles).forEach(k => {
  const b = document.createElement("button");
  b.textContent = k;
  b.onclick = () => { S.style = k; draw(); };
  seg.appendChild(b);
});
document.getElementById("tAxes").onclick = () => { S.axes = !S.axes; draw(); };
document.getElementById("tLab").onclick = () => { S.labels = !S.labels; draw(); };
document.getElementById("tFill").onclick = () => { S.fill = !S.fill; draw(); };
document.getElementById("tMark").onclick = () => { S.mark = !S.mark; draw(); };
document.getElementById("tReset").onclick = () => { v.zoomTo(); v.render(); };

// one chip per plane so u can look at them one by one
const box = document.getElementById("planes");
if (D.planes.length === 0) {
  box.innerHTML = '<span class="muted">No mirror planes in this point group</span>';
} else {
  box.innerHTML = '<span class="lbl">Mirror planes</span>' +
    '<button class="link" id="all">Show all</button><button class="link" id="none">Hide all</button>';
  D.planes.forEach((p, i) => {
    const b = document.createElement("button");
    b.className = "chip";
    b.innerHTML = '<span class="d" style="background:' + p.color + '"></span>' + p.name +
      ' <span class="k">' + p.on.length + (p.on.length === 1 ? ' atom' : ' atoms') + '</span>';
    b.title = p.on.length + " atoms lie on this plane";
    b.onclick = () => { S.planes[i] = !S.planes[i]; draw(); };
    box.appendChild(b);
  });
  document.getElementById("all").onclick = () => { S.planes = S.planes.map(() => true); draw(); };
  document.getElementById("none").onclick = () => { S.planes = S.planes.map(() => false); draw(); };
}

function ui() {
  [...seg.children].forEach(b => b.classList.toggle("on", b.textContent === S.style));
  document.getElementById("tAxes").classList.toggle("off", !S.axes);
  document.getElementById("tLab").classList.toggle("off", !S.labels);
  document.getElementById("tFill").classList.toggle("off", !S.fill);
  document.getElementById("tMark").classList.toggle("off", !S.mark);
  box.querySelectorAll(".chip").forEach((c, i) => c.classList.toggle("off", !S.planes[i]));
}

v.zoomTo();
draw();
</script>
"""
    html = html.replace("__H__", str(height)).replace("__DATA__", json.dumps(data))
    components.html(html, height=height + 170)


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
    viewer(cmol, axes, planes, tol)

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
            {"atom": [str(s.specie) for s in cmol],
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
