"""
bridge.py — connects the web page to symmetry_core. Every function takes plain strings and numbers and
returns JSON text: {"ok": true, ...} or {"ok": false, "error": "..."}.
"""
import json
import os

import numpy as np

import symmetry_core as sc

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = ["water", "ammonia", "methane", "ethane", "allene", "benzene", "co2", "sf6", "cubane", "adamantane", "c60"]
_state = {"frames": None, "name": None, "cache": None}


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def _clean(v):
    if isinstance(v, float):
        return v if np.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def _ok(**kw):
    return json.dumps(_clean({"ok": True, **kw}), allow_nan=False, default=_default)


def _err(msg):
    return json.dumps({"ok": False, "error": str(msg)})


def options():
    return _ok(examples=EXAMPLES, colors={"axes": sc.AXIS_COLORS, "planes": sc.PLANE_COLORS},
               radii=sc.COVALENT_RADII)


def load(name, text):
    try:
        frames = sc.parse_xyz(text)
    except Exception as e:  # noqa: BLE001
        return _err(e)
    _state.update(frames=frames, name=name, cache=None)
    return _ok(name=name, frames=[{"comment": f.comment, "n": len(f.symbols), "formula": sc.formula(f.symbols)}
                                  for f in frames])


def example(key):
    if key not in EXAMPLES:
        return _err(f"No example called {key}.")
    with open(os.path.join(HERE, key + ".xyz"), encoding="utf-8") as fh:
        return load(key + ".xyz", fh.read())


def _frame(frame):
    frames = _state["frames"]
    if not frames:
        raise ValueError("Open an .xyz file first.")
    return frames[max(0, min(int(frame), len(frames) - 1))]


def _result(frame, tol):
    key = (int(frame), round(float(tol), 4))
    if _state["cache"] and _state["cache"][0] == key:
        return _state["cache"][1]
    tol = float(tol)
    if not 0.001 <= tol <= 2.0:
        raise ValueError("The tolerance must be between 0.001 and 2 Å.")
    res = sc.analyse(_frame(frame), tol)
    _state["cache"] = (key, res)
    return res


def analyse(frame, tol):
    try:
        return _ok(**_result(frame, tol))
    except Exception as e:  # noqa: BLE001
        return _err(e)


def scan(frame):
    try:
        return _ok(rows=sc.tolerance_scan(_frame(frame)))
    except Exception as e:  # noqa: BLE001
        return _err(e)


def exports(frame, tol, stem):
    try:
        res = _result(frame, tol)
        name = _state["name"] or "molecule.xyz"
        xyz = [str(res["n"]), f"{res['comment']} | centred on the centre of mass, point group {res['pg'].replace('*', '∞')}"]
        xyz += [f"{s:<2} {x:14.8f} {y:14.8f} {z:14.8f}" for s, (x, y, z) in zip(res["symbols"], res["coords"])]
        return _ok(files={f"{stem}_pointgroup.txt": sc.report_text(res, name),
                          f"{stem}_centred.xyz": "\n".join(xyz) + "\n"})
    except Exception as e:  # noqa: BLE001
        return _err(e)
