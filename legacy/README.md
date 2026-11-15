# The original app (kept for reference)

`app.py` is the first version: Streamlit + pymatgen (`PointGroupAnalyzer`) + 3Dmol.js. It is unchanged.

What moved into the new version:
- `sub`, `axis_name`, `rot_angle`, `angle_to_order`, `extract_axis`, `get_axes`, `get_planes`, `label_planes`,
  `count_ops`, `ops_string`, `linear_stuff`, `rodrigues`, `describe_ops`, `atom_labels`: copied into
  `symmetry_core.py` as they were (colours changed to match the new interface).
- The animation maths (`pos`, `rotate`, `reflect`, `ease`) is in `web/index.src.html`, unchanged.

What changed and why:
- pymatgen cannot run in the browser, so `symmetry_core.py` finds the point group itself with NumPy
  (candidate axes and planes, tested atom by atom, group built from atom permutations).
- `count_ops` groups operations by axis order only, so it merges classes that character tables keep apart
  (Oh: 9C2 instead of 6C2 + 3C2(=C4²); Ih: 24C5 instead of 12C5 + 12C5²; Th: 8C3 instead of 4C3 + 4C3²).
  The new version computes the true conjugacy classes and labels them with the same names.
- `sub("σv'")` produced σv with a subscripted prime; the regex now takes one base letter.
