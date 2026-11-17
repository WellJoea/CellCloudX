# -*- coding: utf-8 -*-
"""Generate _OTM_fixed.py from _OTM.py by patching ONLY the buggy lines.

The original file is never modified. Every replacement is asserted to occur
exactly once, so any drift between the original source and this patch fails
loudly instead of silently producing a wrong file.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
M3LIB = os.path.join(HERE, '..', 'cellcloudx', 'alignment', 'm3lib')
SRC = os.path.normpath(os.path.join(M3LIB, '_OTM.py'))
DST = os.path.normpath(os.path.join(M3LIB, '_OTM_fixed.py'))

with open(SRC, 'r', encoding='utf-8') as f:
    text = f.read()

header = ("# NOTE: this file is a generated, corrected copy of _OTM.py. "
          "The original source is unchanged.\n"
          "# Fixes applied (everything else is byte-identical to the original):\n"
          "#   1. feat_normal 'cos'/'cosine': recompute per-row L2 norms for new query\n"
          "#      samples instead of reusing training-row norms (old call\n"
          "#      normalize(iF, *stats) raised TypeError: Xs is keyword-only, and the\n"
          "#      stored norms belong to the training rows, not to the query rows).\n"
          "#   2. feat_normal 'ccos'/'ccosine': reuse only the fitted column center\n"
          "#      (Xm=stats[0]) and recompute per-row norms (old call passed the stale\n"
          "#      training row-norm tensor positionally -> TypeError / wrong values).\n"
          "#   3. feat_normal 'zc': pass only Xm=stats[0], Xs=stats[1]; the stored affine\n"
          "#      matrix Xf is bookkeeping, not a centerlize argument (old call passed\n"
          "#      4 positional arguments -> TypeError).\n")

replacements = [
    # --- 1. cos / cosine -------------------------------------------------
    ("                    outs =  normalize(iF, *stats)",
     "                    outs = normalize(iF)  # Row norms are sample-wise: recompute for each query row."),
    # --- 2. zc ------------------------------------------------------------
    ("                elif inorm == 'zc':\n"
     "                    outs = centerlize(iF, *stats)",
     "                elif inorm == 'zc':\n"
     "                    # stats = [Xm, Xs, Xf]: reuse the fitted column center and\n"
     "                    # global scale for new samples; Xf is affine bookkeeping,\n"
     "                    # not a centerlize argument.\n"
     "                    outs = centerlize(iF, Xm=stats[0], Xs=stats[1]) if stats else centerlize(iF)"),
    # --- 3. ccos / ccosine -------------------------------------------------
    ("                elif inorm in ['ccosine', 'ccos']:\n"
     "                    outs = center_normalize(iF, *stats)",
     "                elif inorm in ['ccosine', 'ccos']:\n"
     "                    # stats = [Xm, l2x]: reuse the fitted column center only;\n"
     "                    # row norms are sample-wise and must be recomputed.\n"
     "                    outs = center_normalize(iF, Xm=stats[0]) if stats else center_normalize(iF)"),
]

patched = text
for old, new in replacements:
    n = patched.count(old)
    assert n == 1, f"expected exactly 1 occurrence, found {n} for:\n{old[:80]}..."
    patched = patched.replace(old, new, 1)

# Insert the provenance header right after the module docstring.
anchor = 'including for dense execution."""'
assert patched.count(anchor) == 1
patched = patched.replace(anchor, anchor + '\n' + header, 1)

with open(DST, 'w', encoding='utf-8') as f:
    f.write(patched)

print("wrote:", DST)
print("original lines:", text.count('\n') + 1, "| fixed lines:", patched.count('\n') + 1)

# Quick diff summary of the changed regions.
import difflib
diff = list(difflib.unified_diff(
    text.splitlines(), patched.splitlines(), lineterm='', n=2))
print('\n'.join(diff[:120]))
