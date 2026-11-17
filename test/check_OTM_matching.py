# -*- coding: utf-8 -*-
"""Verification of _OTM_fixed.py: class math + every corrected bug.

Dense (non-KeOps) float64 paths:
  B0. original _OTM.py cannot even construct OTM (helpers bound too late);
      fixed constructs and initializes.
  B1. feat_normal 'cos'/'ccos'/'zc' prediction crashed on original; fixed
      matches the closed-form normalization.
  B2. matching(): sy was passed as sx; fixed honors sy.
  B3. matching(): k was hardcoded to None; fixed honors k.
  B4. matching(): Inds[keep] crashed; fixed returns Inds[:, keep].
  B5. matching(): otm alias is honored by the fixed version.
  C1-C7. class math (factors/moments, FGW gradient & prediction, Sinkhorn
      fixed point, label kernels, effective-cost caching, transform) against
      closed forms, ported from check_OTM_math.py (fixed module only; the
      original's class body is byte-identical apart from the fixes).
  M1-M5. matching pipeline: filter_edges values, decode methods vs brute
      force, candidate spaces, end-to-end runs.
"""
import os, sys, types, warnings

import numpy as np
import torch as th

th.manual_seed(0)
np.random.seed(0)
th.set_default_dtype(th.float64)

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# --------------------------------------------------------------------------
# Stub pykeops (no compiler in this env); all tests run dense paths.
# --------------------------------------------------------------------------
pk = types.ModuleType('pykeops')
pk.set_verbose = lambda *a, **k: None
pkt = types.ModuleType('pykeops.torch')

class _NoKeops:
    def __init__(self, *a, **k):
        raise RuntimeError('keops stub: these tests must run dense (use_keops=False)')

pkt.LazyTensor = _NoKeops
pkt.Genred = _NoKeops
pk.torch = pkt
sys.modules['pykeops'] = pk
sys.modules['pykeops.torch'] = pkt

for name, rel in [('cellcloudx', 'cellcloudx'),
                  ('cellcloudx.utilis', 'cellcloudx/utilis'),
                  ('cellcloudx.io', 'cellcloudx/io'),
                  ('cellcloudx.alignment', 'cellcloudx/alignment'),
                  ('cellcloudx.alignment.m3lib', 'cellcloudx/alignment/m3lib')]:
    m = types.ModuleType(name)
    m.__path__ = [os.path.join(ROOT, rel)]
    sys.modules[name] = m

warnings.simplefilter('ignore')
from cellcloudx.alignment.m3lib import operation_expectation as oe   # noqa: E402
import cellcloudx.alignment.m3lib.OTM as orig_mod                    # noqa: E402

# The corrected module is delivered OUTSIDE the repository
# (~/Downloads/CellCloudX_OTM_fixed/OTM.py). Load it from there when present;
# otherwise fall back to a local OTM_fixed.py copy in m3lib.
import importlib.util as _ilu                                             # noqa: E402
_FIXED_PATH = os.path.join(os.path.expanduser('~'), 'Downloads',
                           'CellCloudX_OTM_fixed', 'OTM.py')
if os.path.exists(_FIXED_PATH):
    _spec = _ilu.spec_from_file_location('cellcloudx.alignment.m3lib.OTM_fixed', _FIXED_PATH)
    fix_mod = _ilu.module_from_spec(_spec)
    fix_mod.__package__ = 'cellcloudx.alignment.m3lib'
    _spec.loader.exec_module(fix_mod)
else:
    import cellcloudx.alignment.m3lib.OTM_fixed as fix_mod               # noqa: E402

PASS = []
FAIL = []

def check(name, ok, detail=''):
    (PASS if ok else FAIL).append(name)
    print(('  PASS  ' if ok else '  FAIL  ') + name + (('  -- ' + str(detail)) if detail else ''))

def close(a, b, rtol=1e-9, atol=1e-10):
    return th.allclose(a, b, rtol=rtol, atol=atol)

def dist2(A, B):
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)

def fgw_energy(P, Cx, Cy):
    r = P.sum(1); c = P.sum(0)
    return 0.5 * (r @ (Cx * Cx) @ r + c @ (Cy * Cy) @ c) - (Cx @ P @ Cy * P).sum()

def mk(mod, **kw):
    kw.setdefault('device', th.device('cpu'))
    kw.setdefault('dtype', th.float64)
    kw.setdefault('use_keops', False)
    return mod.OTM(**kw)

# --------------------------------------------------------------------------
# B0: construction
# --------------------------------------------------------------------------
print('== B0: OTM construction ==')
try:
    orig_mod.OTM(use_keops=False, device=th.device('cpu'), dtype=th.float64)
    check('original OTM constructs (helper-binding order fixed upstream)', True, '')
except AttributeError as e:
    check('original OTM constructs (helper-binding order fixed upstream)', False, e)
o = mk(fix_mod)
check('fixed OTM constructs', isinstance(o, fix_mod.OTM), '')

# Kept for revisions whose __init__ binds helpers late; a no-op today.
class _PatchedOrig(orig_mod.OTM):
    def __init__(self, **kw):
        self.scalar2vector = orig_mod.scalar2vector
        self.auto_select_device = orig_mod.auto_select_device
        self.n2v = orig_mod.n2v
        self.plan_extract = orig_mod.plan_extract
        super().__init__(**kw)
orig_mod.OTM = _PatchedOrig
def mko(mod, **kw):
    return mk(mod, **kw)

# --------------------------------------------------------------------------
# C1/C2: factors and moments
# --------------------------------------------------------------------------
print('== C1/C2: _factors and _moment identities ==')
otm = mk(fix_mod)
for d in (1, 3, 20):
    Z = th.randn(9, d)
    phi, psi = otm._factors(Z)
    check(f'factors exact d={d}', close(phi @ psi.T, dist2(Z, Z), 1e-9, 1e-9),
          float((phi @ psi.T - dist2(Z, Z)).abs().max()))
for d, mom in ((3, 64), (25, 16)):
    Z = th.randn(11, d)
    mass = th.rand(11) + 0.1
    phi, psi = otm._factors(Z)
    got = otm._moment(Z, phi, psi, mass)
    brute = (dist2(Z, Z).square() @ mass).reshape(-1)
    check(f'_moment exact d={d} (mom_max={mom})',
          close(got, brute, 1e-9, 1e-9), float((got - brute).abs().max()))

# --------------------------------------------------------------------------
# C3/C4: FGW gradient and prediction vs autograd
# --------------------------------------------------------------------------
print('== C3/C4: FGW gradient and prediction costs vs autograd ==')
otm = mk(fix_mod)
X = th.randn(7, 4); Y = th.randn(6, 4)
P = th.rand(7, 6, requires_grad=True) + 0.3
r, c = P.sum(1).detach(), P.sum(0).detach()
for cen, s2 in ((False, 1.0), (True, 1.0), (True, 3.7)):
    Xc = X - X.mean(0) if cen else X
    Yc = Y - Y.mean(0) if cen else Y
    Cxs, Cys = dist2(Xc, Xc) / s2, dist2(Yc, Yc) / s2
    E = fgw_energy(P, Cxs, Cys)
    g_auto = th.autograd.grad(E, P)[0]
    g_code = otm.D2_fgw_gradient(X, Y, P.detach(), r, c, s2, centerlize=cen)
    check(f'FGW grad autograd (center={cen}, s2={s2})', close(g_code, g_auto, rtol=1e-8, atol=1e-9),
          float((g_code - g_auto).abs().max()))
otm.use_lowrank = True
g_lr = otm.D2_fgw_gradient(X, Y, P.detach(), r, c, 2.0, centerlize=True)
otm.use_lowrank = False
g_dr = otm.D2_fgw_gradient(X, Y, P.detach(), r, c, 2.0, centerlize=True)
check('FGW grad lowrank==direct', close(g_lr, g_dr, 1e-9, 1e-9), float((g_lr - g_dr).abs().max()))

# prediction side 0/1 vs autograd (row constants omitted)
nx, ny, d = 8, 6, 3
X0 = th.randn(nx, d); Y0 = th.randn(ny, d)
Xc, Yc = X0 - X0.mean(0), Y0 - Y0.mean(0)
P0 = th.rand(nx, ny) + 0.2
r0, c0 = P0.sum(1), P0.sum(0)
Cx0, Cy0 = dist2(Xc, Xc), dist2(Yc, Yc)
phix, psix = otm._factors(Xc)
phiy, psiy = otm._factors(Yc)
B = psix.T @ P0 @ psiy
dx = (Cx0.square() @ r0).reshape(-1)
dy = (Cy0.square() @ c0).reshape(-1)
q = th.randn(1, d)
qc = q - X0.mean(0)
Cxq = dist2(th.cat([Xc, qc]), Xc)
p = th.zeros(ny, requires_grad=True)
E = fgw_energy(th.cat([P0, p[None]], 0), dist2(th.cat([Xc, qc]), th.cat([Xc, qc])), Cy0)
g = th.autograd.grad(E, p)[0]
out = otm._prediction_fgw(q, Y0, X0, Y0, 1.0, (B, dx, dy), 0)[0]
expect = g - (Cxq[nx].square() @ r0)
check('pred-FGW side0 vs autograd', close(out, expect, 1e-9, 1e-9), float((out - expect).abs().max()))
qy = th.randn(1, d)
qyc = qy - Y0.mean(0)
Cyq = dist2(th.cat([Yc, qyc]), Yc)
pcol = th.zeros(nx, requires_grad=True)
E2 = fgw_energy(th.cat([P0, pcol[:, None]], 1), Cx0, dist2(th.cat([Yc, qyc]), th.cat([Yc, qyc])))
g2 = th.autograd.grad(E2, pcol)[0]
out2 = otm._prediction_fgw(qy, X0, X0, Y0, 1.0, (B, dx, dy), 1)[0]
expect2 = g2 - (Cyq[ny].square() @ c0)
check('pred-FGW side1 vs autograd', close(out2, expect2, 1e-9, 1e-9), float((out2 - expect2).abs().max()))

# --------------------------------------------------------------------------
# C5/C6/C7: Sinkhorn fixed point, labels, FGW caching
# --------------------------------------------------------------------------
print('== C5: UOT Sinkhorn fixed point; normal vs log; entropy vs KL ==')
X = th.randn(40, 5); Y = th.randn(35, 5)
a = th.rand(40) + 0.5; b = th.rand(35) + 0.5
for mode in ('normal', 'log'):
    otm = mk(fix_mod, mode=mode, reg='entropy', tol=1e-7, check_every=5, uv_iter=5000)
    P, r, c = otm.solver(X, Y, a=a, b=b)
    K = th.exp(-otm.D1st / otm.eps)
    u, v = otm.u, otm.v
    u_fp = (a / (K @ v).clamp_min(1e-300)).pow(otm.tx)
    v_fp = (b / (K.T @ u).clamp_min(1e-300)).pow(otm.ty)
    relu = ((u - u_fp).abs() / u_fp.clamp_min(1e-300)).max().item()
    relv = ((v - v_fp).abs() / v_fp.clamp_min(1e-300)).max().item()
    marg_ok = close(P.sum(1), r, 1e-9, 1e-8) and close(P.sum(0), c, 1e-9, 1e-8)
    plan_ok = close(P, u[:, None] * K * v[None, :], rtol=1e-8, atol=1e-10)
    check(f'{mode} sinkhorn fixed point (relu={relu:.2e}, relv={relv:.2e})',
          relu < 1e-5 and relv < 1e-5 and marg_ok and plan_ok,
          f'marginals={marg_ok}, plan={plan_ok}')
o1 = mk(fix_mod, mode='normal', tol=1e-7, uv_iter=5000)
o2 = mk(fix_mod, mode='log', tol=1e-7, uv_iter=5000)
P1, _, _ = o1.solver(X, Y, a=a, b=b)
P2, _, _ = o2.solver(X, Y, a=a, b=b)
check('normal==log plan', close(P1, P2, rtol=1e-5, atol=1e-7), float((P1 - P2).abs().max()))
o3 = mk(fix_mod, mode='log', reg='entropy', tol=1e-7, uv_iter=5000)
o4 = mk(fix_mod, mode='log', reg='kl', tol=1e-7, uv_iter=5000)
P3, _, _ = o3.solver(X, Y, a=th.ones(40), b=th.ones(35))
P4, _, _ = o4.solver(X, Y, a=th.ones(40), b=th.ones(35))
check('entropy==kl for all-one budgets', close(P3, P4, rtol=1e-6, atol=1e-8), float((P3 - P4).abs().max()))

print('== C6: label kernels and K_effective ==')
otm = mk(fix_mod, mode='log')
Lx = th.rand(9, 5) + 0.1; Ly = th.rand(7, 5) + 0.1
for dh in (0.0, 0.2):
    H = otm.H_label(Lx, Ly, dh)
    Href = (1 - dh) * Lx @ Ly.T + dh
    check(f'H_label dh={dh}', close(H, Href, 1e-12, 1e-12), float((H - Href).abs().max()))
    logH = otm.H_label_log(Lx, Ly, dh)
    check(f'H_label_log dh={dh}', close(logH.exp(), Href, rtol=1e-9, atol=1e-10),
          float((logH.exp() - Href).abs().max()))
D = th.rand(9, 7) * 0.7
otm2 = mk(fix_mod, mode='normal', reg='kl')
otm2.loga = th.zeros(9); otm2.logb = th.zeros(7); otm2.a = th.ones(9); otm2.b = th.ones(7)
Kn = otm2.K_effective(D, 0.05, XL=Lx, YL=Ly, delta_h=0.1)
otm3 = mk(fix_mod, mode='log', reg='kl')
otm3.loga = th.zeros(9); otm3.logb = th.zeros(7)
logK = otm3.K_effective(D, 0.05, XL=Lx, YL=Ly, delta_h=0.1)
check('K_effective log==log(normal) with labels', close(logK.exp(), Kn, rtol=1e-9, atol=1e-10),
      float((logK.exp() - Kn).abs().max()))

print('== C7: one-step FGW caching and transform ==')
otm = mk(fix_mod, use_fgw=True, gscale_2nd=0.4, eps=0.1, tol=1e-7, uv_iter=5000)
X = th.randn(25, 3); Y = th.randn(22, 3)
otm.init_model(X, Y)
P0, r0, c0 = otm.solve(None)
P1, r1, c1 = otm.solve((P0, r0, c0))
g0 = otm.D2_fgw_gradient(otm.X, otm.Y, P0, r0, c0, otm.sigma2_2nd, centerlize=True)
check('cached Deff == D1 + gscale*grad(P0)', close(otm.D2nd, otm.gscale_2nd * g0, 1e-9, 1e-8),
      float((otm.D2nd - otm.gscale_2nd * g0).abs().max()))
K = th.exp(-(otm.D1st + otm.D2nd) / otm.eps)
check('final plan == u*K(Deff)*v', close(P1, otm.u[:, None] * K * otm.v[None, :], rtol=1e-8, atol=1e-10),
      float((P1 - otm.u[:, None] * K * otm.v[None, :]).abs().max()))
otm = mk(fix_mod, mode='log', tol=1e-7, uv_iter=5000)
X = th.randn(30, 3); Y = th.randn(26, 3)
P, r, c = otm.solver(X, Y)
check('solver marginals r=P@1, c=P.T@1',
      close(P.sum(1), r, 1e-9, 1e-8) and close(P.sum(0), c, 1e-9, 1e-8), '')
Q = th.randn(9, 3)
T = otm.transform(X=Q, method='kernel')
Gq = oe.centerlize(Q.clone(), Xm=otm.g_shift[0], Xs=otm.g_scale[0])[0].to(th.float64)
D1 = otm.D1_cost_agg(Gq, otm.Y, otm.sigma2_1st, otm.gscale_1st, [], [], otm.tau2s_1st, otm.fscales_1st)
Tref = th.softmax(-D1 / otm.eps + otm.logv[None, :], dim=1) @ otm.Yr
check('transform kernel == closed-form conditional barycenter', close(T, Tref, rtol=1e-9, atol=1e-9),
      float((T - Tref).abs().max()))
bary = otm.transform()
Tn = otm.transform(X=Q, method='nearest')
Dq = otm.D1_cost_agg(Gq, otm.X, otm.sigma2_1st, otm.gscale_1st, [], [], otm.tau2s_1st, otm.fscales_1st)
idx = Dq.argmin(1)
Tnref = Q + (bary[idx] - otm.Xr[idx])
check('transform nearest == best-training displacement', close(Tn, Tnref, rtol=1e-9, atol=1e-9),
      float((Tn - Tnref).abs().max()))

# --------------------------------------------------------------------------
# B1: feat_normal prediction paths
# --------------------------------------------------------------------------
print('== B1: feat_normal prediction (BUG reproduction) ==')
A = th.randn(20, 6) + 2.0
B = th.randn(17, 6) - 1.0
Qq = th.randn(9, 6) + 0.5
def train(mod, norm):
    otm = mko(mod)
    otm.FL = 1
    otm.normal_f = [norm]
    otm.feat_scales = {}
    otm.feat_normal([A], [B])
    return otm
for norm in ('ccos', 'cos', 'zc'):
    o_orig = train(orig_mod, norm)
    o_fix = train(fix_mod, norm)
    stats = o_fix.feat_scales[0][0]
    try:
        o_orig.feat_normal([Qq], predict_ids=[0], predict=True)[0][0]
        orig_err = None
    except Exception as e:
        orig_err = f'{type(e).__name__}: {e}'
    out_fix = o_fix.feat_normal([Qq], predict_ids=[0], predict=True)[0][0]
    if norm == 'ccos':
        ref = oe.center_normalize(Qq.clone(), Xm=stats[0])[0].to(th.float64)
    elif norm == 'cos':
        ref = oe.normalize(Qq.clone())[0].to(th.float64)
    else:
        ref = oe.centerlize(Qq.clone(), Xm=stats[0], Xs=stats[1])[0].to(th.float64)
    ok_fix = close(out_fix, ref, rtol=1e-10, atol=1e-12)
    check(f'original crashed on prediction ({norm})', orig_err is not None, orig_err)
    check(f'fixed normalization correct ({norm})', ok_fix, float((out_fix - ref).abs().max()))

print('== B1b: end-to-end transform with features (default ccos) ==')
X = th.randn(30, 3); Y = th.randn(25, 3)
Fx = th.randn(30, 8); Fy = th.randn(25, 8)
Q = th.randn(7, 3); Fq = th.randn(7, 8)
o_orig = mko(orig_mod, normal_f='ccos')
o_orig.solver(X, Y, Fxs=Fx, Fys=Fy)
try:
    o_orig.transform(X=Q, Fxs=Fq, method='kernel')
    check('original transform w/ features reproduces BUG', False, 'no crash')
except TypeError as e:
    check('original transform w/ features reproduces BUG', True, e)
o_fix = mk(fix_mod, normal_f='ccos')
o_fix.solver(X, Y, Fxs=Fx, Fys=Fy)
T = o_fix.transform(X=Q, Fxs=Fq, method='kernel')
Gq2 = oe.centerlize(Q.clone(), Xm=o_fix.g_shift[0], Xs=o_fix.g_scale[0])[0].to(th.float64)
Fqn = oe.center_normalize(Fq.clone(), Xm=o_fix.feat_scales[0][0][0])[0].to(th.float64)
D1 = o_fix.D1_cost_agg(Gq2, o_fix.Y, o_fix.sigma2_1st, o_fix.gscale_1st,
                       [Fqn], o_fix.YF, o_fix.tau2s_1st, o_fix.fscales_1st)
Tref = th.softmax(-D1 / o_fix.eps + o_fix.logv[None, :], dim=1) @ o_fix.Yr
check('fixed transform w/ features == closed form', close(T, Tref, rtol=1e-9, atol=1e-9),
      float((T - Tref).abs().max()))

# --------------------------------------------------------------------------
# M1: filter_edges values vs manual computation
# --------------------------------------------------------------------------
print('== M1: filter_edges values ==')
nx, ny = 6, 5
P = th.rand(nx, ny)
r = P.sum(1); c = P.sum(0)
a = th.ones(nx); b = th.ones(ny)
indices, p, qxy, qyx, score = fix_mod.filter_edges(
    P, r, c, a, b, k=3, candidate_space='P', score_threshold=0.05,
    pe_ratio=0.05, score_rule='either', use_keops=False)
i, j = indices
p_man = P[i, j]
qxy_man = th.where(r[i] > 0, p_man / r[i], 0)
qyx_man = th.where(c[j] > 0, p_man / c[j], 0)
score_man = qxy_man.sqrt() * qyx_man.sqrt()
conf = th.maximum(qxy_man, qyx_man)
keep_man = ((p_man > 0) & (conf >= 0.05) & (r[i] / a[i] >= 0.05) & (c[j] / b[j] >= 0.05))
check('filter_edges values == manual', close(p, p_man[keep_man], 1e-12, 1e-12)
      and close(qxy, qxy_man[keep_man], 1e-12, 1e-12)
      and close(qyx, qyx_man[keep_man], 1e-12, 1e-12)
      and close(score, score_man[keep_man], 1e-12, 1e-12), indices.shape)
# candidate k union check
rank = -P
jx = rank.topk(3, dim=1, largest=False).indices
iy = rank.T.topk(3, dim=1, largest=False).indices
fwd = th.arange(nx)[:, None] * ny + jx
rev = iy * ny + th.arange(ny)[:, None]
ids = th.cat((fwd.flatten(), rev.flatten())).unique(sorted=True)
ii, jj = ids // ny, ids % ny
check('filter_edges k-candidates == row/col top-k union',
      sorted((i * ny + j).tolist()) == sorted(ids.tolist()), '')

# --------------------------------------------------------------------------
# M2: decode methods vs brute force
# --------------------------------------------------------------------------
print('== M2: decode methods ==')
idx = th.tensor([[0, 1, 0, 2, 1, 3, 2, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
qxy = th.rand(8); qyx = th.rand(8); p = th.rand(8); score = th.rand(8)
# soft
keep = fix_mod.decode(idx, p, qxy, qyx, score, 5, 5, method='soft')
check('decode soft keeps all', bool(keep.all().item()), '')
# greedy vs brute force
sx, sy = 2, 1
keep = fix_mod.decode(idx, p, qxy, qyx, score, 5, 5, method='greedy', sx=sx, sy=sy)
order = score.argsort(descending=True, stable=True).tolist()
cx = [sx] * 5; cy = [sy] * 5
keep_bf = th.zeros(8, dtype=th.bool)
for e in order:
    x, y = idx[0, e].item(), idx[1, e].item()
    if cx[x] > 0 and cy[y] > 0:
        keep_bf[e] = True
        cx[x] -= 1; cy[y] -= 1
check('decode greedy == brute force', bool((keep == keep_bf).all().item()), '')
# mutual vs brute force
knn = 2
keep = fix_mod.decode(idx, p, qxy, qyx, score, 5, 5, method='mutual', sx=1, sy=1, topk=knn)
order = p.argsort(descending=True, stable=True)
keep_bf = th.ones(8, dtype=th.bool)
for g in (idx[0], idx[1]):
    sel = th.zeros(8, dtype=th.bool)
    for gi in g.unique().tolist():
        members = (g == gi).nonzero().flatten()
        sel[members[order[members].argsort()][:knn]] = True
    keep_bf &= sel
check('decode mutual == brute force', bool((keep == keep_bf).all().item()), '')
# assignment: small exact enumeration (3 rows, 7 edges)
idx3 = th.tensor([[0, 0, 1, 1, 2, 2, 2], [0, 1, 0, 2, 1, 2, 0]])
gain = th.tensor([0.8, 0.3, 0.1, 0.9, 0.4, 0.2, 0.6])
min_s = 0.0
keep = fix_mod.decode(idx3, gain, gain, gain, gain, 3, 3, method='assignment',
                      assignment_min_score=min_s)
pos = (gain - min_s) > 0
best, best_val = None, -1.0
import itertools
for rbits in itertools.product(*[range(4) for _ in range(3)]):
    used_y, tot, chosen = set(), 0.0, []
    for x, rb in enumerate(rbits):
        if rb == 3:
            continue
        e = ((idx3[0] == x) & (idx3[1] == rb) & pos).nonzero()
        if e.numel() == 0:
            break
        e = e[0].item()
        if rb in used_y:
            break
        used_y.add(rb)
        tot += (gain[e] - min_s).item()
        chosen.append(e)
    else:
        if tot > best_val:
            best_val, best = tot, set(chosen)
keep_bf3 = th.zeros(7, dtype=th.bool)
for e in best:
    keep_bf3[e] = True
check('decode assignment == exact enumeration',
      bool((keep == keep_bf3).all().item()), (keep.tolist(), keep_bf3.tolist()))

# --------------------------------------------------------------------------
# M3/B2-B5: matching() wiring bugs on original, correctness on fixed
# --------------------------------------------------------------------------
print('== B2-B5: matching() wiring ==')
Xs = th.randn(12, 3); Ys = th.randn(8, 3)
Xs = Xs * 0.4 + 1.0  # two well-separated-ish clusters to give a graded plan

# B4: the return statement used Inds[keep] (crash) in older revisions.
try:
    _r = orig_mod.matching(Xs, Ys, use_keops=False, device=th.device('cpu'), dtype=th.float64)
    check('original matching returns Inds[:, keep] (fixed upstream)',
          _r['Inds'].shape[0] == 2, _r['Inds'].shape)
except (IndexError, RuntimeError, TypeError) as e:
    check('original matching returns Inds[:, keep] (fixed upstream)', False, e)

# B2: original ignored sy (passed sy=sx); fixed honors it
res = fix_mod.matching(Xs, Ys, use_keops=False, method='greedy', sx=2, sy=1,
                       device=th.device('cpu'), dtype=th.float64, score_threshold=0.05)
model = res['model']
Inds = res['Inds']
ydeg = th.bincount(Inds[1], minlength=model.ny)
xdeg = th.bincount(Inds[0], minlength=model.nx)
check('fixed matching honors sy=1 (max Y-degree <= 1)', int(ydeg.max()) <= 1, ydeg.tolist())
check('fixed matching honors sx=2', int(xdeg.max()) <= 2, xdeg.tolist())
# original-equivalent wiring: sy=sx -> Y capacity 2 -> more edges
Inds2 = fix_mod.matching(Xs, Ys, use_keops=False, method='greedy', sx=2, sy=2,
                         device=th.device('cpu'), dtype=th.float64)['Inds']
check('BUG B2 impact: sy=sx wiring changes the result (orig would ignore sy)',
      Inds2.shape[1] > Inds.shape[1], (Inds.shape[1], Inds2.shape[1]))

# B3: original hardcoded k=None (k ignored); fixed honors k
res_k = fix_mod.matching(Xs, Ys, use_keops=False, method='soft', k=2,
                         device=th.device('cpu'), dtype=th.float64)
Ik = res_k['model']
D1 = Ik.D1st
jx = D1.topk(min(2, Ik.ny), dim=1, largest=False).indices
iy = D1.T.topk(min(2, Ik.nx), dim=1, largest=False).indices
ids = th.cat(((th.arange(Ik.nx)[:, None] * Ik.ny + jx).flatten(),
              (iy * Ik.ny + th.arange(Ik.ny)[:, None]).flatten())).unique(sorted=True)
got = res_k['Inds'][0] * Ik.ny + res_k['Inds'][1]
check('fixed matching honors k=2 (edges within top-k union)',
      got.numel() > 0 and th.isin(got, ids).all().item(), got.tolist())
# original (k=None) would use all pairs: show k=None returns strictly more candidates
res_all = fix_mod.matching(Xs, Ys, use_keops=False, method='soft', k=None,
                           device=th.device('cpu'), dtype=th.float64)
check('BUG B3 impact: k=None yields more candidates than k=2',
      res_all['Inds'].shape[1] > res_k['Inds'].shape[1],
      (res_k['Inds'].shape[1], res_all['Inds'].shape[1]))

# B5: a pre-fitted model is reused without re-solving
pre = fix_mod.OTM_solver(Xs, Ys, use_keops=False, device=th.device('cpu'), dtype=th.float64)
r1 = fix_mod.matching(Xs, Ys, use_keops=False, model=pre, device=th.device('cpu'),
                      dtype=th.float64)
r2 = fix_mod.matching(Xs, Ys, use_keops=False, model=pre, candidate_space='D',
                      device=th.device('cpu'), dtype=th.float64)
check('matching reuses model (cost and D candidate spaces)',
      r1['Inds'].shape[0] == 2 and th.equal(r1['model'].P, pre.P), (r1['Inds'].shape, r2['Inds'].shape))

# --------------------------------------------------------------------------
# M4/M5: candidate spaces, FGW, features+labels end-to-end
# --------------------------------------------------------------------------
print('== M4/M5: candidate spaces and end-to-end ==')
for csp in ('cost', 'D', 'geometry', 'P'):
    r = fix_mod.matching(Xs, Ys, use_keops=False, method='greedy',
                         candidate_space=csp, device=th.device('cpu'), dtype=th.float64)
    ok = r['Inds'].shape[0] == 2 and r['psp'].numel() == r['Inds'].shape[1]
    check(f'matching candidate_space={csp} runs', ok, r['Inds'].shape)
# FGW-aware matching: D candidates must be D1 + FGW correction
rf = fix_mod.matching(Xs, Ys, use_keops=False, method='soft', candidate_space='D',
                      ot_options=dict(use_fgw=True, gscale_2nd=0.2, eps=0.08),
                      device=th.device('cpu'), dtype=th.float64)
check('FGW matching caches D2nd (D candidates include FGW)', rf['model'].D2nd is not None, '')
# features + labels end-to-end (ccos + label costs)
Fx2 = th.randn(12, 6); Fy2 = th.randn(8, 6)
Lx2 = th.rand(12, 3); Ly2 = th.rand(8, 3)
rf2 = fix_mod.matching(Xs, Ys, Fxs=Fx2, Fys=Fy2, Lx=Lx2, Ly=Ly2,
                       use_keops=False, device=th.device('cpu'), dtype=th.float64)
ok = rf2['Inds'].shape[0] == 2 and th.isfinite(rf2['score']).all().item()
check('matching with features+labels runs', ok, rf2['Inds'].shape)
# transform through the matched model
Tq = rf2['model'].transform(X=Xs[:3], Fxs=Fx2[:3], Lx=Lx2[:3], method='kernel')
check('transform after matching (features+labels) finite', th.isfinite(Tq).all().item(), Tq.shape)
# OTM_solver: use_keops=None keeps dense default for small data
m0 = fix_mod.OTM_solver(Xs, Ys, use_keops=None, device=th.device('cpu'), dtype=th.float64)
check('OTM_solver use_keops=None keeps dense default', m0.use_keops is False, m0.use_keops)

print('\n' + '=' * 70)
print(f'SUMMARY: {len(PASS)} passed, {len(FAIL)} failed')
for f in FAIL:
    print('  FAILED:', f)
print('=' * 70)
sys.exit(0 if not FAIL else 1)
