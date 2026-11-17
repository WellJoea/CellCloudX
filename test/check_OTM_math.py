# -*- coding: utf-8 -*-
"""Numerical verification of _OTM.py / _OTM_fixed.py algorithm math.

Runs on dense (non-KeOps) paths with float64:
  * factor / moment identities,
  * FGW gradient vs autograd of the explicit FGW energy,
  * FGW prediction costs vs autograd (both directions),
  * UOT Sinkhorn fixed point, normal-vs-log and entropy-vs-KL consistency,
  * label kernel identities (normal and log domain),
  * one-step FGW effective cost caching,
  * transform() kernel/nearest vs closed-form conditional barycenters,
  * REPRODUCTION of the feat_normal prediction bug (original crashes,
    fixed version returns the mathematically correct normalization).
"""
import os, sys, types, warnings

import numpy as np
import torch as th

th.manual_seed(0)
np.random.seed(0)
th.set_default_dtype(th.float64)

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# --------------------------------------------------------------------------
# Stub pykeops (unusable Linux wheel in this env); all tests use dense paths.
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

# Load the package with light fake package modules (avoids the heavy root __init__).
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
    fixed_mod = _ilu.module_from_spec(_spec)
    fixed_mod.__package__ = 'cellcloudx.alignment.m3lib'
    _spec.loader.exec_module(fixed_mod)
else:
    import cellcloudx.alignment.m3lib.OTM_fixed as fixed_mod              # noqa: E402

# Current OTM.py binds its helpers before use, so this patch is a no-op kept
# for compatibility with older revisions.
class _PatchedOrig(orig_mod.OTM):
    def __init__(self, **kw):
        self.scalar2vector = orig_mod.scalar2vector
        self.auto_select_device = orig_mod.auto_select_device
        self.n2v = orig_mod.n2v
        self.plan_extract = orig_mod.plan_extract
        super().__init__(**kw)
orig_mod.OTM = _PatchedOrig

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

print('== T1/T2: _factors and _moment identities (both modules) ==')
for mod, tag in ((orig_mod, 'orig'), (fixed_mod, 'fixed')):
    otm = mk(mod)
    for d in (1, 3, 20):
        Z = th.randn(9, d)
        phi, psi = otm._factors(Z)
        check(f'{tag} factors exact d={d}', close(phi @ psi.T, dist2(Z, Z), 1e-9, 1e-9),
              float((phi @ psi.T - dist2(Z, Z)).abs().max()))
    for d, mom in ((3, 64), (25, 16)):
        Z = th.randn(11, d)
        mass = th.rand(11) + 0.1
        phi, psi = otm._factors(Z)
        got = otm._moment(Z, phi, psi, mass)
        brute = (dist2(Z, Z).square() @ mass).reshape(-1)
        check(f'{tag} _moment exact d={d} (mom_max={mom})',
              close(got, brute, 1e-9, 1e-9), float((got - brute).abs().max()))

print('== T3: D2_fgw_gradient vs autograd of the explicit FGW energy ==')
for mod, tag in ((orig_mod, 'orig'), (fixed_mod, 'fixed')):
    otm = mk(mod)
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
        ok = close(g_code, g_auto, rtol=1e-8, atol=1e-9)
        check(f'{tag} FGW grad autograd (center={cen}, s2={s2})', ok,
              float((g_code - g_auto).abs().max()))
    # low-rank vs direct branch equivalence
    otm.use_lowrank = True
    g_lr = otm.D2_fgw_gradient(X, Y, P.detach(), r, c, 2.0, centerlize=True)
    otm.use_lowrank = False
    g_dr = otm.D2_fgw_gradient(X, Y, P.detach(), r, c, 2.0, centerlize=True)
    check(f'{tag} FGW grad lowrank==direct', close(g_lr, g_dr, 1e-9, 1e-9),
          float((g_lr - g_dr).abs().max()))

print('== T4: _prediction_fgw vs autograd of the augmented energy (row constants omitted) ==')
for mod, tag in ((orig_mod, 'orig'), (fixed_mod, 'fixed')):
    otm = mk(mod)
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
    # ---- side 0: query on X side; omitted constant is (Cx^2 r0)_q ----
    q = th.randn(1, d)
    qc = q - X0.mean(0)
    Cxq = dist2(th.cat([Xc, qc]), Xc)                  # (nx+1, nx)
    p = th.zeros(ny, requires_grad=True)
    P = th.cat([P0, p[None]], 0)
    E = fgw_energy(P, dist2(th.cat([Xc, qc]), th.cat([Xc, qc])), Cy0)
    g = th.autograd.grad(E, p)[0]
    out = otm._prediction_fgw(q, Y0, X0, Y0, 1.0, (B, dx, dy), 0)[0]
    expect = g - (Cxq[nx].square() @ r0)
    check(f'{tag} pred-FGW side0 vs autograd', close(out, expect, 1e-9, 1e-9),
          float((out - expect).abs().max()))
    # ---- side 1: query on Y side; omitted constant is (Cy^2 c0)_q ----
    qy = th.randn(1, d)
    qyc = qy - Y0.mean(0)
    Cyq = dist2(th.cat([Yc, qyc]), Yc)
    pcol = th.zeros(nx, requires_grad=True)
    P2 = th.cat([P0, pcol[:, None]], 1)
    E2 = fgw_energy(P2, Cx0, dist2(th.cat([Yc, qyc]), th.cat([Yc, qyc])))
    g2 = th.autograd.grad(E2, pcol)[0]
    out2 = otm._prediction_fgw(qy, X0, X0, Y0, 1.0, (B, dx, dy), 1)[0]
    expect2 = g2 - (Cyq[ny].square() @ c0)
    check(f'{tag} pred-FGW side1 vs autograd', close(out2, expect2, 1e-9, 1e-9),
          float((out2 - expect2).abs().max()))

print('== T5: UOT Sinkhorn fixed point; normal vs log; entropy vs KL ==')
for mod, tag in ((orig_mod, 'orig'), (fixed_mod, 'fixed')):
    X = th.randn(40, 5); Y = th.randn(35, 5)
    a = th.rand(40) + 0.5; b = th.rand(35) + 0.5
    for mode in ('normal', 'log'):
        otm = mk(mod, mode=mode, reg='entropy', tol=1e-7, check_every=5, uv_iter=5000)
        P, r, c = otm.solver(X, Y, a=a, b=b)
        K = th.exp(-otm.D1st / otm.eps)
        u, v = otm.u, otm.v
        u_fp = (a / (K @ v).clamp_min(1e-300)).pow(otm.tx)
        v_fp = (b / (K.T @ u).clamp_min(1e-300)).pow(otm.ty)
        relu = ((u - u_fp).abs() / u_fp.clamp_min(1e-300)).max().item()
        relv = ((v - v_fp).abs() / v_fp.clamp_min(1e-300)).max().item()
        marg_ok = close(P.sum(1), r, 1e-9, 1e-8) and close(P.sum(0), c, 1e-9, 1e-8)
        plan_ok = close(P, u[:, None] * K * v[None, :], rtol=1e-8, atol=1e-10)
        check(f'{tag} {mode} sinkhorn fixed point (relu={relu:.2e}, relv={relv:.2e})',
              relu < 1e-5 and relv < 1e-5 and marg_ok and plan_ok,
              f'marginals={marg_ok}, plan={plan_ok}')
    # normal vs log plans must agree
    o1 = mk(mod, mode='normal', tol=1e-7, uv_iter=5000)
    o2 = mk(mod, mode='log', tol=1e-7, uv_iter=5000)
    P1, _, _ = o1.solver(X, Y, a=a, b=b)
    P2, _, _ = o2.solver(X, Y, a=a, b=b)
    check(f'{tag} normal==log plan', close(P1, P2, rtol=1e-5, atol=1e-7),
          float((P1 - P2).abs().max()))
    # all-one budgets: entropy and KL reference measures coincide
    o3 = mk(mod, mode='log', reg='entropy', tol=1e-7, uv_iter=5000)
    o4 = mk(mod, mode='log', reg='kl', tol=1e-7, uv_iter=5000)
    P3, _, _ = o3.solver(X, Y, a=th.ones(40), b=th.ones(35))
    P4, _, _ = o4.solver(X, Y, a=th.ones(40), b=th.ones(35))
    check(f'{tag} entropy==kl for all-one budgets', close(P3, P4, rtol=1e-6, atol=1e-8),
          float((P3 - P4).abs().max()))

print('== T6: label kernels (normal/log) and K_effective consistency ==')
for mod, tag in ((orig_mod, 'orig'), (fixed_mod, 'fixed')):
    otm = mk(mod, mode='log')
    Lx = th.rand(9, 5) + 0.1; Ly = th.rand(7, 5) + 0.1
    for dh in (0.0, 0.2):
        H = otm.H_label(Lx, Ly, dh)
        Href = (1 - dh) * Lx @ Ly.T + dh
        check(f'{tag} H_label dh={dh}', close(H, Href, 1e-12, 1e-12), float((H - Href).abs().max()))
        logH = otm.H_label_log(Lx, Ly, dh)
        check(f'{tag} H_label_log dh={dh}', close(logH.exp(), Href, rtol=1e-9, atol=1e-10),
              float((logH.exp() - Href).abs().max()))
    D = th.rand(9, 7) * 0.7
    otm2 = mk(mod, mode='normal', reg='kl')
    otm2.loga = th.zeros(9); otm2.logb = th.zeros(7); otm2.a = th.ones(9); otm2.b = th.ones(7)
    Kn = otm2.K_effective(D, 0.05, XL=Lx, YL=Ly, delta_h=0.1)
    otm3 = mk(mod, mode='log', reg='kl')
    otm3.loga = th.zeros(9); otm3.logb = th.zeros(7)
    logK = otm3.K_effective(D, 0.05, XL=Lx, YL=Ly, delta_h=0.1)
    check(f'{tag} K_effective log==log(normal) with labels',
          close(logK.exp(), Kn, rtol=1e-9, atol=1e-10), float((logK.exp() - Kn).abs().max()))

print('== T7: one-step FGW: Deff == D1 + grad(P0), and final plan uses Deff ==')
for mod, tag in ((orig_mod, 'orig'), (fixed_mod, 'fixed')):
    otm = mk(mod, use_fgw=True, gscale_2nd=0.4, eps=0.1, tol=1e-7, uv_iter=5000)
    X = th.randn(25, 3); Y = th.randn(22, 3)
    otm.init_model(X, Y)
    P0, r0, c0 = otm.solve(None)                      # first UOT solve
    P1, r1, c1 = otm.solve((P0, r0, c0))              # FGW-corrected solve
    g0 = otm.D2_fgw_gradient(otm.X, otm.Y, P0, r0, c0, otm.sigma2_2nd, centerlize=True)
    Deff_ref = otm.D1st + otm.gscale_2nd * g0
    check(f'{tag} cached Deff == D1 + gscale*grad(P0)',
          close(otm.D2nd, otm.gscale_2nd * g0, 1e-9, 1e-8), float((otm.D2nd - otm.gscale_2nd * g0).abs().max()))
    K = th.exp(-Deff_ref / otm.eps)
    check(f'{tag} final plan == u*K(Deff)*v',
          close(P1, otm.u[:, None] * K * otm.v[None, :], rtol=1e-8, atol=1e-10),
          float((P1 - otm.u[:, None] * K * otm.v[None, :]).abs().max()))

print('== T8: solver end-to-end marginals; transform kernel/nearest vs closed form ==')
for mod, tag in ((orig_mod, 'orig'), (fixed_mod, 'fixed')):
    otm = mk(mod, mode='log', tol=1e-7, uv_iter=5000)
    X = th.randn(30, 3); Y = th.randn(26, 3)
    P, r, c = otm.solver(X, Y)
    check(f'{tag} solver marginals r=P@1, c=P.T@1',
          close(P.sum(1), r, 1e-9, 1e-8) and close(P.sum(0), c, 1e-9, 1e-8), '')
    # forward kernel prediction on new queries
    Q = th.randn(9, 3)
    T = otm.transform(X=Q, method='kernel')
    Gq = oe.centerlize(Q.clone(), Xm=otm.g_shift[0], Xs=otm.g_scale[0])[0].to(th.float64)
    D1 = otm.D1_cost_agg(Gq, otm.Y, otm.sigma2_1st, otm.gscale_1st, [], [], otm.tau2s_1st, otm.fscales_1st)
    Tref = th.softmax(-D1 / otm.eps + otm.logv[None, :], dim=1) @ otm.Yr
    check(f'{tag} transform kernel == closed-form conditional barycenter',
          close(T, Tref, rtol=1e-9, atol=1e-9), float((T - Tref).abs().max()))
    # inverse kernel prediction
    Qy = th.randn(8, 3)
    Ti = otm.transform(X=Qy, inverse=True, method='kernel')
    Gqy = oe.centerlize(Qy.clone(), Xm=otm.g_shift[1], Xs=otm.g_scale[1])[0].to(th.float64)
    D1i = otm.D1_cost_agg(Gqy, otm.X, otm.sigma2_1st, otm.gscale_1st, [], [], otm.tau2s_1st, otm.fscales_1st)
    Tiref = th.softmax(-D1i / otm.eps + otm.logu[None, :], dim=1) @ otm.Xr
    check(f'{tag} transform kernel inverse == closed form',
          close(Ti, Tiref, rtol=1e-9, atol=1e-9), float((Ti - Tiref).abs().max()))
    # nearest / knn
    bary = otm.transform()   # fitted barycenters
    Tn = otm.transform(X=Q, method='nearest')
    Dq = otm.D1_cost_agg(Gq, otm.X, otm.sigma2_1st, otm.gscale_1st, [], [], otm.tau2s_1st, otm.fscales_1st)
    idx = Dq.argmin(1)
    Tnref = Q + (bary[idx] - otm.Xr[idx])
    check(f'{tag} transform nearest == best-training displacement',
          close(Tn, Tnref, rtol=1e-9, atol=1e-9), float((Tn - Tnref).abs().max()))
    Tk = otm.transform(X=Q, method='knn', k=5)
    check(f'{tag} transform knn shape/finite', Tk.shape == Q.shape and th.isfinite(Tk).all().item(), Tk.shape)

print('== T9: feat_normal prediction paths (BUG reproduction) ==')
# training-phase statistics for one modality, two domains
A = th.randn(20, 6) + 2.0
B = th.randn(17, 6) - 1.0
Q = th.randn(9, 6) + 0.5          # new query, different row count

def train(mod, norm):
    otm = mk(mod)
    otm.FL = 1
    otm.normal_f = [norm]
    otm.feat_scales = {}
    otm.feat_normal([A], [B])      # fit stats on both domains
    return otm

for norm in ('ccos', 'cos', 'zc'):
    o_orig = train(orig_mod, norm)
    o_fix = train(fixed_mod, norm)
    stats = o_fix.feat_scales[0][0]
    try:
        out_orig = o_orig.feat_normal([Q], predict_ids=[0], predict=True)[0][0]
        orig_err = None
    except Exception as e:
        orig_err = f'{type(e).__name__}: {e}'
    out_fix = o_fix.feat_normal([Q], predict_ids=[0], predict=True)[0][0]
    if norm == 'ccos':
        ref = oe.center_normalize(Q.clone(), Xm=stats[0])[0].to(th.float64)
    elif norm == 'cos':
        ref = oe.normalize(Q.clone())[0].to(th.float64)
    else:
        ref = oe.centerlize(Q.clone(), Xm=stats[0], Xs=stats[1])[0].to(th.float64)
    ok_fix = close(out_fix, ref, rtol=1e-10, atol=1e-12)
    print(f'  norm={norm}: original -> {orig_err}')
    print(f'  norm={norm}: fixed    -> correct={ok_fix}, maxerr={float((out_fix-ref).abs().max()):.2e}')
    if norm in ('ccos', 'cos', 'zc'):
        check(f'BUG reproduced on original ({norm})', orig_err is not None, orig_err)
    check(f'fixed normalization correct ({norm})', ok_fix, float((out_fix - ref).abs().max()))

print('== T10: end-to-end transform with features (default ccos) ==')
X = th.randn(30, 3); Y = th.randn(25, 3)
Fx = th.randn(30, 8); Fy = th.randn(25, 8)
Q = th.randn(7, 3); Fq = th.randn(7, 8)

o_orig = mk(orig_mod, normal_f='ccos')
o_orig.solver(X, Y, Fxs=Fx, Fys=Fy)
try:
    o_orig.transform(X=Q, Fxs=Fq, method='kernel')
    check('end-to-end original transform w/ features', True, 'no crash')
except Exception as e:
    check('end-to-end original transform w/ features reproduces BUG',
          isinstance(e, TypeError), f'{type(e).__name__}: {e}')

o_fix = mk(fixed_mod, normal_f='ccos')
o_fix.solver(X, Y, Fxs=Fx, Fys=Fy)
T = o_fix.transform(X=Q, Fxs=Fq, method='kernel')
# independent recomputation
Gq = oe.centerlize(Q.clone(), Xm=o_fix.g_shift[0], Xs=o_fix.g_scale[0])[0].to(th.float64)
Fqn = oe.center_normalize(Fq.clone(), Xm=o_fix.feat_scales[0][0][0])[0].to(th.float64)
D1 = o_fix.D1_cost_agg(Gq, o_fix.Y, o_fix.sigma2_1st, o_fix.gscale_1st,
                       [Fqn], o_fix.YF, o_fix.tau2s_1st, o_fix.fscales_1st)
Tref = th.softmax(-D1 / o_fix.eps + o_fix.logv[None, :], dim=1) @ o_fix.Yr
check('end-to-end fixed transform w/ features == closed form',
      close(T, Tref, rtol=1e-9, atol=1e-9), float((T - Tref).abs().max()))
# other modalities through transform should not crash either
for norm in ('cos', 'zc', 'std', 'pcc', 'image', 'label', 'pass'):
    oo = mk(fixed_mod, normal_f=norm)
    if norm == 'label':
        Fx2 = th.rand(30, 8); Fy2 = th.rand(25, 8); Fq2 = th.rand(7, 8)
    else:
        Fx2, Fy2, Fq2 = Fx, Fy, Fq
    try:
        oo.solver(X, Y, Fxs=Fx2, Fys=Fy2)
        Tt = oo.transform(X=Q, Fxs=Fq2, method='kernel')
        ok = Tt.shape == Q.shape and th.isfinite(Tt).all().item()
        check(f'fixed transform survives normal_f={norm}', ok, Tt.shape)
    except Exception as e:
        check(f'fixed transform survives normal_f={norm}', False, f'{type(e).__name__}: {e}')

print('== T11: sigma_square identity (per-dimension mean pairwise sq. distance) ==')
Xs = th.randn(12, 4); Ys = th.randn(9, 4)
s2 = oe.sigma_square(Xs, Ys)
s2ref = ((Xs[:, None] - Ys[None]) ** 2).sum(-1).mean() / Xs.shape[1]
check('sigma_square == per-dim mean pairwise squared distance', close(s2, s2ref, 1e-9, 1e-9),
      float((s2 - s2ref).abs()))

print('\n' + '=' * 70)
print(f'SUMMARY: {len(PASS)} passed, {len(FAIL)} failed')
for f in FAIL:
    print('  FAILED:', f)
print('=' * 70)
sys.exit(0 if not FAIL else 1)
