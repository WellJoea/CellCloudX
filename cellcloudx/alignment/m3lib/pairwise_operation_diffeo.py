import math
import torch as th
from torch import Tensor

try:
    import pykeops
    pykeops.set_verbose(False)
    from pykeops.torch import LazyTensor
except ImportError:
    LazyTensor = None

def sample_rff_model(Y, scales, weights=None, n_features=256, random_state=19491001, xp=th):
    """Sample one multi-scale Gaussian RFF model shared by all new stages."""
    scales = xp.as_tensor(scales, device=Y.device, dtype=Y.dtype).reshape(-1)
    S = scales.numel()
    if weights is None:
        weights = xp.ones_like(scales) 
    else:
        weights = xp.as_tensor( weights, device=Y.device, dtype=Y.dtype).reshape(-1)
    if S == 0 or weights.numel() != S:
        raise ValueError("Provide one weight per kernel scale.")
    weights = weights / weights.sum()
    n_features = int(n_features)
    if n_features < S:
        raise ValueError("rff_features must be at least the number of kernel scales.")
    
    # if S == 0 or weights.numel() != S or n_features < S:
    #     raise ValueError("rff_features must be at least the number of kernel scales.")
    # if bool(xp.any(~xp.isfinite(scales))) or bool(xp.any(scales <= 0)):
    #     raise ValueError("kernel_scales must be finite and positive.")
    # if bool(xp.any(~xp.isfinite(weights))) or bool(xp.any(weights < 0)):
    #     raise ValueError("kernel_weights must be finite and non-negative.")
    # if float(weights.sum()) <= 0:
    #     raise ValueError("kernel_weights must have a positive sum.")

    counts = [n_features // S + int(s < n_features % S) for s in range(S)]
    gen = th.Generator(device=Y.device)
    gen.manual_seed(int(random_state))

    freq, phase, amp = [], [], []
    for scale, weight, count in zip(scales, weights, counts):
        w = xp.randn(count, Y.shape[1], device=Y.device, dtype=Y.dtype, generator=gen)
        b = xp.rand(count, device=Y.device, dtype=Y.dtype, generator=gen)
        freq.append(w / scale)
        phase.append(2.0 * xp.pi * b)
        amp.append(xp.sqrt(2.0 * weight / count).expand(count))

    return {
        "origin": Y.mean(0),
        "frequencies": xp.cat(freq),
        "phases": xp.cat(phase),
        "amplitudes": xp.cat(amp),
        "kernel_scales": scales,
        "kernel_weights": weights,
        "random_state": int(random_state),
    }

def fit_rff_stage(TY, P1, PX, rff, lam, delta, n_iter, tol, xp=th):
    """Fit one residual RFF field with matrix-free Jacobi-PCG."""
    n_iter, tol = int(n_iter), float(tol)
    # lam = xp.as_tensor(lam, device=TY.device, dtype=TY.dtype).reshape(())
    # delta = xp.as_tensor(delta, device=TY.device, dtype=TY.dtype).reshape(())
    # if n_iter < 1 or tol <= 0 or float(lam) < 0 or float(delta) < 0:
    #     raise ValueError("RFF solver parameters are invalid.")

    arg = (TY - rff["origin"]) @ rff["frequencies"].T
    Z = xp.cos(arg + rff["phases"]) * rff["amplitudes"]
    B = PX - P1[:, None] * TY
    mass = P1 + delta
    rhs = Z.T @ B

    diag = xp.sum(mass[:, None] * Z.square(), dim=0)
    diag = diag + lam
    if not bool(xp.all(xp.isfinite(diag))) or bool(xp.any(diag <= 0)):
        raise RuntimeError("The RFF diagonal preconditioner is not positive.")

    C = xp.zeros_like(rhs)
    # H = xp.zeros_like(B)
    res = rhs.clone()
    zres = res / diag[:, None]
    p = zres.clone()
    rz = xp.sum(res * zres, dim=0, keepdim=True)
    rhs2 = xp.sum(rhs.square(), dim=0, keepdim=True)
    tiny = xp.finfo(TY.dtype).tiny
    limit = tol ** 2 * rhs2.clamp_min(tiny)
    active = xp.sum(res.square(), dim=0, keepdim=True) > limit

    for _ in range(n_iter):
        if not bool(xp.any(active)):
            break

        pH = Z @ p
        weighted = mass[:, None] * pH
        Ap = Z.T @ weighted + lam * p

        den = xp.sum(p * Ap, dim=0, keepdim=True)
        bad = active & (den <= 0)
        if bool(xp.any(bad)):
            raise RuntimeError("The RFF normal operator is not positive definite.")

        step = xp.zeros_like(den)
        step[active] = rz[active] / den[active]

        C = C + p * step
        H = H + pH * step
        res = res - Ap * step

        res2 = xp.sum(res.square(), dim=0, keepdim=True)
        active_new = res2 > limit

        zres = res / diag[:, None]
        rz_new = xp.sum(res * zres, dim=0, keepdim=True)

        beta = xp.zeros_like(rz)
        valid = active_new & (rz > 0)
        beta[valid] = rz_new[valid] / rz[valid]

        p = zres + p * beta
        p = xp.where(active_new.expand_as(p), p, xp.zeros_like(p))

        rz = rz_new
        active = active_new

    H = Z @ C
    data = 0.5 * xp.sum(P1[:, None] * H.square())
    data = data - xp.sum(B * H)
    raw = 0.5 * xp.sum(C.square())
    point = 0.5 * xp.sum(H.square())
    reg = lam * raw + delta * point
    change = data + reg
    return C, H, change, data, reg, raw, point

@th.no_grad()
def pw_compositional_rkhs_rff_maximization(self, Pt1s, P1s, Nps, PXs, Y, Xs):
    """Append compositional RFF-RKHS stages under one fixed PWR posterior."""

    xp, L = self.xp, self.L
    sigma2_exp = self.sigma2_exp
    itranspara, itmat = self.transparas, self.tmats
    omega = self.omega
    mu = itranspara['alpha_mu']**(self.iteration)
    alpha = itranspara['alpha']

    ols = omega/sigma2_exp # exp 
    wsa = sum( omega[iL] * float(alpha[iL]) * mu for iL in range(L) )
    PX = sum( iPX * iol for iPX, iol in zip(PXs, ols) )
    P1 = sum( iP1 * iol for iP1, iol in zip(P1s, ols) )
    TY = self.TY if getattr(self, "TY", None) is not None else Y
    

    K = int(itranspara.get("n_stages", itranspara.get("K", 3)))
    delta = float(itranspara.get("delta", 0.0))
    if K < 1 or delta < 0:
        raise ValueError("n_stages must be positive and delta non-negative.")

    old_stages = list(itmat.get("rkhs_stages", []))
    old_model = itmat.get("rkhs_model")
    if old_stages and old_model not in {None, "rff"}:
        raise RuntimeError("Different compositional RKHS models cannot share stages.")

    lam_s = K * wsa
    delta_s = K * delta

    scales = itranspara.get("kernel_scales", itranspara.get("beta"))
    weights = itranspara.get("kernel_weights")
    n_features = itranspara.get(
        "rff_features", itranspara.get("n_features", 256)
    )
    seed = int(itranspara.get("random_state", 19491001)) + len(old_stages)
    rff = sample_rff_model(
        TY, scales, weights, n_features=n_features, random_state=seed, xp=xp
    )

    act_r = Y.new_zeros(())
    act_d = Y.new_zeros(())
    new_stages = []
    changes = []
    for _ in range(K):
        result = fit_rff_stage(TY, P1, PX, rff, lam_s, delta_s,
                               itranspara["n_iter"], itranspara["tol"], xp)
        C, H, change, data, reg, raw, point = result

        scale = 1.0 + abs(float(data)) + abs(float(reg))
        eps = 100.0 * xp.finfo(Y.dtype).eps * scale
        if float(change) > eps:
            raise RuntimeError("An RFF compositional stage increased its objective.")
        if abs(float(change)) <= eps:
            break

        TY = TY + H
        act_r = act_r + K * raw
        act_d = act_d + K * point
        changes.append(float(change))

        stage = dict(rff)
        stage["model"] = "rff"
        stage["coeff"] = C
        stage["raw_action"] = K * raw
        stage["delta_action"] = K * point
        new_stages.append(stage)

    old_r = xp.as_tensor(itmat.get("rkhs_raw_action", 0.0), device=Y.device, dtype=Y.dtype)
    old_d = xp.as_tensor(itmat.get("rkhs_delta_action", 0.0), device=Y.device, dtype=Y.dtype)
    act_r = old_r + act_r
    act_d = old_d + act_d
    Qp = wsa * act_r + delta * act_d

    sigma2_exp, sigma2, Q = self.update_sigma2(
        Pt1s, P1s, Nps, PXs, Xs, TY, Qp=Qp
    )

    itmat["rkhs_stages"] = old_stages + new_stages
    itmat["rkhs_raw_action"] = act_r
    itmat["rkhs_delta_action"] = act_d
    itmat["rkhs_path_energy"] = Qp
    itmat["rkhs_stage_changes"] = (list(itmat.get("rkhs_stage_changes", [])) + changes)
    itmat["rkhs_model"] = "rff"
    itmat["W"] = None

    self.TY = TY
    self.sigma2 = sigma2
    self.sigma2_exp = sigma2_exp
    self.Q = Q


@th.no_grad()
def ccf_rkhs_deformable_transform_point(
    X, stages, xp=th, inverse=False, Ym=0, Ys=1, Xm=0, Xs=1,
    n_iter=50, tol=1e-6, relax=1.0, chunk_size=4096,
    use_keops=None, strict=True, return_info=False,
):
    """Apply control/full-kernel RKHS stages and optionally invert them numerically."""
    stages = list(stages)
    is_tensor = th.is_tensor(X)
    src_device = X.device if is_tensor else th.device("cpu")
    src_dtype = X.dtype if is_tensor and X.is_floating_point() else None

    if stages:
        ref = th.as_tensor(stages[0]["coeff"])
        device, dtype = ref.device, ref.dtype
    elif is_tensor:
        device = X.device
        dtype = X.dtype if X.is_floating_point() else th.get_default_dtype()
    else:
        device, dtype = th.device("cpu"), th.get_default_dtype()

    out = xp.as_tensor(X, device=device, dtype=dtype).clone()
    if out.ndim != 2:
        raise ValueError("X must have shape (N,D).")
    if n_iter < 1 or tol <= 0 or not 0 < relax <= 1:
        raise ValueError("n_iter, tol and relax are invalid.")
    if chunk_size is None:
        chunk_size = max(out.shape[0], 1)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive or None.")

    Ym = xp.as_tensor(Ym, device=device, dtype=dtype)
    Ys = xp.as_tensor(Ys, device=device, dtype=dtype)
    Xm = xp.as_tensor(Xm, device=device, dtype=dtype)
    Xs = xp.as_tensor(Xs, device=device, dtype=dtype)
    if bool(xp.any(Ys <= 0)) or bool(xp.any(Xs <= 0)):
        raise ValueError("Ys and Xs normalization scales must be positive.")

    out = (out - Xm) / Xs if inverse else (out - Ym) / Ys
    order = list(reversed(stages)) if inverse else stages
    max_res, max_iter, converged = 0.0, 0, True

    for stage in order:
        controls = xp.as_tensor(stage["controls"], device=device, dtype=dtype)
        coeff = xp.as_tensor(stage["coeff"], device=device, dtype=dtype)
        scales = xp.as_tensor(stage["kernel_scales"], device=device, dtype=dtype)
        weights = stage.get("kernel_weights")
        weights = None if weights is None else xp.as_tensor(weights, device=device, dtype=dtype)
        stage_keops = stage.get("backend") == "keops" if use_keops is None else use_keops

        def displacement(points):
            kernel = kernel_gaussian(
                points, controls, scales, weights, use_keops=bool(stage_keops)
            )
            return kernel @ coeff

        for start in range(0, out.shape[0], chunk_size):
            stop = min(start + chunk_size, out.shape[0])
            target = out[start:stop].clone()
            if not inverse:
                out[start:stop] = target + displacement(target)
                continue

            root = target.clone()
            # for used in range(1, n_iter + 1):
            #     proposal = target - displacement(root)
            #     update = relax * (proposal - root)
            #     root = root + update
            #     scale = 1.0 + xp.linalg.vector_norm(root, dim=1).amax()
            #     error = xp.linalg.vector_norm(update, dim=1).amax()
            #     if float(error) <= tol * float(scale):
            #         break

            # residual = root + displacement(root) - target
            # scale = 1.0 + xp.linalg.vector_norm(target, dim=1).amax()
            # relres = float(xp.linalg.vector_norm(residual, dim=1).amax() / scale)

            scale = 1.0 + float(xp.linalg.vector_norm(target, dim=1).amax())
            for used in range(1, n_iter + 1):
                residual = root + displacement(root) - target
                error = xp.linalg.vector_norm(residual, dim=1).amax()
                if float(error) <= tol * scale:
                    break
                root = root - relax * residual

            residual = root + displacement(root) - target
            relres = float(xp.linalg.vector_norm(residual, dim=1).amax()) / scale

            ok = relres <= tol
            if strict and not ok:
                raise RuntimeError("The numerical RKHS inverse did not converge.")
            out[start:stop] = root
            max_res = max(max_res, relres)
            max_iter = max(max_iter, used)
            converged = converged and ok

    out = out * Ys + Ym if inverse else out * Xs + Xm
    if is_tensor:
        if src_dtype is None:
            out = out.to(device=src_device)
        else:
            out = out.to(device=src_device, dtype=src_dtype)
    else:
        out = out.detach().cpu().numpy()

    info = {
        "converged": converged,
        "max_iterations": max_iter,
        "max_relative_residual": max_res,
    }
    return (out, info) if return_info else out

@th.no_grad()
def ccf_rkhs_rff_deformable_transform_point(
    X, stages, xp=th, inverse=False, Ym=0, Ys=1, Xm=0, Xs=1,
    n_iter=50, tol=1e-6, relax=1.0, chunk_size=50000,
    strict=True, return_info=False,
):
    """Apply sampled RFF stages to full points, with an optional numerical inverse.

    Forward: raw Y -> (Y - Ym) / Ys -> RFF stages -> Z * Xs + Xm.
    Inverse: raw X -> (X - Xm) / Xs -> inverse stages -> Z * Ys + Ym.

    Ys and Xs are normalization scales, not sampled point sets. The inverse is
    solved stage by stage with fixed-point iteration and is accepted only when
    its normalized-domain residual satisfies ``tol``.
    """
    stages = list(stages)
    is_tensor = th.is_tensor(X)
    src_device = X.device if is_tensor else th.device("cpu")
    src_dtype = X.dtype if is_tensor and X.is_floating_point() else None

    if stages:
        ref = th.as_tensor(stages[0]["coeff"])
        device, dtype = ref.device, ref.dtype
    elif is_tensor:
        device = X.device
        dtype = X.dtype if X.is_floating_point() else th.get_default_dtype()
    else:
        device, dtype = th.device("cpu"), th.get_default_dtype()

    out = xp.as_tensor(X, device=device, dtype=dtype).clone()
    if out.ndim != 2:
        raise ValueError("X must have shape (N,D).")
    if n_iter < 1 or not math.isfinite(tol) or tol <= 0:
        raise ValueError("n_iter and tol must be positive.")
    if not math.isfinite(relax) or relax <= 0 or relax > 1:
        raise ValueError("relax must be in (0,1].")
    if chunk_size is None:
        chunk_size = max(out.shape[0], 1)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive or None.")

    Ym = xp.as_tensor(Ym, device=device, dtype=dtype)
    Ys = xp.as_tensor(Ys, device=device, dtype=dtype)
    Xm = xp.as_tensor(Xm, device=device, dtype=dtype)
    Xs = xp.as_tensor(Xs, device=device, dtype=dtype)
    if bool(xp.any(Ys <= 0)) or bool(xp.any(Xs <= 0)):
        raise ValueError("Ys and Xs normalization scales must be positive.")

    if inverse:
        out = (out - Xm) / Xs
        order = list(reversed(stages))
    else:
        out = (out - Ym) / Ys
        order = stages

    bounds = []
    max_res = 0.0
    max_iter = 0
    converged = True

    if return_info:
        for stage in stages:
            freq = xp.as_tensor(stage["frequencies"], device=device, dtype=dtype)
            amp = xp.as_tensor(stage["amplitudes"], device=device, dtype=dtype)
            C = xp.as_tensor(stage["coeff"], device=device, dtype=dtype)
            c_norm = xp.linalg.vector_norm(C, dim=1)
            w_norm = xp.linalg.vector_norm(freq, dim=1)
            bound = xp.sum(amp.abs() * c_norm * w_norm)
            bounds.append(float(bound))

    for stage in order:
        origin = xp.as_tensor(stage["origin"], device=device, dtype=dtype)
        freq = xp.as_tensor(stage["frequencies"], device=device, dtype=dtype)
        phase = xp.as_tensor(stage["phases"], device=device, dtype=dtype)
        amp = xp.as_tensor(stage["amplitudes"], device=device, dtype=dtype)
        C = xp.as_tensor(stage["coeff"], device=device, dtype=dtype)

        for start in range(0, out.shape[0], chunk_size):
            stop = min(start + chunk_size, out.shape[0])
            target = out[start:stop].clone()

            if not inverse:
                arg = (target - origin) @ freq.T
                Z = xp.cos(arg + phase) * amp
                out[start:stop] = target + Z @ C
                continue

            root = target.clone()
            used = 0
            # for used in range(1, n_iter + 1):
            #     arg = (root - origin) @ freq.T
            #     Z = xp.cos(arg + phase) * amp
            #     proposal = target - Z @ C
            #     update = relax * (proposal - root)
            #     root = root + update

            #     scale = 1.0 + xp.linalg.vector_norm(root, dim=1).amax()
            #     error = xp.linalg.vector_norm(update, dim=1).amax()
            #     if float(error) <= tol * float(scale):
            #         break
            scale = 1.0 + float(xp.linalg.vector_norm(target, dim=1).amax())
            for used in range(1, n_iter + 1):
                arg = (root - origin) @ freq.T
                Z = xp.cos(arg + phase) * amp
                residual = root + Z @ C - target
                error = xp.linalg.vector_norm(residual, dim=1).amax()
                if float(error) <= tol * scale:
                    break
                root = root - relax * residual

            arg = (root - origin) @ freq.T
            Z = xp.cos(arg + phase) * amp
            residual = root + Z @ C - target
            # scale = 1.0 + xp.linalg.vector_norm(target, dim=1).amax()
            error = xp.linalg.vector_norm(residual, dim=1).amax()
            rel_error = float(error) / float(scale)
            ok = rel_error <= tol

            max_res = max(max_res, rel_error)
            max_iter = max(max_iter, used)
            converged = converged and ok
            if strict and not ok:
                raise RuntimeError(
                    "RFF inverse did not converge. The fitted map may not be "
                    "contractive or globally invertible."
                )
            out[start:stop] = root

    out = out * Ys + Ym if inverse else out * Xs + Xm
    if is_tensor:
        if src_dtype is None:
            out = out.to(device=src_device)
        else:
            out = out.to(device=src_device, dtype=src_dtype)
    else:
        out = out.detach().cpu().numpy()

    info = {
        "converged": converged,
        "max_iterations": max_iter,
        "max_relative_residual": max_res,
        "lipschitz_bounds": bounds,
        "guaranteed": not bounds or max(bounds) < 1.0,
    }
    return (out, info) if return_info else out


@th.no_grad()
def pw_compositional_rkhs_white_maximization(
    self, Pt1s, P1s, Nps, PXs, Y, Xs
):
    """PWR compositional M-step with data and RKHS regularization only."""
    xp, L = self.xp, self.L
    sigma2_exp = self.sigma2_exp
    itranspara, itmat = self.transparas, self.tmats
    omega = self.omega

    mu = itranspara["alpha_mu"] ** self.iteration
    alpha = itranspara["alpha"]
    delta = float(itranspara.get("delta", 0.0))
    n_stages = int(itranspara.get("n_stages", itranspara.get("K", 3)))
    if n_stages < 1 or delta < 0:
        raise ValueError("n_stages must be positive and delta non-negative.")

    # ols       : (L,)
    # PX        : (M, D)
    # P1        : (M,)
    ols = omega / sigma2_exp
    wsa = sum( omega[iL] * float(alpha[iL]) * mu for iL in range(L) )
    PX = sum(iPX * iol for iPX, iol in zip(PXs, ols))
    P1 = sum(iP1 * iol for iP1, iol in zip(P1s, ols))

    TYc = self.TY if getattr(self, "TY", None) is not None else Y
    if TYc.shape != Y.shape:
        raise ValueError("self.TY and Y must have the same shape.")

    old_stages = list(itmat.get("rkhs_stages", []))
    old_model = itmat.get("rkhs_model")
    if old_stages and old_model not in {None, "control_white"}:
        raise RuntimeError("Different compositional RKHS models cannot share stages.")

    # scales  : (S,)
    # weights : (S,)
    scales = itranspara.get("kernel_scales", itranspara.get("beta"))
    scales = xp.as_tensor(scales, device=Y.device, dtype=Y.dtype).reshape(-1)
    weights = itranspara.get("kernel_weights")
    if weights is None:
        weights = xp.ones_like(scales) 
    else:
        weights = xp.as_tensor(weights, device=Y.device, dtype=Y.dtype).reshape(-1)
    # if scales.numel() == 0 or weights.numel() != scales.numel():
    #     raise ValueError("kernel_scales and kernel_weights are invalid.")
    # if bool(xp.any(~xp.isfinite(scales))) or bool(xp.any(scales <= 0)):
    #     raise ValueError("kernel_scales must be finite and positive.")
    # if bool(xp.any(~xp.isfinite(weights))) or bool(xp.any(weights < 0)):
    #     raise ValueError("kernel_weights must be finite and non-negative.")
    # if float(weights.sum()) <= 0:
    #     raise ValueError("kernel_weights must have a positive sum.")
    weights = weights / weights.sum()

    # controls : (K, D)
    # Kcc : (K, K)
    # Lcc : (K, K)
    n_controls = int(itranspara.get("n_controls", 8000))
    control_method = itranspara.get("control_method", "voxel")
    seed = int(itranspara.get("random_state", 19491001)) + len(old_stages)
    controls = control_points( TYc, n_controls=n_controls, method=control_method, random_state=seed,)
    Kcc = kernel_gaussian(controls, controls, scales=scales, weights=weights, use_keops=False)
    Kcc = 0.5 * (Kcc + Kcc.T)
    jitter = float(itranspara.get("kernel_jitter", 1e-6))
    if jitter < 0:
        raise ValueError("kernel_jitter must be non-negative.")
    Kcc.diagonal().add_(jitter)
    Lcc = xp.linalg.cholesky(Kcc)

    stage_wsa = n_stages * wsa
    stage_delta = n_stages * delta
    raw_action_new = Y.new_zeros(())
    delta_action_new = Y.new_zeros(())
    stages_new, changes = [], []

    for _ in range(n_stages):
        Kyc = kernel_gaussian(
            TYc, controls, scales=scales, weights=weights, use_keops=False
        )
        # Phi : (M, K)

        Phi = xp.linalg.solve_triangular(Lcc, Kyc.T, upper=False).T
        B = PX - P1[:, None] * TYc

        Pg = Phi.T @ ((P1 + stage_delta)[:, None] * Phi)
        Pg.diagonal().add_(stage_wsa)
        Pg = 0.5 * (Pg + Pg.T)
        Ln = xp.linalg.cholesky(Pg)
        theta = xp.linalg.solve_triangular(Ln, Phi.T @ B, upper=False)
        theta = xp.linalg.solve_triangular(Ln.T, theta, upper=True)

        W = xp.linalg.solve_triangular(Lcc.T, theta, upper=True)
        # H = Phi @ theta
        H = Kyc @ W
        data_change = ( 0.5 * xp.sum(P1[:, None] * H.square()) - xp.sum(B * H))
        raw_energy = 0.5 * xp.sum(theta.square())
        delta_energy = 0.5 * xp.sum(H.square())
        stage_energy = stage_wsa * raw_energy + stage_delta * delta_energy
        objective_change = data_change + stage_energy

        scale = 1.0 + abs(float(data_change)) + abs(float(stage_energy))
        tolerance = 100.0 * xp.finfo(Y.dtype).eps * scale
        if float(objective_change) > tolerance:
            raise RuntimeError("A compositional RKHS stage increased its objective.")
        if abs(float(objective_change)) <= tolerance:
            break

        TYc = TYc + H
        raw_action_new += n_stages * raw_energy
        delta_action_new += n_stages * delta_energy
        changes.append(float(objective_change))
        stages_new.append(
            dict(
                model="control_white",
                controls=controls,
                coeff=W,
                kernel_scales=scales,
                kernel_weights=weights,
                raw_action=n_stages * raw_energy,
                delta_action=n_stages * delta_energy,
                backend="dense",
                random_state=seed,
            )
        )

    raw_action = xp.as_tensor(
        itmat.get("rkhs_raw_action", 0.0), device=Y.device, dtype=Y.dtype
    ) + raw_action_new
    delta_action = xp.as_tensor(
        itmat.get("rkhs_delta_action", 0.0), device=Y.device, dtype=Y.dtype
    ) + delta_action_new
    Qp = wsa * raw_action + delta * delta_action

    sigma2_exp, sigma2, Q = self.update_sigma2(
        Pt1s, P1s, Nps, PXs, Xs, TYc, Qp=Qp
    )

    itmat["rkhs_stages"] = old_stages + stages_new
    itmat["rkhs_raw_action"] = raw_action
    itmat["rkhs_delta_action"] = delta_action
    itmat["rkhs_path_energy"] = Qp
    itmat["rkhs_stage_changes"] = (
        list(itmat.get("rkhs_stage_changes", [])) + changes
    )
    itmat["rkhs_model"] = "control_white"
    itmat["W"] = None
    self.TY = TYc
    self.sigma2 = sigma2
    self.sigma2_exp = sigma2_exp
    self.Q = Q


def nonwhite_rkhs_solver(
    TY, P1, B, scales, weights, lam, delta,
    solver="pcg", n_iter=50, tol=1e-5, use_keops=False,
    direct_max_points=3000, xp=th,
):
    """Solve one full-point RKHS stage without kernel whitening.

    The canonical full-control equation is
        [M G + lam I] W = B,
    where M = diag(P1 + delta), G = K(TY, TY), and H = G W.
    With S = sqrt(M) and W = S U, the symmetric system is
        [S G S + lam I] U = S^{-1} B.

    Solvers are direct, cg, pcg and auto. The keops_cg and keops_pcg names
    are convenient aliases; KeOps is the matrix-vector backend, not a new
    mathematical solver.
    """
    solver = str(solver).lower()
    if solver in {"chol", "cholesky"}:
        solver = "direct"
    elif solver in {"keops", "keops_pcg"}:
        solver, use_keops = "pcg", True
    elif solver == "keops_cg":
        solver, use_keops = "cg", True

    direct_max_points = int(direct_max_points)
    if solver == "auto":
        solver = "direct" if TY.shape[0] <= direct_max_points else "pcg"
    n_iter = int(n_iter)
    tol = float(tol)
    lam = xp.as_tensor(lam, device=TY.device, dtype=TY.dtype).reshape(())
    delta = xp.as_tensor(delta, device=TY.device, dtype=TY.dtype).reshape(())
    if solver not in {"direct", "cg", "pcg"}:
        raise ValueError("rkhs_solver must be auto, direct, cg, pcg or a KeOps alias.")
    if n_iter < 1 or tol <= 0:
        raise ValueError("rkhs_inner_steps and rkhs_inner_tol must be positive.")
    if float(lam) <= 0 or float(delta) < 0:
        raise ValueError("lam must be positive and delta non-negative.")

    mass = P1 + delta
    if bool(xp.any(mass < 0)):
        raise ValueError("P1 + stage_delta must be non-negative.")

    root = xp.sqrt(mass)
    positive = mass > 0
    if bool(xp.any(~positive)) and bool(xp.any(B[~positive] != 0)):
        raise ValueError("B must be zero where P1 + stage_delta is zero.")
    rhs = xp.zeros_like(B)
    rhs[positive] = B[positive] / root[positive, None]
    tiny = xp.finfo(TY.dtype).tiny

    if solver == "direct":
        use_keops = False
        G = kernel_gaussian(TY, TY, scales, weights, use_keops=False)
        A = root[:, None] * G * root[None, :]
        A.diagonal().add_(lam)
        A = 0.5 * (A + A.T)
        Ln = xp.linalg.cholesky(A)
        U = xp.linalg.solve_triangular(Ln, rhs, upper=False)
        U = xp.linalg.solve_triangular(Ln.T, U, upper=True)
        W = root[:, None] * U
        H = G @ W
        res = rhs - A @ U
        steps = 1
    else:
        G = kernel_gaussian(TY, TY, scales, weights, use_keops=use_keops)
        if solver == "pcg":
            diag = mass * weights.sum() + lam
            if bool(xp.any(diag <= 0)):
                raise RuntimeError("The non-white Jacobi preconditioner is not positive.")

        U = xp.zeros_like(B)
        # H = xp.zeros_like(B)
        res = rhs.clone()
        zres = res / diag[:, None] if solver == "pcg" else res
        p = zres.clone()
        rz = xp.sum(res * zres, dim=0, keepdim=True)
        rhs2 = xp.sum(rhs.square(), dim=0, keepdim=True)
        limit = tol ** 2 * rhs2.clamp_min(tiny)
        steps = 0

        for iteration in range(1, n_iter + 1):
            active = xp.sum(res.square(), dim=0, keepdim=True) > limit
            if not bool(xp.any(active)):
                break
            steps = iteration

            Sp = root[:, None] * p
            pH = G @ Sp
            Ap = root[:, None] * pH + lam * p
            den = xp.sum(p * Ap, dim=0, keepdim=True)
            bad = active & (den <= 0)
            if bool(xp.any(bad)):
                raise RuntimeError("The non-white RKHS operator is not positive definite.")

            step = xp.zeros_like(den)
            step[active] = rz[active] / den[active]
            U = U + p * step
            # H = H + pH * step
            res = res - Ap * step

            res2 = xp.sum(res.square(), dim=0, keepdim=True)
            active = res2 > limit
            if not bool(xp.any(active)):
                break

            zres = res / diag[:, None] if solver == "pcg" else res
            rz_new = xp.sum(res * zres, dim=0, keepdim=True)
            beta = xp.zeros_like(rz)
            valid = active & (rz > 0)
            beta[valid] = rz_new[valid] / rz[valid]
            p = xp.where(active.expand_as(p), zres + p * beta, xp.zeros_like(p))
            rz = rz_new

        W = root[:, None] * U
        H = G @ W

    res2 = xp.sum(res.square(), dim=0)
    rhs2 = xp.sum(rhs.square(), dim=0).clamp_min(tiny)
    relres = xp.sqrt(xp.max(res2 / rhs2))
    data = 0.5 * xp.sum(P1[:, None] * H.square()) - xp.sum(B * H)
    raw = 0.5 * xp.sum(W * H)
    point = 0.5 * xp.sum(H.square())
    reg = lam * raw + delta * point
    change = data + reg
    backend = "keops" if use_keops else "dense"
    return W, H, change, data, reg, raw, point, steps, relres, solver, backend

@th.no_grad()
def pw_compositional_rkhs_nonwhite_maximization(self, Pt1s, P1s, Nps, PXs, Y, Xs):
    """PWR full-point compositional RKHS M-step without kernel whitening.

    Relevant parameters
    -------------------
    rkhs_solver : {'auto', 'direct', 'cg', 'pcg', 'keops_cg', 'keops_pcg'}
    rkhs_backend : {'dense', 'keops'}
    rkhs_inner_steps, rkhs_inner_tol, rkhs_direct_max_points
    """
    xp, L = self.xp, self.L
    sigma2_exp = self.sigma2_exp
    itranspara, itmat = self.transparas, self.tmats
    omega = self.omega

    mu = itranspara["alpha_mu"] ** self.iteration
    alpha = itranspara["alpha"]
    delta = float(itranspara.get("delta", 0.0))
    n_stages = int(itranspara.get("n_stages", itranspara.get("K", 3)))
    if n_stages < 1 or delta < 0:
        raise ValueError("n_stages must be positive and delta non-negative.")

    ols = omega / sigma2_exp
    wsa = sum(omega[iL] * float(alpha[iL]) * mu for iL in range(L))
    PX = sum(iPX * iol for iPX, iol in zip(PXs, ols))
    P1 = sum(iP1 * iol for iP1, iol in zip(P1s, ols))

    TYc = self.TY if getattr(self, "TY", None) is not None else Y
    if TYc.shape != Y.shape:
        raise ValueError("self.TY and Y must have the same shape.")

    old_stages = list(itmat.get("rkhs_stages", []))
    old_model = itmat.get("rkhs_model")
    if old_stages and old_model not in {None, "full_nonwhite"}:
        raise RuntimeError("Different compositional RKHS models cannot share stages.")

    scales = itranspara.get("kernel_scales", itranspara.get("beta"))
    scales = xp.as_tensor(scales, device=Y.device, dtype=Y.dtype).reshape(-1)
    weights = itranspara.get("kernel_weights")
    weights = xp.ones_like(scales) if weights is None else xp.as_tensor(
        weights, device=Y.device, dtype=Y.dtype
    ).reshape(-1)
    if scales.numel() == 0 or weights.numel() != scales.numel():
        raise ValueError("kernel_scales and kernel_weights are invalid.")
    # if bool(xp.any(~xp.isfinite(scales))) or bool(xp.any(scales <= 0)):
    #     raise ValueError("kernel_scales must be finite and positive.")
    # if bool(xp.any(~xp.isfinite(weights))) or bool(xp.any(weights < 0)):
    #     raise ValueError("kernel_weights must be finite and non-negative.")
    # if not bool(xp.any(weights > 0)):
    #     raise ValueError("Kernel scales must be positive and weights non-negative.")
    weights = weights / weights.sum()

    solver = str(itranspara.get("rkhs_solver", "pcg")).lower()
    n_iter = int(itranspara.get("rkhs_inner_steps", 50))
    tol = float(itranspara.get("rkhs_inner_tol", 1e-5))
    direct_max_points = int(itranspara.get("rkhs_direct_max_points", 3000))
    if "rkhs_backend" in itranspara:
        backend = str(itranspara["rkhs_backend"]).lower()
        if backend not in {"dense", "keops"}:
            raise ValueError("rkhs_backend must be 'dense' or 'keops'.")
        use_keops = backend == "keops"
    else:
        use_keops = bool(itranspara.get("rkhs_use_keops", False))
    stage_wsa = n_stages * wsa
    stage_delta = n_stages * delta
    raw_action_new = Y.new_zeros(())
    delta_action_new = Y.new_zeros(())
    stages_new, changes, inner_steps, residuals = [], [], [], []
    solvers, backends = [], []

    for _ in range(n_stages):
        controls = TYc
        B = PX - P1[:, None] * TYc
        result = nonwhite_rkhs_solver(
            TYc, P1, B, scales, weights, stage_wsa, stage_delta,
            solver=solver, n_iter=n_iter, tol=tol, use_keops=use_keops,
            direct_max_points=direct_max_points, xp=xp,
        )
        W, H, change, data, reg, raw, point, steps, relres, used, backend = result

        data_value = float(data)
        reg_value = float(reg)
        change_value = float(change)
        scale = 1.0 + abs(data_value) + abs(reg_value)
        tolerance = 100.0 * xp.finfo(Y.dtype).eps * scale
        if change_value > tolerance:
            raise RuntimeError("A compositional non-white RKHS stage increased its objective.")
        if abs(change_value) <= tolerance:
            break

        TYc = TYc + H
        raw_action_new = raw_action_new + n_stages * raw
        delta_action_new = delta_action_new + n_stages * point
        changes.append(change_value)
        inner_steps.append(int(steps))
        residuals.append(float(relres))
        solvers.append(used)
        backends.append(backend)
        stages_new.append(
            dict(
                model="full_nonwhite",
                controls=controls,
                coeff=W,
                kernel_scales=scales,
                kernel_weights=weights,
                raw_action=n_stages * raw,
                delta_action=n_stages * point,
                solver=used,
                backend=backend,
            )
        )

    raw_action = xp.as_tensor(itmat.get("rkhs_raw_action", 0.0), device=Y.device, dtype=Y.dtype)
    delta_action = xp.as_tensor(
        itmat.get("rkhs_delta_action", 0.0), device=Y.device, dtype=Y.dtype
    )
    raw_action = raw_action + raw_action_new
    delta_action = delta_action + delta_action_new
    Qp = wsa * raw_action + delta * delta_action

    sigma2_exp, sigma2, Q = self.update_sigma2(
        Pt1s, P1s, Nps, PXs, Xs, TYc, Qp=Qp,
    )

    itmat["rkhs_stages"] = old_stages + stages_new
    itmat["rkhs_raw_action"] = raw_action
    itmat["rkhs_delta_action"] = delta_action
    itmat["rkhs_path_energy"] = Qp
    itmat["rkhs_stage_changes"] = list(itmat.get("rkhs_stage_changes", [])) + changes
    itmat["rkhs_inner_steps"] = list(itmat.get("rkhs_inner_steps", [])) + inner_steps
    itmat["rkhs_inner_residuals"] = list(itmat.get("rkhs_inner_residuals", [])) + residuals
    itmat["rkhs_solvers"] = list(itmat.get("rkhs_solvers", [])) + solvers
    itmat["rkhs_backends"] = list(itmat.get("rkhs_backends", [])) + backends
    itmat["rkhs_model"] = "full_nonwhite"
    itmat["W"] = None
    self.TY = TYc
    self.sigma2 = sigma2
    self.sigma2_exp = sigma2_exp
    self.Q = Q



def svf_flow(
    X, controls, coeff, scales, weights=None,
    n_steps=8, inverse=False, use_keops=True,
):
    """Integrate a stationary Gaussian-kernel velocity field with RK4."""
    if n_steps < 1:
        raise ValueError("n_steps must be positive.")

    dt = (-1.0 if inverse else 1.0) / n_steps
    Z = X
    for _ in range(n_steps):
        K1 = kernel_gaussian(Z, controls, scales=scales, weights=weights, use_keops=use_keops)
        V1 = K1 @ coeff
        Z2 = Z + 0.5 * dt * V1
        K2 = kernel_gaussian(
            Z2, controls, scales=scales, weights=weights, use_keops=use_keops
        )
        V2 = K2 @ coeff
        Z3 = Z + 0.5 * dt * V2
        K3 = kernel_gaussian(
            Z3, controls, scales=scales, weights=weights, use_keops=use_keops
        )
        V3 = K3 @ coeff
        Z4 = Z + dt * V3
        K4 = kernel_gaussian(
            Z4, controls, scales=scales, weights=weights, use_keops=use_keops
        )
        V4 = K4 @ coeff
        Z = Z + (dt / 6.0) * (V1 + 2.0 * V2 + 2.0 * V3 + V4)
    return Z


@th.no_grad()
def pw_svf_maximization(self, Pt1s, P1s, Nps, PXs, Y, Xs):
    """PWR M-step for one fixed-control stationary velocity field.

    Parameters in self.transparas
    ----------------------------
    kernel_scales, kernel_weights, n_controls, control_method
    svf_steps, svf_inner_steps, svf_lr, svf_tol, svf_patience
    svf_use_keops, random_state
    """
    xp, L = self.xp, self.L
    sigma2_exp = self.sigma2_exp
    itranspara, itmat = self.transparas, self.tmats
    omega = self.omega

    mu = itranspara["alpha_mu"] ** self.iteration
    alpha = itranspara["alpha"]
    ols = omega / sigma2_exp
    wsa = sum(omega[iL] * float(alpha[iL]) * mu for iL in range(L))
    PX = sum(iPX * iol for iPX, iol in zip(PXs, ols))
    P1 = sum(iP1 * iol for iP1, iol in zip(P1s, ols))

    scales = itranspara.get("kernel_scales", itranspara.get("beta"))
    scales = xp.as_tensor(scales, device=Y.device, dtype=Y.dtype).reshape(-1)
    weights = itranspara.get("kernel_weights")
    weights = xp.ones_like(scales) if weights is None else xp.as_tensor(
        weights, device=Y.device, dtype=Y.dtype
    ).reshape(-1)
    if scales.numel() == 0 or weights.numel() != scales.numel():
        raise ValueError("kernel_scales and kernel_weights are invalid.")
    # if bool(xp.any(~xp.isfinite(scales))) or bool(xp.any(scales <= 0)):
    #     raise ValueError("Kernel scales must be finite and positive.")
    # if (
    #     bool(xp.any(~xp.isfinite(weights)))
    #     or bool(xp.any(weights < 0))
    #     or float(weights.sum()) <= 0
    # ):
    #     raise ValueError("Kernel scales must be positive and weights non-negative.")
    weights = weights / weights.sum()

    n_steps = int(itranspara.get("svf_steps", 8))
    n_iter = int(itranspara.get("svf_inner_steps", 50))
    lr = float(itranspara.get("svf_lr", 1e-2))
    tol = float(itranspara.get("svf_tol", 1e-6))
    patience = int(itranspara.get("svf_patience", 5))
    use_keops = bool(itranspara.get("svf_use_keops", True))
    if min(n_steps, n_iter, patience) < 1 or lr <= 0 or tol < 0:
        raise ValueError("SVF integration and optimizer parameters are invalid.")

    base = itmat.get("svf_base")
    if base is None:
        base = self.TY if getattr(self, "TY", None) is not None else Y
        base = base.detach().clone()
    else:
        base = xp.as_tensor(base, device=Y.device, dtype=Y.dtype)

    controls = itmat.get("svf_controls")
    seed = int(itmat.get("svf_random_state", itranspara.get(
        "svf_random_state", itranspara.get("random_state", 19491001)
    )))
    if controls is None:
        n_controls = int(itranspara.get(
            "svf_n_controls", itranspara.get("n_controls", 8000)
        ))
        method = itranspara.get(
            "svf_control_method", itranspara.get("control_method", "voxel")
        )
        controls = control_points(
            base, n_controls=n_controls, method=method, random_state=seed
        )
    controls = xp.as_tensor(controls, device=Y.device, dtype=Y.dtype).detach()

    initial = itmat.get("svf_coeff")
    if initial is not None:
        initial = xp.as_tensor(initial, device=Y.device, dtype=Y.dtype)
    if initial is None or initial.shape != controls.shape:
        initial = xp.zeros_like(controls)
    coeff = xp.nn.Parameter(initial.detach().clone())
    optimizer = xp.optim.Adam([coeff], lr=lr)

    B = PX - P1[:, None] * base
    Kcc = kernel_gaussian(
        controls, controls, scales=scales, weights=weights, use_keops=use_keops
    )
    history = []
    best_loss = float("inf")
    previous = None
    stale = 0

    with xp.enable_grad():
        for iteration in range(n_iter + 1):
            optimizer.zero_grad(set_to_none=True)
            TY = svf_flow(
                base, controls, coeff, scales, weights,
                n_steps=n_steps, use_keops=use_keops,
            )
            U = TY - base
            data = 0.5 * xp.sum(P1[:, None] * U.square()) - xp.sum(B * U)
            raw = 0.5 * xp.sum(coeff * (Kcc @ coeff))
            loss = data + wsa * raw

            value = float(loss.detach())
            if not math.isfinite(value):
                raise FloatingPointError("Non-finite SVF objective.")

            history.append(value)
            if value < best_loss:
                best_loss = value
                best_coeff = coeff.detach().clone()
                best_TY = TY.detach()
                best_data = data.detach()
                best_raw = raw.detach()

            if previous is not None and abs(previous - value) <= tol * (1.0 + abs(previous)):
                stale += 1
            else:
                stale = 0
            if iteration == n_iter or stale >= patience:
                break

            loss.backward()
            optimizer.step()
            previous = value

    Qp = wsa * best_raw
    sigma2_exp, sigma2, Q = self.update_sigma2(
        Pt1s, P1s, Nps, PXs, Xs, best_TY, Qp=Qp
    )

    itmat["svf_base"] = base
    itmat["svf_controls"] = controls
    itmat["svf_coeff"] = best_coeff
    itmat["svf_kernel_scales"] = scales
    itmat["svf_kernel_weights"] = weights
    itmat["svf_steps"] = n_steps
    itmat["svf_random_state"] = seed
    itmat["svf_use_keops"] = use_keops
    itmat["svf_raw_action"] = best_raw
    itmat["svf_path_energy"] = Qp
    itmat["svf_data_change"] = best_data
    itmat["svf_objective"] = best_data + Qp
    itmat["svf_loss_history"] = history
    itmat["svf_inner_steps"] = iteration
    itmat["svf_model"] = "stationary"
    itmat["W"] = None

    self.TY = best_TY
    self.sigma2 = sigma2
    self.sigma2_exp = sigma2_exp
    self.Q = Q


def tvvf_flow(
    X, controls, coeff_steps, scales, weights=None,
    substeps=1, inverse=False, use_keops=True,
):
    """Integrate a piecewise-stationary velocity field with RK4."""
    if coeff_steps.ndim != 3 or coeff_steps.shape[1:] != controls.shape:
        raise ValueError("coeff_steps must have shape (n_time, C, D).")
    if coeff_steps.shape[0] == 0:
        raise ValueError("coeff_steps must contain at least one time interval.")

    sequence = coeff_steps.flip(0) if inverse else coeff_steps
    dt = (-1.0 if inverse else 1.0) / (coeff_steps.shape[0] * substeps)
    Z = X
    for coeff in sequence:
        for _ in range(substeps):
            K1 = kernel_gaussian(
                Z, controls, scales=scales, weights=weights, use_keops=use_keops
            )
            V1 = K1 @ coeff
            Z2 = Z + 0.5 * dt * V1
            K2 = kernel_gaussian(
                Z2, controls, scales=scales, weights=weights, use_keops=use_keops
            )
            V2 = K2 @ coeff
            Z3 = Z + 0.5 * dt * V2
            K3 = kernel_gaussian(
                Z3, controls, scales=scales, weights=weights, use_keops=use_keops
            )
            V3 = K3 @ coeff
            Z4 = Z + dt * V3
            K4 = kernel_gaussian(
                Z4, controls, scales=scales, weights=weights, use_keops=use_keops
            )
            V4 = K4 @ coeff
            Z = Z + (dt / 6.0) * (V1 + 2.0 * V2 + 2.0 * V3 + V4)
    return Z


@th.no_grad()
def pw_tvvf_maximization(self, Pt1s, P1s, Nps, PXs, Y, Xs):
    """PWR M-step for one fixed-control piecewise-stationary velocity field."""
    xp, L = self.xp, self.L
    sigma2_exp = self.sigma2_exp
    itranspara, itmat = self.transparas, self.tmats
    omega = self.omega

    mu = itranspara["alpha_mu"] ** self.iteration
    alpha = itranspara["alpha"]
    ols = omega / sigma2_exp
    wsa = sum(omega[iL] * float(alpha[iL]) * mu for iL in range(L))
    PX = sum(iPX * iol for iPX, iol in zip(PXs, ols))
    P1 = sum(iP1 * iol for iP1, iol in zip(P1s, ols))

    scales = itranspara.get("kernel_scales", itranspara.get("beta"))
    scales = xp.as_tensor(scales, device=Y.device, dtype=Y.dtype).reshape(-1)
    weights = itranspara.get("kernel_weights")
    weights = xp.ones_like(scales) if weights is None else xp.as_tensor(
        weights, device=Y.device, dtype=Y.dtype
    ).reshape(-1)
    if scales.numel() == 0 or weights.numel() != scales.numel():
        raise ValueError("kernel_scales and kernel_weights are invalid.")
    # if bool(xp.any(~xp.isfinite(scales))) or bool(xp.any(scales <= 0)):
    #     raise ValueError("Kernel scales must be finite and positive.")
    # if (
    #     bool(xp.any(~xp.isfinite(weights)))
    #     or bool(xp.any(weights < 0))
    #     or float(weights.sum()) <= 0
    # ):
    #     raise ValueError("Kernel scales must be positive and weights non-negative.")
    weights = weights / weights.sum()

    n_time = int(itranspara.get("tvvf_n_time", 4))
    substeps = int(itranspara.get("tvvf_substeps", 1))
    n_iter = int(itranspara.get("tvvf_inner_steps", 60))
    lr = float(itranspara.get("tvvf_lr", 1e-2))
    tol = float(itranspara.get("tvvf_tol", 1e-6))
    patience = int(itranspara.get("tvvf_patience", 5))
    use_keops = bool(itranspara.get("tvvf_use_keops", True))
    if min(n_time, substeps, n_iter, patience) < 1 or lr <= 0 or tol < 0:
        raise ValueError("TVVF integration and optimizer parameters are invalid.")

    base = itmat.get("tvvf_base")
    if base is None:
        base = self.TY if getattr(self, "TY", None) is not None else Y
        base = base.detach().clone()
    else:
        base = xp.as_tensor(base, device=Y.device, dtype=Y.dtype)

    controls = itmat.get("tvvf_controls")
    seed = int(itmat.get("tvvf_random_state", itranspara.get(
        "tvvf_random_state", itranspara.get("random_state", 19491001)
    )))
    if controls is None:
        n_controls = itranspara.get("tvvf_n_controls", itranspara.get("n_controls", 8000))
        n_controls = int(n_controls)
        method = itranspara.get(
            "tvvf_control_method", itranspara.get("control_method", "voxel")
        )
        controls = control_points(
            base, n_controls=n_controls, method=method, random_state=seed
        )
    controls = xp.as_tensor(controls, device=Y.device, dtype=Y.dtype).detach()

    shape = (n_time, *controls.shape)
    initial = itmat.get("tvvf_coeff_steps")
    if initial is not None:
        initial = xp.as_tensor(initial, device=Y.device, dtype=Y.dtype)
    if initial is None or initial.shape != shape:
        initial = xp.zeros(shape, device=Y.device, dtype=Y.dtype)
    coeff_steps = xp.nn.Parameter(initial.detach().clone())
    optimizer = xp.optim.Adam([coeff_steps], lr=lr)

    B = PX - P1[:, None] * base
    Kcc = kernel_gaussian(
        controls, controls, scales=scales, weights=weights, use_keops=use_keops
    )
    history = []
    best_loss = float("inf")
    previous = None
    stale = 0

    with xp.enable_grad():
        for iteration in range(n_iter + 1):
            optimizer.zero_grad(set_to_none=True)
            TY = tvvf_flow(
                base, controls, coeff_steps, scales, weights,
                substeps=substeps, use_keops=use_keops,
            )
            U = TY - base
            data = 0.5 * xp.sum(P1[:, None] * U.square()) - xp.sum(B * U)
            raw = sum(0.5 * xp.sum(a * (Kcc @ a)) for a in coeff_steps) / n_time
            loss = data + wsa * raw
            value = float(loss.detach())
            if not math.isfinite(value):
                raise FloatingPointError("Non-finite TVVF objective.")
            history.append(value)
            if value < best_loss:
                best_loss = value
                best_coeff_steps = coeff_steps.detach().clone()
                best_TY = TY.detach()
                best_data = data.detach()
                best_raw = raw.detach()

            if previous is not None and abs(previous - value) <= tol * (1.0 + abs(previous)):
                stale += 1
            else:
                stale = 0
            if iteration == n_iter or stale >= patience:
                break

            loss.backward()
            optimizer.step()
            previous = value

    Qp = wsa * best_raw
    sigma2_exp, sigma2, Q = self.update_sigma2(
        Pt1s, P1s, Nps, PXs, Xs, best_TY, Qp=Qp
    )

    itmat["tvvf_base"] = base
    itmat["tvvf_controls"] = controls
    itmat["tvvf_coeff_steps"] = best_coeff_steps
    itmat["tvvf_kernel_scales"] = scales
    itmat["tvvf_kernel_weights"] = weights
    itmat["tvvf_n_time"] = n_time
    itmat["tvvf_substeps"] = substeps
    itmat["tvvf_random_state"] = seed
    itmat["tvvf_use_keops"] = use_keops
    itmat["tvvf_raw_action"] = best_raw
    itmat["tvvf_path_energy"] = Qp
    itmat["tvvf_data_change"] = best_data
    itmat["tvvf_objective"] = best_data + Qp
    itmat["tvvf_loss_history"] = history
    itmat["tvvf_inner_steps"] = iteration
    itmat["tvvf_model"] = "piecewise_stationary"
    itmat["W"] = None

    self.TY = best_TY
    self.sigma2 = sigma2
    self.sigma2_exp = sigma2_exp
    self.Q = Q


def lddmm_flow_rk4(
    X, q0, p0, scales, weights=None, n_steps=10,
    inverse=False, use_keops=True, chunk_size=2048,
):
    """Backward-compatible RK4 wrapper around lddmm_flow."""
    return lddmm_flow(
        X, q0, p0, scales, weights,
        n_steps=n_steps, inverse=inverse, use_keops=use_keops,
        chunk_size=chunk_size, method="rk4",
    )


@th.no_grad()
def pw_lddmm_rk4_maximization(self, Pt1s, P1s, Nps, PXs, Y, Xs):
    """Backward-compatible wrapper selecting the unified RK4 method."""
    itranspara = self.transparas
    missing = object()
    previous = itranspara.get("lddmm_method", missing)
    itranspara["lddmm_method"] = "rk4"
    try:
        return pw_lddmm_maximization(self, Pt1s, P1s, Nps, PXs, Y, Xs)
    finally:
        if previous is missing:
            itranspara.pop("lddmm_method", None)
        else:
            itranspara["lddmm_method"] = previous


def _lddmm_operators(scales, weights, use_keops, chunk_size, rff_model=None):
    """Velocity, Hamiltonian RHS and spatial Lipschitz bound for one fixed kernel."""
    inv_scale2 = scales.square().reciprocal()

    def features(x):
        freq, phase, amp = rff_model
        return amp * th.cos(x @ freq.T + phase)

    def velocity(x, q, p):
        if x.shape[0] == 0:
            return x.clone()
        if rff_model is not None:
            A = features(q).T @ p
            return th.cat([features(xi) @ A for xi in x.split(chunk_size)])
        if use_keops:
            return kernel_gaussian(x, q, scales, weights, use_keops=True) @ p
        return th.cat([
            kernel_gaussian(xi, q, scales, weights, use_keops=False) @ p
            for xi in x.split(chunk_size)
        ])

    def system(q, p):
        if rff_model is not None:
            freq, phase, amp = rff_model
            angle = q @ freq.T + phase
            Phi = amp * th.cos(angle)
            A = Phi.T @ p
            qdot = Phi @ A
            pdot = ((p @ A.T) * (amp * th.sin(angle))) @ freq
            return qdot, pdot
        if use_keops:
            qi, qj = LazyTensor(q[:, None, :]), LazyTensor(q[None, :, :])
            pi, pj = LazyTensor(p[:, None, :]), LazyTensor(p[None, :, :])
            diff = qi - qj
            d2 = (diff ** 2).sum(-1)
            kernel = weights[0] * (-0.5 * inv_scale2[0] * d2).exp()
            factor = inv_scale2[0] * kernel
            for inv2, weight in zip(inv_scale2[1:], weights[1:]):
                Ks = weight * (-0.5 * inv2 * d2).exp()
                kernel, factor = kernel + Ks, factor + inv2 * Ks
            return (kernel * pj).sum(1), (factor * (pi * pj).sum(-1) * diff).sum(1)

        qdots, pdots = [], []
        for qi, pi in zip(q.split(chunk_size), p.split(chunk_size)):
            diff = qi[:, None, :] - q[None, :, :]
            d2 = diff.square().sum(-1)
            kernel = weights[0] * th.exp(-0.5 * inv_scale2[0] * d2)
            factor = inv_scale2[0] * kernel
            for inv2, weight in zip(inv_scale2[1:], weights[1:]):
                Ks = weight * th.exp(-0.5 * inv2 * d2)
                kernel, factor = kernel + Ks, factor + inv2 * Ks
            qdots.append(kernel @ p)
            pdots.append((((pi @ p.T) * factor)[:, :, None] * diff).sum(1))
        return th.cat(qdots), th.cat(pdots)

    def energy(q, p):
        if rff_model is not None:
            return 0.5 * (features(q).T @ p).square().sum()
        return 0.5 * (p * velocity(q, q, p)).sum()

    def lipschitz(q, p):
        # Supremum of ||Dv(x)||: Gaussian gradient, or fixed RFF feature gradients.
        if rff_model is not None:
            freq, _, amp = rff_model
            A = features(q).T @ p
            return (amp.abs() * freq.norm(dim=1) * A.norm(dim=1)).sum()
        return math.exp(-0.5) * (weights / scales).sum() * p.norm(dim=1).sum()

    return velocity, system, energy, lipschitz


def _lddmm_tableau(method):
    """Explicit Runge--Kutta coefficients; RFF retains the historical Ralston default."""
    if method in {"ralston", "rff"}:
        return ((), (2.0 / 3.0,)), (0.25, 0.75)
    return ((), (0.5,), (0.0, 0.5), (0.0, 0.0, 1.0)), (1/6, 1/3, 1/3, 1/6)


def lddmm_flow(
    X, q0, p0, scales, weights=None, n_steps=10,
    inverse=False, use_keops=True, chunk_size=2048,
    method="ralston", rff_model=None, max_steps=None, lipschitz_max=0.9,
    inverse_tol=1e-6, inverse_max_iter=200, return_info=False,
):
    """Shoot fixed initial controls/momenta and transport arbitrary passive points.

    With ``max_steps=None`` the integration grid stays fixed, as required inside
    one M-step. Supplying a cap enables step doubling until each spatial RK
    residual has Lipschitz bound below ``lipschitz_max``. The inverse solves the
    *same discrete steps* in reverse order, rather than negating the time step.
    ``inverse_tol`` bounds each step's relative residual, not the full round trip.
    Scales are Gaussian standard deviations; RFF uses its stored approximate kernel.
    """
    method = str(method).lower()
    if method not in {"ralston", "rk4", "rff"}:
        raise ValueError("method must be 'ralston', 'rk4' or 'rff'.")
    if q0.shape != p0.shape or min(n_steps, chunk_size, inverse_max_iter) < 1:
        raise ValueError("Provide matching q0/p0 shapes and positive step counts.")
    scales = th.as_tensor(scales, device=q0.device, dtype=q0.dtype).reshape(-1)
    weights = th.ones_like(scales) if weights is None else th.as_tensor(
        weights, device=q0.device, dtype=q0.dtype
    ).reshape(-1)
    if (scales.numel() == 0 or weights.numel() != scales.numel()
            or not bool(th.isfinite(scales).all()) or bool((scales <= 0).any())
            or not bool(th.isfinite(weights).all()) or bool((weights < 0).any())
            or not bool(weights.sum() > 0)):
        raise ValueError("Provide positive finite scales and non-negative kernel weights.")
    weights = weights / weights.sum()
    if max_steps is not None and (max_steps < n_steps or not 0 < lipschitz_max < 1):
        raise ValueError("max_steps must cover n_steps and lipschitz_max must be in (0, 1).")
    if method == "rff":
        if rff_model is None:
            raise ValueError("rff_model=(freq, phase, amp) is required for RFF.")
        rff_model = tuple(th.as_tensor(v, device=q0.device, dtype=q0.dtype) for v in rff_model)
    else:
        rff_model = None
        if use_keops and LazyTensor is None:
            raise ImportError("PyKeOps is required when use_keops=True.")

    velocity, system, _, lipschitz = _lddmm_operators(
        scales, weights, use_keops, chunk_size, rff_model
    )
    rows, b = _lddmm_tableau(method)
    check_grid = return_info or max_steps is not None or inverse

    def shoot(nt):
        q, p, trajectory = q0, p0, []
        dt, bound = 1.0 / nt, 0.0
        for _ in range(nt):
            stages, dq, dp, spatial_bounds = [], [], [], []
            for row in rows:
                qi = q + dt * sum(a * v for a, v in zip(row, dq))
                pi = p + dt * sum(a * v for a, v in zip(row, dp))
                stages.append((qi, pi))
                vi, fi = system(qi, pi)
                dq.append(vi)
                dp.append(fi)
                if check_grid:
                    with th.no_grad():
                        Ji = 1.0 + dt * sum(abs(a) * v for a, v in zip(row, spatial_bounds))
                        spatial_bounds.append(float(lipschitz(qi, pi)) * Ji)
            q = q + dt * sum(w * v for w, v in zip(b, dq))
            p = p + dt * sum(w * v for w, v in zip(b, dp))
            trajectory.append(stages)
            if check_grid:
                bound = max(dt * sum(abs(w) * v for w, v in zip(b, spatial_bounds)), bound)
        return q, p, trajectory, bound

    while True:
        q, p, trajectory, bound = shoot(n_steps)
        if not math.isfinite(bound):
            raise FloatingPointError("Non-finite LDDMM trajectory; reduce the momentum update.")
        if max_steps is None or bound <= lipschitz_max:
            break
        if n_steps == max_steps:
            raise RuntimeError("LDDMM step bound exceeds the configured integration budget.")
        n_steps = min(2 * n_steps, max_steps)

    dt = 1.0 / n_steps

    def increment(z, stages):
        velocities = []
        for row, (qi, pi) in zip(rows, stages):
            zi = z + dt * sum(a * v for a, v in zip(row, velocities))
            velocities.append(velocity(zi, qi, pi))
        return dt * sum(w * v for w, v in zip(b, velocities))

    Z = X
    for stages in reversed(trajectory) if inverse else trajectory:
        if not inverse:
            Z = Z + increment(Z, stages)
            continue
        target, estimate = Z, Z
        limit = inverse_tol * (1.0 + float(target.detach().abs().max())) if target.numel() else inverse_tol
        for _ in range(inverse_max_iter):
            residual = estimate + increment(estimate, stages) - target
            if residual.numel() == 0 or float(residual.detach().abs().max()) <= limit:
                break
            estimate = estimate - residual
        else:
            raise RuntimeError("LDDMM inverse did not converge; refine the integration grid.")
        Z = estimate

    result = (Z, q, p)
    if return_info:
        return result + ({"n_steps": n_steps, "max_step_bound": bound},)
    return result


def _lddmm_update_sigma2(self, Pt1s, P1s, Nps, PXs, Xs, TY, Qp):
    """Gaussian scale update using only original PWR residuals, including zero-mass sets."""
    residuals = th.stack([
        (pt1 * X.square().sum(1)).sum() - 2.0 * (TY * px).sum()
        + (p1 * TY.square().sum(1)).sum()
        for pt1, p1, px, X in zip(Pt1s, P1s, PXs, Xs)
    ]).clamp_min(0)
    mass = th.stack([th.as_tensor(v, device=TY.device, dtype=TY.dtype) for v in Nps])
    D, tiny = TY.shape[1], th.finfo(TY.dtype).tiny
    total = (self.omega * mass).sum()
    common = (self.omega * residuals).sum() / (D * total.clamp_min(tiny))
    common = th.where(total > 0, common, self.sigma2_exp).clamp_min(self.sigma2_min)
    sigma2 = common.expand_as(mass) if self.sigma2_sync else th.where(
        mass > 0, residuals / (D * mass.clamp_min(tiny)), self.sigma2
    ).clamp_min(self.sigma2_min)
    Q = Qp + (self.omega * (residuals / (2.0 * sigma2)
                           + 0.5 * D * mass * sigma2.log())).sum()
    return common, sigma2, Q


@th.no_grad()
def pw_lddmm_maximization(self, Pt1s, P1s, Nps, PXs, Y, Xs):
    """Fixed-P geometric M-step: optimize initial momenta, then update Gaussian scales."""
    xp, ipara, itmat = self.xp, self.transparas, self.tmats
    # Each data set uses the same variance as its E-step, even when scales are asynchronous.
    precision = self.omega / self.sigma2
    P1 = sum(w * p for w, p in zip(precision, P1s))
    PX = sum(w * px for w, px in zip(precision, PXs))
    lam = (self.omega * ipara["alpha"]).sum() * ipara["alpha_mu"] ** self.iteration
    scales = xp.as_tensor(ipara.get("kernel_scales", ipara.get("beta")), device=Y.device, dtype=Y.dtype).reshape(-1)
    weights = ipara.get("kernel_weights")
    weights = xp.ones_like(scales) if weights is None else xp.as_tensor(weights, device=Y.device, dtype=Y.dtype)
    weights = weights / weights.sum()
    method = ipara.get("lddmm_method", "ralston").lower()
    n_steps = int(ipara.get("lddmm_steps", 10))
    max_steps = int(ipara.get("lddmm_max_steps", 4096))
    bound_limit = float(ipara.get("lddmm_lipschitz_max", 0.9))
    full_steps = int(ipara.get("lddmm_inner_steps", 60))
    warm_steps = int(ipara.get("lddmm_warm_steps", 10))
    lr = float(ipara.get("lddmm_lr", 1e-2))
    tol = float(ipara.get("lddmm_tol", 1e-6))
    patience = int(ipara.get("lddmm_patience", 5))
    use_keops = bool(ipara.get("lddmm_use_keops", False))
    chunk_size = int(ipara.get("lddmm_chunk_size", 2048))
    optimizer_name = ipara.get("lddmm_optimizer", "adam").lower()
    attachment = ipara.get("lddmm_attachment")

    # The initial geometry remains fixed across E/M iterations; only p0 is warm-started.
    base = xp.as_tensor(itmat.get("lddmm_base", Y), device=Y.device, dtype=Y.dtype).detach()
    q0 = itmat.get("lddmm_controls0")
    seed = int(ipara.get("lddmm_random_state", ipara.get("random_state", 19491001)))
    if q0 is None:
        q0 = control_points(base, int(ipara.get("lddmm_n_controls", 1000)),
                            ipara.get("lddmm_control_method", ipara.get("control_method", "voxel")),
                            random_state=seed)
    q0 = xp.as_tensor(q0, device=Y.device, dtype=Y.dtype).detach()
    initial = itmat.get("lddmm_momentum0")
    warm_start = initial is not None
    initial = xp.zeros_like(q0) if initial is None else xp.as_tensor(initial, device=Y.device, dtype=Y.dtype).detach()
    n_iter = min(full_steps, warm_steps) if warm_start else full_steps
    n_steps = max(n_steps, int(itmat.get("lddmm_steps", n_steps)))

    rff_model, rff_origin = None, None
    if method == "rff":
        sampled = sample_rff_model(base, scales, weights,
                                   int(ipara.get("lddmm_rff_features", 512)), seed, xp)
        freq, rff_origin = sampled["frequencies"], sampled["origin"]
        rff_model = (freq, sampled["phases"] - rff_origin @ freq.T, sampled["amplitudes"])
    _, _, energy, _ = _lddmm_operators(scales, weights, use_keops, chunk_size, rff_model)
    B = PX - P1[:, None] * base
    grid_history = []

    while True:
        p0 = xp.nn.Parameter(initial.clone())
        if optimizer_name == "adam":
            optimizer = xp.optim.Adam([p0], lr=lr)
        elif optimizer_name == "lbfgs":
            # One quasi-Newton update per outer inner-loop iteration; closures may be repeated.
            optimizer = xp.optim.LBFGS([p0], lr=lr, max_iter=1, max_eval=20,
                                      history_size=int(ipara.get("lddmm_lbfgs_history", 10)),
                                      line_search_fn="strong_wolfe")
        else:
            raise ValueError("lddmm_optimizer must be 'adam' or 'lbfgs'.")

        def evaluate():
            TY, q1, p1 = lddmm_flow(base, q0, p0, scales, weights, n_steps=n_steps,
                                    use_keops=use_keops, chunk_size=chunk_size,
                                    method=method, rff_model=rff_model)
            U = TY - base
            data = 0.5 * (P1[:, None] * U.square()).sum() - (B * U).sum()
            extra = TY.new_zeros(()) if attachment is None else attachment(TY)
            raw = energy(q0, p0)
            return data + lam * raw + extra, (TY, q1, p1, data, raw, extra)

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss, _ = evaluate()
            loss.backward()
            return loss

        history, previous, stale, best_loss = [], None, 0, float("inf")
        with xp.enable_grad():
            for iteration in range(n_iter + 1):
                loss, _ = evaluate()
                value = float(loss.detach())
                if not math.isfinite(value):
                    raise FloatingPointError("Non-finite LDDMM objective.")
                history.append(value)
                if value < best_loss:
                    best_loss, best_p0 = value, p0.detach().clone()
                stale = stale + 1 if previous is not None and abs(previous - value) <= tol * (1.0 + abs(previous)) else 0
                if iteration == n_iter or stale >= patience:
                    break
                if optimizer_name == "adam":
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                else:
                    optimizer.step(closure)
                previous = value

        # Check the chosen iterate, then freeze a refined grid and restart the optimizer if needed.
        best_TY, best_q1, best_p1, info = lddmm_flow(
            base, q0, best_p0, scales, weights, n_steps=n_steps,
            use_keops=use_keops, chunk_size=chunk_size, method=method,
            rff_model=rff_model, return_info=True,
        )
        grid_history.append({"n_steps": n_steps, "max_step_bound": info["max_step_bound"]})
        if info["max_step_bound"] <= bound_limit:
            break
        if n_steps >= max_steps:
            raise RuntimeError("LDDMM step bound exceeds the configured integration budget.")
        n_steps, initial = min(2 * n_steps, max_steps), best_p0

    U = best_TY - base
    best_data = 0.5 * (P1[:, None] * U.square()).sum() - (B * U).sum()
    best_raw = energy(q0, best_p0)
    best_extra = best_TY.new_zeros(()) if attachment is None else attachment(best_TY)
    Qp = lam * best_raw + best_extra
    sigma2_exp, sigma2, Q = _lddmm_update_sigma2(self, Pt1s, P1s, Nps, PXs, Xs, best_TY, Qp)

    itmat.update({
        "lddmm_base": base, "lddmm_controls0": q0, "lddmm_momentum0": best_p0,
        "lddmm_controls1": best_q1, "lddmm_momentum1": best_p1,
        "lddmm_kernel_scales": scales, "lddmm_kernel_weights": weights,
        "lddmm_steps": n_steps, "lddmm_max_steps": max_steps,
        "lddmm_lipschitz_max": bound_limit, "lddmm_method": method,
        "lddmm_optimizer": optimizer_name, "lddmm_random_state": seed,
        "lddmm_use_keops": use_keops, "lddmm_chunk_size": chunk_size,
        "lddmm_raw_action": best_raw, "lddmm_path_energy": lam * best_raw,
        "lddmm_attachment_loss": best_extra, "lddmm_data_change": best_data,
        "lddmm_inner_objective": best_data + Qp, "lddmm_objective": best_data + Qp,
        "lddmm_loss_history": history, "lddmm_grid_history": grid_history,
        "lddmm_inner_steps": iteration, "lddmm_warm_start": warm_start,
        "lddmm_model": "hamiltonian_rff" if method == "rff" else "hamiltonian", "W": None,
    })
    if rff_model is not None:
        itmat.update({"lddmm_rff_freq": rff_model[0], "lddmm_rff_phase": rff_model[1],
                      "lddmm_rff_amp": rff_model[2], "lddmm_rff_origin": rff_origin,
                      "lddmm_rff_features": rff_model[0].shape[0]})
    self.TY, self.sigma2, self.sigma2_exp, self.Q = best_TY, sigma2, sigma2_exp, Q


def kernel_gaussian(x: Tensor, c: Tensor, scales=(1.0,), weights=None, use_keops=False):
    """
    Isotropic multi-scale Gaussian kernel K(x, c).

    Scales are positive; explicit weights are non-negative and are not
    normalized here. A zero weight disables that scale.
    """
    scales = th.as_tensor( scales, device=x.device, dtype=x.dtype,).reshape(-1)

    if scales.numel() == 0:
        raise ValueError("At least one kernel scale is required.")

    if weights is None:
        weights = th.ones_like(scales) / scales.numel()
    else:
        weights = th.as_tensor( weights, device=x.device, dtype=x.dtype,).reshape(-1)

    if weights.numel() != scales.numel():
        raise ValueError("Provide one weight per kernel scale.")

    if use_keops:
        x_i = LazyTensor(x[:, None, :])  # (M, 1, D)
        c_j = LazyTensor(c[None, :, :])  # (1, N, D)

        d2 = ((x_i - c_j) ** 2).sum(-1)
        K = weights[0] * ( -d2 / (2.0 * scales[0].square())).exp()
        for scale, weight in zip(scales[1:], weights[1:]):
            K = K + weight * ( -d2 / (2.0 * scale.square())).exp()
    else:
        d2 = th.cdist(x, c).square()
        K = weights[0] * th.exp( -d2 / (2.0 * scales[0].square()))
        for scale, weight in zip(scales[1:], weights[1:]):
            K = K + weight * th.exp( -d2 / (2.0 * scale.square()))
    return K

def control_points( Y: Tensor, n_controls=8000, method="voxel", grid=None, n_iter=10,
                    tol=1e-4, chunk_size=65536, random_state=200504,) -> Tensor:
    """Select a fixed number of unique control points from ``Y``.

    Parameters
    ----------
    Y : (N, D) Tensor
        Input point cloud.
    n_controls : int
        Maximum number of returned control points.
    method : {'random', 'voxel', 'fps', 'kmeans'}
        Control-point selection method.
    grid : int or sequence of int, optional
        Voxel-grid resolution. When omitted, it is inferred from
        ``n_controls`` and the effective point-cloud dimension.
    n_iter : int
        Maximum number of KMeans iterations.
    tol : float
        KMeans convergence tolerance in normalized coordinates.
    chunk_size : int
        Maximum input chunk used by KMeans distance calculations.
    random_state : int
        Seed used by random sampling and KMeans initialization.

    Returns
    -------
    C : (K, D) Tensor
        Unique control points, where
        ``K = min(n_controls, number of unique input points)``.
    """
    if Y.ndim != 2 or Y.shape[0] == 0:
        raise ValueError("Y must have shape (N, D) with N > 0.")
    if n_controls is None or int(n_controls) <= 0:
        raise ValueError("n_controls must be a positive integer.")
    # if int(n_iter) < 1 or float(tol) < 0 or int(chunk_size) < 1:
    #     raise ValueError("n_iter, tol or chunk_size is invalid.")
    # if not bool(th.isfinite(Y).all()):
    #     raise ValueError("Y must contain only finite coordinates.")

    method = str(method).lower()
    aliases = {"rand": "random", "k-means": "kmeans"}
    method = aliases.get(method, method)
    if method not in {"random", "voxel", "fps", "kmeans"}:
        raise ValueError(
            "method must be 'random', 'voxel', 'fps', or 'kmeans'."
        )

    # A unique candidate pool guarantees unique random/FPS samples and makes
    # it possible to refill duplicate voxel/KMeans centers exactly.
    unique = th.unique(Y, dim=0, sorted=False)
    if unique.shape[0] != Y.shape[0]:
        Y = unique
    N, D = Y.shape
    K = min(int(n_controls), N)
    if K == N:
        return Y

    wdtype = Y.dtype
    if not Y.is_floating_point() or Y.dtype in (th.float16, th.bfloat16):
        wdtype = th.float32
    Yw = Y.to(wdtype)

    ymin = Yw.amin(0)
    raw_span = Yw.amax(0) - ymin
    active_dim = raw_span > 0
    span = th.where(active_dim, raw_span, th.ones_like(raw_span))
    Yn = (Yw - ymin) / span

    generator = th.Generator(device=Y.device)
    generator.manual_seed(int(random_state))

    def fps_idx(P, k):
        """Deterministic farthest-point indices."""
        k = min(int(k), P.shape[0])
        idx = th.empty(k, dtype=th.long, device=P.device)
        dist = th.full(
            (P.shape[0],), th.inf, dtype=P.dtype, device=P.device
        )

        cur = ((P - P.mean(0)) ** 2).sum(1).argmin()
        for i in range(k):
            idx[i] = cur
            d2 = ((P - P[cur]) ** 2).sum(1)
            dist = th.minimum(dist, d2)
            if i + 1 < k:
                cur = dist.argmax()
        return idx

    def min_d2(P, C):
        """Minimum squared distance to centers without a large N x K matrix."""
        max_entries = 4_000_000
        batch_size = min(
            int(chunk_size),
            max(1, max_entries // max(C.shape[0], 1)),
        )
        out = th.empty(P.shape[0], dtype=P.dtype, device=P.device)
        for start in range(0, P.shape[0], batch_size):
            stop = min(start + batch_size, P.shape[0])
            out[start:stop] = th.cdist(P[start:stop], C).square().amin(1)
        return out

    def resize_centers(C, k):
        """Reduce or refill centers to exactly k unique coordinates."""
        C = th.unique(C, dim=0, sorted=False)
        if C.shape[0] == k:
            return C

        Cn = (C.to(wdtype) - ymin) / span
        if C.shape[0] > k:
            return C[fps_idx(Cn, k)]

        # Y is unique and k <= len(Y). Taking the k farthest candidates
        # guarantees enough points remain after removing center coincidences.
        d2 = min_d2(Yn, Cn)
        add = Y[th.topk(d2, k=k, largest=True).indices]
        C = th.unique(th.cat([C, add], dim=0), dim=0, sorted=False)
        if C.shape[0] < k:
            raise RuntimeError("Unable to construct the requested unique centers.")
        if C.shape[0] > k:
            Cn = (C.to(wdtype) - ymin) / span
            C = C[fps_idx(Cn, k)]
        return C

    if method == "random":
        idx = th.randperm(N, device=Y.device, generator=generator)[:K]
        C = Y[idx]

    elif method == "fps":
        C = Y[fps_idx(Yn, K)]

    elif method == "voxel":
        if grid is None:
            effective_dim = max(int(active_dim.sum().item()), 1)
            g0 = max(1, math.ceil(K ** (1.0 / effective_dim)))
            g = th.ones(D, dtype=th.long, device=Y.device)
            g[active_dim] = g0
        else:
            gv = th.as_tensor(grid, device=Y.device).reshape(-1)
            if gv.numel() == 1:
                gv = gv.expand(D)
            if gv.numel() != D or bool(th.any(gv <= 0)):
                raise ValueError("grid must be positive and scalar or length D.")
            if gv.is_floating_point() and bool(th.any(gv != gv.round())):
                raise ValueError("grid values must be integers.")
            g = gv.to(th.long)

        key = th.floor(Yn * g.to(Yn)).to(th.long)
        key = th.minimum(key, g - 1)
        _, inverse = th.unique(
            key, dim=0, sorted=False, return_inverse=True
        )
        n_voxels = int(inverse.max().item()) + 1
        count = th.bincount(inverse, minlength=n_voxels).to(Yw)[:, None]
        Cw = th.zeros(n_voxels, D, dtype=wdtype, device=Y.device)
        Cw.index_add_(0, inverse, Yw)
        C = (Cw / count).to(Y.dtype)
        C = resize_centers(C, K)

    else:  # kmeans
        idx = th.randperm(N, device=Y.device, generator=generator)[:K]
        Cn = Yn[idx].clone()

        max_entries = 4_000_000
        batch_size = min(int(chunk_size), max(1, max_entries // K))
        for _ in range(int(n_iter)):
            sums = th.zeros_like(Cn)
            count = th.zeros(K, dtype=Yn.dtype, device=Y.device)

            for start in range(0, N, batch_size):
                stop = min(start + batch_size, N)
                P = Yn[start:stop]
                label = th.cdist(P, Cn).argmin(1)
                sums.index_add_(0, label, P)
                count.index_add_(
                    0,
                    label,
                    th.ones(stop - start, dtype=Yn.dtype, device=Y.device),
                )

            nonempty = count > 0
            new = Cn.clone()
            new[nonempty] = sums[nonempty] / count[nonempty, None]

            empty = ~nonempty
            if bool(empty.any()):
                d2 = min_d2(Yn, new[nonempty])
                n_empty = int(empty.sum().item())
                new[empty] = Yn[th.topk(d2, k=n_empty).indices]

            shift = (new - Cn).square().sum(1).sqrt().amax()
            Cn = new
            if float(shift) <= float(tol):
                break

        C = (Cn * span + ymin).to(Y.dtype)
        C = resize_centers(C, K)

    C = th.unique(C, dim=0, sorted=False)
    return C