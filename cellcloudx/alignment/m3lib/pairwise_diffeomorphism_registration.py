"""Pairwise diffeomorphic registration models for PWR.

This module adapts the diffeomorphic M-steps in
``pairwise_operation_diffeo`` to the public ``pwEMRegistration`` lifecycle.
The short transformer codes are:

``K``
    Compositional reproducing-kernel Hilbert space (RKHS) deformation.
``U``
    Stationary velocity field (SVF).
``V``
    Time-varying/piecewise-stationary velocity field (SVVF/TVVF).
``M``
    Landmark LDDMM shooting.

All kernels and velocity fields are fitted in PWR's normalized coordinate
system.  The saved transformer keeps that normalization so that arbitrary
points can subsequently be transformed in their original coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch as th

from .pairwise_emregistration import pwEMRegistration
from . import pairwise_operation_diffeo as diffeo_ops


class pwDiffeoRegistration(pwEMRegistration):
    """PWR wrapper for RKHS, SVF, SVVF and LDDMM registration.

    Parameters
    ----------
    transformer : {'K', 'U', 'V', 'M'}, default='K'
        Diffeomorphic transformation family.  Long aliases such as ``rkhs``,
        ``svf``, ``svvf``/``tvvf`` and ``lddmm`` are accepted.
    diffeo_method : {'K', 'U', 'V', 'M'}, optional
        Select the method for a generic ``diffeo``/``diffeomorphism`` alias
        (default K). For a specific transformer it must select the same family.
    transparas : mapping, optional
        Final parameter overrides for the selected diffeomorphic M-step.
        This mirrors the parameter override convention used by the other PWR
        registration classes.

    Notes
    -----
    ``kernel_scales`` are Gaussian standard deviations expressed in PWR's
    normalized coordinates (the compatibility name ``beta`` has the same
    meaning here, unlike the legacy D transform's squared-bandwidth form).
    Set ``normal=False`` when the supplied scales are already in data units.
    K bounds each residual stage by ``rkhs_lipschitz_max < 1``. U/V
    refine integration steps when necessary to bound the discrete residual
    maps. The configured step caps raise an error if that bound cannot be
    met within the budget. These conservative bounds can increase run time
    and can constrain a K update relative to an unconstrained fit.
    M refines the chosen shooting grid until the spatial RK residuals satisfy
    ``lddmm_lipschitz_max``; its inverse solves those same discrete steps.
    Continuous-flow theory does not make finite-step integration exact.
    ``lddmm_optimizer`` selects Adam or L-BFGS. Optional
    ``lddmm_attachment(endpoint)`` adds a differentiable scalar endpoint loss
    in normalized coordinates (for example fixed landmark/local constraints);
    it does not contribute to the observation-variance update.
    ``transform_point(..., inverse=True)`` numerically inverts the fitted
    discrete maps, with residual checks; it is not an exact continuous ODE
    solution. U/V/M accept ``inverse_tol`` and ``inverse_max_iter``.
    """

    _TRANSFORMERS = frozenset(("K", "U", "V", "M"))
    _ALIASES = {
        "k": "K", "rkhs": "K", "kernel": "K",
        "u": "U", "svf": "U", "stationary": "U",
        "v": "V", "svvf": "V", "tvvf": "V",
        "time-varying": "V", "time_varying": "V",
        "piecewise-stationary": "V", "piecewise_stationary": "V",
        "m": "M", "lddmm": "M", "shooting": "M",
    }
    _GENERIC_ALIASES = frozenset(("diffeo", "diffeomorphism", "diffeomorphic"))
    _METHOD_NAMES = {
        "K": "RKHS",
        "U": "SVF",
        "V": "SVVF",
        "M": "LDDMM",
    }
    def __init__(
        self,
        *args: Any,
        transformer: str = "K",
        diffeo_method: str | None = None,
        kernel_scales: Any = None,
        kernel_weights: Any = None,
        beta: Any = None,
        alpha: Any = None,
        alpha_decayto: float = 1.0,
        delta: float = 0.0,
        n_controls: int = 1000,
        control_method: str = "voxel",
        random_state: int = 19491001,
        # RKHS
        rkhs_model: str = "rff",
        n_stages: int = 3,
        rff_features: int = 256,
        kernel_jitter: float = 1e-6,
        rkhs_solver: str = "pcg",
        rkhs_backend: str | None = None,
        rkhs_inner_steps: int = 50,
        rkhs_inner_tol: float = 1e-5,
        rkhs_direct_max_points: int = 3000,
        rkhs_lipschitz_max: float = 0.9,
        # SVF
        svf_steps: int = 8,
        svf_max_steps: int = 1024,
        svf_inner_steps: int = 50,
        svf_lr: float = 1e-2,
        svf_tol: float = 1e-6,
        svf_patience: int = 5,
        svf_use_keops: bool | None = None,
        # SVVF / TVVF
        tvvf_n_time: int = 4,
        tvvf_substeps: int = 1,
        tvvf_max_substeps: int = 256,
        tvvf_inner_steps: int = 60,
        tvvf_lr: float = 1e-2,
        tvvf_tol: float = 1e-6,
        tvvf_patience: int = 5,
        tvvf_use_keops: bool | None = None,
        # LDDMM
        lddmm_method: str = "ralston",
        lddmm_optimizer: str = "adam",
        lddmm_lbfgs_history: int = 10,
        lddmm_attachment: Any = None,
        lddmm_steps: int = 10,
        lddmm_max_steps: int = 4096,
        lddmm_lipschitz_max: float = 0.9,
        lddmm_inner_steps: int = 60,
        lddmm_warm_steps: int = 10,
        lddmm_lr: float = 1e-2,
        lddmm_tol: float = 1e-6,
        lddmm_patience: int = 5,
        lddmm_n_controls: int = 1000,
        lddmm_chunk_size: int = 2048,
        lddmm_rff_features: int = 512,
        lddmm_use_keops: bool | None = None,
        transparas: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        code = self._resolve_transformer(transformer, diffeo_method)
        super().__init__(*args, **kwargs)

        self.transformer = code
        self.diffeo_method = self._METHOD_NAMES[code]
        self.reg_core = f"diffeomorphic-{self.diffeo_method}"
        self.tol = 1e-9 if self.tol is None else self.tol
        self.maxiter = 50 if self.maxiter is None else self.maxiter
        if not isinstance(self.maxiter, int) or self.maxiter < 1:
            raise ValueError("maxiter must be a positive integer.")
        self.normal = "isoscale" if self.normal is None else self.normal

        self.tau2_decayto = self.scalar2vetor(
            0.15 if self.tau2_decayto is None else self.tau2_decayto,
            L=self.L,
        )
        self.tau2_decaystop = self.scalar2vetor(
            0.7 if self.tau2_decaystop is None else self.tau2_decaystop,
            L=self.L,
        )

        self.normal_XY()
        self.normal_features()

        if kernel_scales is None:
            kernel_scales = beta
        if kernel_scales is None:
            kernel_scales = (0.5, 1.0, 2.0)
        if alpha is None:
            alpha = 1.0

        # Dense evaluation is the portable default.  Importable PyKeOps does
        # not guarantee that a working compiler/toolchain is available.
        keops_default = False
        parameters = {
            "kernel_scales": kernel_scales,
            "kernel_weights": kernel_weights,
            "beta": kernel_scales,
            "alpha": alpha,
            "alpha_decayto": alpha_decayto,
            "delta": delta,
            "n_controls": n_controls,
            "control_method": control_method,
            "random_state": random_state,
            # RKHS
            "rkhs_model": rkhs_model,
            "n_stages": n_stages,
            "rff_features": rff_features,
            "n_features": rff_features,
            "kernel_jitter": kernel_jitter,
            "rkhs_solver": rkhs_solver,
            "rkhs_inner_steps": rkhs_inner_steps,
            "rkhs_inner_tol": rkhs_inner_tol,
            "rkhs_direct_max_points": rkhs_direct_max_points,
            "rkhs_lipschitz_max": rkhs_lipschitz_max,
            # The RFF implementation uses these historical key names.
            "n_iter": rkhs_inner_steps,
            "tol": rkhs_inner_tol,
            # SVF
            "svf_steps": svf_steps,
            "svf_max_steps": svf_max_steps,
            "svf_inner_steps": svf_inner_steps,
            "svf_lr": svf_lr,
            "svf_tol": svf_tol,
            "svf_patience": svf_patience,
            "svf_use_keops": (
                keops_default if svf_use_keops is None else svf_use_keops
            ),
            # SVVF / TVVF
            "tvvf_n_time": tvvf_n_time,
            "tvvf_substeps": tvvf_substeps,
            "tvvf_max_substeps": tvvf_max_substeps,
            "tvvf_inner_steps": tvvf_inner_steps,
            "tvvf_lr": tvvf_lr,
            "tvvf_tol": tvvf_tol,
            "tvvf_patience": tvvf_patience,
            "tvvf_use_keops": (
                keops_default if tvvf_use_keops is None else tvvf_use_keops
            ),
            # LDDMM
            "lddmm_method": lddmm_method,
            "lddmm_optimizer": lddmm_optimizer,
            "lddmm_lbfgs_history": lddmm_lbfgs_history,
            "lddmm_attachment": lddmm_attachment,
            "lddmm_steps": lddmm_steps,
            "lddmm_max_steps": lddmm_max_steps,
            "lddmm_lipschitz_max": lddmm_lipschitz_max,
            "lddmm_inner_steps": lddmm_inner_steps,
            "lddmm_warm_steps": lddmm_warm_steps,
            "lddmm_lr": lddmm_lr,
            "lddmm_tol": lddmm_tol,
            "lddmm_patience": lddmm_patience,
            "lddmm_n_controls": lddmm_n_controls,
            "lddmm_chunk_size": lddmm_chunk_size,
            "lddmm_rff_features": lddmm_rff_features,
            "lddmm_use_keops": (
                keops_default if lddmm_use_keops is None else lddmm_use_keops
            ),
        }
        if rkhs_backend is not None:
            parameters["rkhs_backend"] = rkhs_backend
        if transparas is not None:
            if not isinstance(transparas, Mapping):
                raise TypeError("transparas must be a mapping or None.")
            overrides = dict(transparas)
            parameters.update(overrides)
            # Keep public names and historical backend keys synchronized when
            # users provide overrides through the standard PWR mapping.
            alias_pairs = (
                ("beta", "kernel_scales"),
                ("rkhs_inner_steps", "n_iter"),
                ("rkhs_inner_tol", "tol"),
                ("n_features", "rff_features"),
                ("K", "n_stages"),
            )
            for legacy, public in alias_pairs:
                if legacy in overrides and public not in overrides:
                    parameters[public] = overrides[legacy]
                elif public in overrides and legacy not in overrides:
                    parameters[legacy] = overrides[public]

        self.init_transparas(parameters)
        self.init_transformer()

    @classmethod
    def _resolve_transformer(
        cls, transformer: str, diffeo_method: str | None = None
    ) -> str:
        if not isinstance(transformer, str):
            raise TypeError("transformer must be a string.")
        name = transformer.strip().lower()
        generic = name in cls._GENERIC_ALIASES
        code = "K" if generic else cls._ALIASES.get(name)
        if code is None:
            raise ValueError(
                "transformer must be K/rkhs, U/svf, V/svvf/tvvf, M/lddmm "
                "or a generic diffeomorphism alias."
            )
        if diffeo_method is not None:
            if not isinstance(diffeo_method, str):
                raise TypeError("diffeo_method must be a string or None.")
            method = cls._ALIASES.get(diffeo_method.strip().lower())
            if method is None:
                raise ValueError(
                    "diffeo_method must select K/rkhs, U/svf, V/tvvf or M/lddmm."
                )
            if not generic and method != code:
                raise ValueError(
                    "transformer and diffeo_method must select the same family."
                )
            code = method
        return code

    def init_transparas(self, parameters: Mapping[str, Any]) -> None:
        """Validate and normalize parameters consumed by diffeomorphic M-steps."""
        ipara = dict(parameters)

        scales = self.to_tensor(
            ipara.get("kernel_scales", ipara.get("beta")),
            dtype=self.floatx,
            device=self.device,
        ).reshape(-1)
        if scales.numel() == 0 or not bool(th.isfinite(scales).all()):
            raise ValueError("kernel_scales must be a non-empty finite sequence.")
        if bool(th.any(scales <= 0)):
            raise ValueError("kernel_scales must be positive.")
        ipara["kernel_scales"] = scales
        ipara["beta"] = scales

        weights = ipara.get("kernel_weights")
        if weights is not None:
            weights = self.to_tensor(
                weights, dtype=self.floatx, device=self.device
            ).reshape(-1)
            if weights.numel() != scales.numel():
                raise ValueError(
                    "kernel_weights must have the same length as kernel_scales."
                )
            if (not bool(th.isfinite(weights).all()) or bool(th.any(weights < 0))
                    or not bool(th.isfinite(weights.sum())) or not bool(weights.sum() > 0)):
                raise ValueError("kernel_weights must be finite, non-negative and have positive sum.")
            ipara["kernel_weights"] = weights / weights.sum()

        alpha = ipara.get("alpha", 1.0)
        if th.is_tensor(alpha) and alpha.ndim == 0:
            alpha = alpha.item()
        alpha = self.to_tensor(
            self.scalar2vetor(alpha, L=self.L),
            dtype=self.floatx,
            device=self.device,
        )
        if not bool(th.isfinite(alpha).all()) or bool(th.any(alpha <= 0)):
            raise ValueError("alpha must contain finite positive values.")
        ipara["alpha"] = alpha

        decay_to = float(ipara.get("alpha_decayto", 1.0))
        if not np.isfinite(decay_to) or decay_to <= 0:
            raise ValueError("alpha_decayto must be finite and positive.")
        ipara["alpha_decayto"] = decay_to
        ipara["alpha_mu"] = decay_to ** (1.0 / max(self.maxiter - 1, 1))

        delta = float(ipara.get("delta", 0.0))
        if not np.isfinite(delta) or delta < 0:
            raise ValueError("delta must be finite and non-negative.")
        ipara["delta"] = delta

        lipschitz_max = float(ipara.get("rkhs_lipschitz_max", 0.9))
        if not np.isfinite(lipschitz_max) or not 0 < lipschitz_max < 1:
            raise ValueError("rkhs_lipschitz_max must be finite and in (0, 1).")
        ipara["rkhs_lipschitz_max"] = lipschitz_max

        rkhs_aliases = {
            "rff": "rff",
            "random_fourier": "rff",
            "random-fourier": "rff",
            "control": "control_white",
            "white": "control_white",
            "control_white": "control_white",
            "full": "full_nonwhite",
            "nonwhite": "full_nonwhite",
            "full_nonwhite": "full_nonwhite",
        }
        rkhs_model = str(ipara.get("rkhs_model", "rff"))
        if rkhs_model not in rkhs_aliases:
            raise ValueError(
                "rkhs_model must be 'rff', 'control_white' or 'full_nonwhite'."
            )
        ipara["rkhs_model"] = rkhs_aliases[rkhs_model]

        rkhs_solver = str(ipara.get("rkhs_solver", "pcg"))
        valid_solvers = {
            "auto", "direct", "chol", "cholesky", "cg", "pcg",
            "keops", "keops_cg", "keops_pcg",
        }
        if rkhs_solver not in valid_solvers:
            raise ValueError(
                "rkhs_solver must be auto, direct, cg, pcg or a KeOps alias."
            )
        ipara["rkhs_solver"] = rkhs_solver

        if "rkhs_backend" in ipara:
            rkhs_backend = str(ipara["rkhs_backend"])
            if rkhs_backend not in {"dense", "keops"}:
                raise ValueError("rkhs_backend must be 'dense' or 'keops'.")
            ipara["rkhs_backend"] = rkhs_backend
        else:
            rkhs_backend = "dense"

        for key in (
            "n_stages", "n_controls", "rff_features", "n_features",
            "rkhs_inner_steps", "n_iter", "rkhs_direct_max_points",
            "svf_steps", "svf_max_steps", "svf_inner_steps", "svf_patience",
            "tvvf_n_time", "tvvf_substeps", "tvvf_max_substeps",
            "tvvf_inner_steps", "tvvf_patience", "lddmm_steps",
            "lddmm_max_steps", "lddmm_inner_steps", "lddmm_patience",
            "lddmm_lbfgs_history",
            "lddmm_warm_steps", "lddmm_n_controls", "lddmm_chunk_size", "lddmm_rff_features",
        ):
            value = ipara[key]
            if isinstance(value, bool) or not np.isfinite(float(value)) or int(value) != float(value) or int(value) < 1:
                raise ValueError(f"{key} must be a positive integer.")
            ipara[key] = int(value)
        for key in ("rkhs_inner_tol", "tol", "svf_lr", "tvvf_lr", "lddmm_lr"):
            value = float(ipara[key])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive.")
            ipara[key] = value
        for key in ("svf_tol", "tvvf_tol", "lddmm_tol"):
            value = float(ipara[key])
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and non-negative.")
            ipara[key] = value
        jitter = float(ipara["kernel_jitter"])
        if not np.isfinite(jitter) or jitter < 0:
            raise ValueError("kernel_jitter must be finite and non-negative.")
        ipara["kernel_jitter"] = jitter
        lddmm_bound = float(ipara["lddmm_lipschitz_max"])
        if not np.isfinite(lddmm_bound) or not 0 < lddmm_bound < 1:
            raise ValueError("lddmm_lipschitz_max must be finite and in (0, 1).")
        ipara["lddmm_lipschitz_max"] = lddmm_bound
        for requested, cap in (("svf_steps", "svf_max_steps"), ("tvvf_substeps", "tvvf_max_substeps"), ("lddmm_steps", "lddmm_max_steps")):
            if ipara[cap] < ipara[requested]:
                raise ValueError(f"{cap} must be at least {requested}.")

        method = str(ipara.get("lddmm_method", "ralston"))
        if method not in {"ralston", "rk4", "rff"}:
            raise ValueError("lddmm_method must be 'ralston', 'rk4' or 'rff'.")
        ipara["lddmm_method"] = method

        optimizer = str(ipara.get("lddmm_optimizer", "adam")).lower()
        if optimizer not in {"adam", "lbfgs"}:
            raise ValueError("lddmm_optimizer must be 'adam' or 'lbfgs'.")
        ipara["lddmm_optimizer"] = optimizer
        attachment = ipara.get("lddmm_attachment")
        if attachment is not None and not callable(attachment):
            raise TypeError("lddmm_attachment must be callable or None.")

        for key in ("svf_use_keops", "tvvf_use_keops", "lddmm_use_keops"):
            if bool(ipara.get(key, False)) and diffeo_ops.LazyTensor is None:
                raise ImportError(
                    f"{key}=True requires a working PyKeOps installation."
                )
        if (
            rkhs_backend == "keops" or rkhs_solver.startswith("keops")
            or bool(ipara.get("rkhs_use_keops", False))
        ) and diffeo_ops.LazyTensor is None:
            raise ImportError(
                "The requested RKHS backend requires a working PyKeOps installation."
            )

        self.transparas = ipara

    def init_transformer(self) -> None:
        self.tmats = {
            "transformer": self.transformer,
            "diffeo_method": self.diffeo_method,
        }

    def init_sample(self) -> None:
        """Initialize sampling safely for short diffeomorphic smoke runs."""
        for value in self.sample:
            if value:
                if not 0 < value <= 1:
                    raise ValueError("sample must be in (0, 1].")

        stop = 0.75 if self.sample_stop is None else self.sample_stop
        stop = int(stop) if stop > 1 else int(self.maxiter * stop)
        self.sample_stop = max(stop, 1)
        if self.sample_stop == 1:
            self.sample_grow = 1.0
        else:
            self.sample_grow = self.sample_growto ** (
                -1.0 / float(self.sample_stop - 1)
            )

    def maximization(
        self, *args: Any, transformer: str | None = None, **kwargs: Any
    ) -> Any:
        """Dispatch the PWR M-step to the selected diffeomorphic solver."""
        code = self.transformer
        if transformer is not None and transformer != code:
            code = self._resolve_transformer(transformer)
        if code != self.transformer:
            raise ValueError("The transformer cannot change during registration.")

        if code == "K":
            model = self.transparas["rkhs_model"]
            if model == "rff":
                func = diffeo_ops.pw_compositional_rkhs_rff_maximization
            elif model == "control_white":
                func = diffeo_ops.pw_compositional_rkhs_white_maximization
            elif model == "full_nonwhite":
                func = diffeo_ops.pw_compositional_rkhs_nonwhite_maximization
            else:  # guarded by init_transparas
                raise RuntimeError(f"Unknown RKHS model: {model!r}.")
        elif code == "U":
            func = diffeo_ops.pw_svf_maximization
        elif code == "V":
            func = diffeo_ops.pw_tvvf_maximization
        elif code == "M":
            func = diffeo_ops.pw_lddmm_maximization
        else:  # guarded by _resolve_transformer
            raise RuntimeError(f"Unknown diffeomorphic transformer: {code!r}.")
        return func(self, *args, **kwargs)

    @staticmethod
    def _detach_tree(value: Any, dtype: th.dtype) -> Any:
        if th.is_tensor(value):
            target_dtype = dtype if value.is_floating_point() else value.dtype
            return value.detach().to(device="cpu", dtype=target_dtype)
        if isinstance(value, dict):
            return {
                key: pwDiffeoRegistration._detach_tree(item, dtype)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [pwDiffeoRegistration._detach_tree(item, dtype) for item in value]
        if isinstance(value, tuple):
            return tuple(
                pwDiffeoRegistration._detach_tree(item, dtype) for item in value
            )
        return value

    @th.no_grad()
    def update_normalize(self) -> None:
        """Move the fitted flow to CPU and attach raw-coordinate metadata."""
        dtype = self.floatxx
        Sx = self.Sx.detach().to(device="cpu", dtype=dtype)
        Sy = self.Sy.detach().to(device="cpu", dtype=dtype)
        Tx = self.Mx.detach().to(device="cpu", dtype=dtype)
        Ty = self.My.detach().to(device="cpu", dtype=dtype)

        tmats = self._detach_tree(self.tmats, dtype)
        tmats.update(
            {
                "transformer": self.transformer,
                "diffeo_method": self.diffeo_method,
                "Ym": Ty,
                "Ys": Sy,
                "Xm": Tx,
                "Xs": Sx,
                "normalization": self.normal,
            }
        )
        self.tmats = tmats
        self.TY = self.TY.detach().to(device="cpu", dtype=dtype) * Sx + Tx

    @staticmethod
    def _first_tensor(value: Any) -> th.Tensor | None:
        if th.is_tensor(value):
            return value
        if isinstance(value, dict):
            for item in value.values():
                found = pwDiffeoRegistration._first_tensor(item)
                if found is not None:
                    return found
        if isinstance(value, (list, tuple)):
            for item in value:
                found = pwDiffeoRegistration._first_tensor(item)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _restore_type(
        value: th.Tensor,
        original: Any,
        original_device: th.device | None,
        original_dtype: th.dtype | None,
    ) -> Any:
        if th.is_tensor(original):
            if original_dtype is not None:
                value = value.to(device=original_device, dtype=original_dtype)
            else:
                value = value.to(device=original_device)
            return value
        return value.detach().cpu().numpy()

    def _flow_input(self, points: Any) -> tuple[th.Tensor, th.device | None, th.dtype | None]:
        original_device = points.device if th.is_tensor(points) else None
        original_dtype = (
            points.dtype
            if th.is_tensor(points) and points.is_floating_point()
            else None
        )
        ref = self._first_tensor(self.tmats)
        if ref is None:
            device, dtype = th.device("cpu"), self.floatxx
        else:
            device = ref.device
            dtype = ref.dtype if ref.is_floating_point() else self.floatxx
        value = th.as_tensor(points, device=device, dtype=dtype).clone()
        if value.ndim != 2 or value.shape[1] != self.D:
            raise ValueError(f"points must have shape (N, {self.D}).")
        if not bool(th.isfinite(value).all()):
            raise ValueError("points must contain only finite coordinates.")
        return value, original_device, original_dtype

    def transform_point(
        self,
        Yr: Any,
        inverse: bool = False,
        use_keops: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """Transform arbitrary points with the fitted diffeomorphism.

        Parameters
        ----------
        Yr : array-like, shape (N, D)
            Points in original (unnormalized) coordinates.
        inverse : bool, default=False
            Evaluate the inverse transformation.
        use_keops : bool, optional
            Override the fitted dense/KeOps evaluation backend.
        **kwargs
            K accepts ``n_iter``, ``tol``, ``relax``, ``chunk_size``,
            ``strict`` and ``return_info`` for numerical inversion. U accepts
            ``n_steps``/``max_steps``; V accepts ``substeps``/``max_substeps``;
            M accepts ``n_steps``, ``max_steps`` and ``chunk_size``. U/V/M
            also accept ``inverse_tol`` and ``inverse_max_iter``. Overriding
            integration steps defines a refined discrete map; omit these
            overrides to replay the fitted transformation.
        """
        required = {"Ym", "Ys", "Xm", "Xs"}
        if not required.issubset(self.tmats):
            raise RuntimeError("register() must finish before transforming points.")

        code = self.tmats.get("transformer", self.transformer)
        if code == "K":
            stages = self.tmats.get("rkhs_stages", [])
            common = {
                "inverse": inverse,
                "Ym": self.tmats["Ym"],
                "Ys": self.tmats["Ys"],
                "Xm": self.tmats["Xm"],
                "Xs": self.tmats["Xs"],
                **kwargs,
            }
            model = self.tmats.get(
                "rkhs_model", self.transparas.get("rkhs_model", "rff")
            )
            if model == "rff":
                return diffeo_ops.ccf_rkhs_rff_deformable_transform_point(
                    Yr, stages, **common
                )
            if use_keops and diffeo_ops.LazyTensor is None:
                use_keops = False
            return diffeo_ops.ccf_rkhs_deformable_transform_point(
                Yr, stages, use_keops=use_keops, **common
            )

        value, original_device, original_dtype = self._flow_input(Yr)
        Ym = th.as_tensor(self.tmats["Ym"], device=value.device, dtype=value.dtype)
        Ys = th.as_tensor(self.tmats["Ys"], device=value.device, dtype=value.dtype)
        Xm = th.as_tensor(self.tmats["Xm"], device=value.device, dtype=value.dtype)
        Xs = th.as_tensor(self.tmats["Xs"], device=value.device, dtype=value.dtype)
        value = (value - Xm) / Xs if inverse else (value - Ym) / Ys

        if code == "U":
            allowed = {"n_steps", "max_steps", "inverse_tol", "inverse_max_iter"}
            unknown = set(kwargs) - allowed
            if unknown:
                raise TypeError(f"Unexpected SVF transform options: {sorted(unknown)}")
            controls = th.as_tensor(
                self.tmats["svf_controls"], device=value.device, dtype=value.dtype
            )
            coeff = th.as_tensor(
                self.tmats["svf_coeff"], device=value.device, dtype=value.dtype
            )
            scales = th.as_tensor(
                self.tmats["svf_kernel_scales"],
                device=value.device,
                dtype=value.dtype,
            )
            weights = th.as_tensor(
                self.tmats["svf_kernel_weights"],
                device=value.device,
                dtype=value.dtype,
            )
            requested_keops = (
                self.tmats.get("svf_use_keops", False)
                if use_keops is None
                else use_keops
            )
            if requested_keops and diffeo_ops.LazyTensor is None:
                requested_keops = False
            value = diffeo_ops.svf_flow(
                value,
                controls,
                coeff,
                scales,
                weights,
                n_steps=int(kwargs.get("n_steps", self.tmats["svf_steps"])),
                max_steps=int(kwargs.get("max_steps", self.tmats.get("svf_max_steps", 1024))),
                inverse_tol=float(kwargs.get("inverse_tol", 1e-6)),
                inverse_max_iter=int(kwargs.get("inverse_max_iter", 200)),
                inverse=inverse,
                use_keops=bool(requested_keops),
            )

        elif code == "V":
            allowed = {"substeps", "max_substeps", "inverse_tol", "inverse_max_iter"}
            unknown = set(kwargs) - allowed
            if unknown:
                raise TypeError(f"Unexpected SVVF transform options: {sorted(unknown)}")
            controls = th.as_tensor(
                self.tmats["tvvf_controls"], device=value.device, dtype=value.dtype
            )
            coeff_steps = th.as_tensor(
                self.tmats["tvvf_coeff_steps"],
                device=value.device,
                dtype=value.dtype,
            )
            scales = th.as_tensor(
                self.tmats["tvvf_kernel_scales"],
                device=value.device,
                dtype=value.dtype,
            )
            weights = th.as_tensor(
                self.tmats["tvvf_kernel_weights"],
                device=value.device,
                dtype=value.dtype,
            )
            requested_keops = (
                self.tmats.get("tvvf_use_keops", False)
                if use_keops is None
                else use_keops
            )
            if requested_keops and diffeo_ops.LazyTensor is None:
                requested_keops = False
            value = diffeo_ops.tvvf_flow(
                value,
                controls,
                coeff_steps,
                scales,
                weights,
                substeps=int(kwargs.get("substeps", self.tmats["tvvf_substeps"])),
                max_substeps=int(kwargs.get("max_substeps", self.tmats.get("tvvf_max_substeps", 256))),
                inverse_tol=float(kwargs.get("inverse_tol", 1e-6)),
                inverse_max_iter=int(kwargs.get("inverse_max_iter", 200)),
                inverse=inverse,
                use_keops=bool(requested_keops),
            )

        elif code == "M":
            allowed = {"n_steps", "max_steps", "chunk_size", "inverse_tol", "inverse_max_iter"}
            unknown = set(kwargs) - allowed
            if unknown:
                raise TypeError(f"Unexpected LDDMM transform options: {sorted(unknown)}")
            q0 = th.as_tensor(
                self.tmats["lddmm_controls0"], device=value.device, dtype=value.dtype
            )
            p0 = th.as_tensor(
                self.tmats["lddmm_momentum0"], device=value.device, dtype=value.dtype
            )
            scales = th.as_tensor(
                self.tmats["lddmm_kernel_scales"],
                device=value.device,
                dtype=value.dtype,
            )
            weights = th.as_tensor(
                self.tmats["lddmm_kernel_weights"],
                device=value.device,
                dtype=value.dtype,
            )
            method = self.tmats.get("lddmm_method", "ralston")
            rff_model = None
            if method == "rff":
                rff_model = tuple(
                    th.as_tensor(self.tmats[key], device=value.device, dtype=value.dtype)
                    for key in (
                        "lddmm_rff_freq",
                        "lddmm_rff_phase",
                        "lddmm_rff_amp",
                    )
                )
            requested_keops = (
                self.tmats.get("lddmm_use_keops", False)
                if use_keops is None
                else use_keops
            )
            if requested_keops and diffeo_ops.LazyTensor is None:
                requested_keops = False
            value = diffeo_ops.lddmm_flow(
                value,
                q0,
                p0,
                scales,
                weights,
                n_steps=int(kwargs.get("n_steps", self.tmats["lddmm_steps"])),
                max_steps=int(kwargs.get("max_steps", self.tmats.get("lddmm_max_steps", 4096))),
                lipschitz_max=float(self.tmats.get("lddmm_lipschitz_max", 0.9)),
                inverse_tol=float(kwargs.get("inverse_tol", 1e-6)),
                inverse_max_iter=int(kwargs.get("inverse_max_iter", 200)),
                inverse=inverse,
                use_keops=bool(requested_keops),
                chunk_size=int(
                    kwargs.get("chunk_size", self.tmats["lddmm_chunk_size"])
                ),
                method=method,
                rff_model=rff_model,
            )[0]
        else:
            raise ValueError(f"Unknown diffeomorphic transformer: {code!r}.")

        value = value * Ys + Ym if inverse else value * Xs + Xm
        return self._restore_type(
            value, Yr, original_device=original_device, original_dtype=original_dtype
        )

    def inverse_transform_point(self, Xr: Any, **kwargs: Any) -> Any:
        """Evaluate the inverse fitted diffeomorphism."""
        return self.transform_point(Xr, inverse=True, **kwargs)

    transform_points = transform_point


def transform_diffeomorphic_points(
    points: Any,
    transformer: Mapping[str, Any],
    inverse: bool = False,
    use_keops: bool | None = None,
    **kwargs: Any,
) -> Any:
    """Apply a saved K/U/V/M transformer without a registration instance.

    This is the serialization-friendly counterpart of
    :meth:`pwDiffeoRegistration.transform_point` and is used by the chained
    ``ccf_transform_point`` dispatcher (for example by ``mmPWR``).
    """
    if not isinstance(transformer, Mapping):
        raise TypeError("transformer must be a mapping returned by get_transformer().")
    required = {"transformer", "Ym", "Ys", "Xm", "Xs"}
    missing = required.difference(transformer)
    if missing:
        raise ValueError(f"Saved diffeomorphic transformer is missing {sorted(missing)}.")

    # Transform evaluation depends only on the fitted state below.  Avoiding a
    # normal constructor keeps this helper cheap and prevents it from needing
    # the original X/Y point clouds.
    model = object.__new__(pwDiffeoRegistration)
    model.tmats = dict(transformer)
    model.transformer = pwDiffeoRegistration._resolve_transformer(
        transformer["transformer"]
    )
    model.tmats["transformer"] = model.transformer
    model.diffeo_method = pwDiffeoRegistration._METHOD_NAMES[model.transformer]
    model.D = int(th.as_tensor(transformer["Ym"]).numel())
    model.floatxx = th.float64
    model.transparas = {
        "rkhs_model": transformer.get("rkhs_model", "rff")
    }
    return model.transform_point(
        points, inverse=inverse, use_keops=use_keops, **kwargs
    )


__all__ = ["pwDiffeoRegistration", "transform_diffeomorphic_points"]
