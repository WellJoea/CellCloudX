from .groupwise_emregistration import gwEMRegistration


class gwHierarchicalRegistration(gwEMRegistration):
    """Groupwise registration with a global similarity and local RKHS warp.

    Each point set is transformed as ``s R [Y + v(Y)] + t``. The
    correspondence statistics are estimated against every enabled groupwise
    neighbour and combined before updating the point-set-specific transform.
    """

    def __init__(
        self,
        *args,
        beta=3.0,
        alpha=5e2,
        low_rank=3000,
        low_rank_type="keops",
        fast_low_rank=10000,
        num_eig=100,
        gamma1=0,
        gamma2=0,
        kw=15,
        kl=15,
        alpha_decayto=1,
        use_p1=False,
        p1_thred=0,
        gamma_growto=1,
        kd_method="sknn",
        fix_s=False,
        s_clip=None,
        transparas=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reg_core = "gw-Hierarchical"
        self.normal = "global" if self.normal is None else self.normal

        self.maxiter = self.maxiter or 400
        self.inneriter = self.inneriter or 40
        assert self.maxiter % self.inneriter == 0, (
            "maxiter must be multiple of inneriter"
        )
        self.tol = -1 if self.tol is None else self.tol
        self.tau2_decayto = (
            0.2 if self.tau2_decayto is None else self.tau2_decayto
        )

        self.transformer = self.scalar2vetor("H", self.L)
        self.init_transparas(
            beta=beta,
            alpha=alpha,
            low_rank=low_rank,
            low_rank_type=low_rank_type,
            fast_low_rank=fast_low_rank,
            num_eig=num_eig,
            gamma1=gamma1,
            gamma2=gamma2,
            kw=kw,
            kl=kl,
            alpha_decayto=alpha_decayto,
            use_p1=use_p1,
            p1_thred=p1_thred,
            gamma_growto=gamma_growto,
            kd_method=kd_method,
            fix_s=fix_s,
            s_clip=s_clip,
            transparas=transparas,
        )
        self.init_transformer()
        self.normal_Xs()
        self.normal_Fs()

    def init_transparas(self, s_clip=None, transparas=None, **kwargs):
        s_clips = self.init_sclip(s_clip)
        kwargs = {
            key: self.scalar2vetor(value, self.L) for key, value in kwargs.items()
        }
        self.transparas = {}

        for iL in range(self.L):
            ipara = {key: value[iL] for key, value in kwargs.items()}
            ipara["s_clip"] = s_clips[iL]

            if transparas is not None:
                ipara = {
                    **ipara,
                    **transparas.get("H", {}),
                    **transparas.get(iL, {}).get("H", {}),
                    **transparas.get(iL, {}),
                }

            schedule_steps = max(self.maxiter - 1, 1)
            ipara["alpha_mu"] = ipara["alpha_decayto"] ** (
                1.0 / float(schedule_steps)
            )
            ipara["gamma_nu"] = ipara["gamma_growto"] ** (
                1.0 / float(schedule_steps)
            )
            ipara["use_low_rank"] = (
                ipara["low_rank"]
                if type(ipara["low_rank"]) is bool
                else bool(self.Ns[iL] >= ipara["low_rank"])
            )
            ipara["use_fast_low_rank"] = (
                ipara["fast_low_rank"]
                if type(ipara["fast_low_rank"]) is bool
                else bool(self.Ns[iL] >= ipara["fast_low_rank"])
            )
            ipara["fast_rank"] = (
                ipara["use_low_rank"] or ipara["use_fast_low_rank"]
            )

            for name in ["alpha", "gamma1", "gamma2", "p1_thred"]:
                ipara[name] = self.xp.tensor(
                    self.scalar2vetor(ipara[name], self.L), dtype=self.floatx
                )
            self.transparas[iL] = ipara

    def init_sclip(self, s_clip):
        if s_clip is None:
            return self.scalar2vetor(None, self.L, force=True)

        s_clip_tensor = self.xp.tensor(s_clip)
        if s_clip_tensor.ndim == 1 and s_clip_tensor.shape[0] == 2:
            return self.scalar2vetor(s_clip, self.L, force=True)
        if s_clip_tensor.ndim == 2 and tuple(s_clip_tensor.shape) == (self.L, 2):
            return s_clip
        raise ValueError("s_clip must be a length-2 vector or an (L, 2) matrix")
