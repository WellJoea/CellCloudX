import torch as th

# Insert the methods below into class pwEM_core in pairwise_operation.py

    def _otm_as_list(self, F):
        if F is None:
            return []
        if isinstance(F, (list, tuple)):
            return list(F)
        return [F]

    @th.no_grad()
    def _otm_sqdist(self, A, B):
        """
        Pairwise squared Euclidean distances.
        A: [M, d], B: [N, d] -> [M, N]
        """
        A2 = (A * A).sum(dim=1, keepdim=True)
        B2 = (B * B).sum(dim=1, keepdim=True).transpose(0, 1)
        D2 = A2 + B2 - 2.0 * (A @ B.transpose(0, 1))
        return D2.clamp_min_(0.0)

    @th.no_grad()
    def _uot_sinkhorn_dense(self, C, a, b, eps, rho_x, rho_y,
                            max_iter=200, tol=1e-4, mask=None):
        """
        Entropic UOT in log-domain.
        C   : [M, N] positive cost matrix
        a   : [M] source masses for moving set Y / TY
        b   : [N] target masses for fixed set X
        mask: [M, N] bool, optional feasible edges mask
        return P: [M, N]
        """
        xp = self.xp
        tiny = self.eps
        neg_inf = -1e30

        # log K = -C / eps
        logK = -C / eps
        if mask is not None:
            logK = xp.where(mask, logK, xp.full_like(logK, neg_inf))

        rx = rho_x / (rho_x + eps)
        ry = rho_y / (rho_y + eps)

        log_a = xp.log(a.clamp_min(tiny))
        log_b = xp.log(b.clamp_min(tiny))

        log_u = xp.zeros_like(a)
        log_v = xp.zeros_like(b)

        for _ in range(max_iter):
            log_u_prev = log_u.clone()
            log_v_prev = log_v.clone()

            # log(K v)
            logKv = xp.logsumexp(logK + log_v[None, :], dim=1)
            log_u = rx * (log_a - logKv)

            # log(K^T u)
            logKTu = xp.logsumexp(logK.transpose(0, 1) + log_u[None, :], dim=1)
            log_v = ry * (log_b - logKTu)

            err = max(
                (log_u - log_u_prev).abs().max().item(),
                (log_v - log_v_prev).abs().max().item(),
            )
            if err < tol:
                break

        logP = log_u[:, None] + logK + log_v[None, :]
        if mask is not None:
            P = xp.where(mask, xp.exp(logP), xp.zeros_like(logP))
        else:
            P = xp.exp(logP)
        return P

    @th.no_grad()
    def _otm_build_cost(self, X, TY, XF=None, YF=None,
                        sigma2=None, tau2=None, feat_w=None):
        """
        Build OTM cost matrix C_{mn} with shape [M, N].
        Geometry: 0.5 / sigma2 * ||TY_m - X_n||^2
        Feature : sum_j 0.5 * feat_w[j] / tau2[j] * ||YF_{m,j} - XF_{n,j}||^2
        """
        xp = self.xp
        tiny = self.eps

        if sigma2 is None:
            raise ValueError('sigma2 is required for OTM cost construction.')

        C = 0.5 * self._otm_sqdist(TY, X) / (sigma2 + tiny)

        XFs = self._otm_as_list(XF)
        YFs = self._otm_as_list(YF)
        if len(XFs) == 0 or len(YFs) == 0:
            return C, []

        if not isinstance(tau2, (list, tuple)):
            tau2 = [tau2 for _ in range(len(XFs))]
        if feat_w is None:
            feat_w = [1.0 for _ in range(len(XFs))]
        elif not isinstance(feat_w, (list, tuple)):
            feat_w = [feat_w for _ in range(len(XFs))]

        feat_d2_raw = []
        for xfj, yfj, tj, wj in zip(XFs, YFs, tau2, feat_w):
            d2j = self._otm_sqdist(yfj, xfj)
            feat_d2_raw.append(d2j)
            tj = tj if th.is_tensor(tj) else xp.tensor(tj, dtype=X.dtype, device=X.device)
            C = C + 0.5 * float(wj) * d2j / (tj + tiny)

        return C, feat_d2_raw

    @th.no_grad()
    def pwexpectation_otm(self):
        """
        OTM/UOT version of the E-step for multiple references Xs.
        Returns the same sufficient statistics as pwexpectation:
            Pt1s, P1s, PXs, Nps

        OTM/UOT E-step with implicit pi0 / pi_m update.

        Interpretation
        --------------
        - pi_m (self.wms[iL]) is the moving-side prior mass over Y.
        - pi0  (self.ws[iL]) is the global unmatched/outlier ratio.
        - In OTM, pi0 is injected implicitly by reducing the total source mass:
            sum(a) = (1 - pi0) * U_mass
            sum(b) = U_mass
        so the missing mass corresponds to unmatched/outlier mass.
        """
        xp = self.xp
        tp = self.transparas

        Pt1s, P1s, PXs, Nps = [], [], [], []

        otm_eps = float(tp.get('otm_eps', 0.05))
        otm_rho_x = float(tp.get('otm_rho_x', 1.0))
        otm_rho_y = float(tp.get('otm_rho_y', 1.0))
        otm_max_iter = int(tp.get('otm_max_iter', 200))
        otm_tol = float(tp.get('otm_tol', 1e-4))
        otm_feat_w = tp.get('otm_feat_w', None)
        otm_reg = tp.get('otm_reg', 'entropy')
        otm_mass_mode = tp.get('otm_mass_mode', 'inlier_target')
        otm_mass_scale = float(tp.get('otm_mass_scale', 1.0))

        if otm_reg == 'entropy':
            expectation_otm = expectation_otm_entropy
        elif otm_reg == 'kl':
            expectation_otm = expectation_otm_kl
        else:
            raise ValueError(f'Unsupported otm_reg: {otm_reg}')

        for iL in range(self.L):
            iX, iTY, fexist = self.Xs[iL], self.TY, self.fexists[iL]
            xFa, yFa, wi = self.XF[iL], self.YF[iL], self.ws[iL]
            sigma2i = self.sigma2[iL]

            use_samplex = bool(self.sample[iL]) and (self.iteration < self.sample_stop)
            if use_samplex:
                xsidx = self.sample_index(iX.shape[0], self.iteration, self.sample[iL])
                iX = iX[xsidx]
                if fexist:
                    xFa = [ifa[xsidx] for ifa in self.XF[iL]]

            Ni = iX.shape[0]
            Mi = iTY.shape[0]

            # ----------------------------------------------
            # 2) Build marginals a^{(l)}, b^{(l)} for UOT
            # ----------------------------------------------
            # if otm_mass_mode == 'inlier_target':
            #     mass0 = max((1.0 - float(wi)) * Ni * otm_mass_scale, 1.0)
            # elif otm_mass_mode == 'fixed_N':
            #     mass0 = max(Ni * otm_mass_scale, 1.0)
            # elif otm_mass_mode == 'min_MN':
            #     mass0 = max(min(Mi, Ni) * otm_mass_scale, 1.0)
            # else:
            #     mass0 = max((1.0 - float(wi)) * Ni * otm_mass_scale, 1.0)

            mass0 = max(Ni * otm_mass_scale, 1.0)

            # a: moving(Y) side prior mass, b: fixed(X) side prior mass
            if hasattr(self, 'wms') and (self.wms is not None) and (self.wms[iL] is not None):
                ai = self.wms[iL].clamp_min(self.eps)
                a = mass0 * ai / ai.sum()
            else:
                a = xp.full((Mi,), mass0 / Mi, dtype=iX.dtype, device=iX.device)
            b = xp.full((Ni,), mass0 / Ni, dtype=iX.dtype, device=iX.device)
            u = self.us[iL]
            v = self.vs[iL]

            # Optional feasible-edge mask for sparse candidates
            mask = None #TODO in keops
            
            #dustbin #TODO

            # -------------------
            # 3) Solve UOT/OTM
            # -------------------
            iPt1, iP1, iPX, iNp, tau2s, iu, iv = expectation_otm(
                iX, iTY, self.sigma2[iL],
                C=None, a=a, b=b, eps_c=1.0, rho_y=1.0, rho_x=1.0,
                u=u, v=v,
                uv_iter=200, uv_tol=1e-4,
                d2f=d2f, tau2=self.tau2[iL], 
                tau2_auto=self.tau2_auto[iL], eps=self.eps, 
                tau2_alpha=self.tau2_grow[iL][self.iteration],
                feat_normal=self.feat_normal[iL], 
                XF=xFa, YF=yFa, device=self.device,
                DF=self.DK[iL],
                xp=self.xp, use_keops=self.use_keops[iL],
            )

            # -------------------------------------
            # 4) Optional updates of ws and tau2
            # -------------------------------------
            self.us[iL] = self.otm_mr * u + (1.0 - self.otm_mr) * iu
            self.vs[iL] = self.otm_mr * v + (1.0 - self.otm_mr) * iv

            # pi0_new, piu_new = update_pi(iP1, iNp, Ni, Mi, pi0, 
            #                             alpha=a1, alpha0=a0, 
            #                             U = Ui,
            #                             mrf_solver=mrf,
            #                             w_clip=w_clip,
            #                             xp=self.xp)

            # self.ws[ iL] = pi0_new
            # self.wms[iL] = piu_new
            # self.ws[ iL] = wa *pi0 + (1-wa)*pi0_new
            # self.wms[iL] = wma *piu + (1-wma)*piu_new

            tau2s = self.tau2[iL]
            self.tau2[iL] = tau2s

            Pt1s.append(iPt1)
            P1s.append(iP1)
            PXs.append(iPX)
            Nps.append(iNp)

        return Pt1s, P1s, PXs, Nps


    @th.no_grad()
    def pwexpectation_otm(self):
        """
        OTM/UOT version of the E-step for multiple references Xs.
        Returns the same sufficient statistics as pwexpectation:
            Pt1s, P1s, PXs, Nps

        OTM/UOT E-step with implicit pi0 / pi_m update.

        Interpretation
        --------------
        - pi_m (self.wms[iL]) is the moving-side prior mass over Y.
        - pi0  (self.ws[iL]) is the global unmatched/outlier ratio.
        - In OTM, pi0 is injected implicitly by reducing the total source mass:
            sum(a) = (1 - pi0) * U_mass
            sum(b) = U_mass
        so the missing mass corresponds to unmatched/outlier mass.
        """

        xp = self.xp
        tp = self.transparas

        Pt1s, P1s, PXs, Nps = [], [], [], []

        otm_eps = float(tp.get('otm_eps', 0.05))
        otm_rho_x = float(tp.get('otm_rho_x', 1.0))
        otm_rho_y = float(tp.get('otm_rho_y', 1.0))
        otm_max_iter = int(tp.get('otm_max_iter', 200))
        otm_tol = float(tp.get('otm_tol', 1e-4))
        otm_mass_mode = tp.get('otm_mass_mode', 'target_U')
        otm_mass_scale = float(tp.get('otm_mass_scale', 1.0))

        for iL in range(self.L):
            iX, iTY, fexist = self.Xs[iL], self.TY, self.fexists[iL]
            xFa, yFa = self.XF[iL], self.YF[iL]
            d2f = self.d2f[iL]

            sigma2i = self.sigma2[iL]

            # parameters shared with LGM version
            pi0 = self.ws[iL]
            piu = self.wms[iL]

            wa = self.wa
            wma = self.wma
            w_clip = self.w_clip

            a0 = self.a0[iL]
            a1 = self.a1[iL]
            U = self.U[iL]
            mrf = self.mrf

            if fexist:
                assert d2f is not None

            # ---------------------------------
            # Optional stochastic subsampling on X
            # ---------------------------------
            use_samplex = bool(self.sample[iL]) and (self.iteration < self.sample_stop)
            xsidx = None
            if use_samplex:
                xsidx = self.sample_index(iX.shape[0], self.iteration, self.sample[iL])
                iX = iX[xsidx]

                if fexist:
                    xFa = [ifa[xsidx] for ifa in self.XF[iL]]
                    if self.use_keops[iL]:
                        d2f = features_pdist2(
                            xFa, yFa, tau2s=self.tau2[iL],
                            use_keops=True,
                            tau2_prediv=not self.tau2_auto[iL],
                            device=self.device, dtype=self.floatx
                        )
                    else:
                        d2f = self.d2f[iL][:, xsidx]

            Ni = iX.shape[0]
            Mi = iTY.shape[0]

            # ---------------------------------
            # Choose total target mass U_mass
            # ---------------------------------
            Ui = U if (U is not None) else Ni

            if otm_mass_mode in ['target_U', 'inlier_target']:
                U_mass = max(float(Ui) * otm_mass_scale, 1.0)
            elif otm_mass_mode == 'fixed_N':
                U_mass = max(float(Ni) * otm_mass_scale, 1.0)
            elif otm_mass_mode == 'min_MN':
                U_mass = max(float(min(Mi, Ni)) * otm_mass_scale, 1.0)
            else:
                U_mass = max(float(Ui) * otm_mass_scale, 1.0)

            # ---------------------------------
            # Build OTM marginals from pi0 / piu
            # ---------------------------------
            # target side total mass
            total_b = U_mass

            # moving/source side total inlier mass
            total_a = max((1.0 - float(pi0)) * U_mass, self.eps)

            # a ~ piu, with sum(a)=total_a
            if piu is not None:
                ai = piu.clamp_min(self.eps)
                a = total_a * ai / ai.sum()
            else:
                a = xp.full((Mi,), total_a / Mi, dtype=iX.dtype, device=iX.device)

            # b uniform on X, with sum(b)=total_b
            b = xp.full((Ni,), total_b / Ni, dtype=iX.dtype, device=iX.device)

            # ---------------------------------
            # Warm start for Sinkhorn scalings
            # ---------------------------------
            u = self.us[iL]
            if (u is not None) and (u.shape[0] != Mi):
                u = None

            # If X is subsampled, the old v usually has wrong size.
            if use_samplex:
                v = None
            else:
                v = self.vs[iL]
                if (v is not None) and (v.shape[0] != Ni):
                    v = None

            # ---------------------------------
            # Select OTM solver
            # ---------------------------------
            if tp.get('otm_reg', 'entropy') == 'entropy':
                expectation_otm = expectation_otm_entropy_ko
            elif tp.get('otm_reg', 'entropy') == 'kl':
                expectation_otm = expectation_otm_kl_ko
            else:
                raise ValueError(f"Unsupported otm_reg: {tp.get('otm_reg')}")

            # ---------------------------------
            # Solve OTM / UOT
            # ---------------------------------
            iPt1, iP1, iPX, iNp, tau2s, iu, iv = expectation_otm(
                iX, iTY, sigma2i,
                C=None, a=a, b=b,
                eps_c=otm_eps,
                rho_y=otm_rho_y,
                rho_x=otm_rho_x,
                u=u, v=v,
                uv_iter=otm_max_iter,
                uv_tol=otm_tol,
                d2f=d2f,
                tau2=self.tau2[iL],
                tau2_auto=self.tau2_auto[iL],
                eps=self.eps,
                tau2_alpha=self.tau2_grow[iL][self.iteration],
                feat_normal=self.feat_normal[iL],
                XF=xFa, YF=yFa,
                device=self.device,
                DF=self.DK[iL],
                xp=self.xp,
                use_keops=self.use_keops[iL],
            )

            # ---------------------------------
            # Update pi0 / piu from OTM sufficient statistics
            # ---------------------------------
            pi0_new, piu_new = update_pi(
                iP1, iNp, Ni, Mi, pi0,
                alpha=a1, alpha0=a0,
                U=U_mass,
                mrf_solver=mrf,
                w_clip=w_clip,
                xp=self.xp
            )

            # damping / momentum
            self.ws[iL] = wa * pi0 + (1.0 - wa) * pi0_new
            self.wms[iL] = wma * (piu if piu is not None else a) + (1.0 - wma) * piu_new

            # update tau2
            self.tau2[iL] = tau2s

            # update Sinkhorn warm starts
            self.us[iL] = iu if (u is None) else (self.otm_mr * u + (1.0 - self.otm_mr) * iu)

            if not use_samplex:
                self.vs[iL] = iv if (v is None) else (self.otm_mr * v + (1.0 - self.otm_mr) * iv)

            Pt1s.append(iPt1)
            P1s.append(iP1)
            PXs.append(iPX)
            Nps.append(iNp)

        return Pt1s, P1s, PXs, Nps


    def optimization(self):
        qprev = self.Q
        if getattr(self, 'use_otm', False):
            Pt1s, P1s, PXs, Nps = self.pwexpectation_otm()
        elif self.use_wm:
            Pt1s, P1s, PXs, Nps = self.pwexpectation_d()
        else:
            Pt1s, P1s, PXs, Nps = self.pwexpectation()

        self.maximization(Pt1s, P1s, Nps, PXs, self.Y, self.Xs, transformer=self.transformer)
        self.diff = self.xp.abs(self.Q - qprev)
        self.iteration += 1


import torch as th

# Insert the methods below into class pwEM_core in pairwise_operation.py

    def _otm_as_list(self, F):
        if F is None:
            return []
        if isinstance(F, (list, tuple)):
            return list(F)
        return [F]

    def _otm_scalar(self, x):
        if th.is_tensor(x):
            return float(x.detach().item())
        return float(x)

    @th.no_grad()
    def _otm_sqdist(self, A, B):
        """
        Pairwise squared Euclidean distances.
        A: [M, d], B: [N, d] -> [M, N]
        """
        A2 = (A * A).sum(dim=1, keepdim=True)
        B2 = (B * B).sum(dim=1, keepdim=True).transpose(0, 1)
        D2 = A2 + B2 - 2.0 * (A @ B.transpose(0, 1))
        return D2.clamp_min_(0.0)

    @th.no_grad()
    def _otm_build_cost(self, X, TY, XF=None, YF=None,
                        sigma2=None, tau2=None, feat_w=None):
        """
        Build OTM cost matrix C_{mn} with shape [M, N].
        Geometry: 0.5 / sigma2 * ||TY_m - X_n||^2
        Feature : sum_j 0.5 * feat_w[j] / tau2[j] * ||YF_{m,j} - XF_{n,j}||^2
        """
        xp = self.xp
        tiny = self.eps

        if sigma2 is None:
            raise ValueError('sigma2 is required for OTM cost construction.')

        C = 0.5 * self._otm_sqdist(TY, X) / (sigma2 + tiny)

        XFs = self._otm_as_list(XF)
        YFs = self._otm_as_list(YF)
        if len(XFs) == 0 or len(YFs) == 0:
            return C, []

        if not isinstance(tau2, (list, tuple)):
            tau2 = [tau2 for _ in range(len(XFs))]
        if feat_w is None:
            feat_w = [1.0 for _ in range(len(XFs))]
        elif not isinstance(feat_w, (list, tuple)):
            feat_w = [feat_w for _ in range(len(XFs))]

        feat_d2_raw = []
        for xfj, yfj, tj, wj in zip(XFs, YFs, tau2, feat_w):
            d2j = self._otm_sqdist(yfj, xfj)
            feat_d2_raw.append(d2j)
            tj = tj if th.is_tensor(tj) else xp.tensor(tj, dtype=X.dtype, device=X.device)
            C = C + 0.5 * float(wj) * d2j / (tj + tiny)

        return C, feat_d2_raw

    @th.no_grad()
    def _otm_make_marginals(self, Mi, Ni, wi, iL, tp, dtype, device):
        """
        Build source/target marginal priors a, b and matched mass mass0.

        Semantics in OTM/UOT:
        - w   : global unmatched ratio prior / overlap prior
        - wms : Y-side mass prior / confidence prior
        """
        xp = self.xp

        otm_mass_mode = tp.get('otm_mass_mode', 'inlier_target')
        otm_mass_scale = float(tp.get('otm_mass_scale', 1.0))

        wi_f = self._otm_scalar(wi)

        if otm_mass_mode == 'inlier_target':
            base_mass = Ni
            mass0 = max((1.0 - wi_f) * Ni * otm_mass_scale, 1.0)
        elif otm_mass_mode == 'inlier_source':
            base_mass = Mi
            mass0 = max((1.0 - wi_f) * Mi * otm_mass_scale, 1.0)
        elif otm_mass_mode == 'overlap_min':
            base_mass = min(Mi, Ni)
            mass0 = max((1.0 - wi_f) * min(Mi, Ni) * otm_mass_scale, 1.0)
        elif otm_mass_mode == 'overlap_mean':
            base_mass = 0.5 * (Mi + Ni)
            mass0 = max((1.0 - wi_f) * 0.5 * (Mi + Ni) * otm_mass_scale, 1.0)
        elif otm_mass_mode == 'fixed_N':
            base_mass = Ni
            mass0 = max(Ni * otm_mass_scale, 1.0)
        elif otm_mass_mode == 'min_MN':
            base_mass = min(Mi, Ni)
            mass0 = max(min(Mi, Ni) * otm_mass_scale, 1.0)
        elif otm_mass_mode == 'mean_MN':
            base_mass = 0.5 * (Mi + Ni)
            mass0 = max(0.5 * (Mi + Ni) * otm_mass_scale, 1.0)
        else:
            base_mass = Ni
            mass0 = max((1.0 - wi_f) * Ni * otm_mass_scale, 1.0)

        if hasattr(self, 'wms') and (self.wms is not None) and (self.wms[iL] is not None):
            ai = self.wms[iL].to(device=device, dtype=dtype).clamp_min(self.eps)
            a = mass0 * ai / ai.sum().clamp_min(self.eps)
        else:
            a = xp.full((Mi,), mass0 / max(Mi, 1), dtype=dtype, device=device)

        # X-side is kept uniform by default. You can extend this similarly if a target prior exists.
        b = xp.full((Ni,), mass0 / max(Ni, 1), dtype=dtype, device=device)

        return a, b, mass0, float(base_mass)

    @th.no_grad()
    def _otm_prepare_dustbin(self, C, a, b, Mi, Ni, tp, dtype, device):
        """
        Augment the OT problem with optional dustbin row/column.

        dustbin_mode:
            'none' : no dustbin
            'x'    : add source-side dustbin row to absorb unmatched X mass
            'y'    : add target-side dustbin column to absorb unmatched Y mass
            'both' : add both

        Note:
            Returned sufficient statistics are always computed on the real-real block only.
            Therefore the M-step remains unchanged.
        """
        xp = self.xp

        dustbin_mode = tp.get('otm_dustbin_mode', 'none')
        if dustbin_mode is None:
            dustbin_mode = 'none'
        dustbin_mode = str(dustbin_mode).lower()

        if dustbin_mode not in {'none', 'x', 'y', 'both'}:
            dustbin_mode = 'none'

        if dustbin_mode == 'none':
            return C, a, b, Mi, Ni

        # Costs are in the normalized transport-cost scale.
        c_db_x = float(tp.get('otm_dustbin_cost_x', 2.5))
        c_db_y = float(tp.get('otm_dustbin_cost_y', 2.5))
        c_db_xy = float(tp.get('otm_dustbin_cost_xy', 0.0))

        # Extra masses associated with unmatched priors.
        db_row_mass = float(tp.get('otm_dustbin_mass_x', max(float(Ni) - float(a.sum().item()), 0.0)))
        db_col_mass = float(tp.get('otm_dustbin_mass_y', max(float(Mi) - float(b.sum().item()), 0.0)))

        Cout = C
        aout = a
        bout = b
        Mext, Next = Mi, Ni

        add_row = dustbin_mode in {'x', 'both'}
        add_col = dustbin_mode in {'y', 'both'}

        if add_row:
            row = xp.full((1, Next), c_db_x, dtype=dtype, device=device)
            Cout = xp.cat([Cout, row], dim=0)
            aout = xp.cat([aout, xp.tensor([db_row_mass], dtype=dtype, device=device)], dim=0)
            Mext += 1

        if add_col:
            col = xp.full((Mext, 1), c_db_y, dtype=dtype, device=device)
            if add_row:
                col[-1, 0] = c_db_xy
            Cout = xp.cat([Cout, col], dim=1)
            bout = xp.cat([bout, xp.tensor([db_col_mass], dtype=dtype, device=device)], dim=0)
            Next += 1

        return Cout, aout, bout, Mext, Next

    @th.no_grad()
    def _uot_sinkhorn_dense_general(self, C, a, b, eps, rho_y, rho_x,
                                    max_iter=200, tol=1e-4, mask=None,
                                    reg_mode='entropy', ref=None,
                                    return_log=True):
        """
        Generalized dense UOT in log-domain.

        Solve:
            min_P <C,P>
                + eps * Omega(P)
                + rho_y * KL(P1 || a)
                + rho_x * KL(P^T1 || b)

        where:
            - reg_mode='entropy': Omega(P) = sum P(log P - 1)
            - reg_mode='kl'     : Omega(P) = KL(P || R), ref=R

        The optimal plan keeps the scaling form:
            P = diag(u) K diag(v),
        with:
            K = exp(-C/eps)                 for entropy mode,
            K = R * exp(-C/eps)            for KL-reference mode.
        """
        xp = self.xp
        tiny = self.eps
        neg_inf = -1e30

        epsv = max(float(eps), tiny)
        ry = float(rho_y) / (float(rho_y) + epsv)
        rx = float(rho_x) / (float(rho_x) + epsv)

        log_a = xp.log(a.clamp_min(tiny))
        log_b = xp.log(b.clamp_min(tiny))

        reg_mode = str(reg_mode).lower()
        if reg_mode == 'kl':
            if ref is None:
                raise ValueError("ref must be provided when reg_mode='kl'.")
            logK = xp.log(ref.clamp_min(tiny)) - C / epsv
        else:
            logK = -C / epsv

        if mask is not None:
            logK = xp.where(mask, logK, xp.full_like(logK, neg_inf))

        log_u = xp.zeros_like(a)
        log_v = xp.zeros_like(b)

        for _ in range(max_iter):
            log_u_prev = log_u.clone()
            log_v_prev = log_v.clone()

            # Y-side update: P1 ~ a
            logKv = xp.logsumexp(logK + log_v[None, :], dim=1)
            log_u = ry * (log_a - logKv)

            # X-side update: Pt1 ~ b
            logKTu = xp.logsumexp(logK.transpose(0, 1) + log_u[None, :], dim=1)
            log_v = rx * (log_b - logKTu)

            err = max(
                (log_u - log_u_prev).abs().max().item(),
                (log_v - log_v_prev).abs().max().item(),
            )
            if err < tol:
                break

        if return_log:
            return log_u, log_v, logK

        logP = log_u[:, None] + logK + log_v[None, :]
        if mask is not None:
            P = xp.where(mask, xp.exp(logP), xp.zeros_like(logP))
        else:
            P = xp.exp(logP)
        return P

    @th.no_grad()
    def _uot_stats_realblock(self, log_u, log_v, logK, X, M_real, N_real,
                             mask=None, chunk_size=0):
        """
        Recover sufficient statistics only on the real-real block:
            P_real = P[:M_real, :N_real].

        This excludes any optional dustbin row/column from the statistics used in M-step.
        """
        xp = self.xp

        logP_rr = log_u[:M_real, None] + logK[:M_real, :N_real] + log_v[None, :N_real]
        if mask is not None:
            P_rr = xp.where(mask[:M_real, :N_real], xp.exp(logP_rr), xp.zeros_like(logP_rr))
        else:
            P_rr = xp.exp(logP_rr)

        P1 = P_rr.sum(dim=1)      # [M_real]
        Pt1 = P_rr.sum(dim=0)     # [N_real]
        Np = P1.sum()             # scalar

        if (chunk_size is None) or (int(chunk_size) <= 0) or (N_real <= int(chunk_size)):
            PX = P_rr @ X
        else:
            D = X.shape[1]
            PX = xp.zeros((M_real, D), dtype=X.dtype, device=X.device)
            bs = int(chunk_size)
            for j0 in range(0, N_real, bs):
                j1 = min(N_real, j0 + bs)
                logP_blk = log_u[:M_real, None] + logK[:M_real, j0:j1] + log_v[None, j0:j1]
                if mask is not None:
                    P_blk = xp.where(mask[:M_real, j0:j1], xp.exp(logP_blk), xp.zeros_like(logP_blk))
                else:
                    P_blk = xp.exp(logP_blk)
                PX = PX + P_blk @ X[j0:j1]

        return Pt1, P1, PX, Np

    @th.no_grad()
    def _otm_update_w(self, wi, iNp, Mi, Ni, tp):
        """
        Update the global unmatched-ratio prior.

        w_update_mode:
            'x'    : 1 - Np / N
            'y'    : 1 - Np / M
            'min'  : 1 - Np / min(M, N)
            'max'  : 1 - Np / max(M, N)
            'mean' : 1 - 2Np / (M + N)

        In pairwise settings, 'mean' or 'min' usually provides the most symmetric behavior.
        """
        mode = str(tp.get('otm_w_update_mode', 'x')).lower()

        if mode == 'y':
            denom = max(float(Mi), 1.0)
            nwi = 1.0 - iNp / denom
        elif mode == 'min':
            denom = max(float(min(Mi, Ni)), 1.0)
            nwi = 1.0 - iNp / denom
        elif mode == 'max':
            denom = max(float(max(Mi, Ni)), 1.0)
            nwi = 1.0 - iNp / denom
        elif mode == 'mean':
            denom = max(0.5 * float(Mi + Ni), 1.0)
            nwi = 1.0 - iNp / denom
        else:  # default: x
            denom = max(float(Ni), 1.0)
            nwi = 1.0 - iNp / denom

        nwi = nwi.clamp(*self.w_clip)
        return self.wa * wi + (1.0 - self.wa) * nwi

    @th.no_grad()
    def pwexpectation_otm(self):
        """
        Final OTM/UOT E-step for multiple references Xs.

        Supported switches:
            - otm_reg_mode      : 'entropy' | 'kl'
            - otm_w_update_mode : 'x' | 'y' | 'min' | 'max' | 'mean'
            - otm_dustbin_mode  : 'none' | 'x' | 'y' | 'both'
            - otm_ref_mode      : 'outer' | 'uniform'   (used when reg_mode='kl')

        Returns the same sufficient statistics as pwexpectation:
            Pt1s, P1s, PXs, Nps
        so that the current M-step can be reused without modification.
        """
        xp = self.xp
        tp = self.transparas

        Pt1s, P1s, PXs, Nps = [], [], [], []

        otm_eps = float(tp.get('otm_eps', 0.05))
        otm_rho_x = float(tp.get('otm_rho_x', 1.0))
        otm_rho_y = float(tp.get('otm_rho_y', 1.0))
        otm_max_iter = int(tp.get('otm_max_iter', 200))
        otm_tol = float(tp.get('otm_tol', 1e-4))
        otm_feat_w = tp.get('otm_feat_w', None)
        otm_reg_mode = str(tp.get('otm_reg_mode', 'entropy')).lower()
        otm_ref_mode = str(tp.get('otm_ref_mode', 'outer')).lower()
        otm_chunk_size = int(tp.get('otm_chunk_size', 0))

        for iL in range(self.L):
            iX = self.Xs[iL]
            iTY = self.TY
            fexist = self.fexists[iL]
            wi = self.ws[iL]
            sigma2i = self.sigma2[iL]

            xFa = self._otm_as_list(self.XF[iL]) if fexist else []
            yFa = self._otm_as_list(self.YF[iL]) if fexist else []

            use_samplex = bool(self.sample[iL]) and (self.iteration < self.sample_stop)
            if use_samplex:
                xsidx = self.sample_index(iX.shape[0], self.iteration, self.sample[iL])
                iX = iX[xsidx]
                if fexist:
                    xFa = [ifa[xsidx] for ifa in xFa]

            Mi = int(iTY.shape[0])
            Ni = int(iX.shape[0])
            dtype = iX.dtype
            device = iX.device

            # -------------------------------------------------
            # 1) Build real-real cost matrix C^{(l)} in [M, N]
            # -------------------------------------------------
            iC, _ = self._otm_build_cost(
                iX, iTY,
                XF=xFa if fexist else None,
                YF=yFa if fexist else None,
                sigma2=sigma2i,
                tau2=self.tau2[iL],
                feat_w=otm_feat_w,
            )

            # -------------------------------------------------
            # 2) Build marginal priors a^{(l)}, b^{(l)}
            # -------------------------------------------------
            a, b, mass0, _ = self._otm_make_marginals(Mi, Ni, wi, iL, tp, dtype, device)

            # -------------------------------------------------
            # 3) Optional sparse candidate mask
            # -------------------------------------------------
            mask = None
            if hasattr(self, 'otm_masks') and (self.otm_masks is not None) and (self.otm_masks[iL] is not None):
                mask = self.otm_masks[iL]
                if use_samplex:
                    mask = mask[:, xsidx]
                mask = mask.to(device=device)

            # -------------------------------------------------
            # 4) Optional dustbin augmentation
            # -------------------------------------------------
            iC_ext, a_ext, b_ext, Mext, Next = self._otm_prepare_dustbin(
                iC, a, b, Mi, Ni, tp, dtype, device
            )
            if mask is not None and ((Mext != Mi) or (Next != Ni)):
                mask_ext = xp.ones((Mext, Next), dtype=th.bool, device=device)
                mask_ext[:Mi, :Ni] = mask
                mask = mask_ext

            # -------------------------------------------------
            # 5) Choose the reference measure for KL-plan regularization
            # -------------------------------------------------
            ref = None
            if otm_reg_mode == 'kl':
                if otm_ref_mode == 'uniform':
                    ref = xp.ones_like(iC_ext)
                else:
                    # outer-product reference consistent with marginal priors
                    ref = a_ext[:, None] * b_ext[None, :]
                    ref = ref / ref.sum().clamp_min(self.eps)

            # -------------------------------------------------
            # 6) Solve generalized UOT/OTM in log-domain
            # -------------------------------------------------
            log_u, log_v, logK = self._uot_sinkhorn_dense_general(
                iC_ext, a_ext, b_ext,
                eps=otm_eps,
                rho_y=otm_rho_y,
                rho_x=otm_rho_x,
                max_iter=otm_max_iter,
                tol=otm_tol,
                mask=mask,
                reg_mode=otm_reg_mode,
                ref=ref,
                return_log=True,
            )

            # -------------------------------------------------
            # 7) Recover sufficient statistics on the real-real block only
            # -------------------------------------------------
            iPt1, iP1, iPX, iNp = self._uot_stats_realblock(
                log_u, log_v, logK,
                X=iX,
                M_real=Mi,
                N_real=Ni,
                mask=mask,
                chunk_size=otm_chunk_size,
            )

            # -------------------------------------------------
            # 8) Update the global unmatched prior w
            # -------------------------------------------------
            self.ws[iL] = self._otm_update_w(wi, iNp, Mi, Ni, tp)

            # -------------------------------------------------
            # 9) Keep tau2 unchanged in the safe default version
            # -------------------------------------------------
            self.tau2[iL] = self.tau2[iL]

            Pt1s.append(iPt1)
            P1s.append(iP1)
            PXs.append(iPX)
            Nps.append(iNp)

        return Pt1s, P1s, PXs, Nps

    def optimization(self):
        qprev = self.Q
        if getattr(self, 'use_otm', False):
            Pt1s, P1s, PXs, Nps = self.pwexpectation_otm()
        elif self.use_wm:
            Pt1s, P1s, PXs, Nps = self.pwexpectation_d()
        else:
            Pt1s, P1s, PXs, Nps = self.pwexpectation()

        self.maximization(Pt1s, P1s, Nps, PXs, self.Y, self.Xs, transformer=self.transformer)
        self.diff = self.xp.abs(self.Q - qprev)
        self.iteration += 1


# -*- coding: utf-8 -*-
"""
Simple updates for mixture weights pi = (pi0, pi1, ..., piM)
under simplex constraint:
    pi_i >= 0, sum_i pi_i = 1.

This script provides:
1) baseline MLE update
2) KL-regularized update with KL(q || pi)  -> closed-form pseudo-count smoothing
3) entropy-regularized update              -> 1D root + Lambert W
4) tied version pi_m = (1 - pi0) / M

Author: OpenAI ChatGPT
"""

import numpy as np

try:
    from scipy.optimize import bisect
    from scipy.special import lambertw
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


EPS = 1e-12


def _normalize_simplex(x, eps=EPS):
    """Project a nonnegative vector to the simplex by normalization."""
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, eps)
    s = x.sum()
    if s <= 0:
        raise ValueError("Sum of entries must be positive.")
    return x / s


# ---------------------------------------------------------------------
# 1) Baseline update: pi_i = c_i / sum_i c_i
# ---------------------------------------------------------------------
def update_pi_mle(c0, cm, eps=EPS):
    """
    Baseline MLE update without regularization.

    Parameters
    ----------
    c0 : float
        Soft count for the background/outlier component.
    cm : array-like, shape (M,)
        Soft counts for M inlier components.

    Returns
    -------
    pi0 : float
    pim : ndarray, shape (M,)
    pi  : ndarray, shape (M+1,)
    """
    cm = np.asarray(cm, dtype=float)
    c = np.concatenate([[float(c0)], cm])
    c = np.maximum(c, 0.0)
    C = c.sum()
    if C <= 0:
        raise ValueError("Total soft count must be positive.")
    pi = np.maximum(c / C, eps)
    pi = pi / pi.sum()
    return pi[0], pi[1:], pi


# ---------------------------------------------------------------------
# 2) KL-regularized update: J = -sum c_i log pi_i + eta * KL(q || pi)
#    Closed-form: pi_i = (c_i + eta q_i) / (C + eta)
# ---------------------------------------------------------------------
def update_pi_kl(c0, cm, eta=1.0, q=None, q0=None, qm=None, eps=EPS):
    """
    KL-regularized update with KL(q || pi).

    Objective:
        min_pi  -sum_i c_i log pi_i + eta * KL(q || pi)
        s.t.    pi in simplex

    Closed-form solution:
        pi_i = (c_i + eta q_i) / (C + eta)

    Parameters
    ----------
    c0 : float
        Soft count for component 0.
    cm : array-like, shape (M,)
        Soft counts for components 1..M.
    eta : float
        Regularization strength.
    q : array-like, optional
        Prior distribution of shape (M+1,).
        If given, q0 and qm are ignored.
    q0 : float, optional
        Prior weight for pi0.
    qm : array-like or None, optional
        Prior weights for pi_m, shape (M,).
        If None and q is None, uses uniform prior over inlier part.
    eps : float
        Numerical floor.

    Returns
    -------
    pi0 : float
    pim : ndarray
    pi  : ndarray
    """
    cm = np.asarray(cm, dtype=float)
    M = len(cm)
    c = np.concatenate([[float(c0)], cm])
    c = np.maximum(c, 0.0)
    C = c.sum()
    if C <= 0:
        raise ValueError("Total soft count must be positive.")
    if eta < 0:
        raise ValueError("eta must be nonnegative.")

    if q is not None:
        q = _normalize_simplex(q, eps=eps)
        if len(q) != M + 1:
            raise ValueError("q must have length M+1.")
    else:
        if q0 is None:
            q0 = 1.0 / (M + 1)
        if qm is None:
            qm = np.full(M, (1.0 - q0) / M, dtype=float)
        q = np.concatenate([[q0], np.asarray(qm, dtype=float)])
        q = _normalize_simplex(q, eps=eps)

    pi = (c + eta * q) / (C + eta)
    pi = _normalize_simplex(pi, eps=eps)
    return pi[0], pi[1:], pi


# ---------------------------------------------------------------------
# 3) Entropy-regularized update:
#       J = -sum c_i log pi_i + tau * sum pi_i log pi_i
#    KKT gives Lambert W form:
#       pi_i = c_i / (tau * W((c_i/tau) * exp(s)))
#    where s is found from sum_i pi_i = 1.
# ---------------------------------------------------------------------
def _pi_from_s_entropy(c, tau, s, eps=EPS):
    """Compute pi(s) from the Lambert-W closed form."""
    c = np.asarray(c, dtype=float)
    pi = np.zeros_like(c)
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if not SCIPY_OK:
        raise ImportError("scipy is required for the entropy vector update.")

    pos = c > 0
    arg = (c[pos] / tau) * np.exp(s)
    w = lambertw(arg)
    w = np.real(w)
    pi[pos] = c[pos] / (tau * w)
    pi[~pos] = 0.0
    pi = np.maximum(pi, eps)
    pi = pi / pi.sum()
    return pi



def update_pi_entropy(c0, cm, tau=1.0, eps=EPS, max_expand=200):
    """
    Entropy-regularized update for the free vector pi.

    Objective:
        min_pi  -sum_i c_i log pi_i + tau * sum_i pi_i log pi_i
        s.t.    pi in simplex

    Parameters
    ----------
    c0 : float
    cm : array-like, shape (M,)
    tau : float
        Entropy regularization strength.
    eps : float
    max_expand : int
        Maximum bracket expansion steps for bisection.

    Returns
    -------
    pi0 : float
    pim : ndarray
    pi  : ndarray
    s   : float
        Dual scalar in the Lambert-W expression.
    """
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if not SCIPY_OK:
        raise ImportError("scipy is required for update_pi_entropy.")

    cm = np.asarray(cm, dtype=float)
    c = np.concatenate([[float(c0)], cm])
    c = np.maximum(c, 0.0)
    if c.sum() <= 0:
        raise ValueError("Total soft count must be positive.")

    def f(s):
        # sum_i pi_i(s) - 1
        pos = c > 0
        vals = np.zeros_like(c)
        arg = (c[pos] / tau) * np.exp(s)
        w = np.real(lambertw(arg))
        vals[pos] = c[pos] / (tau * w)
        return vals.sum() - 1.0

    # f(s) is decreasing in s. Build a bracket [lo, hi] with sign change.
    lo, hi = -20.0, 20.0
    flo, fhi = f(lo), f(hi)

    n_expand = 0
    while flo < 0 and n_expand < max_expand:
        lo -= 10.0
        flo = f(lo)
        n_expand += 1

    n_expand = 0
    while fhi > 0 and n_expand < max_expand:
        hi += 10.0
        fhi = f(hi)
        n_expand += 1

    if flo < 0 or fhi > 0:
        raise RuntimeError("Failed to bracket the root for entropy update.")

    s_star = bisect(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200)
    pi = _pi_from_s_entropy(c, tau, s_star, eps=eps)
    return pi[0], pi[1:], pi, s_star


# ---------------------------------------------------------------------
# 4) Tied version: pi_m = (1 - pi0) / M
# ---------------------------------------------------------------------
def update_pi_tied_kl(c0, cm, eta=1.0, q0=None, eps=EPS):
    """
    Tied KL update with pi_m = (1-pi0)/M.

    Closed-form:
        pi0 = (c0 + eta q0) / (C + eta)
        pi_m = (1 - pi0) / M

    Parameters
    ----------
    c0 : float
    cm : array-like, shape (M,)
    eta : float
    q0 : float or None
        Prior on pi0. If None, uses uniform prior q0 = 1/(M+1).
    eps : float

    Returns
    -------
    pi0 : float
    pim : ndarray
    pi  : ndarray
    """
    cm = np.asarray(cm, dtype=float)
    M = len(cm)
    cin = np.maximum(cm, 0.0).sum()
    c0 = max(float(c0), 0.0)
    C = c0 + cin
    if C <= 0:
        raise ValueError("Total soft count must be positive.")
    if q0 is None:
        q0 = 1.0 / (M + 1)
    q0 = float(np.clip(q0, eps, 1.0 - eps))
    pi0 = (c0 + eta * q0) / (C + eta)
    pi0 = float(np.clip(pi0, eps, 1.0 - eps))
    pim = np.full(M, (1.0 - pi0) / M)
    pi = np.concatenate([[pi0], pim])
    pi = _normalize_simplex(pi, eps=eps)
    return pi[0], pi[1:], pi



def update_pi_tied_entropy(c0, cm, tau=1.0, eps=EPS):
    """
    Tied entropy update with pi_m = (1-pi0)/M.

    Solve the 1D equation on pi0 in (0,1):
        -c0/pi0 + cin/(1-pi0) + tau * log(M*pi0/(1-pi0)) = 0

    Parameters
    ----------
    c0 : float
    cm : array-like, shape (M,)
    tau : float
    eps : float

    Returns
    -------
    pi0 : float
    pim : ndarray
    pi  : ndarray
    """
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if not SCIPY_OK:
        raise ImportError("scipy is required for update_pi_tied_entropy.")

    cm = np.asarray(cm, dtype=float)
    M = len(cm)
    cin = np.maximum(cm, 0.0).sum()
    c0 = max(float(c0), 0.0)

    def g(pi0):
        pi0 = np.clip(pi0, eps, 1.0 - eps)
        return -c0 / pi0 + cin / (1.0 - pi0) + tau * np.log(M * pi0 / (1.0 - pi0))

    lo, hi = eps, 1.0 - eps
    glo, ghi = g(lo), g(hi)
    if not (glo < 0 and ghi > 0):
        raise RuntimeError("The tied entropy equation does not have a valid bracket.")

    pi0 = bisect(g, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200)
    pi0 = float(np.clip(pi0, eps, 1.0 - eps))
    pim = np.full(M, (1.0 - pi0) / M)
    pi = np.concatenate([[pi0], pim])
    pi = _normalize_simplex(pi, eps=eps)
    return pi[0], pi[1:], pi


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Example soft counts
    c0 = 12.0
    cm = np.array([20.0, 18.0, 15.0, 10.0, 25.0], dtype=float)

    print("=== Baseline MLE ===")
    pi0, pim, pi = update_pi_mle(c0, cm)
    print("pi0 =", pi0)
    print("pim =", pim)
    print("sum =", pi.sum())
    print()

    print("=== KL regularization (closed-form) ===")
    pi0, pim, pi = update_pi_kl(c0, cm, eta=5.0, q0=0.10)
    print("pi0 =", pi0)
    print("pim =", pim)
    print("sum =", pi.sum())
    print()

    if SCIPY_OK:
        print("=== Entropy regularization (free vector) ===")
        pi0, pim, pi, s_star = update_pi_entropy(c0, cm, tau=1.0)
        print("pi0 =", pi0)
        print("pim =", pim)
        print("s* =", s_star)
        print("sum =", pi.sum())
        print()

        print("=== Tied KL regularization ===")
        pi0, pim, pi = update_pi_tied_kl(c0, cm, eta=5.0, q0=0.10)
        print("pi0 =", pi0)
        print("pim =", pim)
        print("sum =", pi.sum())
        print()

        print("=== Tied entropy regularization ===")
        pi0, pim, pi = update_pi_tied_entropy(c0, cm, tau=1.0)
        print("pi0 =", pi0)
        print("pim =", pim)
        print("sum =", pi.sum())
    else:
        print("scipy not found: entropy-based examples are skipped.")


def expectation_otm_kl_ko(
    X, TY, sigma2,
    C=None, a=None, b=None, eps_c=1.0, rho_y=1.0, rho_x=1.0,
    u=None, v=None,
    uv_iter=200, uv_tol=1e-4,
    d2f=None, tau2=None,
    tau2_auto=False, eps=1e-9, tau2_alpha=1.0,
    feat_normal=['cos'], XF=None, YF=None, device=None,
    xp=th, DF=[1], use_keops=True,
):
    """
    KL-regularized separable-reference UOT:
        P = diag(a*u) K diag(b*v)

    where
        K = exp(-C/eps_c)               if external cost C is given,
        K = exp(S/eps_c)                if internal score S is built.

    Returns
    -------
    Pt1, P1, PX, Np, tau2s, u, v
    """

    if a is None or b is None:
        raise ValueError("a and b must be provided.")

    tiny = max(float(eps), 1e-12)
    epsv = max(float(eps_c), tiny)

    device = X.device if device is None else device
    dtype = X.dtype

    a = a.to(device=device, dtype=dtype)
    b = b.to(device=device, dtype=dtype)

    rho_y_f = float(rho_y)
    rho_x_f = float(rho_x)
    ry = 1.0 if math.isinf(rho_y_f) else rho_y_f / (rho_y_f + epsv)
    rx = 1.0 if math.isinf(rho_x_f) else rho_x_f / (rho_x_f + epsv)

    # ------------------------------------------------------------
    # Build logK
    # external true OT cost C:   logK = -C / eps
    # internal score S:          logK =  S / eps
    # ------------------------------------------------------------
    if C is None:
        if use_keops:
            logK = update_Cost_ko(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            ) / epsv
        else:
            logK = update_Cost_xp(
                X, TY, sigma2, xp=xp, d2f=d2f, tau2=tau2,
                tau2_auto=tau2_auto, eps=eps, tau2_alpha=tau2_alpha
            ) / epsv
    else:
        if use_keops and ('LazyTensor' in str(type(C))):
            logK = -C / epsv
        else:
            C = C.to(device=device, dtype=dtype)
            logK = -C / epsv
            if use_keops:
                logK = LazyTensor(logK[:, :, None].contiguous())

    K = logK.exp()
    is_lazy = ('LazyTensor' in str(type(K)))

    if u is None:
        u = th.ones_like(a)
    else:
        u = u.to(device=device, dtype=dtype).clone()

    if v is None:
        v = th.ones_like(b)
    else:
        v = v.to(device=device, dtype=dtype).clone()

    for _ in range(uv_iter):
        u_prev = u.clone()
        v_prev = v.clone()

        if is_lazy:
            bv = b * v
            bv_j = LazyTensor(bv[None, :, None])                  # (1, N, 1)
            Kbv = (K * bv_j).sum(dim=1).flatten().clamp_min(tiny)
            u = Kbv.pow(-ry)

            au = a * u
            au_i = LazyTensor(au[:, None, None])                  # (M, 1, 1)
            KTau = (K * au_i).sum(dim=0).flatten().clamp_min(tiny)
            v = KTau.pow(-rx)

        else:
            bv = b * v
            Kbv = (K @ bv).clamp_min(tiny)
            u = Kbv.pow(-ry)

            au = a * u
            KTau = (K.transpose(0, 1) @ au).clamp_min(tiny)
            v = KTau.pow(-rx)

        err = max(
            (u - u_prev).abs().max().item(),
            (v - v_prev).abs().max().item(),
        )
        if err < uv_tol:
            break

    au = a * u
    bv = b * v

    if is_lazy:
        au_i = LazyTensor(au[:, None, None])                      # (M,1,1)
        bv_j = LazyTensor(bv[None, :, None])                      # (1,N,1)
        X_j  = LazyTensor(X[None, :, :])                          # (1,N,D)

        P1  = au * (K * bv_j).sum(dim=1).flatten()                # (M,)
        Pt1 = bv * (K * au_i).sum(dim=0).flatten()                # (N,)
        PX  = (au_i * K * bv_j * X_j).sum(dim=1)                  # (M,D)

    else:
        P1  = au * (K @ bv)                                       # (M,)
        Pt1 = bv * (K.transpose(0, 1) @ au)                       # (N,)
        PX  = au[:, None] * (K @ (bv[:, None] * X))               # (M,D)

    Np = xp.sum(P1)

    if tau2_auto and (XF is not None) and (YF is not None):
        tau2s = update_tau2_otm(
            Pt1, P1, (au, K, bv), Np, XF, YF,
            feat_normal=feat_normal, DK=DF, device=device, xp=xp
        )
    else:
        tau2s = tau2

    return Pt1, P1, PX, Np, tau2s, u, v