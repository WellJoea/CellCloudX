# _OTM.py 算法公式推导与代码一致性核对

> 行号以当前 `cellcloudx/alignment/m3lib/_OTM.py`（1104 行版）为准；`_OTM_fixed.py` 仅在第 416–421 行区间（feat_normal）不同，其余完全一致。

---

## §1 总体问题设定

每步求解的非平衡最优传输（UOT）子问题：

$$\min_{P\ge 0}\;\underbrace{\langle C,P\rangle}_{\text{线性代价}} + \varepsilon\,\Omega(P) + \rho_x\,\mathrm{KL}(P\mathbf{1}\,\|\,a) + \rho_y\,\mathrm{KL}(P^{\top}\mathbf{1}\,\|\,b)$$

- $C = D_{\mathrm{eff}}$：有效代价（一阶代价 + 可选的 FGW 梯度修正），见 §6。
- $\Omega$：`reg='entropy'` 时为香农熵 $H(P)$；`reg='kl'` 时为 $\mathrm{KL}(P\|a\otimes b)$。
- $a,b$：行/列质量预算（默认全 1 向量，不是归一化直方图）。
- 返回的 $P$ 是**运输质量**，$r=P\mathbf{1}$、$c=P^{\top}\mathbf{1}$ 是其边缘。

---

## §2 一阶代价与核矩阵

$$D_{1,ij}=\frac{g_1}{\sigma_1^2}\|x_i-y_j\|_2^2+\sum_k\frac{f_k}{\tau_k^2}\|F^k x_i-F^k y_j\|_2^2$$

**代码核对**：`D1_cost_agg`（L540–549）
`D1st = (gscale_1st/sigma2_1st)*dist2(X,Y) + Σ_k (fscales_1st[k]/tau2s_1st[k])*dist2(Fx[k],Fy[k])` ✓ 逐项对应，权重为 0 的项被跳过（`if gscale_1st>0`、`if fscales_1st[i]>0`）。

核矩阵（熵正则情形）：$K_{ij}=e^{-D_{ij}/\varepsilon}$。

---

## §3 两种正则的等价性（KL 参考测度 ⇒ 修改代价的熵正则）

$$\mathrm{KL}(P\|ab^{\top})=\sum_{ij}\Big[P_{ij}\log\frac{P_{ij}}{a_i b_j}-P_{ij}+a_i b_j\Big]$$

展开：

$$\mathrm{KL}(P\|ab^{\top})=\sum P_{ij}\log P_{ij}-\sum P_{ij}\log a_i-\sum P_{ij}\log b_j-\sum P_{ij}+\sum a_i b_j$$

因此

$$\langle C,P\rangle+\varepsilon\,\mathrm{KL}(P\|ab^{\top})=\langle C',P\rangle+\varepsilon H(P)+\mathrm{const},\qquad
C'_{ij}=C_{ij}-\varepsilon\log a_i-\varepsilon\log b_j$$

**结论**：KL 情形等价于熵正则 + 修改代价 $C'$，对应核

$$\tilde K_{ij}=e^{-C'_{ij}/\varepsilon}=e^{-C_{ij}/\varepsilon}\,a_i\,b_j$$

**代码核对**：`K_effective`（L592–615）
- log 域：`logK = -D/eps; if reg=='kl': logK += loga[:,None] + logb[None,:]`（L605–606）✓
- 普通域：`K *= a[:,None] * b[None,:]`（L613–614）✓
- 推论：$a=b=\mathbf{1}$ 时 $\log a=\log b=0$，两种正则**完全等价**——文档声称一致，数值验证（T5）通过 ✓

---

## §4 最优性条件（驻点方程）

$$J(P)=\langle C,P\rangle+\varepsilon H(P)+\rho_x\mathrm{KL}(P\mathbf{1}\|a)+\rho_y\mathrm{KL}(P^{\top}\mathbf{1}\|b)$$

$$H(P)=\sum P_{ij}(\log P_{ij}-1),\qquad
\frac{\partial H}{\partial P_{ij}}=\log P_{ij};\qquad
\mathrm{KL}(P\mathbf{1}\|a)=\sum_i\big[(P\mathbf{1})_i\log\frac{(P\mathbf{1})_i}{a_i}-(P\mathbf{1})_i+a_i\big],\quad
\frac{\partial\,\mathrm{KL}(P\mathbf{1}\|a)}{\partial P_{ij}}=\log(P\mathbf{1})_i-\log a_i$$

驻点条件：

$$C_{ij}+\varepsilon\log P_{ij}+\rho_x\big(\log(P\mathbf{1})_i-\log a_i\big)+\rho_y\big(\log(P^{\top}\mathbf{1})_j-\log b_j\big)=0$$

代入分解形式 $P_{ij}=u_i K_{ij}v_j$，$K_{ij}=e^{-C_{ij}/\varepsilon}$：

$$\log P_{ij}=\log u_i+\log v_j-\frac{C_{ij}}{\varepsilon},\qquad
\log(P\mathbf{1})_i=\log u_i+\log(Kv)_i,\qquad
\log(P^{\top}\mathbf{1})_j=\log v_j+\log(K^{\top}u)_j$$

代入后 $C_{ij}$ 与 $-\varepsilon\cdot(C_{ij}/\varepsilon)$ 抵消，方程分离为行/列两族：

$$\underbrace{\varepsilon\log u_i+\rho_x\big(\log u_i+\log(Kv)_i-\log a_i\big)}_{=0}+\underbrace{\varepsilon\log v_j+\rho_y\big(\log v_j+\log(K^{\top}u)_j-\log b_j\big)}_{=0}=0$$

行方程给出：

$$(\varepsilon+\rho_x)\log u_i=\rho_x\big(\log a_i-\log(Kv)_i\big)\quad\Longrightarrow\quad
\log u_i = t_x\big(\log a_i-\log(Kv)_i\big),\qquad t_x=\frac{\rho_x}{\rho_x+\varepsilon}$$

普通域等价地 $u_i=\big(a_i/(Kv)_i\big)^{t_x}$；列同理 $v_j=\big(b_j/(K^{\top}u)_j\big)^{t_y}$。

**代码核对**：
- 指数 $t_x,t_y$：`init_config` L262–263 `tx,ty = rho/(rho+eps)` ✓
- log 域更新：`sinkhorn_log` L801 `logu = tx*(loga - Kv)` ✓
- 普通域更新：`sinkhorn_normal` L732 `u = (a/Kv.clamp_min(tiny)).pow(tx)`（clamp 仅防除零）✓

---

## §5 Sinkhorn 交替迭代与收敛判据

**迭代格式**（非平衡 Sinkhorn，Chizat et al. 2018）：固定 $v$ 精确解行方程、固定 $u$ 精确解列方程，交替进行。

普通域（L728–735）：`u=(a/Kv)^tx → Ktu=Kᵀu → v=(b/Ktu)^ty → Kv=K·v` ✓
log 域（L797–804）：`logu=tx(loga−Kv) → Ktu=logsumexp(logK+logu,0) → logv=ty(logb−Ktu) → Kv=logsumexp(logK+logv,1)` ✓

**检查时刻**：`Kv` 用新 $v$、`Ktu` 用新 $u$，因此"下一轮 u"公式
$$\mathrm{nextu}_i=\big(a_i/(Kv)_i\big)^{t_x}$$
与当前 $u$（由上一轮 $v$ 算出）之差就是一步前瞻固定点残差——驻点处二者相等。

**残差判据推导**：
- 正常域（`_uv_conver_normal` L744–781）：`err_u = max|u−nextu|/max(|u|,|nextu|)`（L769–770），`err_v` 同理（L777–778）。`step` 只查行；`hybrid` 先行后列；`kkt = max(err_u,err_v)` ✓
- log 域（`_uv_conver_log` L815–855）：
$$e^x_i=\varepsilon\log u_i+\rho_x(\log u_i+\log(Kv)_i-\log a_i)=(\varepsilon+\rho_x)\big[\log u_i-t_x(\log a_i-\log(Kv)_i)\big]$$
即 $(\varepsilon+\rho_x)\times$（当前与下一轮 logu 之差），驻点处为 0 ✓（L838）。$e^y$ 同理（L847）。
- log-kkt 用原始梯度的 sup 范数：
$$\sup_{i,j}|e^x_i+e^y_j|=\max\Big(\big|\max_i e^x_i+\max_j e^y_j\big|,\;\big|\min_i e^x_i+\min_j e^y_j\big|\Big)$$
（$e^x_i+e^y_j$ 的最大值在各自最大值处取到，最小值在各自最小值处取到）——代码 L849–852 完全一致 ✓。除以 `scale=max(1,eps,rho_x,rho_y)` 为量纲化处理（文档注明，合理）。
- `pot/rel/rel_mean/log/log_mean` 为迭代变化量判据（启发式，L752–764、L822–835）。

---

## §6 FGW 能量与梯度

结构项（文档采用 half-sum 约定）：

$$G(P)=\frac12\sum_{i,j,k,l}\big(Cx_{ik}-Cy_{jl}\big)^2 P_{ij}P_{kl},
\qquad Cx_{ik}=\frac{\|x_i-x_k\|^2}{\sigma_2^2},\;Cy_{jl}=\frac{\|y_j-y_l\|^2}{\sigma_2^2}$$

**中心化不改变两两距离**：$(x_i-m)-(x_k-m)=x_i-x_k$，故 `centerlize=True` 只改善数值条件，不改变 $Cx$（L517–520）✓。

**展开**（$r=P\mathbf{1},\;c=P^{\top}\mathbf{1}$）：

$$(Cx_{ik}-Cy_{jl})^2=Cx_{ik}^2-2Cx_{ik}Cy_{jl}+Cy_{jl}^2$$

- 第一项：$\frac12\sum_{ik}Cx_{ik}^2(\sum_j P_{ij})(\sum_l P_{kl})=\frac12 r^{\top}(Cx\circ Cx)r$
- 第三项：$\frac12 c^{\top}(Cy\circ Cy)c$
- 第二项：$\sum_{ijkl}Cx_{ik}P_{ij}Cy_{jl}P_{kl}=\mathrm{tr}(Cx\,P\,Cy\,P^{\top})$

**梯度**：

$$\frac{\partial}{\partial P_{ij}}\Big[\tfrac12 r^{\top}Cx^2 r\Big]=\sum_k Cx_{ik}^2 r_k=(Cx^2 r)_i\qquad(\text{对称性使两项合并})$$

$$\frac{\partial}{\partial P}\mathrm{tr}(Cx P Cy P^{\top})=Cx P Cy+Cx^{\top}P Cy^{\top}=2\,Cx P Cy$$

$$\Longrightarrow\quad \nabla_P G=\underbrace{Cx^2 r}_{\text{dx}}\mathbf{1}^{\top}+\mathbf{1}\underbrace{(Cy^2 c)^{\top}}_{\text{dy}}-2\,Cx P Cy$$

**代码核对**：`D2_fgw_gradient`（L508–538）
- `dx=_moment(X,...,r)` = $Cx^2 r$（L526/529），`dy` = $Cy^2 c$；
- `cross=psix.T@(P@psiy)`（L523），`left=phix@cross, right=phiy`（L525）→ `left@rightᵀ = Cx P Cy`；
- 返回 `dx[:,None] + dy[None,:] − 2*dot(left,right)`（L538）✓ 与上式逐项一致。
- 系数 (1, 1, −2) 恰是 half-sum 约定；数值上与该显式能量的 autograd 完全一致（误差 ~1e-12，验证 T3）✓

---

## §7 距离因子分解与低秩矩恒等式

因子：$\phi=[q,\mathbf{1},Z]$，$\psi=[\mathbf{1},q,-2Z]$，$q=\|Z_i\|^2$（按行）。

$$\phi_i\cdot\psi_k=q_i+q_k-2\langle Z_i,Z_k\rangle=\|Z_i-Z_k\|^2=C_{ik}$$

**代码核对**：`_factors`（L475–482）✓（数值验证 T1：与 `cdist²` 误差 ~1e-15）。

**矩恒等式** $C^2m=(C\circ C)m$：

$$(C\circ C)_{ik}=(\phi_i\cdot\psi_k)^2=\Big(\sum_a\phi_{ia}\psi_{ka}\Big)\Big(\sum_b\phi_{ib}\psi_{kb}\Big)$$

$$\sum_k m_k C_{ik}^2=\sum_{ab}\phi_{ia}\phi_{ib}\Big[\sum_k m_k\psi_{ka}\psi_{kb}\Big]=\phi_i\,\big[\psi^{\top}\mathrm{diag}(m)\psi\big]\,\phi_i^{\top}$$

**代码核对**：`_moment`（L484–491）
`((phi @ (psi.T @ (mass[:,None]*psi))) * phi).sum(1)` ✓ 与上式一致；$d>\text{moment\_max\_dim}$ 时改用直接归约 `dist2(Z,Z)²@mass`（L491）——数学恒等、仅计算路线不同（文档注明，验证 T2 通过）。

---

## §8 预测（transform）的代价推导

### 8.1 kernel 条件重心

拟合计划 $P_{ij}=u_i K_{ij} v_j$。给定查询 $i$ 的条件分布：

$$p(j|i)=\frac{P_{ij}}{(P\mathbf{1})_i}=\frac{K_{ij}v_j}{(Kv)_i}$$

核含标签与参考测度：$K_{ij}=e^{-D_{ij}/\varepsilon}H_{ij}\,[a_i b_j]$（kl 时）。行归一化中与 $i$ 有关的因子 $u_i,a_i$ 消去：

$$p(j|i)\propto b_j v_j H_{ij}e^{-D_{ij}/\varepsilon}\qquad\Longrightarrow\qquad
\log w_{ij}=-\frac{D_{ij}}{\varepsilon}+\log v_j+\big[\log b_j\big]$$

而标签代价已并入 $D$：$D\leftarrow D-\varepsilon\log H$（见 §9），故

$$\log w_{ij}=-\frac{D_{1,ij}+[\text{FGW 项}]}{\varepsilon}+\log H_{ij}+\log v_j+\big[\log b_j\big]$$

重心映射：$\hat y_i=\sum_j p(j|i)\,y_j=\mathrm{softmax}_j(\log w_{ij})\,Y$。

**代码核对**：`transform` L1037–1039（`offset` 取 anchor 侧 `logu/logv`，kl 时加 `loga/logb`）、L1053–1057（`logw=-D/eps+offset`，softmax 乘 `Recenter`）✓。与闭式条件重心数值一致（验证 T8，误差 ~1e-12）。

### 8.2 nearest / knn

$$\hat y=x+\sum_{k}w_k\big(\hat y_{\mathrm{train}}(i_k)-x_{\mathrm{train}}(i_k)\big),\qquad
w=\mathrm{softmax}\Big(-\frac{d_k-\min_k d_k}{\mathrm{temp}}\Big)$$

减去行最小值是 softmax 平移不变性的数值稳定写法。代码 L1059–1066 ✓（验证 T8 通过）。

### 8.3 FGW 预测项（行常数省略）

查询行 $i$ 加入后，由 §6：$\partial G/\partial P_{ij}=(Cx^2r)_i+(Cy^2c)_j-2(Cx P_0 Cy)_{ij}$。
对固定查询 $i$，$(Cx^2r)_i$ 与 $j$ 无关 → 在行 softmax 中**抵消**，只需保留

$$(Cy^2c)_j-2\,(Cx P_0 Cy)_{ij}$$

其中 $(Cx P_0 Cy)_{ij}=\phi_q(i)\,\underbrace{\psi_x^{\top}P_0\psi_y}_{B}\,\phi_y(j)^{\top}$，$B$ 在拟合时缓存（`_fgw_last`/`_fgw_prediction`）。

**代码核对**：`_prediction_fgw`（L1088–1104）
- side=0：`other=dy`（=Cy²c），B 不变 → `dy[None,:] − 2·dot(phiq@B, phir)` ✓
- side=1：交换 X↔Y、B→Bᵀ、other=dx（=Cx²r），保留 $(Cx^2r)_i-2(CxP_0Cy)_{ij}$ ✓
- 与"增广能量对查询行/列质量的 autograd 梯度（差一个行/列常数）"数值一致（验证 T4，误差 ~1e-12）✓

---

## §9 标签核（普通域与 log 域）

普通域：$H_{ij}=(1-\delta)(Lx\,Ly^{\top})_{ij}+\delta$

**代码核对**：`H_label`（L617–644）`H = lh*A@B.T + dh`（keops 分支等价）✓

log 域目标：$\log\big[(1-\delta)\sum_k A_{ik}B_{jk}+\delta\big]$。构造增广列：

$$\log A'=\big[\log A+\tfrac12\log(1-\delta)\;\big|\;\tfrac12\log\delta\big],\qquad \log B'\text{ 同理}$$

则

$$\mathrm{logsumexp}_k(\log A'_{ik}+\log B'_{jk})
=\log\Big[\sum_k e^{\log A_{ik}+\log B_{jk}+\log(1-\delta)}+e^{\log\delta}\Big]
=\log\big[(1-\delta)AB^{\top}+\delta\big]$$

**代码核对**：`H_label_log`（L646–673）：L660–662 的增广列构造、L663–673 的 logsumexp 归约 ✓（数值验证 T6：与普通域 exp 后误差 ~1e-15）。keops 分支的 `shift+(Z-shift).exp().sum().log()` 是稳定的 logsumexp 写法 ✓。

**标签代价等价性**：$K=e^{-D/\varepsilon}H=e^{-(D-\varepsilon\log H)/\varepsilon}$ ⇒ 标签等价于额外代价 $-\varepsilon\log H$，与 `_prediction_cost` L1085–1086（`D -= eps*H_label_log(...)`）一致 ✓。

备选实现 `H_label_log0`（L675–708）用 softplus 恒等式：

$$\log(dh+lh\,e^x)=\log dh+\log\big(1+e^{x+\log lh-\log dh}\big)=\log dh+\mathrm{relu}(t)+\log\big(1+e^{-|t|}\big),\quad t=x+\log lh-\log dh$$

代码 L705–706 一致 ✓（未被调用，仅为备选）。

---

## §10 尺度估计器

$$\sigma^2(X,Y)=\overline{X\circ X}+\overline{Y\circ Y}-2\,\overline{\bar X\circ\bar Y}$$

恒等式：$\sum_{ij}\|x_i-y_j\|^2=m\sum_i\|x_i\|^2+n\sum_j\|y_j\|^2-2nm\,\bar x\cdot\bar y$，且 $\overline{X\circ X}=\frac{1}{nd}\sum_i\|x_i\|^2$、$\overline{\bar X\circ\bar Y}=\frac{1}{d}\bar x\cdot\bar y$，故

$$\sigma^2=\frac{1}{d}\cdot\frac{1}{nm}\sum_{ij}\|x_i-y_j\|^2=\text{每维度平均成对平方距离}$$

**代码核对**：`operation_expectation.sigma_square`（L225–236）✓——作为 `dist2/σ²` 的标量尺度正确（dist2 对 d 维求和）。数值验证 T11 通过。
`sigma_square_cos`（L209–223）乘 $d/2$：行归一化特征下 $\overline{X\circ X}=1/d$，故 $\sigma^2_{\cos}=1-\bar x\cdot\bar y=1-\text{平均余弦相似度}\in[0,2]$，是合理尺度 ✓。

---

## §11 一致性结论汇总

| 公式 | 推导结论 | 代码位置 | 一致性 |
|---|---|---|---|
| 一阶代价 D1 | 加权平方距离和 | `D1_cost_agg` L540–549 | ✓ |
| KL 参考测度 ⇒ 核乘 a⊗b | $K'=e^{-C/\varepsilon}a b^{\top}$ | `K_effective` L605–606, 613–614 | ✓（T5 验证） |
| UOT 驻点方程 | $\log u=t_x(\log a-\log Kv)$，$t_x=\rho/(\rho+\varepsilon)$ | L262–263, L728–732, L797–801 | ✓ |
| Sinkhorn 交替迭代 | 行/列精确解交替 | L728–735, L797–804 | ✓（T5 不动点残差 <1e-7） |
| 收敛残差（log-kkt） | sup 范数 $\max(\|\max e^x+\max e^y\|,\|\min e^x+\min e^y\|)$ | L838–852 | ✓ |
| FGW 梯度 | $Cx^2r\,\mathbf{1}^{\top}+\mathbf{1}(Cy^2c)^{\top}-2CxPCy$ | L508–538 | ✓（T3 autograd 1e-12） |
| 因子分解 | $\phi\psi^{\top}=\|Z_i-Z_k\|^2$ | L475–482 | ✓（T1） |
| 矩 $C^2m$ | $\phi[\psi^{\top}\mathrm{diag}(m)\psi]\phi^{\top}$ | L484–491 | ✓（T2） |
| kernel 条件重心 | $\mathrm{softmax}(-D/\varepsilon+\log v+\log b)\,Y$ | L1037–1057 | ✓（T8） |
| FGW 预测行常数省略 | 保留 $Cy^2c-2CxP_0Cy$（双向） | L1088–1104 | ✓（T4） |
| 标签 log 混合 | logsumexp 增广列 | L646–673 | ✓（T6） |
| 尺度 σ² | 每维度平均成对平方距离 | `sigma_square` | ✓（T11） |

**唯一发现的代码-公式不一致**：`feat_normal` 预测路径对 `cos/ccos/zc` 复用训练行统计量并错误展开参数（已在 `_OTM_fixed.py` 修复，见主报告）。
