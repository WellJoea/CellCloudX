import numpy as np
from numpy.linalg import norm
from copy import copy
import cvxpy as cp
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

np.random.seed(999)


smallest_float = np.nextafter(np.float32(0), np.float32(1))
float_type = np.longdouble

def max_norm(x):
    return np.amax(np.abs(x))

def compute_entropy(P):
    logP = np.log(P + 1e-20)
    return -1 * np.sum(logP * P - P)


def compute_KL(P, Q):
    log_ratio = np.log(P) - np.log(Q)
    return np.sum(P * log_ratio - P + Q)


def dot(x, y):
    return np.sum(x * y)


def compute_B(C, u, v, eta):
    return np.exp((u + v.T - C) / eta)


def compute_f(C, X, a, b, tau1, tau2):
    Xa = X.sum(axis=1).reshape(-1, 1)
    Xb = X.sum(axis=0).reshape(-1, 1)

    return dot(C, X) + tau1 * compute_KL(Xa, a) + tau2 * compute_KL(Xb, b)


def compute_g_primal(C, u, v, a, b, eta, tau1, tau2):
    B = compute_B(C, u, v, eta)

    Ba = B.sum(axis=1).reshape(-1, 1)
    Bb = B.sum(axis=0).reshape(-1, 1)

    return dot(C, B) + tau1 * compute_KL(Ba, a) + tau2 * compute_KL(Bb, b) - eta * compute_entropy(B)


def compute_g_dual(C, u, v, a, b, eta, tau1, tau2):
    B = compute_B(C, u, v, eta)
    f = eta * np.sum(B) + tau1 * dot(np.exp(- u / tau1), a) + tau2 * dot(np.exp(- v / tau2), b)

    return f


def solve_g_dual_cp(C, a, b, eta, tau):
    dim_a = a.shape[0]
    dim_b = b.shape[0]

    u = cp.Variable(shape=a.shape)
    v = cp.Variable(shape=b.shape)

    u_stack = cp.vstack([u.T for _ in range(dim_b)])
    v_stack = cp.hstack([v for _ in range(dim_a)])

    obj = eta * cp.sum(cp.exp((u_stack + v_stack - C) / eta))
    obj += tau * cp.sum(cp.multiply(cp.exp(- u / tau), a))
    obj += tau * cp.sum(cp.multiply(cp.exp(- v / tau), b))

    prob = cp.Problem(cp.Minimize(obj))
    # prob.solve(solver=cp.SCS, eps=smallest_float)
    prob.solve()

    return prob.value, u.value, v.value


def solve_f_cp(C, a, b, tau1=1.0, tau2=1.0):
    """
    Convex programming solver for standard Unbalanced Optimal Transport.

    :param C:
    :param a:
    :param b:
    :param tau1:
    :param tau2:
    :return:
    """

    X = cp.Variable((a.shape[0], b.shape[0]), nonneg=True)

    row_sums = cp.sum(X, axis=1)
    col_sums = cp.sum(X, axis=0)

    obj = cp.sum(cp.multiply(X, C))

    obj -= tau1 * cp.sum(cp.entr(row_sums))
    obj -= tau2 * cp.sum(cp.entr(col_sums))

    obj -= tau1 * cp.sum(cp.multiply(row_sums, cp.log(a.reshape(-1, ))))
    obj -= tau2 * cp.sum(cp.multiply(col_sums, cp.log(b.reshape(-1, ))))

    obj -= (tau1 + tau2) * cp.sum(X)
    obj += tau1 * cp.sum(a.reshape(-1, )) + tau2 * cp.sum(b.reshape(-1, ))

    prob = cp.Problem(cp.Minimize(obj))
    # prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-6)
    prob.solve()

    return prob.value, X.value
    

def sinkhorn_uot(C, a, b, eta=1.0, tau1=1.0, tau2=1.0, k=100, compute_optimal=True):
    """
    Sinkhorn algorithm for entropic-regularized Unbalanced Optimal Transport.

    :param C:
    :param a:
    :param b:
    :param eta:
    :param tau1:
    :param tau2:
    :param k:
    :param epsilon:
    :return:
    """

    output = {
        "u": list(),
        "v": list(),
        "f": list(),
        "g_dual": list()
    }

    # Compute optimal value and X for unregularized UOT
    if compute_optimal:
        f_optimal, X_optimal = solve_f_cp(C, a, b, tau1=tau1, tau2=tau2)
        output["f_optimal"] = f_optimal
        output["X_optimal"] = X_optimal

    # Initialization
    u = np.zeros_like(a).astype(float_type)
    v = np.zeros_like(b).astype(float_type)

    output["u"].append(copy(u))
    output["v"].append(copy(v))

    # # Compute initial value of f
    # B = compute_B(C, u, v, eta)
    # f = compute_f_primal(C=C, X=B, a=a, b=b, tau1=tau1, tau2=tau2)
    # output["f"].append(f)

    for i in range(k):
        u_old = copy(u)
        v_old = copy(v)
        B = compute_B(C, u, v, eta)

        f = compute_f(C=C, X=B, a=a, b=b, tau1=tau1, tau2=tau2)

        output["f"].append(f)

        # Sinkhorn update
        if i % 2 == 0:
            Ba = B.sum(axis=1).reshape(-1, 1)
            u = (u / eta + np.log(a) - np.log(Ba)) * (tau1 * eta / (eta + tau1))
        else:
            Bb = B.sum(axis=0).reshape(-1, 1)
            v = (v / eta + np.log(b) - np.log(Bb)) * (tau2 * eta / (eta + tau2))

        g_dual = compute_g_dual(C=C, u=u, v=v, a=a, b=b, eta=eta, tau1=tau1, tau2=tau2)

        output["u"].append(copy(u))
        output["v"].append(copy(v))
        output["g_dual"].append(g_dual)

        # err = norm(u - u_old, ord=1) + norm(v - v_old, ord=1)

        # if err < 1e-10:
        #     break
        #
        # if np.abs(f - output["f_optimal"]) < epsilon:
        #     break

    return output


def find_k_sinkhorn(C, a, b, epsilon, f_optimal, eta=1.0, tau1=1.0, tau2=1.0, momentum=100, max_trial=100000):
    # Initialization
    u = np.zeros_like(a)
    v = np.zeros_like(b)

    i = 0
    count = 0
    trial = 0
    start_trial = False

    while True:
        B = compute_B(C, u, v, eta)

        f_primal = compute_f(C=C, X=B, a=a, b=b, tau1=tau1, tau2=tau2)

        # Sinkhorn update
        if i % 2 == 0:
            Ba = B.sum(axis=1).reshape(-1, 1)
            u = (u / eta + np.log(a) - np.log(Ba)) * (tau1 * eta / (eta + tau1))
        else:
            Bb = B.sum(axis=0).reshape(-1, 1)
            v = (v / eta + np.log(b) - np.log(Bb)) * (tau2 * eta / (eta + tau2))

        if np.abs(f_primal - f_optimal) < epsilon:
            # print(f"{i} ------ {np.abs(f_primal - f_optimal)}")
            if count == 0:
                true_flag = i

            counting_flag = i
            count += 1
            start_trial = True

            if count > momentum:
                return true_flag

            # if start_trial:
            #     print("X.")
        else:
            if start_trial:
                counting_flag = -np.inf
                count = 0
                trial += 1

        if trial > max_trial:
            break

        i += 1

    return counting_flag


range_a = 10
range_b = 10
range_C = 100
dim_a = 100
dim_b = 100
eta = 0.05
tau1 = 1.0
tau2 = 1.0
k = 500

C = np.random.uniform(low=1, high=range_C, size=(dim_a, dim_b)).astype("float128")
C = (C + C.T) / 2
a = np.random.uniform(low=0.1, high=range_a, size=(dim_a, 1)).astype("float128")
b = np.random.uniform(low=0.1, high=range_b, size=(dim_b, 1)).astype("float128")

n = dim_a
tau = tau1

alpha = np.sum(a)
beta = np.sum(b)

exp_name = f"[n={dim_a}]_[tau={'{0:.1f}'.format(np.mean(tau1))}]_[rC={range_C}]_[ra={range_a}]_[rb={range_b}]_[eta={'{0:.2f}'.format(np.mean(eta))}]"

g_dual_optimal, u_optimal, v_optimal = solve_g_dual_cp(C=C, a=a, b=b, eta=eta, tau=tau)
output = sinkhorn_uot(C=C, a=a, b=b, eta=eta, tau1=tau1, tau2=tau2, k=k)

delta_u = [max_norm(u - u_optimal) for u in output["u"]]
delta_v = [max_norm(v - v_optimal) for v in output["v"]]

"""
PLOTTING
"""
fig, axs = plt.subplots(2, 2, figsize=(18, 12))
plt.subplots_adjust(hspace=0.5)

fig.suptitle(exp_name)

axs[0, 0].plot(list(range(k)), output["f"], "r", label="f")
axs[0, 0].plot([0, k - 1], [output["f_optimal"], output["f_optimal"]], "b", label="f optimal")
axs[0, 0].set_xlabel("k (iterations)")
axs[0, 0].set_title("f")
axs[0, 0].legend()
axs[0, 0].text(x=0.5, y=0.7, s=f"min_f={'{0:.2f}'.format(min(output['f']))}\noptimal_f={'{0:.2f}'.format(output['f_optimal'])}", horizontalalignment='center', verticalalignment='center', transform=axs[0, 0].transAxes)

b1 = 500
b2 = 750
axs[0, 1].plot(list(range(k)), output["g_dual"], "r", label="g dual")
axs[0, 1].plot([0, k - 1], [g_dual_optimal, g_dual_optimal], "b", label="g dual optimal")
axs[0, 1].set_xlabel("k (iterations)")
axs[0, 1].set_title("g dual")
axs[0, 1].legend()
axs[0, 1].text(x=0.5, y=0.7, s=f"min_g={'{0:.2f}'.format(min(output['g_dual']))}\noptimal_g={'{0:.2f}'.format(g_dual_optimal)}", horizontalalignment='center', verticalalignment='center', transform=axs[0, 1].transAxes)

axs[1, 0].plot(list(range(k + 1)), delta_u, "r", label="f optimal")
axs[1, 0].set_xlabel("k (iterations)")
axs[1, 0].set_title(r"$\| u - u^* \|_\infty$")
axs[1, 0].text(x=0.7, y=0.7, s=f"min={'{0:.2f}'.format(delta_u[-1])}", horizontalalignment='center', verticalalignment='center', transform=axs[1, 0].transAxes)

axs[1, 1].plot(list(range(k + 1)), delta_v, "r", label="f optimal")
axs[1, 1].set_xlabel("k (iterations)")
axs[1, 1].set_title(r"$\| v - v^* \|_\infty$")
axs[1, 1].text(x=0.7, y=0.7, s=f"min={'{0:.2f}'.format(delta_v[-1])}", horizontalalignment='center', verticalalignment='center', transform=axs[1, 1].transAxes)

plt.show()

B_star = compute_B(C=C, u=u_optimal, v=v_optimal, eta=eta)
a_star = np.sum(B_star, axis=1, keepdims=True)
b_star = np.sum(B_star, axis=0, keepdims=True).T

R = max(max_norm(np.log(a)), max_norm(np.log(b))) + max(np.log(n), 1 / eta * max_norm(C) - np.log(n))