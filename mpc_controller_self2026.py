import numpy as np
import cvxpy as cp

def mpc_controller(i, A, B, init, Np, U_ref, Q, R, U_min, U_max):
    # A_bar
    A_bar = np.zeros((3 * Np, 3))
    A_k = A(i)
    A_bar[0:3, :] = A_k
    for j in range(1, Np):
        A_k = A_k @ A(i + j)
        A_bar[3*j:3*(j+1), :] = A_k

    # B_bar
    B_bar = np.zeros((3 * Np, 2 * Np))
    for k in range(Np):
        for l in range(k + 1):
            B_bar_kl = np.eye(3)
            for m in range(l, k):
                B_bar_kl = B_bar_kl @ A(i + m + 1)
            B_bar_kl = B_bar_kl @ B(i + l)
            B_bar[3*k:3*(k+1), 2*l:2*(l+1)] = B_bar_kl

    H_k = 2 * (B_bar.T @ Q @ B_bar + R)
    H_k = 0.5 * (H_k + H_k.T)
    f_k = 2 * B_bar.T @ Q @ A_bar @ init.reshape(3, 1)
    # Constraints: U_min <= U_ref + delta_U <= U_max
    
    d = np.zeros((4 * Np, 1))
    for n in range(Np):
        d[2*n:2*(n+1), :] = U_max.reshape(2,1) - U_ref[n, :].reshape(2,1)
        d[2*(n+Np):2*(n+Np+1), :] = U_ref[n, :].reshape(2,1) - U_min.reshape(2,1)

    I = np.eye(2 * Np)
    D = np.vstack((I, -I))

    delta_U = cp.Variable((2 * Np, 1))
    objective = cp.Minimize(0.5 * cp.quad_form(delta_U, H_k) + f_k.T @ delta_U)
    constraints = [D @ delta_U <= d]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if delta_U.value is None:
        return 0.0, 0.0

    U = delta_U.value[0:2, :] + U_ref[0, :].reshape(2, 1)
    vel = float(U[0, 0])
    steering = float(U[1, 0])

    return vel, steering
