import numpy as np
import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev, nn
from jax import config
from cyipopt import minimize_ipopt

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)
# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
config.update('jax_platform_name', 'cpu')
# z = (x0(t0) .... x0(tN) x1(t0) .... x1(tN) 
#      x2(t0) .... x2(tN) y(t0) .... y(tN)
#      u0(t0) .... u0(tN) u1(t0) .... u1(tN) T)

# time grid
N = 200

# initial point
x0i = 0.
x1i = 0.
x2i = 0.
yi = 0.

# final point
x0f = -1.
x1f = 1.
x2f = - np.pi / 2

# until : always avoid the zone
x0c_1 = -0.5
x1c_1 = 0.
R_1 = 0.3

# eventually : enter the zone at least one time
x0c_2 = 0.
x1c_2 = 1.
R_2 = 0.5

delta = 0.1
c = 1e-8

def objective(z):
    T = z[-1]
    # x0, x1, x2, y, u0, u1 = jnp.split(z[:-1], 6)
    # pen = jnp.max(jnp.array([0, delta - y[-1]])) # penalization when y(T) < delta
    # distance = jnp.sum(jnp.sqrt(jnp.diff(x0)**2 + jnp.diff(x1)**2))
    # energy = jnp.sum(u0**2 + u1**2)*T/x0.size
    return T

def eq_constraint(z):
    x0, x1, x2, y, u0, u1 = jnp.split(z[:-1], 6)
    return (x0[-1] - x0f)**2 + (x1[-1] - x1f)**2

def phi_1(x0, x1):
    # always avoid the zone
    return ((x0 - x0c_1) ** 2 + (x1 - x1c_1) ** 2) - R_1 ** 2

def phi_2(x0, x1):
    # enter the zone at least one time
    return R_2 ** 2 - ((x0 - x0c_2) ** 2 + (x1 - x1c_2) ** 2)

def ineq_constraint_1(z):
    x0, x1, x2, y, u0, u1 = jnp.split(z[:-1], 6)
    return jnp.maximum(phi_1(x0, x1), y - delta)

def ineq_constraint_2(z):
    x0, x1, x2, y, u0, u1 = jnp.split(z[:-1], 6)
    return y[-1] - delta

def ode_rhs(x, v):
    t_a = 1
    t_b = 6
    x0, x1, x2, y = x
    u0, u1 = v
    xdot0 = jnp.cos(x2) * u0
    xdot1 = jnp.sin(x2) * u0
    xdot2 = u1
    ydot  = (jnp.maximum(jnp.zeros(x0.size), phi_2(x0, x1))) ** 2
    return jnp.asarray([xdot0, xdot1, xdot2, ydot])

def constraint(z):
    T = z[-1]
    x0, x1, x2, y, u0, u1 = jnp.split(z[:-1], 6)
    x = jnp.asarray([x0, x1, x2, y])
    v = jnp.asarray([u0, u1])
    res = jnp.zeros((4, x0.size))
    # initial values
    res = res.at[:, 0].set(x.at[:, 0].get() - jnp.array([x0i, x1i, x2i, yi]))
    # 'solve' the ode-system
    # direct method (explicite euler scheme)
    res = res.at[:, 1:].set(x.at[:, 1:].get()
                            - x.at[:, :-1].get()
                            - T/x0.size * ode_rhs(
                                x.at[:, :-1].get(), v.at[:, :-1].get()))
    return res.flatten()

# jit the functions
obj_jit = jit(objective)
con_jit = jit(constraint)
con_eq_jit = jit(eq_constraint)
con_ineq_1_jit = jit(ineq_constraint_1)
con_ineq_2_jit = jit(ineq_constraint_2)

# build the derivatives and jit them
obj_grad = jit(grad(obj_jit)) # objective gradient
# obj_hess = jit(jacrev(jacfwd(obj_jit))) # objective hessian

con_jac = jit(jacfwd(con_jit)) # jacobian
# con_hess = jacrev(jacfwd(con_jit)) # hessian
# con_hessvp = jit(lambda x, v: con_hess(x) * v[0]) # hessian vector-product

con_eq_jac = jit(jacfwd(con_eq_jit)) # jacobian
# con_eq_hess = jacrev(jacfwd(con_eq_jit)) # hessian
# con_eq_hessvp = jit(lambda x, v: con_eq_hess(x) * v[0]) # hessian vector-product

con_ineq_1_jac = jit(jacfwd(con_ineq_1_jit))  # jacobian
# con_ineq_1_hess = jacrev(jacfwd(con_ineq_1_jit))  # hessian
# con_ineq_1_hessvp = jit(lambda x, v: con_ineq_1_hess(x) * v[0]) # hessian vector-product

con_ineq_2_jac = jit(jacfwd(con_ineq_2_jit))  # jacobian
# con_ineq_2_hess = jacrev(jacfwd(con_ineq_2_jit))  # hessian
# con_ineq_2_hessvp = jit(lambda x, v: con_ineq_2_hess(x) * v[0]) # hessian vector-product

# initial point: random point
# np.random.seed(32)
# z0 = - np.random.random(N * 6 + 1)

# initial point: solution of the problem under "enventually" constraint
# z0 = np.zeros(N * 7 + 1) - 0.01
# with open('res_01.npy', 'rb') as f:
#     res_org = np.load(f)
# z0[: 4*N] = res_org[:4*N]
# z0[5*N :] = res_org[5*N :]

# initial point: straight line
# z0 = - np.ones(N * 6 + 1)
# z0[:2*N] = np.append(
#     np.arange(-1., 0., 1./N)[::-1],
#     np.arange(0., 1., 1./N))
# z0[2*N : 4*N] = np.zeros(2 * N) + 0.01
# z0[-1] = 100

# initial point: a feasible solution
z0 = np.ones(N * 6 + 1)
z0[ : 2*N] = np.concatenate((
    0.5 * np.sin(np.arange(0., np.pi, 2 * np.pi / N)),
    np.arange(- 1., 0., 2 * 1. / N)[::-1],
    0.5 - 0.5 * np.cos(np.arange(0., np.pi, 2 * np.pi / N)),
    np.ones(int(N / 2))))
z0[2*N : 3*N] = np.concatenate((
    np.arange(0., np.pi, 2 * np.pi / N),
    np.pi * np.ones(int(N / 2))))
z0[3*N : 4*N] = np.zeros(N) + 0.01
z0[-1] = 10

# variable bounds
bnds = [(None, None) for i in range(z0.size)]
for i in range(z0.size):
    if i > 4 * N - 1 :
        bnds[i] = (-1, 1)
    elif i > 3 * N - 1 :
        bnds[i] = (0, None)
bnds[-1] = (0.1, None)

# constraints:
cons = [{'type': 'eq', 'fun': con_jit, 'jac':con_jac},
        {'type': 'eq', 'fun': con_eq_jit, 'jac': con_eq_jac},
        {'type': 'ineq', 'fun': con_ineq_1_jit, 'jac': con_ineq_1_jac},
        {'type': 'ineq', 'fun': con_ineq_2_jit, 'jac': con_ineq_2_jac}]

# call the solver
res = minimize_ipopt(obj_jit, jac=obj_grad, x0=z0, bounds=bnds,
                     constraints=cons,
                     options = {'max_iter': 50000,
                                'constr_viol_tol': 1e-16,
                                'acceptable_constr_viol_tol': 1e-12})
print(res)

with open('res_01_until.npy', 'wb') as f:
    np.save(f, res.x)
