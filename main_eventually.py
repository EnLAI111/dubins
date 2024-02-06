import numpy as np
import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev
from jax import config
from cyipopt import minimize_ipopt

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)
# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
config.update('jax_platform_name', 'cpu')
# z = (x1(t0) .... x1(tN) x2(t0) .... x2(tN) x3(t0) .... x3(tN) u1(t0) .... u1(tN) u2(t0) .... u2(tN))

# time grid
N = 1000

x0i = 0
x1i = 0
x2i = 0

x0f = -1
x1f = 1
x2f = np.pi

x0c = 0
x1c = 1
R = 0.5
delta = 0.1

def objective(z):
    T = z[-1]
    # x0, x1, x2, y, u0, u1 = jnp.split(z[:-1], 6)
    # distance = jnp.sum(jnp.sqrt(jnp.diff(x0)**2 + jnp.diff(x1)**2))
    # energy = jnp.sum(u0**2 + u1**2)*T/x0.size
    return T

def eq_constraint_i(z):
    x0, x1, x2, y, u0, u1 = jnp.split(z[:-1], 6)
    return (x0[0] - x0i)**2 + (x1[0] - x1i)**2 + (x2[0] - x2i)**2

def eq_constraint_f(z):
    x0, x1, x2, y, u0, u1 = jnp.split(z[:-1], 6)
    return (x0[-1] - x0f)**2 + (x1[-1] - x1f)**2

def ineq_constraint(z):
    x0, x1, x2, y, u0, u1 = jnp.split(z[:-1], 6)
    return y[-1] - delta

def g(x0, x1):
    return R - jnp.sqrt((x0 - x0c) ** 2 + (x1 - x1c) ** 2)

def ode_rhs(t, x, v):
    t_a = 1
    t_b = 6
    x0, x1, x2, y = x
    u0, u1 = v
    xdot0 = jnp.cos(x2) * u0
    xdot1 = jnp.sin(x2) * u0
    xdot2 = u1
    ydot  = jnp.max(jnp.asarray([0, g(x0, x1)]))**2
    return jnp.asarray([xdot0, xdot1, xdot2, ydot])

def constraint(z):
    T = z[-1]
    x0, x1, x2, y, u0, u1 = jnp.split(z[:-1], 6)
    x = jnp.asarray([x0, x1, x2, y])
    v = jnp.asarray([u0, u1])
    res = jnp.zeros((4, x0.size))
    # 'solve' the ode-system
    for j in range(x0.size-1):
        # explicite euler scheme
        res = res.at[:, j+1].set(x.at[:, j+1].get() 
                                 - x.at[:, j].get() 
                                 - T/x0.size * ode_rhs(T/x0.size*(j+1),
                                     x.at[:, j].get(), v.at[:, j].get()))
    return res.flatten()

# jit the functions
obj_jit = jit(objective)
con_jit = jit(constraint)
con_eq_i_jit = jit(eq_constraint_i)
con_eq_f_jit = jit(eq_constraint_f)
con_ineq_jit = jit(ineq_constraint)

# build the derivatives and jit them
obj_grad = jit(grad(obj_jit)) # objective gradient
# obj_hess = jit(jacrev(jacfwd(obj_jit))) # objective hessian

con_jac = jit(jacfwd(con_jit)) # jacobian
# con_hess = jacrev(jacfwd(con_jit)) # hessian
# con_hessvp = jit(lambda x, v: con_hess(x) * v[0]) # hessian vector-product

con_eq_i_jac = jit(jacfwd(con_eq_i_jit)) # jacobian
# con_eq_i_hess = jacrev(jacfwd(con_eq_i_jit)) # hessian
# con_eq_i_hessvp = jit(lambda x, v: con_eq_i_hess(x) * v[0]) # hessian vector-product

con_eq_f_jac = jit(jacfwd(con_eq_f_jit)) # jacobian
# con_eq_f_hess = jacrev(jacfwd(con_eq_f_jit)) # hessian
# con_eq_f_hessvp = jit(lambda x, v: con_eq_f_hess(x) * v[0]) # hessian vector-product

con_ineq_jac = jit(jacfwd(con_ineq_jit))  # jacobian
# con_ineq_hess = jacrev(jacfwd(con_ineq_jit))  # hessian
# con_ineq_hessvp = jit(lambda x, v: con_ineq_hess(x) * v[0]) # hessian vector-product

# initial point
np.random.seed(32)
z0 = - np.random.random(N * 6 + 1)
# z0 = - np.ones(N * 6 + 1)
# with open('res_org.npy', 'rb') as f:
#     res_org = np.load(f)
# z0[:3*N] = res_org[:3*N]
# z0[3*N:4*N] = np.zeros(N)
# z0[4*N-1:] = res_org[3*N-1:]

z0[ : 2*N] = np.concatenate((
    np.sin(np.arange(0., np.pi, np.pi * 2 / N)),
    np.arange(-1., 0., 2 * 1. / N)[::-1],
    0.5 - np.cos(np.arange(0., np.pi, np.pi * 2 / N)),
    - np.ones(int(N / 2))))
z0[2*N : 3*N] = np.zeros(N) + 0.1
z0[-1] = 10

# variable bounds
bnds = [(None, None) for i in range(z0.size)]
for i in range(z0.size):
    if i > 4 * N - 1 :
        bnds[i] = (-1, 1)
    elif i > 3 * N - 1 :
        bnds[-1] = (0, None)
bnds[-1] = (0, None)

# constraints:
cons = [{'type': 'eq', 'fun': con_jit, 'jac':con_jac},
        {'type': 'eq', 'fun': con_eq_i_jit, 'jac': con_eq_i_jac},
        {'type': 'eq', 'fun': con_eq_f_jit, 'jac': con_eq_f_jac},
        {'type': 'ineq', 'fun': con_ineq_jit, 'jac': con_ineq_jac}]

# call the solver
res = minimize_ipopt(obj_jit, jac=obj_grad, x0=z0, bounds=bnds,
                     constraints=cons, options = {'max_iter': 50000,
                                                  'constr_viol_tol': 1e-12,
                                                  'acceptable_constr_viol_tol': 1e-10})
print(res)

with open('res_01.npy', 'wb') as f:
    np.save(f, res.x)
