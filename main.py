import numpy as np
import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev
from jax import config
from cyipopt import minimize_ipopt

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)
# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
config.update('jax_platform_name', 'cpu')
# z = (x1(t0) .... x1(tN) x2(t0) .... x2(tN) x3(t0) .... x3(tN) u1(t0) .... u1(tN) u2(t0) .... u2(tN) T)

# time grid
N = 100

x0f = -1
x1f = 1
x2f = np.pi

def objective(z):
    T = z[-1]
    # x0, x1, x2, u0, u1 = jnp.split(z[:-1], 5)
    # distance = jnp.sum(jnp.sqrt(jnp.diff(x0)**2 + jnp.diff(x1)**2))
    # energy = jnp.sum(u0**2 + u1**2)*T/x0.size
    return T

def eq_constraint_i(z):
    x0, x1, x2, u0, u1 = jnp.split(z[:-1], 5)
    return x0[0]**2 + x1[0]**2

def eq_constraint_f(z):
    x0, x1, x2, u0, u1 = jnp.split(z[:-1], 5)
    return (x0[-1] - x0f)**2 + (x1[-1] - x1f)**2

def ode_rhs(x, v):
    x0, x1, x2= x
    u0, u1 = v
    xdot0 = jnp.cos(x2) * u0
    xdot1 = jnp.sin(x2) * u0
    xdot2 = u1
    return jnp.array([xdot0, xdot1, xdot2])

def constraint(z):
    T = z[-1]
    x0, x1, x2, u0, u1 = jnp.split(z[:-1], 5)
    x = jnp.array([x0, x1, x2])
    v = jnp.array([u0, u1])
    res = jnp.zeros((3, x0.size))
    # 'solve' the ode-system
    for j in range(x0.size-1):
        # explicite euler scheme
        res = res.at[:, j+1].set(x.at[:, j+1].get() 
                                 - x.at[:, j].get() 
                                 - T/x0.size * ode_rhs(
                                     x.at[:, j].get(), v.at[:, j].get()))
    return res.flatten()

# jit the functions
obj_jit = jit(objective)
con_jit = jit(constraint)
con_eq_i_jit = jit(eq_constraint_i)
con_eq_f_jit = jit(eq_constraint_f)

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

# initial point
np.random.seed(32)
z0 = np.random.random(N * 5 + 1)
# z0 = np.ones(N * 5 + 1)
# z0[:2*N] = np.append(
#     np.arange(-1., 0., 1./N)[::-1],
#     np.arange(0., 1., 1./N))
# z0[2*N:3*N] = np.zeros(N) + 0.1
# z0[-1] = 10

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
    if i > 3 * N - 1 :
        bnds[i] = (-1, 1)
bnds[-1] = (0, None)

# constraints:
cons = [{'type': 'eq', 'fun': con_jit, 'jac': con_jac},
        {'type': 'eq', 'fun': con_eq_i_jit, 'jac': con_eq_i_jac},
        {'type': 'eq', 'fun': con_eq_f_jit, 'jac': con_eq_f_jac}]

# call the solver
res = minimize_ipopt(obj_jit, jac=obj_grad, x0=z0, bounds=bnds,
constraints=cons, options = {'max_iter': 50000,
                             'constr_viol_tol': 1e-12,
                             'acceptable_constr_viol_tol': 1e-10})
print(res)

with open('res_org.npy', 'wb') as f:
    np.save(f, res.x)
