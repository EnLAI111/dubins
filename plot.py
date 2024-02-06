import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt

# initial point
x0i = 0
x1i = 0
x2i = 0

# final point
x0f = -1
x1f = 1
x2f = np.pi

# eventually
x0c = 0
x1c = 1
R = 0.5
delta = 0.1
c = 1e-6

with open('res_org.npy', 'rb') as f:
    res_org = np.load(f)
T_org = res_org[-1]
x0_org, x1_org, x2_org, u0_org, u1_org = jnp.split(res_org[:-1], 5)

with open('res_001.npy', 'rb') as f:
    res_001 = np.load(f)
T_001 = res_001[-1]
x0_001, x1_001, x2_001, y_001, u0_001, u1_001 = jnp.split(res_001[:-1], 6)

with open('res_005.npy', 'rb') as f:
    res_005 = np.load(f)
T_005 = res_005[-1]
x0_005, x1_005, x2_005, y_005, u0_005, u1_005 = jnp.split(res_005[:-1], 6)

with open('res_01.npy', 'rb') as f:
    res_01 = np.load(f)
T_01 = res_01[-1]
x0_01, x1_01, x2_01, y_01, u0_01, u1_01 = jnp.split(res_01[:-1], 6)

def g(x0, x1):
    return R ** 2 - ((x0 - x0c) ** 2 + (x1 - x1c) ** 2)

def ode_rhs_6(t, x, v):
    t_a = 1
    t_b = 6
    x0, x1, x2, y = x
    u0, u1 = v
    xdot0 = jnp.cos(x2) * u0
    xdot1 = jnp.sin(x2) * u0
    xdot2 = u1
    ydot  = jnp.max(jnp.asarray([0, g(x0, x1)]))**2
    return jnp.asarray([xdot0, xdot1, xdot2, ydot])

def ode_rhs_5(x, v):
    x0, x1, x2 = x
    u0, u1 = v
    xdot0 = jnp.cos(x2) * u0
    xdot1 = jnp.sin(x2) * u0
    xdot2 = u1
    return jnp.array([xdot0, xdot1, xdot2])

def constraint(z):
    T = z[-1]
    if (z.size - 1) % 6 == 0:
        x0, x1, x2, y, u0, u1 = jnp.split(z[:-1], 6)
        res = jnp.zeros((4, x0.size))
        x = jnp.array([x0, x1, x2, y])
        v = jnp.array([u0, u1])
        # initial values
        res = res.at[:, 0].set(x.at[:, 0].get() - jnp.array([0., 0., 0., 0.]))
        # 'solve' the ode-system
        for j in range(x0.size-1):
            # direct method (explicite euler scheme)
            res = res.at[:, j+1].set(x.at[:, j+1].get() 
                                     - x.at[:, j].get() 
                                     - T/x0.size * ode_rhs_6(T/x0.size*(j+1),
                                         x.at[:, j].get(), v.at[:, j].get()))
    elif(z.size - 1) % 5 == 0:
        x0, x1, x2, u0, u1 = jnp.split(z[:-1], 5)
        res = jnp.zeros((3, x0.size))
        x = jnp.array([x0, x1, x2])
        v = jnp.array([u0, u1])
        # initial values
        res = res.at[:, 0].set(x.at[:, 0].get() - jnp.array([0., 0., 0.]))
        # 'solve' the ode-system
        for j in range(x0.size-1):
            # direct method (explicite euler scheme)
            res = res.at[:, j+1].set(x.at[:, j+1].get() 
                                     - x.at[:, j].get() 
                                     - T/x0.size * ode_rhs_5(
                                         x.at[:, j].get(), v.at[:, j].get()))
    return jnp.mean(abs(res), axis=0)


fig = plt.figure(figsize=(16, 8))
axs = fig.subplot_mosaic([['Left', 'TopRight'],['Left', 'BottomRight']],
                          gridspec_kw={'width_ratios':[2, 1]})

axs['Left'].plot(x0_org, x1_org, label = 'Original Problem')
axs['Left'].plot(x0_001, x1_001, label = 'Eventually: '+ r'$\delta = 0.01$')
axs['Left'].plot(x0_005, x1_005, label = 'Eventually: '+ r'$\delta = 0.05$')
axs['Left'].plot(x0_01, x1_01, label = 'Eventually: '+ r'$\delta = 0.1$')
circle = plt.Circle((x0c, x1c), R, color = 'r', alpha = 0.5)
axs['Left'].add_patch(circle)
axs['Left'].set_xlim([-2, 2])
axs['Left'].set_ylim([-2, 2])
axs['Left'].legend()
axs['Left'].set_xlabel(r'$x_0$')
axs['Left'].set_ylabel(r'$x_1$')
axs['Left'].set_title('Minisation of time')

axs['TopRight'].plot(np.arange(0, T_org, T_org / x0_org.size)[:x0_org.size], 
                     u0_org)
axs['TopRight'].plot(np.arange(0, T_001, T_001 / x0_001.size)[:x0_001.size], 
                     u0_001)
axs['TopRight'].plot(np.arange(0, T_005, T_005 / x0_005.size)[:x0_005.size], 
                     u0_005)
axs['TopRight'].plot(np.arange(0, T_01, T_01 / x0_01.size)[:x0_01.size],
                     u0_01)
axs['TopRight'].set_ylim([-1.1, 1.1])
axs['TopRight'].set_xlabel('t (s)')
axs['TopRight'].set_ylabel(r'$u_0$')

axs['BottomRight'].plot(np.arange(0, T_org, T_org / x0_org.size)[:x0_org.size], 
                        u1_org)
axs['BottomRight'].plot(np.arange(0, T_001, T_001 / x0_001.size)[:x0_001.size],
                        u1_001)
axs['BottomRight'].plot(np.arange(0, T_005, T_005 / x0_005.size)[:x0_005.size], 
                        u1_005)
axs['BottomRight'].plot(np.arange(0, T_01, T_01 / x0_01.size)[:x0_01.size],
                        u1_01)
axs['BottomRight'].set_ylim([-1.1, 1.1])
axs['BottomRight'].set_xlabel('t (s)')
axs['BottomRight'].set_ylabel(r'$u_1$')

fig.savefig('dubins_time.png')
fig.show()

fig = plt.figure(figsize=(16, 8))
axs = fig.subplot_mosaic([['TopLeft', 'TopRight'],['BottomLeft', 'BottomRight']],
                          gridspec_kw={'width_ratios':[1, 1]})
axs['TopLeft'].plot(np.arange(0, T_org, T_org / x0_org.size)[:x0_org.size], 
                 R - np.sqrt((x0_org - x0c) ** 2 + (x1_org - x1c) ** 2), 
                 label = 'Original Problem')
axs['TopLeft'].plot(np.arange(0, T_001, T_001 / x0_001.size)[:x0_001.size], 
                 R - np.sqrt((x0_001 - x0c) ** 2 + (x1_001 - x1c) ** 2), 
                 label = 'Eventually: '+ r'$\delta = 0.01$')
axs['TopLeft'].plot(np.arange(0, T_005, T_005 / x0_005.size)[:x0_005.size], 
                 R - np.sqrt((x0_005 - x0c) ** 2 + (x1_005 - x1c) ** 2), 
                 label = 'Eventually: '+ r'$\delta = 0.05$')
axs['TopLeft'].plot(np.arange(0, T_01, T_01 / x0_01.size)[:x0_01.size], 
                 R - np.sqrt((x0_01 - x0c) ** 2 + (x1_01 - x1c) ** 2), 
                 label = 'Eventually: '+ r'$\delta = 0.1$')
axs['TopLeft'].axhline(y=0, color='r', linestyle='--')
axs['TopLeft'].set_xlabel('t (s)')
axs['TopLeft'].set_ylabel(r'$g(x(t))$')
axs['TopLeft'].legend()

axs['BottomLeft'].plot(np.arange(0, T_org, T_org / x0_org.size)[:x0_org.size], 
                        x2_org, label = 'Original Problem')
axs['BottomLeft'].plot(np.arange(0, T_001, T_001 / x0_001.size)[:x0_001.size], 
                        x2_001, label = 'Eventually: '+ r'$\delta = 0.01$')
axs['BottomLeft'].plot(np.arange(0, T_005, T_005 / x0_005.size)[:x0_005.size], 
                        x2_005, label = 'Eventually: '+ r'$\delta = 0.05$')
axs['BottomLeft'].plot(np.arange(0, T_01, T_01 / x0_01.size)[:x0_01.size], 
                        x2_01, label = 'Eventually: '+ r'$\delta = 0.1$')
axs['BottomLeft'].set_xlabel('t (s)')
axs['BottomLeft'].set_ylabel(r'$x_2$')

axs['TopRight'].plot(np.arange(0, T_org, T_org / x0_org.size)[:x0_org.size], 
                     x0_org, label = 'Original Problem')
axs['TopRight'].plot(np.arange(0, T_001, T_001 / x0_001.size)[:x0_001.size], 
                     x0_001, label = 'Eventually: '+ r'$\delta = 0.01$')
axs['TopRight'].plot(np.arange(0, T_005, T_005 / x0_005.size)[:x0_005.size], 
                     x0_005, label = 'Eventually: '+ r'$\delta = 0.05$')
axs['TopRight'].plot(np.arange(0, T_01, T_01 / x0_01.size)[:x0_01.size], 
                     x0_01, label = 'Eventually: '+ r'$\delta = 0.1$')
axs['TopRight'].set_xlabel('t (s)')
axs['TopRight'].set_ylabel(r'$x_0$')

axs['BottomRight'].plot(np.arange(0, T_org, T_org / x0_org.size)[:x0_org.size], 
                        x1_org, label = 'Original Problem')
axs['BottomRight'].plot(np.arange(0, T_001, T_001 / x0_001.size)[:x0_001.size], 
                        x1_001, label = 'Eventually: '+ r'$\delta = 0.01$')
axs['BottomRight'].plot(np.arange(0, T_005, T_005 / x0_005.size)[:x0_005.size], 
                        x1_005, label = 'Eventually: '+ r'$\delta = 0.05$')
axs['BottomRight'].plot(np.arange(0, T_01, T_01 / x0_01.size)[:x0_01.size], 
                        x1_01, label = 'Eventually: '+ r'$\delta = 0.1$')
axs['BottomRight'].set_xlabel('t (s)')
axs['BottomRight'].set_ylabel(r'$x_1$')

fig.savefig('dubins_time_2.png')
fig.show()

fig = plt.figure(figsize=(16, 8))
axs = fig.subplot_mosaic([['Left', 'Right']],
                          gridspec_kw={'width_ratios':[1, 1]})

axs['Left'].plot(np.arange(0, T_org, T_org / x0_org.size)[:x0_org.size],
                 constraint(res_org), label = 'Original Problem')
axs['Left'].plot(np.arange(0, T_001, T_001 / x0_001.size)[:x0_001.size],
                 constraint(res_001), label = 'Eventually: '+ r'$\delta = 0.01$')
axs['Left'].plot(np.arange(0, T_005, T_005 / x0_005.size)[:x0_005.size],
                 constraint(res_005), label = 'Eventually: '+ r'$\delta = 0.05$')
axs['Left'].plot(np.arange(0, T_01, T_01 / x0_01.size)[:x0_01.size],
                 constraint(res_01), label = 'Eventually: '+ r'$\delta = 0.1$')
axs['Left'].legend()
axs['Left'].set_ylabel('dynamic constriant')
axs['Left'].set_xlabel('time (s)')
                       
axs['Right'].plot(np.arange(0, T_org, T_org / x0_org.size)[:x0_org.size],
                  np.zeros(x0_org.size))
axs['Right'].plot(np.arange(0, T_001, T_001 / x0_001.size)[:x0_001.size],
                  y_001)
axs['Right'].plot(np.arange(0, T_005, T_005 / x0_005.size)[:x0_005.size],
                  y_005)
axs['Right'].plot(np.arange(0, T_01, T_01 / x0_01.size)[:x0_01.size],
                  y_01)
axs['Right'].set_xlabel('time (s)')
axs['Right'].set_ylabel(r'$y$')

fig.savefig('dubins_time_3.png')
fig.show()
