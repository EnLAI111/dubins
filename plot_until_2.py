import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt

# initial point
x0i = 0.
x1i = 0.
x2i = 0.
yi = 0.
etai = 0.

# final point
x0f = -1
x1f = 1
x2f = np.pi

# until : always avoid the zone
x0c_1 = -0.5
x1c_1 = 0.
R_1 = 0.3

# eventually : enter the zone at least one time
x0c_2 = 0.
x1c_2 = 1.
R_2 = 0.5

delta_1 = 0.1
delta_2 = 0.1

def phi_1(x0, x1):
    # always avoid the zone
    return ((x0 - x0c_1) ** 2 + (x1 - x1c_1) ** 2) - R_1 ** 2

def phi_2(x0, x1):
    # enter the zone at least one time
    return R_2 ** 2 - ((x0 - x0c_2) ** 2 + (x1 - x1c_2) ** 2)

def ode_rhs_7(x, v):
    t_a = 1
    t_b = 6
    x0, x1, x2, y, eta = x
    u0, u1 = v
    xdot0 = jnp.cos(x2) * u0
    xdot1 = jnp.sin(x2) * u0
    xdot2 = u1
    ydot  = (jnp.maximum(jnp.zeros(x0.size), phi_2(x0, x1))) ** 2
    etadot = jnp.minimum(jnp.zeros(x0.size), -phi_1(x0, x1)) * (y - delta_2 > 0)
    return jnp.asarray([xdot0, xdot1, xdot2, ydot, etadot])

def ode_rhs_5(x, v):
    x0, x1, x2 = x
    u0, u1 = v
    xdot0 = jnp.cos(x2) * u0
    xdot1 = jnp.sin(x2) * u0
    xdot2 = u1
    return jnp.array([xdot0, xdot1, xdot2])

def constraint(z):
    T = z[-1]
    if (z.size - 1) % 7 == 0:
        x0, x1, x2, y, eta, u0, u1 = jnp.split(z[:-1], 7)
        x = jnp.asarray([x0, x1, x2, y, eta])
        v = jnp.asarray([u0, u1])
        res = jnp.zeros((5, x0.size))
        # initial values
        res = res.at[:, 0].set(x.at[:, 0].get() - jnp.array([x0i, x1i, x2i, yi, etai]))
        # 'solve' the ode-system
        # direct method (explicite euler scheme)
        res = res.at[:, 1:].set(x.at[:, 1:].get()
                                - x.at[:, :-1].get()
                                - T/x0.size * ode_rhs_7(
                                    x.at[:, :-1].get(), v.at[:, :-1].get()))
    elif(z.size - 1) % 5 == 0:
        x0, x1, x2, u0, u1 = jnp.split(z[:-1], 5)
        res = jnp.zeros((3, x0.size))
        x = jnp.array([x0, x1, x2])
        v = jnp.array([u0, u1])
        # initial values
        res = res.at[:, 0].set(x.at[:, 0].get() - jnp.array([0., 0., 0.]))
        # 'solve' the ode-system
        # direct method (explicite euler scheme)
        res = res.at[:, 1:].set(x.at[:, 1:].get()
                                - x.at[:, :-1].get()
                                - T/x0.size * ode_rhs_5(
                                    x.at[:, :-1].get(), v.at[:, :-1].get()))
    return jnp.mean(abs(res), axis=0)


file_list = ['org', '01']
file_name = ['Original Problem',
             'Until: '+ r'$\delta = 0.1$']
d = {}

with open('res_org.npy', 'rb') as f:
    res_org = np.load(f)
T_org = res_org[-1]
x0_org, x1_org, x2_org, u0_org, u1_org = jnp.split(res_org[:-1], 5)
d['res_org'] = res_org
d['T_org'] = res_org[-1]
d['x0_org'], d['x1_org'], d['x2_org'], d['u0_org'], d['u1_org'] = jnp.split(res_org[:-1], 5)

for i in range(1, len(file_list)):
    with open('res_' + file_list[i] + '_until_2.npy', 'rb') as f:
        d['res_{0}'.format(file_list[i])] = np.load(f)
    d['T_{0}'.format(file_list[i])] = d['res_{0}'.format(file_list[i])][-1]
    d['x0_{0}'.format(file_list[i])], \
        d['x1_{0}'.format(file_list[i])], \
            d['x2_{0}'.format(file_list[i])], \
                d['y_{0}'.format(file_list[i])], \
                    d['eta_{0}'.format(file_list[i])], \
                        d['u0_{0}'.format(file_list[i])], \
                            d['u1_{0}'.format(file_list[i])] = \
                                jnp.split(d['res_{0}'.format(file_list[i])][:-1], 7)

fig = plt.figure(figsize=(16, 8))
axs = fig.subplot_mosaic([['Left', 'TopRight'],['Left', 'BottomRight']],
                          gridspec_kw={'width_ratios':[2, 1]})

for i in range(len(file_list)):
    axs['Left'].plot(d['x0_{0}'.format(file_list[i])],
                     d['x1_{0}'.format(file_list[i])],
                     label = file_name[i])
circle_1 = plt.Circle((x0c_1, x1c_1), R_1, color = 'r', alpha = 0.5)
circle_2 = plt.Circle((x0c_2, x1c_2), R_2, color = 'g', alpha = 0.5)
axs['Left'].add_patch(circle_1)
axs['Left'].add_patch(circle_2)
axs['Left'].set_xlim([-2, 2])
axs['Left'].set_ylim([-2, 2])
axs['Left'].set_xlabel(r'$x_0$')
axs['Left'].set_ylabel(r'$x_1$')
axs['Left'].set_title('Minisation of time')

for i in range(len(file_list)):
    T_tmp = d['T_{0}'.format(file_list[i])]
    x0_size_tmp = d['x0_{0}'.format(file_list[i])].size
    u0_tmp = d['u0_{0}'.format(file_list[i])]
    axs['TopRight'].plot(np.arange(0, T_tmp, T_tmp / x0_size_tmp)[:x0_size_tmp], 
                     u0_tmp)
axs['TopRight'].set_ylim([-1.1, 1.1])
axs['TopRight'].set_xlabel('t (s)')
axs['TopRight'].set_ylabel(r'$u_0$')

for i in range(len(file_list)):
    T_tmp = d['T_{0}'.format(file_list[i])]
    x0_size_tmp = d['x0_{0}'.format(file_list[i])].size
    u1_tmp = d['u1_{0}'.format(file_list[i])]
    axs['BottomRight'].plot(np.arange(0, T_tmp, T_tmp / x0_size_tmp)[:x0_size_tmp], 
                     u1_tmp)
axs['BottomRight'].set_ylim([-1.1, 1.1])
axs['BottomRight'].set_xlabel('t (s)')
axs['BottomRight'].set_ylabel(r'$u_1$')

fig.legend(loc="outside upper center", ncol=2)
fig.savefig('dubins_until2_1.png', bbox_inches = 'tight', pad_inches = 0.1)
fig.show()

fig = plt.figure(figsize=(16, 8))
axs = fig.subplot_mosaic([['TopLeft', 'TopRight'],['BottomLeft', 'BottomRight']],
                          gridspec_kw={'width_ratios':[1, 1]})

for i in range(len(file_list)):
    T_tmp = d['T_{0}'.format(file_list[i])]
    x0_size_tmp = d['x0_{0}'.format(file_list[i])].size
    x0_tmp = d['x0_{0}'.format(file_list[i])]
    axs['TopLeft'].plot(np.arange(0, T_tmp, T_tmp / x0_size_tmp)[:x0_size_tmp], 
                     x0_tmp, label = file_name[i])
axs['TopLeft'].set_xlabel('t (s)')
axs['TopLeft'].set_ylabel(r'$x_0$')

for i in range(len(file_list)):
    T_tmp = d['T_{0}'.format(file_list[i])]
    x0_size_tmp = d['x0_{0}'.format(file_list[i])].size
    x1_tmp = d['x1_{0}'.format(file_list[i])]
    axs['BottomLeft'].plot(np.arange(0, T_tmp, T_tmp / x0_size_tmp)[:x0_size_tmp], 
                     x1_tmp)
axs['BottomLeft'].set_xlabel('t (s)')
axs['BottomLeft'].set_ylabel(r'$x_1$')

for i in range(len(file_list)):
    T_tmp = d['T_{0}'.format(file_list[i])]
    x0_size_tmp = d['x0_{0}'.format(file_list[i])].size
    x2_tmp = d['x2_{0}'.format(file_list[i])]
    axs['TopRight'].plot(np.arange(0, T_tmp, T_tmp / x0_size_tmp)[:x0_size_tmp], 
                     x2_tmp)
axs['TopRight'].set_xlabel('t (s)')
axs['TopRight'].set_ylabel(r'$x_2$')

for i in range(len(file_list)):
    T_tmp = d['T_{0}'.format(file_list[i])]
    x0_size_tmp = d['x0_{0}'.format(file_list[i])].size
    res_tmp = d['res_{0}'.format(file_list[i])]
    axs['BottomRight'].plot(np.arange(0, T_tmp, T_tmp / x0_size_tmp)[:x0_size_tmp], 
                     constraint(res_tmp))
axs['BottomRight'].set_ylabel('dynamic constriant')
axs['BottomRight'].set_xlabel('time (s)')

fig.legend(loc="outside upper center", ncol=2)
fig.savefig('dubins_until2_2.png', bbox_inches = 'tight', pad_inches = 0.1)
fig.show()

fig = plt.figure(figsize=(16, 8))
axs = fig.subplot_mosaic([['TopLeft', 'TopRight'],['BottomLeft', 'BottomRight']],
                          gridspec_kw={'width_ratios':[1, 1]})

for i in range(len(file_list)):
    T_tmp = d['T_{0}'.format(file_list[i])]
    x0_size_tmp = d['x0_{0}'.format(file_list[i])].size
    x0_tmp = d['x0_{0}'.format(file_list[i])]
    x1_tmp = d['x1_{0}'.format(file_list[i])]
    axs['TopLeft'].plot(np.arange(0, T_tmp, T_tmp / x0_size_tmp)[:x0_size_tmp], 
                     phi_1(x0_tmp, x1_tmp), label = file_name[i])
axs['TopLeft'].axhline(y=0, color='r', linestyle='--')
axs['TopLeft'].set_xlabel('t (s)')
axs['TopLeft'].set_ylabel(r'$\varphi_1(x(t))$')

for i in range(len(file_list)):
    T_tmp = d['T_{0}'.format(file_list[i])]
    x0_size_tmp = d['x0_{0}'.format(file_list[i])].size
    x0_tmp = d['x0_{0}'.format(file_list[i])]
    x1_tmp = d['x1_{0}'.format(file_list[i])]
    axs['BottomLeft'].plot(np.arange(0, T_tmp, T_tmp / x0_size_tmp)[:x0_size_tmp], 
                     phi_2(x0_tmp, x1_tmp))
axs['BottomLeft'].axhline(y=0, color='r', linestyle='--')
axs['BottomLeft'].set_xlabel('t (s)')
axs['BottomLeft'].set_ylabel(r'$\varphi_2(x(t))$')

axs['TopRight'].plot(np.arange(0, T_org, T_org / x0_org.size)[:x0_org.size],
                  np.zeros(x0_org.size))
for i in range(1, len(file_list)):
    T_tmp = d['T_{0}'.format(file_list[i])]
    x0_size_tmp = d['x0_{0}'.format(file_list[i])].size
    eta_tmp = d['eta_{0}'.format(file_list[i])]
    axs['TopRight'].plot(np.arange(0, T_tmp, T_tmp / x0_size_tmp)[:x0_size_tmp], 
                     eta_tmp)
axs['TopRight'].set_xlabel('time (s)')
axs['TopRight'].set_ylabel(r'$y$')

axs['BottomRight'].plot(np.arange(0, T_org, T_org / x0_org.size)[:x0_org.size],
                  np.zeros(x0_org.size))
for i in range(1, len(file_list)):
    T_tmp = d['T_{0}'.format(file_list[i])]
    x0_size_tmp = d['x0_{0}'.format(file_list[i])].size
    y_tmp = d['y_{0}'.format(file_list[i])]
    axs['BottomRight'].plot(np.arange(0, T_tmp, T_tmp / x0_size_tmp)[:x0_size_tmp], 
                     y_tmp)
axs['BottomRight'].set_xlabel('time (s)')
axs['BottomRight'].set_ylabel(r'$y$')

fig.legend(loc="outside upper center", ncol=2)
fig.savefig('dubins_until2_3.png', bbox_inches = 'tight', pad_inches = 0.1)
fig.show()