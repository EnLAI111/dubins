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
    ydot  = (jnp.maximum(jnp.zeros(x0.size), g(x0, x1))) ** 2
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
        # direct method (explicite euler scheme)
        res = res.at[:, 1:].set(x.at[:, 1:].get()
                                - x.at[:, :-1].get()
                                - T/x0.size * ode_rhs_6(
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


file_list = ['org', '00005', '0001', '0005', '001', '005', '01']
file_name = ['Original Problem', 
             'Eventually: '+ r'$\delta = 0.0005$',
             'Eventually: '+ r'$\delta = 0.001$',
             'Eventually: '+ r'$\delta = 0.005$',
             'Eventually: '+ r'$\delta = 0.01$',
             'Eventually: '+ r'$\delta = 0.05$',
             'Eventually: '+ r'$\delta = 0.1$']
d = {}

with open('res_org.npy', 'rb') as f:
    res_org = np.load(f)
T_org = res_org[-1]
x0_org, x1_org, x2_org, u0_org, u1_org = jnp.split(res_org[:-1], 5)
d['res_org'] = res_org
d['T_org'] = res_org[-1]
d['x0_org'], d['x1_org'], d['x2_org'], d['u0_org'], d['u1_org'] = jnp.split(res_org[:-1], 5)

for i in range(1, len(file_list)):
    with open('res_' + file_list[i] + '.npy', 'rb') as f:
        d['res_{0}'.format(file_list[i])] = np.load(f)
    d['T_{0}'.format(file_list[i])] = d['res_{0}'.format(file_list[i])][-1]
    d['x0_{0}'.format(file_list[i])], \
        d['x1_{0}'.format(file_list[i])], \
            d['x2_{0}'.format(file_list[i])], \
                d['y_{0}'.format(file_list[i])], \
                    d['u0_{0}'.format(file_list[i])], \
                        d['u1_{0}'.format(file_list[i])] = \
                            jnp.split(d['res_{0}'.format(file_list[i])][:-1], 6)

fig = plt.figure(figsize=(16, 8))
axs = fig.subplot_mosaic([['Left', 'TopRight'],['Left', 'BottomRight']],
                          gridspec_kw={'width_ratios':[2, 1]})

for i in range(len(file_list)):
    axs['Left'].plot(d['x0_{0}'.format(file_list[i])],
                     d['x1_{0}'.format(file_list[i])],
                     label = file_name[i])
circle = plt.Circle((x0c, x1c), R, color = 'g', alpha = 0.5)
axs['Left'].add_patch(circle)
axs['Left'].set_xlim([-2, 2])
axs['Left'].set_ylim([-2, 2])
axs['Left'].legend()
axs['Left'].set_xlabel(r'$x_0$')
axs['Left'].set_ylabel(r'$x_1$')
axs['Left'].set_title('Minisation of time')

for i in range(1, len(file_list)):
    T_tmp = d['T_{0}'.format(file_list[i])]
    x0_size_tmp = d['x0_{0}'.format(file_list[i])].size
    u0_tmp = d['u0_{0}'.format(file_list[i])]
    axs['TopRight'].plot(np.arange(0, T_tmp, T_tmp / x0_size_tmp)[:x0_size_tmp], 
                     u0_tmp)
axs['TopRight'].set_ylim([-1.1, 1.1])
axs['TopRight'].set_xlabel('t (s)')
axs['TopRight'].set_ylabel(r'$u_0$')

for i in range(1, len(file_list)):
    T_tmp = d['T_{0}'.format(file_list[i])]
    x0_size_tmp = d['x0_{0}'.format(file_list[i])].size
    u1_tmp = d['u1_{0}'.format(file_list[i])]
    axs['BottomRight'].plot(np.arange(0, T_tmp, T_tmp / x0_size_tmp)[:x0_size_tmp], 
                     u1_tmp)
axs['BottomRight'].set_ylim([-1.1, 1.1])
axs['BottomRight'].set_xlabel('t (s)')
axs['BottomRight'].set_ylabel(r'$u_1$')

fig.savefig('dubins_time_1.png', bbox_inches = 'tight', pad_inches = 0)
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
                     x1_tmp, label = file_name[i])
axs['BottomLeft'].set_xlabel('t (s)')
axs['BottomLeft'].set_ylabel(r'$x_1$')

for i in range(len(file_list)):
    T_tmp = d['T_{0}'.format(file_list[i])]
    x0_size_tmp = d['x0_{0}'.format(file_list[i])].size
    x2_tmp = d['x2_{0}'.format(file_list[i])]
    axs['TopRight'].plot(np.arange(0, T_tmp, T_tmp / x0_size_tmp)[:x0_size_tmp], 
                     x2_tmp, label = file_name[i])
axs['TopRight'].set_xlabel('t (s)')
axs['TopRight'].set_ylabel(r'$x_2$')

axs['BottomRight'].plot(np.arange(0, T_org, T_org / x0_org.size)[:x0_org.size],
                  np.zeros(x0_org.size))
for i in range(1, len(file_list)):
    T_tmp = d['T_{0}'.format(file_list[i])]
    x0_size_tmp = d['x0_{0}'.format(file_list[i])].size
    y_tmp = d['y_{0}'.format(file_list[i])]
    axs['BottomRight'].plot(np.arange(0, T_tmp, T_tmp / x0_size_tmp)[:x0_size_tmp], 
                     y_tmp, label = file_name[i])
axs['BottomRight'].set_xlabel('time (s)')
axs['BottomRight'].set_ylabel(r'$y$')

fig.savefig('dubins_time_2.png', bbox_inches='tight')
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
                     g(x0_tmp, x1_tmp), label = file_name[i])
axs['TopLeft'].axhline(y=0, color='r', linestyle='--')
axs['TopLeft'].set_xlabel('t (s)')
axs['TopLeft'].set_ylabel(r'$g(x(t))$')
axs['TopLeft'].legend()

for i in range(len(file_list)):
    T_tmp = d['T_{0}'.format(file_list[i])]
    x0_size_tmp = d['x0_{0}'.format(file_list[i])].size
    res_tmp = d['res_{0}'.format(file_list[i])]
    axs['TopRight'].plot(np.arange(0, T_tmp, T_tmp / x0_size_tmp)[:x0_size_tmp], 
                     constraint(res_tmp), label = file_name[i])
axs['TopRight'].legend()
axs['TopRight'].set_ylabel('dynamic constriant')
axs['TopRight'].set_xlabel('time (s)')

fig.savefig('dubins_time_3.png', bbox_inches = 'tight', pad_inches = 0)
fig.show()
