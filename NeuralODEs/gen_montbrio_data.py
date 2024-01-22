import jax.numpy as jnp
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import sys
sys.path.append('/Users/martinbreyton/INS_Code/PhD/scripts/python')
import re



def df(X, Delta=1, tau=1, eta=-5, J = 15, Iext = 0):
    r = X[0]
    V = X[1]
    dr = 1 / tau * (Delta / (jnp.pi * tau) + 2 * V * r)
    dV = 1 / tau * (V ** 2 - jnp.pi ** 2 * tau ** 2 * r ** 2 + eta + J * tau * r + Iext)
    return np.array([dr, dV])

def run(X0, dt, N, eta=None):
    rslt = []
    for i in range(N):
        rslt.append(X0)
        X0 = X0 + df(X0, eta=eta)*dt
    return jnp.array(rslt).reshape(len(rslt), 2)


# Plot example trajectory
X0 = np.array([1., 1.])
traj = run(X0, 0.01, 1000, eta=-5)
plt.plot(traj[:,0], traj[:,1])
plt.show()


r_i, v_i = np.mgrid[0.:8:13j, -10:10:13j]
X = jnp.vstack([r_i.ravel(), v_i.ravel()])
for eta in np.arange(-10, 8, .25):
    rvstack = []
    for i, X0 in enumerate(X.T):
        rvstack.append(run(X0, .01, 1000, eta=eta))
    rve = np.array(rvstack)
    # rves.append(rve)
    np.savez(f'data/montbrio_data/mpr_data_eta{eta}.npz', data=rve)
    print('done: ', eta)


# Merge rv times series with eta values to build training data
rve_stack = []
for i, file_i in enumerate(os.listdir('data/montbrio_data/phase_plane_normal')):
    if file_i.startswith('.DS_Store'):
        continue
    eta = np.round(float(re.findall(r"[-+]?(?:\d*\.*\d+)", file_i)[0]),1)
    rv = np.load(f'data/montbrio_data/phase_plane_normal/{file_i}')['data']
    rv = rv.transpose((1, 2, 0))
    eta_mat = np.ones((rv.shape[0], 1, rv.shape[-1]))*eta
    rve = np.concatenate([rv, eta_mat], axis=1)
    rve_stack.append(rve)

rves = np.concatenate(rve_stack, axis=-1)
np.savez('data/montbrio_data/rves_etas_no_noise_normal.npz', data=rves)


# Plot sample trajectories
fig, (ax, ax2) = plt.subplots(1,2, figsize=(8,4))
for i in np.random.choice(rves.shape[-1], size=(10,)):
    ax.plot(np.array(rves[:100,0,i]), np.array(rves[:100,1,i]))
    ax.plot(np.array(rves[0,0,i]), np.array(rves[0,1,i]), 'go')
    ax2.plot(np.array(rves[:100,0,i]))
ax.set_title('Montbrio model')
ax.set_xlabel(r'$r$')
ax.set_ylabel(r'$V$')
ax2.set_ylabel(r'$r$')
ax2.set_ylabel(r'$Time$')
ax.set_xlim((0,4))
ax.set_ylim((-4, 4))
ax2.set_ylim((0, 3))
# plt.savefig('montbrio_flow_with_data_N100_short_noise.png', bbox_inches='tight', facecolor='None')
plt.show()