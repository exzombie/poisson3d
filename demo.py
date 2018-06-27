#!/usr/bin/env python3

from __init__ import EwaldField, Poisson3D
import matplotlib.pylab as plt
import numpy as np
import pymatgen as mg
from scipy.interpolate import interpn


#mode = 'size'
mode = 'N'


def norm(x):
    return np.sqrt(x.dot(x))

eps = 1e-10 * 8.854187817620e-12 / 1.602176565e-19

sizes = (5, 10, 20, 40)
Ns = (16, 32, 64)
angles = (110, 100, 70)
msize = min(sizes)
N = 128

pos = np.array([[1.495519237,   1.994488309,   1.000041210],
                [2.473452185,   1.992651076,   0.999979232],
                [1.250228578,   2.940960614,   0.999979560]])
spec = ['O', 'H', 'H']
chgs = [-0.8, 0.4, 0.4]

# Line for drawing.
start = np.array([0.7, 3.1, 1.0])
end = np.array([4.5, 0.0, 1.0])
step = (end - start) / N
length = norm(end - start)

Tref = np.linspace(0, 1, N)
Xref = start + Tref[:, None] * (end - start)
pot = lambda x: 1 / 4 / np.pi / eps * \
    (chgs[0] / norm(x - pos[0, :]) +
     chgs[1] / norm(x - pos[1, :]) +
     chgs[2] / norm(x - pos[2, :]))
Vref = [pot(x) for x in Xref]


def wrap(lattice, cart):
    frac = lattice.get_fractional_coords(cart)
    return lattice.get_cartesian_coords(frac % 1.0)

if mode == 'size':
    V = []
    for S in sizes:
        lat = mg.Lattice.from_parameters(S, S, S, *angles)
        ewald = EwaldField(mg.Structure(lat, spec, pos,
                                        coords_are_cartesian=True,
                                        site_properties={'charge': chgs}))
        v = np.empty_like(Tref)
        for i, x in enumerate(Xref):
            v[i] = ewald.potential_at(wrap(lat, x))
        V.append(v)

    #plt.plot(Tref*length, Vref, label='1/r')
    for v, s in zip(V, sizes):
        plt.plot(Tref*length, v, label='s'+str(s))
        #plt.plot(Tref*length, v-Vref, label='diff ' + str(s))

elif mode == 'N':
    VE = []
    VP = []
    lat = mg.Lattice.from_parameters(msize, msize, msize, *angles)
    for N in Ns:
        rho = np.zeros((N, N, N))
        shape = np.array(rho.shape)
        voxel = lat.matrix / shape[:, None]
        voxvol = voxel[0, :].dot(np.cross(voxel[1, :], voxel[2, :]))
        toint = lambda X: [int(i) for i in X]
        indices = [toint(lat.get_fractional_coords(X) * shape)
                   for X in pos]

        # align charges to grid
        newpos = [X / shape for X in indices]
        indices = [tuple(i) for i in indices]
        for i, c in zip(indices, chgs):
            rho[i] = -c / eps / voxvol
        phi = Poisson3D(lat.matrix).solve(rho)

        # interpolate the values along the line
        VP.append(interpn([np.linspace(0, 1, N+1)[:-1]] * 3, phi,
                          [lat.get_fractional_coords(x) % 1.0 for x in Xref],
                          fill_value=np.nan, bounds_error=False))

        # do the ewald reference
        ewald = EwaldField(mg.Structure(lat, spec, newpos,
                                        coords_are_cartesian=False,
                                        site_properties={'charge': chgs}))
        v = np.empty_like(Tref)
        for i, x in enumerate(Xref):
            v[i] = ewald.potential_at(x)
        VE.append(v)

    for ve, vp, i in zip(VE, VP, Ns):
        plt.plot(Tref*length, vp, label='poiss ' + str(i))
        plt.plot(Tref*length, ve, label='ewald ' + str(i))
else:
    raise RuntimeError('mode undefined!')


plt.legend()
plt.show()
