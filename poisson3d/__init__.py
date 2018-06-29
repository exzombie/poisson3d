#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import ctypes as ct
import numpy as np
from os.path import dirname, abspath
from os import stat
from subprocess import check_call
import inspect
import pymatgen as mg
from numpy import pi, sqrt, log
from math import erfc
from pymatgen.core.physical_constants import ELECTRON_CHARGE, EPSILON_0
from numba import jit, float64, int64


class Poisson3D:
    """This class implements a solver for the Poisson equation

             $$\nabla^2 \phi = \rho$$

    on a 3-dimensional domain with straight but not neccessarily
    orthogonal boundaries and periodic boundary conditions. The domain
    is specified by a 3x3 matrix representing the basis vectors. Both
    $\rho$ and $\phi$ have the mean value of zero; even if rho.mean()
    is nonzero on input, it is treated as if it were zero. The Poisson
    equation on a periodic domain does not allow for a non-zero
    average values.
    """

    def __init__(self, basis, domain_shape=None):
        """The solver is constructed for the domain specified by the basis and
        shape parameters. basis is a 3x3 matrix whose rows are the
        right-handed basis vectors describing the shape of the domain,
        i.e. they represent the edges of the parallelepiped:

                          +------------------------+
                         /|                        |
                        / |                        |
                       /  +------------------------+
                      /  /                        /
                     +  /                        /
                     | /  basis[2,:]            /
          basis[1,:] |/                        /
                     +------------------------+
                              basis[0,:]

        The domain_shape parameter specifies the shape of the input
        and output arrays of the solver. If set to None, the shape is
        read when solving for the first time, and subsequently
        enforced.
        """
        if basis.shape != (3, 3):
            raise ValueError('basis is not a 3x3 matrix.')
        if basis.dtype != np.float:
            raise ValueError('basis data type is not float.')

        self._basis = basis.copy(order='C')
        if domain_shape is not None:
            if len(domain_shape) != 3:
                raise ValueError('domain is not 3-dimensional.')
            self._shape = np.asarray(domain_shape, dtype=np.uint32)
            self._check = tuple(domain_shape[:])
        else:
            self._shape = None
        self._phase = None
        self._invphase = None

    def solve(self, rhs):
        """This function solves the Poisson equation for the domain this
        instance of the solver was initialized with. rhs is a
        3-dimensional numpy array whose shape must match the
        domain. This function returns a numpy array of the same shape
        as rhs.
        """
        if len(rhs.shape) != 3:
            raise ValueError('rhs is not 3-dimensional.')
        if rhs.dtype != np.float:
            raise ValueError('rhs data type is not float.')

        if self._shape is None:
            self._shape = np.asarray(rhs.shape, dtype=np.uint32, order='C')
            self._check = rhs.shape[:]

        if self._check != rhs.shape:
            raise ValueError('rhs shape does not match this solver\'s domain.')

        copy = rhs.copy(order='C')
        if self._phase is None:
            self._phase = np.ndarray(rhs.shape, dtype=np.float_, order='C')
            Poisson3D._impl.poisson3d(
                self._shape.ctypes.data_as(ct.POINTER(ct.c_uint32)),
                copy.ctypes.data_as(ct.POINTER(ct.c_double)),
                self._basis.ctypes.data_as(ct.POINTER(ct.c_double)),
                self._phase.ctypes.data_as(ct.POINTER(ct.c_double)))
        else:
            Poisson3D._impl.poisson3d_precomputed(
                self._shape.ctypes.data_as(ct.POINTER(ct.c_uint32)),
                copy.ctypes.data_as(ct.POINTER(ct.c_double)),
                self._phase.ctypes.data_as(ct.POINTER(ct.c_double)))
        return copy

    def apply(self, lhs):
        """This function applies the Laplace operator, i.e. it is the inverse
           of solving the Poisson equation. solve() and apply() are
           inverses, with the caveat that they force the average value
           of the field to be zero.
        """
        if self._invphase is None:
            if self._phase is None:
                self.solve(lhs)
            with np.errstate(divide='ignore', invalid='ignore'):
                self._invphase = 1.0 / self._phase
                self._invphase[0, 0, 0] = 0.0
        self._phase, self._invphase = self._invphase, self._phase
        rhs = self.solve(lhs)
        self._phase, self._invphase = self._invphase, self._phase
        return rhs

    def _getImplementation():
        mypath = dirname(abspath(inspect.getfile(inspect.currentframe())))
        impl = mypath + '/poisson3d.so'
        source = mypath + '/poisson3d.cpp'
        sourcestat = stat(source)
        try:
            implstat = stat(impl)
        except:
            class Object:
                pass
            implstat = Object()
            implstat.st_mtime = 0
        if sourcestat.st_mtime >= implstat.st_mtime:
            cflags = '--std=c++11 -ggdb -Wall -Wextra -fPIC -shared'.split()
            optim = '-O2 -march=native'.split()
            define = ['-DNDEBUG']
            ldflags = ['-lfftw3']
            cmd = ['c++'] + cflags + optim + define + ldflags + \
                  ['-o', impl, source]
            check_call(cmd)
        return ct.cdll.LoadLibrary(impl)

    _impl = _getImplementation()


class EwaldField(object):
    """Calculates the electrostatic field of a periodic array of charges using
    the Ewald technique.
    Ref: http://www.ee.duke.edu/~ayt/ewaldpaper/ewaldpaper.html

    Atomic units used in the code, then converted to volts.

    This is an adaptation of the EwaldSummation class from
    pymatgen. Instead of computing the electrostatic energy, it
    computes the electrostatic potential at a given location.

    There is a jellium term which is neccessary to make the potential
    independent of eta when the cell is charged.

    There is no polarization term, which corresponds to the tin-foil
    macroscopic boundary condition.

    Refs: http://pymatgen.org/_modules/pymatgen/analysis/ewald.html
          http://www.pymatgen.org/
    """

    # Converts unit of q*q/r into eV
    CONV_FACT = 1e10 * ELECTRON_CHARGE / (4 * pi * EPSILON_0)

    def __init__(self, structure, real_space_cut=None, recip_space_cut=None,
                 eta=None, acc_factor=8.0):
        """Initializes the structures needed for Ewald summation. Default
        convergence parameters have been specified, but you can
        override them if you wish.

        Args:
            structure (Structure): Input structure that must have proper
                Specie on all sites, i.e. Element with oxidation state. Use
                Structure.add_oxidation_state... for example.
            real_space_cut (float): Real space cutoff radius dictating how
                many terms are used in the real space sum. Defaults to None,
                which means determine automagically using the formula given
                in gulp 3.1 documentation.
            recip_space_cut (float): Reciprocal space cutoff radius.
                Defaults to None, which means determine automagically using
                the formula given in gulp 3.1 documentation.
            eta (float): The screening parameter. Defaults to None, which means
                determine automatically.
            acc_factor (float): No. of significant figures each sum is
                converged to.
        """
        self._s = structure
        self._s.add_site_property('index', list(range(len(self._s))))
        self._vol = structure.volume

        self._acc_factor = acc_factor

        # set screening length
        self._eta = eta if eta \
            else (len(structure) * 0.01 / self._vol) ** (1 / 3) * pi
        self._sqrt_eta = sqrt(self._eta)

        # acc factor used to automatically determine the optimal real and
        # reciprocal space cutoff radii
        self._accf = sqrt(log(10 ** acc_factor))

        self._rmax = real_space_cut if real_space_cut \
            else self._accf / self._sqrt_eta
        self._gmax = recip_space_cut if recip_space_cut \
            else 2 * self._sqrt_eta * self._accf

        self._precomputed_positions = None

        # The next few lines pre-compute certain quantities and store them.
        # Ewald summation is rather expensive, and these shortcuts are
        # necessary to obtain several factors of improvement in speedup.
        coords = np.array(self._s.cart_coords)
        rcp_latt = self._s.lattice.reciprocal_lattice
        recip_nn = rcp_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0],
                                                 self._gmax)
        recip_nn = [x for x in recip_nn if x[1] != 0]
        frac_to_cart = rcp_latt.get_cartesian_coords
        self._gvects = np.zeros((len(recip_nn), 3))
        self._gvectdots = np.zeros((len(recip_nn), len(coords)))
        self._gsquares = np.zeros((len(recip_nn),))
        for i, (fcoords, dist, _) in enumerate(recip_nn):
            gvect = frac_to_cart(fcoords)
            self._gvects[i] = gvect
            self._gvectdots[i] = np.sum(gvect[None, :] * coords, 1)
            self._gsquares[i] = gvect.dot(gvect)
        self._expvals = np.exp(-self._gsquares / 4.0 / self._eta)
        self._jellium = np.pi / (self._vol * self._eta) * \
            sum([s.charge for s in self._s]) * EwaldField.CONV_FACT

    def _calc_recip(self, position, idx=None):
        prefactor = 4 * pi / self._vol
        charges = np.array([s.charge for s in self._s])
        position = np.asarray(position)
        position = position[None, :]

        posdot = np.sum(self._gvects * position, 1)
        # expargs is indexed as [k-vector, j-index]
        expargs = (posdot[:, None] - self._gvectdots)
        # We now sum complex exponentials over j into structure factors
        cos = np.cos(expargs)
        sin = np.sin(expargs)
        sfactors = cos + 1.0j * sin
        if idx is not None:
            self._precomputed_positions['recip'][idx] = sfactors.copy()
        sfactors = np.sum(charges[None, :] * sfactors, 1)
        # sum over k-vectors
        erecip = np.sum(sfactors / self._gsquares * self._expvals)
        return prefactor * EwaldField.CONV_FACT * erecip.real

    def _calc_real(self, position, idx=None):
        NN = self._s.get_sites_in_sphere(position, self._rmax)
        nn = np.array([[x, s.charge] for s, x in NN])
        r = nn[:, 0]
        q = nn[:, 1]
        erfcval = np.array([erfc(k) for k in self._sqrt_eta * r])
        tmp = erfcval / r
        if idx is not None:
            self._precomputed_positions['real pos'][idx] = tmp.copy()
            self._precomputed_positions['real chg'][idx] = \
                [s.properties['index'] for s, _ in NN]
        ereal = q * tmp
        return EwaldField.CONV_FACT * ereal.sum()

    def potential_at(self, position):
        """Calculate the potential at the cartesian coordinates specified py
           the position argument."""
        return self._calc_real(position) + self._calc_recip(position) \
            - self._jellium

    def change_charges(self, charges):
        """Change the charges of the underlying structure. charges is a list
           of charges for each atom in the structure."""
        for i, s in enumerate(self._s):
            self._s[i] = s.specie, s.frac_coords, {'charge': charges[i],
                                                   'index': i}
        self._jellium = np.pi / (self._vol * self._eta) * \
            sum([s.charge for s in self._s]) * EwaldField.CONV_FACT

    def precompute_positions(self, positions):
        """If the potential is to be computed several times for the same set
           of points but different charges, it is much faster to
           precompute certain factors. Call this function with a numpy
           array of cartesian coordinates as rows, then call
           recompute_positions() to perform the computation with a set
           of charges."""
        NN = len(positions)
        self._precomputed_positions = {'recip': [None] * NN,
                                       'real pos': [None] * NN,
                                       'real chg': [None] * NN}
        for i, p in enumerate(positions):
            self._calc_real(p, i)
            self._calc_recip(p, i)

    def recompute_positions(self, charges):
        """If the potential is to be computed several times for the same set
           of points but different charges, it is much faster to
           precompute certain factors. Call precompute_charges() first
           with a numpy array of cartesian coordinates as rows, then
           call this function to perform the computation with a set of
           charges. It returns an array of potential values for the
           precomputed positions."""
        if self._precomputed_positions is None:
            raise RuntimeError('precompute_positions() must be called before '
                               'recompute_positions().')
        P = self._precomputed_positions
        NN = len(P['recip'])
        charges = np.asarray(charges)
        jellium = np.pi / (self._vol * self._eta) * \
            np.sum(charges) * EwaldField.CONV_FACT
        potentials = np.empty((NN,))
        K = P['recip']
        R = P['real pos']
        C = P['real chg']
        prefactor = 4 * pi / self._vol
        for i in range(NN):
            sfactors = np.sum(charges[None, :] * K[i], 1)
            erecip = np.sum(sfactors / self._gsquares * self._expvals)
            potentials[i] = prefactor * EwaldField.CONV_FACT * erecip.real
            q = charges[C[i]]
            potentials[i] += EwaldField.CONV_FACT * (q * R[i]).sum()
        potentials -= jellium
        return potentials


def _norm(x):
    return np.sqrt(x.dot(x))


class EwaldFieldDirect_UNFINISHED:
    """Adaptation of http://li.mit.edu/A/CourseWork/Ju_Li/Simulation/Ewald/simple_ewald.c"""

    class Params:
        pass

    def __init__(self, lattice, frac_coords, accuracy):
        NP = frac_coords.shape[0]
        PI = np.pi
        ew = EwaldFieldDirect.Params()
        ew.latt = lattice.matrix.copy()
        ew.klatt = lattice.reciprocal_lattice.matrix.copy()
        ew.volume = lattice.volume

        ew.alpha = np.power(NP * PI * PI * PI * 5.5 /
                            ew.volume / ew.volume, 1. / 6.)
        ew.rcut = np.sqrt(-np.log(accuracy)) / ew.alpha * 1.2
        ew.kcut = 2 * ew.alpha * np.sqrt(-np.log(accuracy)) * 1.2

        # the shortest distances to respective surfaces
        rd1 = np.fabs(ew.volume) / _norm(ew.klatt[0, :])
        rd2 = np.fabs(ew.volume) / _norm(ew.klatt[1, :])
        rd3 = np.fabs(ew.volume) / _norm(ew.klatt[2, :])
        kd1 = 2 * PI / _norm(ew.latt[0, :])
        kd2 = 2 * PI / _norm(ew.latt[1, :])
        kd3 = 2 * PI / _norm(ew.latt[2, :])
        ew.max_n1 = int(np.ceil(ew.rcut / rd1))
        ew.max_n2 = int(np.ceil(ew.rcut / rd2))
        ew.max_n3 = int(np.ceil(ew.rcut / rd3))
        ew.max_k1 = int(np.ceil(ew.kcut / kd1))
        ew.max_k2 = int(np.ceil(ew.kcut / kd2))
        ew.max_k3 = int(np.ceil(ew.kcut / kd3))

        ew.s1 = frac_coords[:, 0].copy()
        ew.s2 = frac_coords[:, 1].copy()
        ew.s3 = frac_coords[:, 2].copy()
        EwaldFieldDirect._adjustCoords(ew.s1, ew.s2, ew.s3)

        self.ew = ew

    # Kludge until we update numba.
    @jit(nopython=True, nogil=True)
    def _adjustCoords(s1, s2, s3):
        for i in range(s1.size):
            while ((s1[i] > 0.5) or (s1[i] < -0.5) or
                   (s2[i] > 0.5) or (s2[i] < -0.5) or
                   (s3[i] > 0.5) or (s3[i] < -0.5)):
                if s1[i] < -0.5:
                    s1[i] += 1
                if s1[i] > 0.5:
                    s1[i] -= 1
                if s2[i] < -0.5:
                    s2[i] += 1
                if s2[i] > 0.5:
                    s2[i] -= 1
                if s3[i] < -0.5:
                    s3[i] += 1
                if s3[i] > 0.5:
                    s3[i] -= 1

    def potential_at(self, frac_coords, charges):
        return EwaldFieldDirect._potential_at(np.asarray(frac_coords),
                                              np.asarray(charges),
                                              self.ew.latt, self.ew.klatt,
                                              self.ew.volume, self.ew.alpha,
                                              self.ew.rcut, self.ew.kcut,
                                              self.ew.max_n1, self.ew.max_n2,
                                              self.ew.max_n3, self.ew.max_k1,
                                              self.ew.max_k2, self.ew.max_k3,
                                              self.ew.s1, self.ew.s2, self.ew.s3)

    @jit(nopython=True, nogil=True)
    def _potential_at(frac_coords, charges, latt, klatt, volume, alpha,
                      rcut, kcut, max_n1, max_n2, max_n3,
                      max_k1, max_k2, max_k3,
                      s1, s2, s3):
        CONV_FACT = 1e10 * ELECTRON_CHARGE / (4 * pi * EPSILON_0)
        potential = 0.0
        h11 = latt[0, 0]
        h12 = latt[0, 1]
        h13 = latt[0, 2]
        h21 = latt[1, 0]
        h22 = latt[1, 1]
        h23 = latt[1, 2]
        h31 = latt[2, 0]
        h32 = latt[2, 1]
        h33 = latt[2, 2]
        g11 = klatt[0, 0]
        g12 = klatt[0, 1]
        g13 = klatt[0, 2]
        g21 = klatt[1, 0]
        g22 = klatt[1, 1]
        g23 = klatt[1, 2]
        g31 = klatt[2, 0]
        g32 = klatt[2, 1]
        g33 = klatt[2, 2]
        for j in range(s1.size):
            pot = 0.0
            ds1 = s1[j] - frac_coords[0]
            ds2 = s2[j] - frac_coords[1]
            ds3 = s3[j] - frac_coords[2]

            # real space lattice sum
            for n1 in range(-max_n1, max_n1 + 1):
                for n2 in range(-max_n2, max_n2 + 1):
                    for n3 in range(-max_n3, max_n3 + 1):
                        dx = (n1 + ds1) * h11 + (n2 + ds2) * \
                            h21 + (n3 + ds3) * h31
                        dy = (n1 + ds1) * h12 + (n2 + ds2) * \
                            h22 + (n3 + ds3) * h32
                        dz = (n1 + ds1) * h13 + (n2 + ds2) * \
                            h23 + (n3 + ds3) * h33
                        r2 = dx * dx + dy * dy + dz * dz
                        if r2 < rcut * rcut:
                            r = np.sqrt(r2)
                            pot += erfc(alpha * r) / r if r != 0. \
                                else 0.0  # -2.*alpha/np.sqrt(np.pi)

            # reciprocal space lattice sum
            for k1 in range(-max_k1, max_k1 + 1):
                for k2 in range(-max_k2, max_k2 + 1):
                    for k3 in range(-max_k3, max_k3 + 1):
                        if not ((k1 == 0) and (k2 == 0) and (k3 == 0)):
                            dx = k1 * g11 + k2 * g21 + k3 * g31
                            dy = k1 * g12 + k2 * g22 + k3 * g32
                            dz = k1 * g13 + k2 * g23 + k3 * g33
                            r2 = dx * dx + dy * dy + dz * dz
                            # use inversion symmetry
                            # WTF!? TODO
                            if r2 <= kcut * kcut:
                                pot += 8. * np.pi / r2 / volume * \
                                    np.exp(-r2 / alpha / alpha / 4.) \
                                    * np.cos(2 * np.pi * (k1 * ds1 + k2 * ds2 + k3 * ds3))  # \
                                # if k3 > 0 else \
                                # 4.*np.pi/r2/volume * \
                                #np.exp(-r2/alpha/alpha/4.) \
                                #* np.cos(2*np.pi*(k1*ds1+k2*ds2+k3*ds3))

            potential += charges[j] * pot
        jellium = np.pi / (volume * alpha * alpha) * \
            np.sum(charges) * CONV_FACT
        return potential * CONV_FACT - jellium


def _test_poisson3d():
    print('Testing Poisson3D by comparing to Ewald summation.')
    print(

        """
    Four point charges are placed in a small triclinic unit cell.
    Three sample points are chosen in the cell and the fields there
    are computed using the Poisson3D and EwaldField methods. Relative
    differences between the computed values are printed. Do not expect
    miracles as the Poisson3D method is ill-suited to represent fields
    of point charges. The errors should become smaller with increasing
    grid size.
    """)

    # Create a unit cell about 4Å in size.
    basis = mg.Lattice.from_lengths_and_angles([4.2, 3.5, 3.8], [80, 92, 111])

    # Create several charges in non-symmetric positions. All charges
    # must come in pairs to maintain total charge neutrality. Note,
    # however, that these positions are not exactly the same as the
    # positions actually used later, because those are tweaked to lie
    # exactly on the grid.
    positions = [[0.0, 0.0, 0.0],
                 [2.0, 0.0, 0.0],
                 [1.6, 2.4, 0.5],
                 [1.6, 2.4, 1.7]]
    props = {'charge': [2.8, -2.8, 1.0, -1.0]}
    species = ['Mg', 'O', 'H', 'H']  # unimportant
    struct = mg.Structure(basis, species, positions,
                          coords_are_cartesian=True,
                          site_properties=props)
    probes = np.array([[0.5, 0.5, 0.5],
                       [0.0, 1.0, 0.0],
                       [1.0, 1.0, 1.0]])

    # Compute potential using Ewald summation.
    def ewd(gridShape):
        # Clamp sites to the grid.
        newstruct = struct.copy()
        sh = np.asarray(gridShape)
        for i, s in enumerate(newstruct):
            c = (np.floor(sh * s.frac_coords)) / sh
            newstruct.replace(i, s.specie, c, properties=s.properties)
        P = [basis.get_fractional_coords(p) for p in probes]
        P = [(np.floor(sh * p)) / sh for p in P]
        P = [basis.get_cartesian_coords(p) for p in P]

        # Compute the potential as the difference between full Ewald
        # energy and both reference states (i.e. isolated
        # field-generating charges and isolated test charges)
        ewald = EwaldField(newstruct)
        refpotential = np.array([ewald.potential_at(p) for p in P])

        # A hack to test the unfinished Direct method.
        #ewald = EwaldFieldDirect(newstruct.lattice, newstruct.frac_coords, 1e-6)
        #refpotential = np.array([ewald.potential_at(newstruct.lattice.get_fractional_coords(p), newstruct.site_properties['charge']) for p in P])

        return refpotential

    # Compute potential using Poisson3D.
    poisson = None

    def p3d(gridShape):
        # Permittivity in units of e0 / V / Å
        eps = 1e-10 * 8.854187817620e-12 / 1.602176565e-19
        rho = np.zeros(gridShape)
        indices = [[int(x * g) for x, g in
                    zip(site.frac_coords, gridShape)]
                   for site in struct]
        voxel = basis.matrix.copy()
        for i in range(3):
            voxel[i, :] /= gridShape[i]
        voxvol = voxel[0, :].dot(np.cross(voxel[1, :], voxel[2, :]))
        for i, site in enumerate(indices):
            rho[site[0], site[1], site[2]] = -props['charge'][i] / voxvol / eps
        phi = poisson.solve(rho)

        P = [basis.get_fractional_coords(p) for p in probes]
        potential = np.array([phi[tuple([int(x * g) for x, g in
                                         zip(p, gridShape)])]
                              for p in P])
        return potential

    print('# grid           relative errors')
    for grid in ([31, 32, 33], [61, 70, 80], [81, 82, 83],
                 [128, 128, 128], [256, 256, 256]):
        print(grid, end=' ', flush=True)
        poisson = Poisson3D(basis.matrix, grid)
        pot = p3d(grid)
        refpot = ewd(grid)
        print((pot - refpot) / refpot)

    print("""
    Running again with memoization enabled. Each run will be repeated
    twice; the second time should be faster and produce identical
    results.
    """)
    print('# grid           relative errors')
    for grid in ([31, 32, 33], [61, 70, 80], [81, 82, 83],
                 [128, 128, 128], [256, 256, 256]):
        poisson = Poisson3D(basis.matrix, grid)
        for repeat in range(2):
            print(grid, end=' ', flush=True)
            pot = p3d(grid)
            refpot = ewd(grid)
            print((pot - refpot) / refpot)


if __name__ == '__main__':
    _test_poisson3d()
