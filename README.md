# poisson3d

A solver for poisson equation on a 3-dimensional periodic domain with
non-orthogonal boundaries. An example of use:
````python
# Calculating electrostatic field
from poisson3d import Poisson3D
from numpy import array
import pymatgen

charge_density = get_charge_density_from_somewhere()
cif = pymatgen.Structure.from_file('structure.cif')
basis = cif.lattice
solver = Poisson3D(basis)
field = solver.solve(-charge_density / eps0)
````

More information is available in doc strings while the maths are
described in the `maths.html` document.

## Authors

This package is written by Jure Varlec <jure.varlec@ki.si> and is
available under the MIT license.
