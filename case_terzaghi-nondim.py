# --- Imports ---
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from typing import List

import basix
import dolfinx
import ufl

import dolfinx.fem as dfem
import dolfinx.fem.petsc as dfem_petsc
import dolfinx.mesh as dmesh
from dolfinx.mesh import CellType, DiagonalType, create_rectangle


# --- Input parameters ---
# Material
pi_1 = 1

# Id for incompressible problem
is_incomp = True

# Discretisation
sdisc_nelmt = [9, 72]
# sdisc_nelmt = [9 * 3, 72 * 3]
tdisc_dt = 0.005

# Load
bc_qtop = -0.1


# --- Auxiliaries ---
def create_geometry_rectangle(
    l_domain: List[float],
    n_elmt: List[int],
    diagonal: DiagonalType = DiagonalType.left,
):
    # --- Create mesh
    mesh = create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([l_domain[0], l_domain[1]])],
        [n_elmt[0], n_elmt[1]],
        cell_type=CellType.triangle,
        diagonal=diagonal,
    )
    tol = 1.0e-14
    boundaries = [
        (1, lambda x: np.isclose(x[0], 0)),
        (2, lambda x: np.isclose(x[1], 0)),
        (3, lambda x: np.isclose(x[0], l_domain[0])),
        (4, lambda x: np.isclose(x[1], l_domain[1])),
    ]

    facet_indices, facet_markers = [], []
    for marker, locator in boundaries:
        facets = dmesh.locate_entities(mesh, 1, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full(len(facets), marker))

    facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
    facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = dmesh.meshtags(
        mesh, 1, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)

    return mesh, facet_tag, ds


def set_boundary_conditions(facet_tags, V, bc_qtop):
    # Compute required connectivity's
    domain.topology.create_connectivity(1, 2)

    # Extract subspaces
    V_u, _ = V.sub(0).collapse()
    V_p, _ = V.sub(1).collapse()
    V_sig, _ = V.sub(2).collapse()
    V_wfs, _ = V.sub(4).collapse()

    # Interpolate dirichlet conditions
    class TractionBC:
        def __init__(self, load_qtop):
            self.load = load_qtop

        def __call__(self, x):
            traction = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
            traction[0] = 0
            traction[1] = self.load

            return traction

    u_zero = dfem.Function(V_u)
    p_zero = dfem.Function(V_p)
    sig_zero = dfem.Function(V_sig)
    sig_top = dfem.Function(V_sig)
    wfs_zero = dfem.Function(V_wfs)

    traction_bc = TractionBC(bc_qtop)
    sig_top.interpolate(traction_bc)

    list_bc = []

    # --- Displacement boundaries
    # Bottom
    facets = facet_tags.indices[facet_tags.values == 2]
    dofs = dfem.locate_dofs_topological((V.sub(0).sub(1), V_u.sub(1)), 1, facets)
    list_bc.append(dfem.dirichletbc(u_zero, dofs, V.sub(0)))

    # Left
    facets = facet_tags.indices[facet_tags.values == 1]
    dofs = dfem.locate_dofs_topological((V.sub(0).sub(0), V_u.sub(0)), 1, facets)
    list_bc.append(dfem.dirichletbc(u_zero, dofs, V.sub(0)))

    # Right
    facets = facet_tags.indices[facet_tags.values == 3]
    dofs = dfem.locate_dofs_topological((V.sub(0).sub(0), V_u.sub(0)), 1, facets)
    list_bc.append(dfem.dirichletbc(u_zero, dofs, V.sub(0)))

    # --- Traction boundaries
    # Top
    facets = facet_tags.indices[facet_tags.values == 4]
    dofs = dfem.locate_dofs_topological((V.sub(2), V_sig), 1, facets)
    list_bc.append(dfem.dirichletbc(sig_zero, dofs, V.sub(2)))

    facets = facet_tags.indices[facet_tags.values == 4]
    dofs = dfem.locate_dofs_topological((V.sub(3), V_sig), 1, facets)
    list_bc.append(dfem.dirichletbc(sig_top, dofs, V.sub(3)))

    # Left/ Right
    facets = facet_tags.indices[facet_tags.values == 1]
    dofs = dfem.locate_dofs_topological((V.sub(3), V_sig), 1, facets)
    list_bc.append(dfem.dirichletbc(sig_zero, dofs, V.sub(3)))

    facets = facet_tags.indices[facet_tags.values == 3]
    dofs = dfem.locate_dofs_topological((V.sub(3), V_sig), 1, facets)
    list_bc.append(dfem.dirichletbc(sig_zero, dofs, V.sub(3)))

    # Bottom
    facets = facet_tags.indices[facet_tags.values == 2]
    dofs = dfem.locate_dofs_topological((V.sub(2), V_sig), 1, facets)
    list_bc.append(dfem.dirichletbc(sig_zero, dofs, V.sub(2)))

    # --- Pressure boundaries
    # Top
    facets = facet_tags.indices[facet_tags.values == 4]
    dofs = dfem.locate_dofs_topological((V.sub(1), V_p), 1, facets)
    list_bc.append(dfem.dirichletbc(p_zero, dofs, V.sub(1)))

    # --- Flux boundaries
    # Bottom
    facets = facet_tags.indices[facet_tags.values == 2]
    dofs = dfem.locate_dofs_topological((V.sub(4), V_wfs), 1, facets)
    list_bc.append(dfem.dirichletbc(wfs_zero, dofs, V.sub(4)))

    # Left
    facets = facet_tags.indices[facet_tags.values == 1]
    dofs = dfem.locate_dofs_topological((V.sub(4), V_wfs), 1, facets)
    list_bc.append(dfem.dirichletbc(wfs_zero, dofs, V.sub(4)))

    # Right
    facets = facet_tags.indices[facet_tags.values == 3]
    dofs = dfem.locate_dofs_topological((V.sub(4), V_wfs), 1, facets)
    list_bc.append(dfem.dirichletbc(wfs_zero, dofs, V.sub(4)))

    return list_bc


# --- Problem setup ---
# --- Set geometry
domain, facet_tags, ds = create_geometry_rectangle([1.0, 5.0], sdisc_nelmt)

# --- Set function-space
P_u = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(2,))
P_p = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
P_sig = basix.ufl.element("BDM", domain.basix_cell(), 1)
P_wfs = basix.ufl.element("BDM", domain.basix_cell(), 1)
P_l2 = basix.ufl.element("DG", domain.basix_cell(), 0)

V = dfem.functionspace(
    domain, basix.ufl.mixed_element([P_u, P_p, P_sig, P_sig, P_wfs, P_l2, P_l2])
)

V_u, uh_to_u = V.sub(0).collapse()

# Solution and history functions
uh = dfem.Function(V)
uh_n = dfem.Function(V_u)

# Interpolation of stresses
P_sig_expt = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(2, 2))
V_sig_expt = dfem.functionspace(domain, P_sig_expt)

V_sig1, uh_to_sig1 = V.sub(2).collapse()
V_sig2, uh_to_sig2 = V.sub(3).collapse()

sigma_1 = dfem.Function(V_sig1)
sigma_2 = dfem.Function(V_sig2)

sigma_expt = dfem.Function(V_sig_expt)

expr_sigma = dfem.Expression(
    ufl.as_matrix([[sigma_1[0], sigma_1[1]], [sigma_2[0], sigma_2[1]]]),
    V_sig_expt.element.interpolation_points(),
)

# --- Set weak form
# Trial- and test functions
u, p, sig1, sig2, wfs, l1, l2 = ufl.TrialFunctions(V)
v_u, v_p, v_sig1, v_sig2, v_wfs, v_l1, v_l2 = ufl.TestFunctions(V)

# Definition stress tensor
sig = ufl.as_matrix([[sig1[0], sig1[1]], [sig2[0], sig2[1]]])
v_sig = ufl.as_matrix([[v_sig1[0], v_sig1[1]], [v_sig2[0], v_sig2[1]]])


# Kinematics
def EtS_u(u):
    return ufl.sym(ufl.grad(u))


# Definition constitutive laws
def EtS_sig(sig, p, pi_1):
    h_pi1 = 2 * (pi_1 + 1)

    A_sig = 0.5 * (sig - (pi_1 / (2 * h_pi1)) * ufl.tr(sig) * ufl.Identity(2))
    corr_p = (1 / h_pi1) * p * ufl.Identity(2)

    return A_sig + corr_p


# Definition weak form
dt = tdisc_dt
dvol = ufl.dx

res = (
    ufl.inner(ufl.div(u - uh_n) + ufl.div(wfs), ufl.div(v_u) + ufl.div(v_wfs))
    + ufl.inner(dt * ufl.grad(p) + wfs, dt * ufl.grad(v_p) + v_wfs)
    + ufl.inner(
        EtS_u(u) - EtS_sig(sig, p, pi_1),
        EtS_u(v_u) - EtS_sig(v_sig, v_p, pi_1),
    )
    + ufl.inner(ufl.div(sig), ufl.as_vector([v_l1, v_l2]))
    + ufl.inner(ufl.as_vector([l1, l2]), ufl.div(v_sig))
) * dvol

a = dfem.form(ufl.lhs(res))
l = dfem.form(ufl.rhs(res))

list_bcs = set_boundary_conditions(facet_tags, V, bc_qtop)

# --- Initialise solver
# Stiffness matrix
A = dfem_petsc.assemble_matrix(a, bcs=list_bcs)
A.assemble()

# Initialise RHS
L = dfem_petsc.create_vector(l)

# Initialise solver
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setOperators(A)
solver.setTolerances(rtol=1e-12, atol=1e-12)

# Configure mumps
solver.setType(PETSc.KSP.Type.PREONLY)
pc = solver.getPC()
pc.setType(PETSc.PC.Type.LU)
pc.setFactorSolverType("mumps")

# --- Solve problem ---
# Initialize physical time
time = 0.0

# Initialize history values
uh_n.x.array[:] = 0.0

# Initialize export ParaView
outfile = dolfinx.io.XDMFFile(
    MPI.COMM_WORLD, "test_terzaghi-LeastSquares-nondim.xdmf", "w"
)
outfile.write_mesh(domain)

# Time loop
duration_solve = 0.0
for n in range(10):
    # Update time
    time = time + tdisc_dt

    # Calculate current solution
    duration_solve -= MPI.Wtime()

    # Assemble RHS
    with L.localForm() as loc_L:
        loc_L.set(0)

    dfem.petsc.assemble_vector(L, l)
    dfem.apply_lifting(L, [a], [list_bcs])
    L.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    dfem.set_bc(L, list_bcs)

    # Solve equation system
    solver(L, uh.vector)
    uh.x.scatter_forward()

    duration_solve += MPI.Wtime()

    PETSc.Sys.Print("Phys. Time {:.4f}, Calc. Time {:.4f}".format(time, duration_solve))

    uh_n.x.array[:] = uh.x.array[uh_to_u]
    sigma_1.x.array[:] = uh.x.array[uh_to_sig1]
    sigma_2.x.array[:] = uh.x.array[uh_to_sig2]

    u = uh.sub(0).collapse()
    p = uh.sub(1).collapse()
    sigma_expt.interpolate(expr_sigma)

    u.name = "u_h"
    outfile.write_function(u, time)
    p.name = "p_h"
    outfile.write_function(p, time)
    sigma_expt.name = "sig_h"
    outfile.write_function(sigma_expt, time)

outfile.close()
