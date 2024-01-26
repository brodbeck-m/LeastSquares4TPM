# --- Imports ---
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
import typing

import basix
import dolfinx
import ufl

import dolfinx.fem as dfem
import dolfinx.fem.petsc as dfem_petsc
import dolfinx.mesh as dmesh

# --- Input parameters ---
# Material
nuhS = 0.25

# Discretisation
sdisc_eorder = 1

sdisc_nelmt = 2
sdisc_nref = 8

# --- Auxiliaries ---
pi_1 = 2 * nuhS / (1 - 2 * nuhS)
ktD_tilde = 1.0e-0


# Interpolate ufl-function into dolfinx-function
def interpolate_ufl_to_function(f_ufl: typing.Any, f: dfem.Function):
    # Create expression
    expr = dfem.Expression(f_ufl, f.function_space.element.interpolation_points())

    # Perform interpolation
    f.interpolate(expr)


# Calculate error (u, p, wtFS)
def calculate_error(uh, u_ex, norm="l2", degree_raise=3):
    # The mesh
    mesh = uh.function_space.mesh
    sdim = mesh.topology.dim

    # Create higher order function space
    degree = uh.function_space.ufl_element().degree()
    family = uh.function_space.ufl_element().family()

    if uh.function_space.num_sub_spaces == 0:
        felmt = basix.ufl.element(family, mesh.basix_cell(), degree + degree_raise)
    elif uh.function_space.num_sub_spaces == sdim:
        felmt = basix.ufl.element(
            family, mesh.basix_cell(), degree + degree_raise, shape=(sdim,)
        )
    else:
        raise NotImplementedError("Unknown function space!")

    W = dfem.functionspace(mesh, felmt)

    # Interpolate approximate solution
    u_W = dfem.Function(W)
    u_W.interpolate(uh)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = dfem.Function(W)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = dfem.Expression(u_ex, W.element.interpolation_points())
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)

    # Compute the error in the higher order function space
    e_W = dfem.Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    if norm == "l2":
        error = dfem.form(ufl.inner(e_W, e_W) * ufl.dx)
        ref_val = dfem.form(ufl.inner(u_ex_W, u_ex_W) * ufl.dx)
    elif norm == "h1":
        error = dfem.form(ufl.inner(ufl.grad(e_W), ufl.grad(e_W)) * ufl.dx)
        ref_val = dfem.form(ufl.inner(ufl.grad(u_ex_W), ufl.grad(u_ex_W)) * ufl.dx)
    elif norm == "hdiv":
        error = dfem.form(ufl.inner(ufl.div(e_W), ufl.div(e_W)) * ufl.dx)
        ref_val = dfem.form(ufl.inner(ufl.div(u_ex_W), ufl.div(u_ex_W)) * ufl.dx)
    else:
        raise ValueError("Unknown error norm")

    error_local = dfem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)

    ref_val_local = dfem.assemble_scalar(ref_val)
    ref_val_global = mesh.comm.allreduce(ref_val_local, op=MPI.SUM)

    return np.sqrt(error_global / ref_val_global)


def calculate_error_ufl(uh, u_ex, norm="l2", degree_raise=3):
    # The mesh
    domain = uh.function_space.mesh
    sdim = domain.topology.dim

    # Check if u_ex is a ufl expression
    if not isinstance(u_ex, ufl.core.expr.Expr):
        raise ValueError("Exact solution has to be of ufl-type!")

    # Compute the error in the higher order function space
    err_u = uh - u_ex

    # Integrate the error
    dvol = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 7})
    if norm == "l2":
        error = dfem.form(ufl.inner(err_u, err_u) * dvol)
        ref_val = dfem.form(ufl.inner(u_ex, u_ex) * dvol)
    elif norm == "h1":
        error = dfem.form(ufl.inner(ufl.grad(err_u), ufl.grad(err_u)) * dvol)
        ref_val = dfem.form(ufl.inner(ufl.grad(u_ex), ufl.grad(u_ex)) * dvol)
    elif norm == "hdiv":
        error = dfem.form(ufl.inner(ufl.div(err_u), ufl.div(err_u)) * dvol)
        ref_val = dfem.form(ufl.inner(ufl.div(u_ex), ufl.div(u_ex)) * dvol)
    else:
        raise ValueError("Unknown error norm")

    error_local = dfem.assemble_scalar(error)
    error_global = domain.comm.allreduce(error_local, op=MPI.SUM)

    ref_val_local = dfem.assemble_scalar(ref_val)
    ref_val_global = domain.comm.allreduce(ref_val_local, op=MPI.SUM)

    return np.sqrt(error_global / ref_val_global)


def calculate_stress_error(sigh, u_ex, p_ex, pi_1, norm="l2", degree_raise=3):
    # Approximation details
    quad_degree = uh.function_space.ufl_element().degree() + degree_raise
    domain = uh.function_space.mesh

    # The exact stress
    sig_ex = exact_stress(u_ex, p_ex, pi_1)

    # Recast tensor form of sigh
    sig = ufl.as_matrix([[sigh[0][0], sigh[0][1]], [sigh[1][0], sigh[1][1]]])

    # Compute error
    err_sig = sig - sig_ex
    dvol = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": quad_degree})

    if norm == "l2":
        error = dfem.form(ufl.inner(err_sig, err_sig) * dvol)
    elif norm == "hdiv":
        error = dfem.form(ufl.inner(ufl.div(err_sig), ufl.div(err_sig)) * dvol)
    else:
        raise ValueError("Unknown error norm")

    error_local = dfem.assemble_scalar(error)
    error_global = domain.comm.allreduce(error_local, op=MPI.SUM)

    return np.sqrt(error_global)


# --- Analytic solution ---
def exact_displacement(x, pi_1):
    a = 1e-1
    h = 1 / (2 * pi_1)
    return ufl.as_vector(
        [
            a * (ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]) + h * x[0] * x[0]),
            a * (-ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) + h * x[1] * x[1]),
        ]
    )


def exact_pressure(x):
    b = 5e-1
    return b * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def exact_stress(u_ex, p_ex, pi_1):
    EtS = ufl.sym(ufl.grad(u_ex))
    return 2 * EtS + pi_1 * ufl.div(u_ex) * ufl.Identity(2) - p_ex * ufl.Identity(2)


def exact_seepage(u_ex, p_ex, ktD_tilde):
    return -ktD_tilde * ufl.grad(p_ex)


# --- Numerical solution ---
def solve_problem(sdisc_nelmt, sdisc_eorder, pi_1, ktD_tilde):
    # --- Create mesh
    # The mesh
    domain = dmesh.create_unit_square(
        MPI.COMM_WORLD, sdisc_nelmt, sdisc_nelmt, dmesh.CellType.triangle
    )

    # The boundary facets
    domain.topology.create_connectivity(1, 2)
    boundary_facets = dmesh.exterior_facet_indices(domain.topology)

    # --- The exact solution
    # The spacial positions
    x = ufl.SpatialCoordinate(domain)

    # Displacement and pressure
    u_ext = exact_displacement(x, pi_1)
    p_ext = exact_pressure(x)
    sig_ext = exact_stress(u_ext, p_ext, pi_1)
    wtfs_ext = exact_seepage(u_ext, p_ext, ktD_tilde)

    # --- Function spaces
    P_u = basix.ufl.element("Lagrange", domain.basix_cell(), sdisc_eorder, shape=(2,))
    P_p = basix.ufl.element("Lagrange", domain.basix_cell(), sdisc_eorder)
    P_sig = basix.ufl.element("RT", domain.basix_cell(), sdisc_eorder)
    P_wfs = basix.ufl.element("RT", domain.basix_cell(), sdisc_eorder)
    P_l2 = basix.ufl.element("DG", domain.basix_cell(), 0, shape=(2,))

    V = dfem.functionspace(
        domain, basix.ufl.mixed_element([P_u, P_p, P_sig, P_sig, P_wfs, P_l2])
    )

    # Solution and history functions
    uh = dfem.Function(V)

    # --- Weak form
    # Trial- and test functions
    u, p, sig1, sig2, wfs, l = ufl.TrialFunctions(V)
    v_u, v_p, v_sig1, v_sig2, v_wfs, v_l = ufl.TestFunctions(V)

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

    # RHS for manufactured solution
    rhs_blm = ufl.div(sig_ext)
    rhs_bmo = ufl.div(u_ext) + ufl.div(wtfs_ext)

    # Definition weak form
    dvol = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 10})

    res = (
        ufl.inner(ufl.div(u) + ufl.div(wfs) - rhs_bmo, ufl.div(v_u) + ufl.div(v_wfs))
        + ufl.inner(ktD_tilde * ufl.grad(p) + wfs, ktD_tilde * ufl.grad(v_p) + v_wfs)
        + ufl.inner(
            EtS_u(u) - EtS_sig(sig, p, pi_1),
            EtS_u(v_u) - EtS_sig(v_sig, v_p, pi_1),
        )
        + ufl.inner(ufl.div(sig) - rhs_blm, v_l)
        + ufl.inner(l, ufl.div(v_sig))
    ) * dvol

    a = dfem.form(ufl.lhs(res))
    l = dfem.form(ufl.rhs(res))

    # --- Boundary conditions
    # Initialise list
    list_bcs = []

    # Collapsed function spaces (displacement and pressure)
    V_u, _ = V.sub(0).collapse()
    V_p, _ = V.sub(1).collapse()

    # Boundary values from exact solution
    uD = dfem.Function(V_u)
    interpolate_ufl_to_function(u_ext, uD)

    pD = dfem.Function(V_p)
    interpolate_ufl_to_function(p_ext, pD)

    # Set BCs
    dofs = dfem.locate_dofs_topological((V.sub(0), V_u), 1, boundary_facets)
    list_bcs.append(dfem.dirichletbc(uD, dofs, V.sub(0)))

    dofs = dfem.locate_dofs_topological((V.sub(1), V_p), 1, boundary_facets)
    list_bcs.append(dfem.dirichletbc(pD, dofs, V.sub(1)))

    # --- Initialise solver
    # Assemble system matrix and RHS
    A = dfem_petsc.assemble_matrix(a, bcs=list_bcs)
    A.assemble()

    # Initialise RHS
    L = dfem_petsc.create_vector(l)

    # Initialise solver
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)
    solver.setTolerances(rtol=1e-14, atol=1e-14)

    # Configure mumps
    solver.setType(PETSc.KSP.Type.PREONLY)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")

    # --- Solve system
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

    # Collapse functions
    uh_u = uh.sub(0).collapse()
    uh_p = uh.sub(1).collapse()
    uh_sig1 = uh.sub(2).collapse()
    uh_sig2 = uh.sub(3).collapse()
    uh_wfs = uh.sub(4).collapse()

    # Evaluate error
    sig_h = ufl.as_matrix([[uh_sig1[0], uh_sig1[1]], [uh_sig2[0], uh_sig2[1]]])

    form_err_divsig = dfem.form(
        ufl.inner(ufl.div(sig_h) - rhs_blm, ufl.div(sig_h) - rhs_blm) * dvol
    )
    form_rhsblm = dfem.form(ufl.inner(rhs_blm, rhs_blm) * dvol)
    form_lsfunc = dfem.form(
        (
            ufl.inner(
                ufl.div(uh_u) + ufl.div(uh_wfs) - rhs_bmo,
                ufl.div(uh_u) + ufl.div(uh_wfs) - rhs_bmo,
            )
            + ufl.inner(
                ktD_tilde * ufl.grad(uh_p) + uh_wfs, ktD_tilde * ufl.grad(uh_p) + uh_wfs
            )
            + ufl.inner(
                EtS_u(uh_u) - EtS_sig(sig_h, uh_p, pi_1),
                EtS_u(uh_u) - EtS_sig(sig_h, uh_p, pi_1),
            )
            + ufl.inner(ufl.div(sig_h) - rhs_blm, ufl.div(sig_h) - rhs_blm)
        )
        * dvol
    )

    lsfunc_local = dfem.assemble_scalar(form_lsfunc)
    lsfunc_global = domain.comm.allreduce(lsfunc_local, op=MPI.SUM)

    error_divsig_local = dfem.assemble_scalar(form_err_divsig)
    resblm_local = dfem.assemble_scalar(form_rhsblm)
    error_divsig_global = domain.comm.allreduce(error_divsig_local, op=MPI.SUM)
    resblm_global = domain.comm.allreduce(resblm_local, op=MPI.SUM)

    print(
        "Error stress (pure/normalised): {}, {} | LS functional: {}".format(
            error_divsig_global, error_divsig_global / resblm_global, lsfunc_global
        )
    )

    # Output results
    out_name = "convstudy_nelmt-" + str(int(sdisc_nelmt)) + ".xdmf"
    outfile = dolfinx.io.XDMFFile(MPI.COMM_WORLD, out_name, "w")
    outfile.write_mesh(domain)

    uh_u.name = "uh"
    uh_p.name = "ph"
    uD.name = "u_ext"
    pD.name = "p_ext"

    outfile.write_function(uh_u, 0.0)
    outfile.write_function(uh_p, 0.0)
    outfile.write_function(uD, 0.0)
    outfile.write_function(pD, 0.0)
    outfile.close()

    return uh, u_ext, p_ext, lsfunc_global


# --- Convergence study ---
# Initialise storage for results
results = np.zeros((sdisc_nref, 12))

# Perform simulations
for n in range(0, sdisc_nref):
    # Number of mesh elements
    n_elmt = sdisc_nelmt * (2**n)

    results[n, 0] = n_elmt
    results[n, 1] = 1 / n_elmt

    # Solve problem
    uh, u_ext, p_ext, value_lsfunc = solve_problem(
        n_elmt, sdisc_eorder, pi_1, ktD_tilde
    )

    # --- Evaluate errors,
    # Collapse function
    sub_u = uh.sub(0).collapse()
    sub_p = uh.sub(1).collapse()
    sub_sig1 = uh.sub(2).collapse()
    sub_sig2 = uh.sub(3).collapse()
    sub_wtfs = uh.sub(4).collapse()

    # Evaluate errors
    results[n, 2] = calculate_error_ufl(sub_u, u_ext, norm="h1")
    results[n, 4] = calculate_error_ufl(sub_p, p_ext, norm="h1")
    results[n, 6] = calculate_stress_error(
        [sub_sig1, sub_sig2], u_ext, p_ext, pi_1, norm="hdiv"
    )
    results[n, 8] = calculate_error_ufl(
        sub_wtfs, exact_seepage(u_ext, p_ext, ktD_tilde), norm="hdiv"
    )
    results[n, 10] = value_lsfunc

# Compute convergence rates
results[1:, 3] = np.log(results[1:, 2] / results[:-1, 2]) / np.log(
    results[1:, 1] / results[:-1, 1]
)
results[1:, 5] = np.log(results[1:, 4] / results[:-1, 4]) / np.log(
    results[1:, 1] / results[:-1, 1]
)
results[1:, 7] = np.log(results[1:, 6] / results[:-1, 6]) / np.log(
    results[1:, 1] / results[:-1, 1]
)
results[1:, 9] = np.log(results[1:, 8] / results[:-1, 8]) / np.log(
    results[1:, 1] / results[:-1, 1]
)
results[1:, 11] = np.log(results[1:, 10] / results[:-1, 10]) / np.log(
    results[1:, 1] / results[:-1, 1]
)

# Results to csv
out_name = "convstudy_pi1-{}_ktD-{}".format(round(pi_1), ktD_tilde)
out_name = out_name.replace(".", "d")
out_name += ".csv"

header = "nelmt, h, err_u, rate_u, err_p, rate_p, err_sig, rate_sig, err_wtFS, rate_wtFS, lsfunc, rate_lsfunc"
np.savetxt(out_name, results, header=header, delimiter=",")
