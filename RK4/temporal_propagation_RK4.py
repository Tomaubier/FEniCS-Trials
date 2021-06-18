import fenics.gmsh_helpers as gmsh_helpers
import dolfinx, gmsh, ufl, time, os
from petsc4py import PETSc
from mpi4py import MPI
from tqdm import tqdm
import numpy as np

# %%

# Physical quantities
tx_freq = 200e3 # Hz
c0 = 1481 # m.s-1
omega = 2*np.pi*tx_freq
wave_len = c0 / tx_freq

# Transducer properties
tx_radius = 15e-2
tx_aperture_radius = 15e-2
alpha_aperture = np.arcsin(tx_aperture_radius / (2*tx_radius))

# Domain parameters
topology_dim = 2
dims = (20e-2, 20e-2)
tx_marker, boundary_marker, ac_domain_marker = 1, 2, 3

# Spatial discretisation parameters
degree = 3
n_wave = 3
h_elem = wave_len / n_wave

# Temporal discretisation parameters
CFL = .5 #.5
dt = (CFL * h_elem) / c0
simulation_duration = 15e-5
t_mesh = np.arange(0, simulation_duration, dt)
n_rk_steps = 4

# --- Check the CFL (Courant Friecrishs Lewy) value for the discretization ---

print(f'ndt_wave = {1/(tx_freq * dt):.0f}')
print(f'The scheme {"should be" if CFL <= np.pi/2 else "is not"} stable as CFL = {CFL:.2f} {"<=" if CFL <= np.pi/2 else ">"} pi/2.')

# %% --- Mesh generation ---

gmsh.initialize()

model = gmsh.model
model.add("DummyRectDomain")

# Points defining the geometry of the domain
points = []
# Corners of the acoustic domain
rect_domain_geom = [[0., -1/2], [1., -1/2], [1., 1/2], [0., 1/2]]
for rect_geom in rect_domain_geom:
    points.append(model.geo.addPoint(*[gg*dd for gg, dd in zip(rect_geom, dims)],0, meshSize=h_elem, tag=len(points)))

# Lines defining the limits of the domain
lines = []
# Acoustic domain boundaries
for pt in points:
    lines.append(model.geo.addLine(points[pt], points[(pt+1)%4]))

# Junction of the individual boubaries and definition of the acoustic domain surface
curveloop = model.geo.addCurveLoop(lines)
domain = model.geo.addPlaneSurface([curveloop])

# This command is mandatory and synchronize CAD with GMSH Model. The less you launch it, the better it is for performance purpose
gmsh.model.geo.synchronize()

# Assigns the various geometry elements to physical groups
gmsh.model.addPhysicalGroup(1, [lines[-1]], tx_marker)
gmsh.model.setPhysicalName(1, tx_marker, "Tx")
gmsh.model.addPhysicalGroup(1, lines[:-1], boundary_marker)
gmsh.model.setPhysicalName(1, boundary_marker, "Domain boundary")
gmsh.model.addPhysicalGroup(2, [domain], ac_domain_marker)
gmsh.model.setPhysicalName(2, ac_domain_marker, "Acoustic domain")

# Mesh generation and
model.mesh.generate(topology_dim)
gmsh.write("DummyRectDomain.msh")

# Gmsh to dolfinx mesh
mesh, cell_tags, facet_tags = gmsh_helpers.gmsh_model_to_mesh(model, cell_data=True, facet_data=True, gdim=topology_dim)

# Finalize GMSH
gmsh.finalize()

# %% --- Variational problem ---

# v_cg1 = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)
# p_cg1 = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)
# V = dolfinx.FunctionSpace(mesh, v_cg1)
# P = dolfinx.FunctionSpace(mesh, p_cg1)
V = dolfinx.FunctionSpace(mesh, ("Lagrange", degree))
P = dolfinx.FunctionSpace(mesh, ("Lagrange", degree))

v = ufl.TrialFunction(V)
u = ufl.TestFunction(V)
p = ufl.TrialFunction(P)
q = ufl.TestFunction(P)

# --- Boundary Conditions ---

# Retreive the transducer facet indexes to apply the bc later on
tx_facets = facet_tags.indices[facet_tags.values == tx_marker]
tx_dofs = dolfinx.fem.locate_dofs_topological(P, topology_dim - 1, tx_facets)
p_source = dolfinx.Function(P)
with p_source.vector.localForm() as loc:
    loc.set(np.sin(omega*0))
tx_bc = dolfinx.DirichletBC(p_source, tx_dofs)

boundary_facets = facet_tags.indices[facet_tags.values == boundary_marker]
boundary_dofs = dolfinx.fem.locate_dofs_topological(P, topology_dim - 1, boundary_facets)
p_boundary = dolfinx.Function(V)
with p_boundary.vector.localForm() as loc:
    loc.set(0)
boundary_bc = dolfinx.DirichletBC(p_boundary, boundary_dofs)

bc_v = []
bc_p = [tx_bc] # , boundary_bc

# TODO: use n = ufl.FacetNormal(mesh) to impose a boundary displacement velocity

rk_c = [dolfinx.Function(V) for ii in range(n_rk_steps)]
rk_k = [dolfinx.Function(P) for ii in range(n_rk_steps)]

# k_n.name = 'k_n'

ufl_f = dolfinx.Constant(mesh, 0)
ufl_c0 = dolfinx.Constant(mesh, c0)

v_sol = dolfinx.Function(V)
with v_sol.vector.localForm() as loc:
    loc.set(0)
p_sol = dolfinx.Function(P)
with p_sol.vector.localForm() as loc:
    loc.set(0)

# Step 1 solver setup

F1 = ufl_c0**2 * ufl.inner(ufl.grad(p), ufl.grad(q))*ufl.dx - ufl.inner(rk_k[0], q)*ufl.dx
a1 = ufl.lhs(F1)
L1 = ufl.rhs(F1)
# Convert a to the matrix form
A1 = dolfinx.fem.assemble_matrix(a1, bcs=bc_p)
A1.assemble()
b1 = dolfinx.fem.create_vector(L1)
# Create the linear solver using PETSc
step1_solver = PETSc.KSP().create(mesh.mpi_comm())
step1_solver.setOperators(A1)
step1_solver.setType(PETSc.KSP.Type.PREONLY)
step1_solver.getPC().setType(PETSc.PC.Type.LU)

# Step 2 solver setup

F2 = ufl_c0**2 * ufl.inner(ufl.grad(p + (dt*rk_c[0])/2), ufl.grad(q))*ufl.dx - ufl.inner(rk_k[1], q)*ufl.dx
a2 = ufl.lhs(F2)
L2 = ufl.rhs(F2)
# Convert a to the matrix form
A2 = dolfinx.fem.assemble_matrix(a2, bcs=bc_p)
A2.assemble()
b2 = dolfinx.fem.create_vector(L2)
# Create the linear solver using PETSc
step2_solver = PETSc.KSP().create(mesh.mpi_comm())
step2_solver.setOperators(A2)
step2_solver.setType(PETSc.KSP.Type.PREONLY)
step2_solver.getPC().setType(PETSc.PC.Type.LU)

# Step 3 solver setup

F3 = ufl_c0**2 * ufl.inner(ufl.grad(p + (dt*rk_c[1])/2), ufl.grad(q)) * ufl.dx - ufl.inner(rk_k[2], q)*ufl.dx
a3 = ufl.lhs(F3)
L3 = ufl.rhs(F3)
# Convert a to the matrix form
A3 = dolfinx.fem.assemble_matrix(a3, bcs=bc_p)
A3.assemble()
b3 = dolfinx.fem.create_vector(L3)
# Create the linear solver using PETSc
step3_solver = PETSc.KSP().create(mesh.mpi_comm())
step3_solver.setOperators(A3)
step3_solver.setType(PETSc.KSP.Type.PREONLY)
step3_solver.getPC().setType(PETSc.PC.Type.LU)

# Step 4 solver setup

F4 = ufl_c0**2 * ufl.inner(ufl.grad(p + dt*rk_c[2]), ufl.grad(q)) * ufl.dx - ufl.inner(rk_k[3], q)*ufl.dx
a4 = ufl.lhs(F4)
L4 = ufl.rhs(F4)
# Convert a to the matrix form
A4 = dolfinx.fem.assemble_matrix(a4, bcs=bc_p)
A4.assemble()
b4 = dolfinx.fem.create_vector(L4)
# Create the linear solver using PETSc
step4_solver = PETSc.KSP().create(mesh.mpi_comm())
step4_solver.setOperators(A4)
step4_solver.setType(PETSc.KSP.Type.PREONLY)
step4_solver.getPC().setType(PETSc.PC.Type.LU)

# --- Tome stepping ---

xdmf_file = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "RK4_plane_wave_benchmark2.xdmf", "w")
xdmf_file.write_mesh(mesh)

for ii, tt in enumerate(tqdm(t_mesh)):

    # Compute the pressure at the transducer surface & update BCs
    with p_source.vector.localForm() as loc:
        loc.set(np.sin(omega*tt))
    tx_bc = dolfinx.DirichletBC(p_source, tx_dofs)
    bc_p = [tx_bc] # , boundary_bc

    # RK step 1 @ t

    # Update c1 = v
    with rk_c[0].vector.localForm() as rk_c_loc, v_sol.vector.localForm() as v_loc:
        v_loc.copy(rk_c_loc)
    # Update the right hand side reusing the initial vector
    with b1.localForm() as loc_b:
        loc_b.set(0)
    dolfinx.fem.assemble_vector(b1, L1)
    # Apply Dirichlet boundary condition to the vector
    dolfinx.fem.apply_lifting(b1, [a1], [bc_p])
    b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b1, bc_p)
    # Solve linear problem
    step1_solver.solve(b1, rk_k[0].vector)
    dolfinx.cpp.la.scatter_forward(rk_k[0].x)

    # RK step 2 @ t + dt/2

    # Update c2 = v + (dt*k1)/2
    with rk_c[1].vector.localForm() as rk_c_loc, rk_k[0].vector.localForm() as rk_k_loc, v_sol.vector.localForm() as v_loc:
        (v_loc + (dt*rk_k_loc) / 2).copy(rk_c_loc)
    # Update the right hand side reusing the initial vector
    with b2.localForm() as loc_b:
        loc_b.set(0)
    dolfinx.fem.assemble_vector(b2, L2)
    # Apply Dirichlet boundary condition to the vector
    dolfinx.fem.apply_lifting(b2, [a2], [bc_p])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b2, bc_p)
    # Solve linear problem
    step2_solver.solve(b2, rk_k[1].vector)
    dolfinx.cpp.la.scatter_forward(rk_k[1].x)

    # RK step 3 @ t + dt/2

    # Update c3 = v + (dt*k2)/2
    with rk_c[2].vector.localForm() as rk_c_loc, rk_k[1].vector.localForm() as rk_k_loc, v_sol.vector.localForm() as v_loc:
        (v_loc + (dt*rk_k_loc) / 2).copy(rk_c_loc)
    # Update the right hand side reusing the initial vector
    with b3.localForm() as loc_b:
        loc_b.set(0)
    dolfinx.fem.assemble_vector(b3, L3)
    # Apply Dirichlet boundary condition to the vector
    dolfinx.fem.apply_lifting(b3, [a3], [bc_p])
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b3, bc_p)
    # Solve linear problem
    step3_solver.solve(b3, rk_k[2].vector)
    dolfinx.cpp.la.scatter_forward(rk_k[2].x)

    # RK step 4 @ t + dt

    # Update c4 = v + dt*k3
    with rk_c[3].vector.localForm() as rk_c_loc, rk_k[2].vector.localForm() as rk_k_loc, v_sol.vector.localForm() as v_loc:
        (v_loc + dt*rk_k_loc).copy(rk_c_loc)
    # Update the right hand side reusing the initial vector
    with b4.localForm() as loc_b:
        loc_b.set(0)
    dolfinx.fem.assemble_vector(b4, L4)
    # Apply Dirichlet boundary condition to the vector
    dolfinx.fem.apply_lifting(b4, [a4], [bc_p])
    b4.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b4, bc_p)
    # Solve linear problem
    step4_solver.solve(b4, rk_k[3].vector)
    dolfinx.cpp.la.scatter_forward(rk_k[3].x)

    # Combine coeficients c_n & k_n to update the de pressure & velocity

    with p_sol.vector.localForm() as p_loc, rk_c[0].vector.localForm() as c1, rk_c[1].vector.localForm() as c2, rk_c[2].vector.localForm() as c3, rk_c[3].vector.localForm() as c4:
         (p_loc + dt * (c1 + 2*c2 + 2*c3 + c4) / 6).copy(p_loc)

    with v_sol.vector.localForm() as v_loc, rk_k[0].vector.localForm() as k1, rk_k[1].vector.localForm() as k2, rk_k[2].vector.localForm() as k3, rk_k[3].vector.localForm() as k4:
        (v_loc + dt * (k1 + 2*k2 + 2*k3 + k4) / 6).copy(v_loc)

    p_sol.name = "pressure"
    xdmf_file.write_function(p_sol, tt)

xdmf_file.close()
