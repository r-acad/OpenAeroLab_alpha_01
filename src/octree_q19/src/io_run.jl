# Run outputs in the app's file contract (plan section 3.10, DESIGN.md section 9): csv/convergence.csv,
# csv/forces.csv (the header of src/forces/io.jl), csv/resource_history.csv, field planes in the shared lbmslice
# format, the LBMP surface with its sidecar, VTU volume output, and checkpoints with resume.

# ---------------------------------------------------------------- histories
const FORCES_HEADER = "Step,Time,U_inlet,Fx_N,Fy_N,Fz_N,Fx_p_N,Fy_p_N,Fz_p_N,Fx_v_N,Fy_v_N,Fz_v_N,Mx,My,Mz,Cd,Cl,Cs,Cd_p,Cl_p,Cs_p,Cd_v,Cl_v,Cs_v,Cmx,Cmy,Cmz,N_avg,Cd_mean,Cl_mean,Cs_mean,Cd_p_mean,Cl_p_mean,Cs_p_mean,Cd_v_mean,Cl_v_mean,Cs_v_mean"

mutable struct Histories
    conv::IOStream
    forces::IOStream
    resources::IOStream
    n_avg::Int
    sums::Vector{Float64}        # running sums of the nine coefficients
    t_wall0::Float64
end

function open_histories(out::AbstractString; append::Bool=false)
    dir = joinpath(out, "csv"); mkpath(dir)
    mode = append ? "a" : "w"
    conv = open(joinpath(dir, "convergence.csv"), mode); forces = open(joinpath(dir, "forces.csv"), mode); res = open(joinpath(dir, "resource_history.csv"), mode)
    if !append
        println(conv, "Step,Time_phys_s,Domain_travel_pct,u_inlet_lattice,Rho_min,Rho_max,U_max_lattice,MLUPS,Elapsed_s,Mass_rel,Cd,Cl,Cs")
        println(forces, FORCES_HEADER)
        println(res, "Step,Elapsed_s,RAM_used_MiB,Iter_per_s,MLUPS")
    end
    Histories(conv, forces, res, 0, zeros(9), time())
end
close_histories(h::Histories) = (close(h.conv); close(h.forces); close(h.resources))

"Density and velocity extrema over the fluid leaves, and the relative mass."
function _state_extrema_chunk(Lv, S, chunk)
    rmin = Inf; rmax = -Inf; umax = 0.0
    @inbounds for oi in chunk
        o = Int(S.real_octs[oi])
        for k in 1:8
            Lv.kind[k, o] == K_FLUID || continue
            r = Float64(S.rho[k, o]); rmin = min(rmin, r); rmax = max(rmax, r)
            umax = max(umax, Float64(S.u[k, 1, o])^2 + Float64(S.u[k, 2, o])^2 + Float64(S.u[k, 3, o])^2)
        end
    end
    rmin, rmax, umax
end

function state_extrema(st::State)
    rmin = Inf; rmax = -Inf; umax = 0.0
    for (l, Lv) in enumerate(st.mesh.levels)
        S = st.levels[l]; chunks = S.chunks
        if length(chunks)==1
            mn,mx,um = _state_extrema_chunk(Lv,S,chunks[1])
            rmin=min(rmin,mn);rmax=max(rmax,mx);umax=max(umax,um)
            continue
        end
        # Independent min/max reductions preserve NaN/Inf detection and do not
        # modify the every-step stability guard. Empty chunks are neutral.
        partial=Vector{NTuple{3,Float64}}(undef,length(chunks))
        @threads :static for t in eachindex(chunks)
            partial[t]=_state_extrema_chunk(Lv,S,chunks[t])
        end
        for (mn,mx,um) in partial
            rmin=min(rmin,mn);rmax=max(rmax,mx);umax=max(umax,um)
        end
    end
    rmin, rmax, sqrt(umax)
end

"Write one row of the convergence and resource histories; `mlups` is the rate measured over the last interval."
function write_convergence!(h::Histories, st::State, coefficients::NTuple{3,Float64}, mlups::Float64, iter_per_s::Float64)
    sc = st.scaling; L = nlevels(st.mesh)
    t_phys = st.step * sc.dt_finest * 2.0^(L - 1)
    travel = 100 * t_phys * st.cfg.flow.velocity / st.mesh.domain[1]
    rmin, rmax, umax = state_extrema(st)
    mass_rel = st.mass0 > 0 ? total_mass(st) / st.mass0 - 1 : 0.0
    elapsed = time() - h.t_wall0
    println(h.conv, join([st.step, @sprintf("%.9g", t_phys), @sprintf("%.4f", travel), @sprintf("%.6f", st.u_inlet), @sprintf("%.6f", rmin), @sprintf("%.6f", rmax), @sprintf("%.6f", umax),
                          @sprintf("%.3f", mlups), @sprintf("%.1f", elapsed), @sprintf("%.3e", mass_rel), @sprintf("%.6f", coefficients[1]), @sprintf("%.6f", coefficients[2]), @sprintf("%.6f", coefficients[3])], ","))
    println(h.resources, join([st.step, @sprintf("%.1f", elapsed), @sprintf("%.0f", Sys.maxrss() / 2^20), @sprintf("%.4f", iter_per_s), @sprintf("%.3f", mlups)], ","))
    flush(h.conv); flush(h.resources)
end

"""
Write one row of forces.csv from the last collected momentum exchange. The pressure/viscous split is not
available from the momentum exchange alone: the viscous columns are reported as the full force's components
projected on the local flow direction only when a wall model gives them (phase 4); until then the `_p` columns
carry the total and the `_v` columns are zero, and the header says so in the solver log once.
"""
function write_forces!(h::Histories, st::State, F::Vector{Float64}, M::Vector{Float64}, coefficients::Vector{Float64})
    sc = st.scaling; L = nlevels(st.mesh)
    t_phys = st.step * sc.dt_finest * 2.0^(L - 1)
    h.n_avg += 1; h.sums .+= coefficients[1:9]
    means = h.sums ./ h.n_avg
    f(x) = @sprintf("%.9g", x)
    println(h.forces, join(vcat([string(st.step), f(t_phys), f(st.u_inlet * sc.velocity_scale)], f.(F), f.(F), f.(zeros(3)), f.(M), f.(coefficients), [string(h.n_avg)], f.(means)), ","))
    flush(h.forces)
end

"Force coefficients in X-flow/Z-up wind axes (ground: solver Y-up); moment coefficients retain solver axes. Inputs are full-model loads."
function coefficients(st::State, F::Vector{Float64}, M::Vector{Float64})
    q = 0.5 * st.cfg.flow.density * st.cfg.flow.velocity^2
    A = st.cfg.reference_area; c = st.cfg.reference_chord
    q > 0 && A > 0 && c > 0 && all(isfinite, (q, A, c)) ||
        throw(ArgumentError("Force coefficients require positive finite dynamic pressure, reference area and chord"))
    ground = st.cfg.mesh.faces[3] == :ground
    lift, side = ground ? (F[2], -F[3]) : (F[3], F[2])
    [F[1] / (q * A), lift / (q * A), side / (q * A), F[1] / (q * A), lift / (q * A), side / (q * A), 0.0, 0.0, 0.0, M[1] / (q * A * c), M[2] / (q * A * c), M[3] / (q * A * c)]
end

"Decode the app's chord-relative moment setting into the same placed domain frame as the surface."
function moment_center_domain(st::State)
    rel = st.cfg.moment_center; chord = st.cfg.reference_chord
    isempty(st.mesh.triangles) && return ntuple(a -> st.mesh.offset[a] + rel[a] * chord, 3)
    lo, hi = bounds(st.mesh.triangles)
    # The app expresses a symmetry reference about the retained half's bbox.
    # A centered full STL is retained for two-sided intersections, but its
    # negative-Y part contributes no fluid cells or mirrored moments.
    ymin = st.cfg.mesh.faces[3] == :symmetry ? max(0.0, lo[2]) : lo[2]
    (lo[1] + rel[1] * chord,
     (ymin + hi[2]) / 2 + rel[2] * chord,
     (lo[3] + hi[3]) / 2 + rel[3] * chord)
end

"Moments about the configured centre from the per-triangle forces (newtons), in the domain frame."
function moments_newton(st::State, tri_force_newton::Matrix{Float64})
    M = zeros(3); c = moment_center_domain(st)
    for (t, T) in enumerate(st.mesh.triangles)
        r = vsub(vmul(vadd(vadd(T[1], T[2]), T[3]), 1 / 3), c)
        Fx = tri_force_newton[1, t]; Fy = tri_force_newton[2, t]; Fz = tri_force_newton[3, t]
        M[1] += r[2] * Fz - r[3] * Fy; M[2] += r[3] * Fx - r[1] * Fz; M[3] += r[1] * Fy - r[2] * Fx
    end
    # Mirrored full-model moments: the polar force reflects across Y=0,
    # while its axial moment retains only the pitch component.
    st.cfg.mesh.faces[3] == :symmetry ? [0.0, 2M[2], 0.0] : M
end

# ---------------------------------------------------------------- field planes
"The sampler used by `export_planes` for the flow fields: leaf values at the finest spacing, SI units."
function field_sampler(st::State)
    sc = st.scaling; vs = Float32(sc.velocity_scale); rho0 = Float32(st.cfg.flow.density)
    cs2_phys = Float32(vs^2 / 3)
    function (mesh, an, coord, a1, a2, n1, n2, h, arrays)
        names = ("density", "pressure", "u", "v", "w", "velocity_magnitude")
        out = Dict(name => fill(NaN32, n1, n2) for name in names)
        vort = fill(NaN32, n1, n2)
        nt = Threads.nthreads(); ranges = chunk_ranges(n2, nt)
        Threads.@threads :static for t in 1:nt
            for k in ranges[t], i in 1:n1
                p = ntuple(a -> a == an ? coord : a == a1 ? (i - 0.5) * h : (k - 0.5) * h, 3)
                l, s = locate_leaf(mesh, p)
                l == 0 && continue
                Lv = mesh.levels[l]; S = st.levels[l]; kk = slot_k(s); o = slot_oct(s)
                Lv.kind[kk, o] == K_FLUID || continue
                r = Float32(S.rho[kk, o]); ux = Float32(S.u[kk, 1, o]) * vs; uy = Float32(S.u[kk, 2, o]) * vs; uz = Float32(S.u[kk, 3, o]) * vs
                out["density"][i, k] = r * rho0; out["pressure"][i, k] = (r - 1f0) * cs2_phys * rho0
                out["u"][i, k] = ux; out["v"][i, k] = uy; out["w"][i, k] = uz; out["velocity_magnitude"][i, k] = sqrt(ux * ux + uy * uy + uz * uz)
                # vorticity component normal to the plane from the leaf's central differences of the stored velocity
                g = velocity_gradient(Lv, S.u, kk, o)
                dxl = Float32(Lv.dx)
                wn = an == 1 ? (g[6] - g[8]) : an == 2 ? (g[7] - g[3]) : (g[2] - g[4])     # (dw/dy - dv/dz), (du/dz - dw/dx), (dv/dx - du/dy)
                vort[i, k] = Float32(wn) * vs / dxl
            end
        end
        push!(arrays, ("density", out["density"], "kg/m^3", "fluid density"), ("pressure", out["pressure"], "Pa", "pressure relative to the reference, rho0 cs^2 (rho - 1)"),
              ("u", out["u"], "m/s", "velocity x"), ("v", out["v"], "m/s", "velocity y"), ("w", out["w"], "m/s", "velocity z"),
              ("velocity_magnitude", out["velocity_magnitude"], "m/s", "velocity magnitude"),
              ("vorticity_" * ("x", "y", "z")[an], vort, "1/s", "vorticity component normal to the plane"))
    end
end

# ---------------------------------------------------------------- surface (LBMP)
"""
Legacy surface fields per triangle: pressure from the unweighted mean density of cutting links (a wall cell
can contribute more than once), shear from per-triangle momentum exchange projected tangentially (Pa),
Cp. Written as VTK/surface_NNNNNN.lbmp with its TOML sidecar, in the solver's triangle order.
"""
function write_surface!(st::State, out::AbstractString, step::Int, tri_force_newton::Matrix{Float64})
    mesh = st.mesh; tri = mesh.triangles; nt = length(tri)
    nt == 0 && return nothing
    sc = st.scaling; rho0 = st.cfg.flow.density; vs = sc.velocity_scale; cs2 = vs^2 / 3
    q = 0.5 * rho0 * st.cfg.flow.velocity^2
    psum = zeros(nt); pcnt = zeros(Int, nt)
    for (l, Lv) in enumerate(mesh.levels)
        S = st.levels[l]
        for (ci, s) in enumerate(Lv.cut_slot)
            r = Float64(S.rho[slot_k(Int(s)), slot_oct(Int(s))])
            for li in Lv.cut_off[ci]:Lv.cut_off[ci + 1]-1
                t = Int(Lv.cut_tri[li]); t > 0 || continue
                psum[t] += r; pcnt[t] += 1
            end
        end
    end
    vtk = joinpath(out, "VTK"); mkpath(vtk)
    path = joinpath(vtk, @sprintf("surface_%06d.lbmp", step))
    open(path * ".tmp", "w") do io
        write(io, codeunits("LBMP")); write(io, UInt32(1)); write(io, UInt32(nt)); write(io, UInt32(1))
        name = codeunits("body"); write(io, UInt32(length(name))); write(io, name)
        for T in tri, v in T, a in 1:3; write(io, Float32(v[a])); end
        for T in tri
            n, _ = tri_normal_area(T); write(io, Float32(n[1]), Float32(n[2]), Float32(n[3]))
        end
        for t in 1:nt
            p = pcnt[t] > 0 ? (psum[t] / pcnt[t] - 1) * cs2 * rho0 : NaN
            write(io, Float32(p))
        end
        for t in 1:nt
            n, area = tri_normal_area(tri[t])
            F = (tri_force_newton[1, t], tri_force_newton[2, t], tri_force_newton[3, t])
            fn = vdot(F, n); tang = vsub(F, vmul(n, fn))
            write(io, Float32(area > 0 && pcnt[t] > 0 ? vnorm(tang) / area : NaN))
        end
        for t in 1:nt
            p = pcnt[t] > 0 ? (psum[t] / pcnt[t] - 1) * cs2 * rho0 : NaN
            write(io, Float32(p / q))
        end
        for t in 1:nt; write(io, Int32(1)); end
    end
    mv(path * ".tmp", path; force=true)
    open(path * ".toml", "w") do io
        TOML.print(io, Dict("format" => "LBMP", "format_version" => 1, "geometry_frame" => "domain", "triangle_order" => "solver_geometry",
                            "mesh_offset_m" => collect(mesh.offset), "triangles" => nt, "step" => step, "solver" => SOLVER_NAME,
                            "moment_center_m"=>collect(moment_center_domain(st)),"reference_area_m2"=>st.cfg.reference_area,"reference_chord_m"=>st.cfg.reference_chord,
                            "pressure_note" => "pressure from the mean density of the wall cells cutting each triangle; shear from the tangential momentum exchange per triangle area (development)",
                            "cp_reference" => "Cp = pressure / (0.5 rho0 U^2)"))
    end
    if st.cfg.outputs.surface_vtk
        points=Matrix{Float32}(undef,3,3nt)
        cells=WriteVTK.MeshCell[];pressures=Vector{Float32}(undef,nt);shear=similar(pressures)
        for t in 1:nt
            for k in 1:3,a in 1:3;points[a,3(t-1)+k]=tri[t][k][a];end
            push!(cells,WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_TRIANGLE,3(t-1).+(1:3)))
            pressures[t]=pcnt[t]>0 ? (psum[t]/pcnt[t]-1)*cs2*rho0 : NaN32
            n,area=tri_normal_area(tri[t]);F=Tuple(tri_force_newton[:,t]);tangent=vsub(F,vmul(n,vdot(F,n)))
            shear[t]=area>0&&pcnt[t]>0 ? vnorm(tangent)/area : NaN32
        end
        grid=WriteVTK.vtk_grid(joinpath(vtk,@sprintf("surface_%06d",step)),points,cells;compress=false,append=false)
        grid["Pressure"]=pressures;grid["Cp"]=pressures./Float32(q);grid["Shear"]=shear
        WriteVTK.vtk_save(grid)
    end
    path
end

# ---------------------------------------------------------------- volume (VTU)
"Fluid leaves as voxels with SI fields, through WriteVTK (inline binary)."
function write_volume!(st::State, out::AbstractString, step::Int)
    mesh = st.mesh; sc = st.scaling; vs = sc.velocity_scale; rho0 = st.cfg.flow.density; cs2 = vs^2 / 3
    pts = Float32[]; cells = WriteVTK.MeshCell[]; vel = Float32[]; dens = Float32[]; pres = Float32[]; lev = UInt8[]
    npts = 0
    for (l, Lv) in enumerate(mesh.levels)
        S = st.levels[l]
        for o in S.real_octs, k in 1:8
            Lv.kind[k, o] == K_FLUID || continue
            c = slot_cell(Lv, slot_id(k, o)); x0 = c[1] * Lv.dx; y0 = c[2] * Lv.dx; z0 = c[3] * Lv.dx; d = Lv.dx
            for (dz, dy, dx) in ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))
                push!(pts, Float32(x0 + dx * d), Float32(y0 + dy * d), Float32(z0 + dz * d))
            end
            push!(cells, WriteVTK.MeshCell(WriteVTK.VTKCellTypes.VTK_VOXEL, npts .+ (1:8)))
            npts += 8
            r = Float64(S.rho[k, o])
            push!(vel, Float32(S.u[k, 1, o] * vs), Float32(S.u[k, 2, o] * vs), Float32(S.u[k, 3, o] * vs))
            push!(dens, Float32(r * rho0)); push!(pres, Float32((r - 1) * cs2 * rho0)); push!(lev, UInt8(l))
        end
    end
    dir = joinpath(out, "VTK"); mkpath(dir)
    base = joinpath(dir, @sprintf("flow_%06d", step))
    vtk = WriteVTK.vtk_grid(base, reshape(pts, 3, :), cells; compress=false, append=false)
    vtk["Velocity"] = reshape(vel, 3, :); vtk["Density"] = dens; vtk["Pressure"] = pres; vtk["Level"] = lev
    vtk["Obstacle"] = zeros(UInt8, length(dens))
    WriteVTK.vtk_save(vtk)
    base * ".vtu"
end

# ---------------------------------------------------------------- checkpoint
const CHECKPOINT_MAGIC = "OCTQ19C\0"

"Write a checkpoint (the current buffers of every level, the step, the averages) atomically; keeps the last `keep`."
function write_checkpoint!(st::State{T}, out::AbstractString, h::Histories; keep::Int=2) where T
    dir = joinpath(out, "checkpoints"); mkpath(dir)
    path = joinpath(dir, @sprintf("checkpoint_%09d.bin", st.step))
    open(path * ".tmp", "w") do io
        write(io, codeunits(CHECKPOINT_MAGIC)); write(io, Int64(1)); write(io, Int64(sizeof(T)))
        write(io, codeunits(rpad(st.mesh.config_sha, 64)[1:64])); write(io, codeunits(rpad(st.mesh.stl_sha, 64)[1:64]))
        write(io, Int64(st.step)); write(io, Float64(st.u_inlet)); write(io, Float64(st.mass0)); write(io, Int64(h.n_avg)); write(io, h.sums)
        write(io, Int64(nlevels(st.mesh)))
        for S in st.levels
            write(io, Int64(size(S.f, 3))); write(io, S.f); write(io, S.g); write(io, S.u); write(io, S.v); write(io, S.rho); write(io, S.rhob); write(io, Int64(length(S.tau_w))); write(io, S.tau_w)
        end
    end
    mv(path * ".tmp", path; force=true)
    files = sort(filter(f -> startswith(f, "checkpoint_") && endswith(f, ".bin"), readdir(dir)))
    for f in files[1:max(0, length(files) - keep)]; rm(joinpath(dir, f); force=true); end
    path
end

"Read one explicitly named checkpoint; its case and precision must match the state."
function read_checkpoint!(st::State{T}, path::AbstractString, h::Union{Histories,Nothing}) where T
    open(path, "r") do io
        String(read(io, length(CHECKPOINT_MAGIC))) == CHECKPOINT_MAGIC || error("not a checkpoint")
        read(io, Int64) == 1 || error("checkpoint format")
        read(io, Int64) == sizeof(T) || error("checkpoint precision differs from the run's")
        csha = String(read(io, 64)); ssha = String(read(io, 64))
        (strip(csha) == st.mesh.config_sha && strip(ssha) == st.mesh.stl_sha) || error("checkpoint belongs to another case (config or STL hash differs)")
        st.step = Int(read(io, Int64)); st.u_inlet = T(read(io, Float64)); st.mass0 = read(io, Float64)
        navg=Int(read(io,Int64));sums=Vector{Float64}(undef,9);read!(io,sums)
        if h !== nothing; h.n_avg=navg;h.sums .= sums; end
        Int(read(io, Int64)) == nlevels(st.mesh) || error("checkpoint level count differs")
        for S in st.levels
            Int(read(io, Int64)) == size(S.f, 3) || error("checkpoint oct count differs")
            read!(io, S.f); read!(io, S.g); read!(io, S.u); read!(io, S.v); read!(io, S.rho); read!(io, S.rhob)
            n = Int(read(io, Int64)); n == length(S.tau_w) || error("checkpoint wall-model size differs"); read!(io, S.tau_w)
        end
        eof(io) || error("checkpoint has unexpected trailing data")
    end
    st.step
end

"Restore the latest checkpoint into `st` (same mesh and precision); returns the step, or nothing when none exists."
function resume_checkpoint!(st::State{T}, out::AbstractString, h::Histories) where T
    dir = joinpath(out, "checkpoints"); isdir(dir) || return nothing
    files = sort(filter(f -> startswith(f, "checkpoint_") && endswith(f, ".bin"), readdir(dir)))
    isempty(files) && return nothing
    read_checkpoint!(st,joinpath(dir,files[end]),h)
end
