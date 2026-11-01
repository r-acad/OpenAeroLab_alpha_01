# CMeshMakie — reusable CairoMakie flow-field renderer for the CMESH2D kinetic solver.
# Renders physics-informative multi-panel field slices (density / Mach / pressure / vorticity / vapor)
# on the CURVILINEAR C-mesh in PHYSICAL space, and a spanwise view for the 3D (span3d) state, then
# ships them to Telegram. Reuses the PROVEN bilinear pixel-map rasterizer from tools/cmesh_case9_movie.jl
# (build_pixmap / build_bilinear / gather), swapping Plots -> CairoMakie so it composes with a harness
# that already `using Plots`. Env: CairoMakie loads under --project=src via the stacked @v1.12 env (no
# Project.toml change); `CairoMakie.save` is always fully qualified to avoid the Plots `save`/`plot` clash.
#
# Usage from any cmesh harness (after building solver state `s`, mesh `mesh`, metrics `met`):
#   include(joinpath(ROOT,"tools","render","cmesh_makie.jl")); using .CMeshMakie
#   png = render_cmesh_2d(s, mesh, met; title="NACA0015 a8 sig0.80", vapor=(rho_sat, rho_v))
#   tg_send(png, "caption")                       # or render_and_send(s, mesh, met; caption=..., ...)
module CMeshMakie
using CairoMakie, Printf
import KernelAbstractions as KA

export PixMap, build_pixmap, render_cmesh_2d, render_cmesh_span3d, tg_send, render_and_send

const CS2 = 2.0/3.0                       # lattice cs^2 (matches cmesh_flux_core CS2)

# ------------------------------------------------------------------ rasterizer (proven, parameterized)
struct PixMap
    XR::Vector{Float64}; YR::Vector{Float64}
    PIX::Matrix{Int32}; BI4::Array{Int32,3}; BW4::Array{Float32,3}
    NI::Int; NJ::Int; PX::Int; PY::Int; win::NTuple{4,Float64}
end

# Build the pixel->cell bilinear interpolation map ONCE for a static mesh (X,Y are node coords, size (NI+1,NJ+1)).
function build_pixmap(X, Y, NI::Int, NJ::Int; win=(-0.35, 1.5, -0.6, 0.6), PX::Int=1000)
    x0, x1, y0, y1 = win
    PY = round(Int, PX * (y1 - y0) / (x1 - x0))
    PIX = zeros(Int32, PY, PX)
    for j in NJ:-1:1, i in 1:NI
        xs = (X[i,j], X[i+1,j], X[i,j+1], X[i+1,j+1]); ys = (Y[i,j], Y[i+1,j], Y[i,j+1], Y[i+1,j+1])
        xmin, xmax = extrema(xs); ymin, ymax = extrema(ys)
        (xmax < x0 || xmin > x1 || ymax < y0 || ymin > y1) && continue
        pa = max(1, floor(Int,(xmin-x0)/(x1-x0)*PX)+1); pb = min(PX, ceil(Int,(xmax-x0)/(x1-x0)*PX))
        qa = max(1, floor(Int,(ymin-y0)/(y1-y0)*PY)+1); qb = min(PY, ceil(Int,(ymax-y0)/(y1-y0)*PY))
        v = Int32(i + (j-1)*NI)
        for qq in qa:qb, pp in pa:pb; PIX[qq,pp] = v; end
    end
    I4 = zeros(Int32, PY, PX, 4); W4 = zeros(Float32, PY, PX, 4)
    for p in 1:PX, q in 1:PY
        c = PIX[q,p]; c == 0 && continue
        i = Int(mod1(c, NI)); j = Int((c-1) ÷ NI) + 1
        xp = x0 + (p-0.5)*(x1-x0)/PX; yp = y0 + (q-0.5)*(y1-y0)/PY
        x00=X[i,j]; y00=Y[i,j]; x10=X[i+1,j]; y10=Y[i+1,j]; x01=X[i,j+1]; y01=Y[i,j+1]; x11=X[i+1,j+1]; y11=Y[i+1,j+1]
        a = 0.5; b = 0.5
        for _ in 1:4
            fx = (1-a)*(1-b)*x00 + a*(1-b)*x10 + (1-a)*b*x01 + a*b*x11 - xp
            fy = (1-a)*(1-b)*y00 + a*(1-b)*y10 + (1-a)*b*y01 + a*b*y11 - yp
            dxa = -(1-b)*x00 + (1-b)*x10 - b*x01 + b*x11; dxb = -(1-a)*x00 - a*x10 + (1-a)*x01 + a*x11
            dya = -(1-b)*y00 + (1-b)*y10 - b*y01 + b*y11; dyb = -(1-a)*y00 - a*y10 + (1-a)*y01 + a*y11
            det = dxa*dyb - dxb*dya; abs(det) < 1e-30 && break
            a -= ( dyb*fx - dxb*fy)/det; b -= (-dya*fx + dxa*fy)/det
            a = clamp(a, 0.0, 1.0); b = clamp(b, 0.0, 1.0)
        end
        sa = a - 0.5; tb = b - 0.5
        i2 = clamp(i + (sa >= 0 ? 1 : -1), 1, NI); j2 = clamp(j + (tb >= 0 ? 1 : -1), 1, NJ)
        ws = abs(sa); wt = abs(tb)
        I4[q,p,1] = Int32(i  + (j -1)*NI); W4[q,p,1] = Float32((1-ws)*(1-wt))
        I4[q,p,2] = Int32(i2 + (j -1)*NI); W4[q,p,2] = Float32(ws*(1-wt))
        I4[q,p,3] = Int32(i  + (j2-1)*NI); W4[q,p,3] = Float32((1-ws)*wt)
        I4[q,p,4] = Int32(i2 + (j2-1)*NI); W4[q,p,4] = Float32(ws*wt)
    end
    XR = collect(range(x0, x1, length=PX)); YR = collect(range(y0, y1, length=PY))
    PixMap(XR, YR, PIX, I4, W4, NI, NJ, PX, PY, win)
end

# gather a cell field (NI*NJ flattened, col-major i + (j-1)*NI) onto the pixel grid -> (PX,PY) for Makie
function gather(pm::PixMap, cellf::Vector{Float64})
    img = fill(NaN64, pm.PX, pm.PY)
    Threads.@threads for p in 1:pm.PX
        for q in 1:pm.PY
            pm.PIX[q,p] == 0 && continue
            img[p,q] = pm.BW4[q,p,1]*cellf[pm.BI4[q,p,1]] + pm.BW4[q,p,2]*cellf[pm.BI4[q,p,2]] +
                       pm.BW4[q,p,3]*cellf[pm.BI4[q,p,3]] + pm.BW4[q,p,4]*cellf[pm.BI4[q,p,4]]
        end
    end
    img
end

# ------------------------------------------------------------------ field extraction (host arrays)
_sync(s) = (try; KA.synchronize(s.backend); catch; end)

# Compute per-cell fields from host arrays. hrho/hux/huy/hT include ghost pad g; hxc/hyc are interior (NI,NJ).
# Returns flattened Vector{Float64} (i+(j-1)*NI) for each field. gam_eos = (5+drot)/(3+drot).
function cellfields2d(hrho, hux, huy, hT, hxc, hyc, NI, NJ, g, gam_eos; vapor=nothing,
                      halpha=nothing, hpexc=nothing, hpressure=nothing, hsound_speed=nothing)
    if hsound_speed !== nothing
        size(hsound_speed) == size(hrho) && all(c -> isfinite(c) && c > 0, hsound_speed) ||
            throw(ArgumentError("rendered EOS sound speed must be finite, positive and full-ghost"))
    end
    n = NI*NJ
    fR = zeros(n); fM = zeros(n); fP = zeros(n); fW = fill(NaN64, n); fV = fill(NaN64, n)
    fA = fill(NaN64, n); fUx = zeros(n)
    rho_sat = vapor === nothing ? 0.0 : Float64(vapor[1]); rho_v = vapor === nothing ? 0.0 : Float64(vapor[2])
    rho_l = 1.0
    Threads.@threads for j in 1:NJ
        for i in 1:NI
            k = i + (j-1)*NI
            r = Float64(hrho[i+g,j+g]); u = Float64(hux[i+g,j+g]); v = Float64(huy[i+g,j+g]); t = Float64(hT[i+g,j+g])
            # Under the PDP split (2026-08) the carrier T is railed and rho*cs2*T is a pedestal; the physical
            # pressure is p_mix = rho*cs2*T + pexc. When pexc is supplied, report p_mix.
            fR[k] = r; fP[k] = hpressure===nothing ? r*CS2*t + (hpexc === nothing ? 0.0 : Float64(hpexc[i+g,j+g])) : Float64(hpressure[i+g,j+g])
            fUx[k] = u
            fM[k] = hsound_speed===nothing ? sqrt(u*u+v*v)/sqrt(gam_eos*CS2*max(t,1e-3)) :
                    sqrt(u*u+v*v)/Float64(hsound_speed[i+g,j+g])
            if vapor !== nothing
                fV[k] = clamp((rho_l - r)/(rho_l - rho_v + 1e-9), 0.0, 1.0)
            end
            if halpha !== nothing
                fA[k] = Float64(halpha[i+g,j+g])   # supplied composition or current beta, never a rho proxy
            end
        end
    end
    # vorticity + Jacobian (interior only; borders left NaN)
    Threads.@threads for j in 2:NJ-1
        for i in 2:NI-1
            k = i + (j-1)*NI
            xxi = 0.5*(hxc[i+1,j]-hxc[i-1,j]); yxi = 0.5*(hyc[i+1,j]-hyc[i-1,j])
            xet = 0.5*(hxc[i,j+1]-hxc[i,j-1]); yet = 0.5*(hyc[i,j+1]-hyc[i,j-1])
            J = xxi*yet - xet*yxi; abs(J) < 1e-30 && (J = copysign(1e-30, J==0 ? 1.0 : J))
            duxi = 0.5*(hux[i+1+g,j+g]-hux[i-1+g,j+g]); duet = 0.5*(hux[i+g,j+1+g]-hux[i+g,j-1+g])
            dvxi = 0.5*(huy[i+1+g,j+g]-huy[i-1+g,j+g]); dvet = 0.5*(huy[i+g,j+1+g]-huy[i+g,j-1+g])
            dudy = (-duxi*xet + duet*xxi)/J; dvdx = (dvxi*yet - dvet*yxi)/J
            fW[k] = dvdx - dudy
        end
    end
    (rho=fR, mach=fM, p=fP, vort=fW, vapor=fV, alpha=fA, ux=fUx)
end

# robust symmetric range for a diverging field: high percentile of |v| (near-wall extremes clip so
# the wake/shear-layer structure stays visible instead of being washed out by the BL peak).
function _absrange(v; pct=0.97)
    a = sort!(abs.(filter(isfinite, v))); isempty(a) && return (-1.0, 1.0)
    m = a[max(1, round(Int, pct*length(a)))]; m <= 0 && (m = 1.0); (-m, m)
end
function _range(v, lo_d, hi_d)
    fin = filter(isfinite, v); isempty(fin) && return (lo_d, hi_d)
    (min(lo_d, minimum(fin)), max(hi_d, maximum(fin)))
end

# ------------------------------------------------------------------ 2D renderer
"""
    render_cmesh_2d(s, mesh, met; title, out, win, PX, vapor, pm, alpha, pexc, Uref, pref) -> png path

Multi-panel (density, Mach, pressure, vorticity[, vapor][, alpha][, u_x/U]) on the curvilinear C-mesh,
physical space. `vapor=(rho_sat, rho_v)` adds a rho-proxy void-fraction panel (HEM states).
`alpha=true` (or an array with the same ghost pad as `s.rho`) adds the TRANSPORTED void fraction
panel from `s.alpha` (TEM/PDP states) and a u_x/U panel with reversed flow in blue.
`pexc=true` (or an array) makes the pressure panel the physical p_mix = ρcs²T + pexc, and if
`Uref`/`pref` are given it is shown as Cp = (p_mix − pref)/(½ρ_l U²). `pm` reuses a prebuilt PixMap.
`pressure` overrides the pedestal with supplied physical pressure. `sound_speed`
supplies the full-ghost EOS acoustic speed for Mach; otherwise the legacy carrier
reference is retained. `alpha_is_beta=true` labels the supplied alpha array as
current vapor volume fraction, without treating it as a transported reference label.
"""
function render_cmesh_2d(s, mesh, met; title::String="CMESH flow field", out::String,
                         win=(-0.35, 1.5, -0.6, 0.6), PX::Int=1000, vapor=nothing, pm=nothing,
                         alpha=nothing, pexc=nothing, pressure=nothing, sound_speed=nothing,
                         alpha_is_beta::Bool=false, Uref=nothing, pref=nothing)
    _sync(s)
    NI, NJ, g = Int(met.NI), Int(met.NJ), Int(s.g)
    gam_eos = (5.0 + Float64(s.drot))/(3.0 + Float64(s.drot))
    hrho=Array(s.rho); hux=Array(s.ux); huy=Array(s.uy); hT=Array(s.T); hxc=Array(s.xc); hyc=Array(s.yc)
    halpha = alpha === nothing ? nothing : (alpha === true ? Array(s.alpha) : Array(alpha))
    hpexc  = pexc  === nothing ? nothing : (pexc  === true ? Array(s.pexc)  : Array(pexc))
    hpressure=pressure===nothing ? nothing : Array(pressure)
    hsound_speed=sound_speed===nothing ? nothing : Array(sound_speed)
    alpha_is_beta && halpha===nothing && throw(ArgumentError("beta label requires a supplied vapor-volume field"))
    ff = cellfields2d(hrho, hux, huy, hT, hxc, hyc, NI, NJ, g, gam_eos;
                     vapor=vapor, halpha=halpha, hpexc=hpexc, hpressure, hsound_speed)
    P = pm === nothing ? build_pixmap(mesh.X, mesh.Y, NI, NJ; win=win, PX=PX) : pm
    if (hpexc !== nothing || hpressure !== nothing) && Uref !== nothing && pref !== nothing
        rho_ref=hpressure===nothing ? 1.0 : Float64(s.rho_inf)
        cp = (ff.p .- Float64(pref)) ./ (0.5*rho_ref*Float64(Uref)^2)
        pressure_panel = ("Cp = (p_mix − p_∞)/(½ρ_l U²)", cp, :plasma, (max(-2.0, minimum(filter(isfinite, cp))), 1.0))
    elseif hpexc !== nothing || hpressure !== nothing
        pressure_panel = (hpressure===nothing ? "Pressure  p_mix = ρcs²T + pexc" : "Pressure  p_mix (EOS)",
                          ff.p, :plasma, _range(ff.p, minimum(ff.p), maximum(ff.p)))
    else
        pressure_panel = ("Pressure  p=ρcs²T", ff.p, :plasma, _range(ff.p, 0.45, 0.75))
    end
    panels = Any[("Density  ρ", ff.rho, :viridis, _range(ff.rho, 0.6, 1.05)),
                 (hsound_speed===nothing ? "Mach" : "Mach  |u| / c_EOS", ff.mach, :turbo,
                  (0.0, max(1.0, 1.05*maximum(filter(isfinite,ff.mach))))),
                 pressure_panel,
                 ("Vorticity  ω_z", ff.vort, :balance, _absrange(ff.vort))]
    vapor !== nothing && push!(panels, ("Vapor fraction  α_v", ff.vapor, :dense, (0.0, 1.0)))
    if halpha !== nothing
        push!(panels, (alpha_is_beta ? "Current vapor volume fraction  β" : "Void fraction  α (transported)",
                       ff.alpha, :dense, (0.0, 1.0)))
        U = Uref === nothing ? max(1e-9, maximum(filter(isfinite, abs.(ff.ux)))) : Float64(Uref)
        push!(panels, ("u_x / U   (blue = reversed)", ff.ux ./ U, :balance, (-1.0, 1.0)))
    end
    ncol = 2; nrow = ceil(Int, length(panels)/ncol)
    fig = Figure(size=(700*ncol, 300*nrow + 60), backgroundcolor=:white)
    try; Label(fig[0, 1:ncol], title, fontsize=17, font=:bold); catch; end
    for (idx, (ttl, fld, cmap, cr)) in enumerate(panels)
        r = (idx-1)÷ncol + 1; c = (idx-1)%ncol + 1
        img = gather(P, fld)
        ax = Axis(fig[r, c]; title=ttl, xlabel="x/c", ylabel="y/c", titlesize=13)
        try; ax.aspect = DataAspect(); catch; end
        hm = heatmap!(ax, P.XR, P.YR, img; colormap=cmap, colorrange=cr, nan_color=:transparent)
        try; Colorbar(fig[r, c][1,2], hm); catch; end
        try; xlims!(ax, win[1], win[2]); ylims!(ax, win[3], win[4]); catch; end
    end
    mkpath(dirname(out)); CairoMakie.save(out, fig)
    @printf("[CMeshMakie] wrote %s (%d KB)\n", out, filesize(out)÷1024); flush(stdout)
    out
end

# ------------------------------------------------------------------ 3D span renderer
# span3d state: rho/ux/uy/(uz) are 3D (…,…,NK). Render span-AVERAGE panels (reuse 2D pipeline) + an
# (x-along-foil, span-z) suction-wall vapor/density map that exposes the 3D structure.
"""
    render_cmesh_span3d(s, mesh, met; title, out, win, PX, vapor, wall_j) -> png path
"""
function render_cmesh_span3d(s, mesh, met; title::String="CMESH 3D flow field", out::String,
                             win=(-0.35, 1.5, -0.6, 0.6), PX::Int=1000, vapor=nothing, wall_j::Int=1)
    _sync(s)
    NI, NJ, g = Int(met.NI), Int(met.NJ), Int(s.g)
    gam_eos = (5.0 + Float64(s.drot))/(3.0 + Float64(s.drot))
    R = Array(s.rho); UX = Array(s.ux); UY = Array(s.uy); T = Array(s.T)
    NK = size(R, 3)
    # span-average the ghosted arrays over k -> 2D ghosted arrays for the 2D pipeline
    ravg = dropdims(sum(R, dims=3), dims=3) ./ NK
    uavg = dropdims(sum(UX, dims=3), dims=3) ./ NK
    vavg = dropdims(sum(UY, dims=3), dims=3) ./ NK
    tavg = dropdims(sum(T, dims=3), dims=3) ./ NK
    hxc=Array(s.xc); hyc=Array(s.yc)
    ff = cellfields2d(ravg, uavg, vavg, tavg, hxc, hyc, NI, NJ, g, gam_eos; vapor=vapor)
    P = build_pixmap(mesh.X, mesh.Y, NI, NJ; win=win, PX=PX)
    # spanwise structure map: density at the first wall row over (i-index, k)
    NIW = met.NIW; NIF = met.NIF   # CMetrics struct fields (was a buggy dict-style get(::CMetrics,...))
    foil = (NIW+1):(NIW+NIF)
    xwall = [Float64(hxc[i, 1]) for i in foil]
    wallmap = [Float64(R[i+g, wall_j+g, k]) for i in foil, k in 1:NK]   # (foil, NK)
    panels = Any[("Span-avg Density  ⟨ρ⟩", ff.rho, :viridis, _range(ff.rho, 0.6, 1.05)),
                 ("Span-avg Mach", ff.mach, :turbo, (0.0, max(1.0, 1.05*maximum(filter(isfinite,ff.mach))))),
                 ("Span-avg Vorticity  ⟨ω_z⟩", ff.vort, :balance, _absrange(ff.vort))]
    vapor !== nothing && push!(panels, ("Span-avg Vapor  ⟨α_v⟩", ff.vapor, :dense, (0.0, 1.0)))
    fig = Figure(size=(1400, 300*2 + 360 + 60), backgroundcolor=:white)
    try; Label(fig[0, 1:2], title, fontsize=17, font=:bold); catch; end
    for (idx,(ttl, fld, cmap, cr)) in enumerate(panels)
        r=(idx-1)÷2 + 1; c=(idx-1)%2 + 1; img=gather(P, fld)
        ax=Axis(fig[r,c]; title=ttl, xlabel="x/c", ylabel="y/c", titlesize=13)
        try; ax.aspect=DataAspect(); catch; end
        hm=heatmap!(ax, P.XR, P.YR, img; colormap=cmap, colorrange=cr, nan_color=:transparent)
        try; Colorbar(fig[r,c][1,2], hm); catch; end
        try; xlims!(ax, win[1], win[2]); ylims!(ax, win[3], win[4]); catch; end
    end
    # spanwise (x, z) wall map spanning the full bottom row
    rr = length(panels)÷2 + 1
    axw = Axis(fig[rr, 1:2]; title="Suction-wall density over (chord, span) — 3D structure", xlabel="x/c", ylabel="span k", titlesize=13)
    hw = heatmap!(axw, xwall, collect(1:NK), wallmap; colormap=:viridis, colorrange=_range(vec(wallmap), 0.6, 1.05))
    try; Colorbar(fig[rr, 3], hw); catch; end
    mkpath(dirname(out)); CairoMakie.save(out, fig)
    @printf("[CMeshMakie] wrote %s (%d KB)\n", out, filesize(out)÷1024); flush(stdout)
    out
end

# ------------------------------------------------------------------ Telegram
const _TG = normpath(joinpath(@__DIR__, "..", "tg_send.jl"))
function tg_send(png::String, caption::String)
    length(caption) > 1000 && (caption = caption[1:1000])
    try
        project=joinpath(dirname(dirname(_TG)),"third_party","JXfoil");
        run(`$(Base.julia_cmd()) --startup-file=no --project=$project $_TG $png $caption photo`); true
    catch e; @warn "tg_send failed" e; false; end
end

"""render + send in one call. `mode=:auto` picks 3D if s.rho is 3D."""
function render_and_send(s, mesh, met; caption::String, out::String, mode::Symbol=:auto, kwargs...)
    is3d = mode == :span3d || (mode == :auto && ndims(s.rho) == 3)
    png = is3d ? render_cmesh_span3d(s, mesh, met; out=out, kwargs...) :
                 render_cmesh_2d(s, mesh, met; out=out, kwargs...)
    tg_send(png, caption); png
end

end # module
