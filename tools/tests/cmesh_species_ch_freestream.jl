# julia --startup-file=no --project=src tools/tests/cmesh_species_ch_freestream.jl
# The conservative Species strip pair (src/cmesh/cmesh_species_strip.jl) with a FREE-STREAM C block: a small C+H mesh of
# a blunt NACA 0012 in a 60-chord frame, every outer face of the C block characteristic (the Species carrier's
# free-fluid contract: outer_mode 1, all tags 1), fixed static pressure on the downstream ends of both blocks. This is
# the configuration Airfoil Lab's two-block Species worker runs (apps/airfoil_web/src/workers/species_ch_worker.jl; the
# carrier below repeats its `_species_carrier` statement for statement). CPU, short marches. Development checks, not
# validation. The solver source is unchanged: the pair is the production one, only its C block's outer tags differ.
using Test,KernelAbstractions,Printf,Logging
const LBM_F=Float32
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
include(joinpath(ROOT,"src","collision","lattice.jl"))
include(joinpath(ROOT,"src","cmesh","cmesh_gen.jl"))
include(joinpath(ROOT,"src","cmesh","cmesh_metrics.jl"))
include(joinpath(ROOT,"src","compressible","cmesh_solver_ka.jl"))
include(joinpath(ROOT,"src","cavitation","cmesh_cav_tem.jl"))
include(joinpath(ROOT,"src","cavitation","cmesh_cav_species.jl"))
const G=CMeshSolverKA;const P=CMeshCavSpecies;const T=CMeshCavTEM
const CX,CY,CZ,WT,_,_=build_d3q39_static()
G.GLOBAL_DT[]=1
P.LEDGER_ENABLED[]=true          # the balances below read the pair's conservation ledger

"NACA 0012 truncated at 95 % chord (a blunt base of 1.7 % chord), unit chord."
function foil(;cut=0.95,n=60)
    yt(x)=5*.12*(.2969*sqrt(x)-.126x-.3516x^2+.2843x^3-.1036x^4)
    xs=[cut*.5*(1-cospi(k/n)) for k in 0:n];y=[yt(x) for x in xs]
    CMeshGen.AirfoilPts(vcat(reverse(xs),xs[2:end])./cut,vcat(-reverse(y),y[2:end])./cut)
end
"The C+H mesh: 128 foil cells, 64 wake stations, 64 rows, 4 base cells per half, in a frame of 60 chords."
function small_mesh(;nb=4)
    af=foil();gap=hypot(af.x[1]-af.x[end],af.y[1]-af.y[end])
    frame=CMeshGen.ChannelSpec(height=60.,xi_in=-29.5,xi_out=30.5,angle_deg=0.)
    with_logger(SimpleLogger(IOBuffer(),Logging.Warn)) do
        CMeshGen.generate_cmesh(af;n_foil=128,n_wake=64,n_normal=64,dn1=4e-4,growth_max=1.25,smooth_iters=80,bl_freeze=6,
            channel=frame,te_topology=:strip,n_base=nb,te_ds=1.5gap/(2nb))
    end
end
"Both blocks' metrics. `free=true`: the C block's outer faces all characteristic (a free stream); the strip's all interface."
function metrics(m;free=true)
    base=[(m.X[m.NIW+1,1],m.Y[m.NIW+1,1],m.X[m.NIW+m.NIF+1,1],m.Y[m.NIW+m.NIF+1,1])]
    c=CMeshMetrics._build_metrics(m.X,m.Y,m.NIW,m.NIF,fill(UInt8(0),m.NI);extra_segments=base)
    if free
        raw=NamedTuple{fieldnames(typeof(c))}(Tuple(getfield(c,k) for k in fieldnames(typeof(c))))
        c=merge(raw,(outer_face_kind=fill(UInt8(1),c.NI),channel_provenance=false,free_fluid_provenance=true))
    end
    h=CMeshMetrics._build_metrics(m.strip.X,m.strip.Y,m.strip.NI,0,fill(UInt8(4),m.strip.NI);channel_provenance=true,wall_segments=base)
    (;c,h)
end
"One carrier and its TEM at the uniform ambient state: isothermal air, or water with the production calibration."
function carrier(met;mach,alpha,re,sigma,air)
    s=G.build_sol_ka(CPU(),met;drot=2f0,cxv=CX,cyv=CY,czv=CZ,wv=WT)
    s.wall_flux=true;s.recon=Int32(1);s.fullvisc=Int32(1)
    G.set_freestream_ka!(s,mach,alpha;pg_correction=false)
    speed=hypot(Float64(s.uxi),Float64(s.uyi));s.nu_lam=Float32(speed/re)
    G.set_sst_ka!(s;tu_inf=0.001,mu_t_ratio=0.1)
    s.cav=Int32(4)
    t=air ? T.build_tem(s;sigma,ratio=1000.0,vfloor=false,eos=1,wallis=1,vaploc=1.0,exact_reference=true) :
        T.build_tem(s;sigma,Cprod=100.,Cdest=5.,ratio=43197.,a2l_frac=879.8,eps_pv=.1,Tmin=1.2f0,rate_ref_pv=true,
            vfloor=false,eos=1,wallis=1,vaploc=1.0,exact_reference=true)
    t.omega=1f0;t.src_jac=Int32(0)
    p,a2=if air
        c2=Float32((speed/mach)^2)
        t.p_v=c2*s.rho_inf;t.rho_l_sat=s.rho_inf;t.rho_v_sat=0.001f0*s.rho_inf;t.a2l=1f30
        t.lq=1f0;t.psm=0f0;t.am_floor_frac=0f0;t.kprod=0f0;t.kdest=0f0;fill!(t.alpha,0f0)
        (Float64(c2*s.rho_inf),Float64(c2))
    else
        T.tem_ambient(t,s)
    end
    G.set_farfield_mixture!(s;p_ref=p,a2m=a2,Tmin=t.Tmin,Tmax=t.Tmax)
    fill!(s.T,clamp(Float32(p)/(s.rho_inf*G.CS2),t.Tmin,t.Tmax))
    s.pexc .= Float32(p).-s.rho_inf*G.CS2.*s.T
    fill!(s.am,sqrt(Float32(a2)))
    G.set_pressure_outlet!(s,t;p_out=p,checked_mixture=true)
    (;s,t,p,speed)
end
"""The free-stream pair. `interior_speed` scales the start velocity of both blocks against the far field's (a start
that is not the ambient state, so mass crosses the outer boundary from the first step); `kw` reaches build_species_ka."""
function pair(m;mach=.2,alpha=2.,re=1e6,sigma=1.5,air=true,interior_speed=1.,kw...)
    met=metrics(m)
    C=carrier(met.c;mach,alpha,re,sigma,air);H=carrier(met.h;mach,alpha,re,sigma,air)
    if interior_speed!=1
        for q in (C.s,H.s);q.ux .*= Float32(interior_speed);q.uy .*= Float32(interior_speed);end
    end
    G.seed_channel_reference_ka!(C.s)
    c=P.build_species_ka(C.s,C.t;wallmode=Int32(2),reconstruction=true,eos_precision=:double,nut_density=:none,
        sst_correction_limit=.5,strip=(solver=H.s,tem=H.t),kw...)
    (;c,s=C.s,t=C.t,ss=H.s,ts=H.t,p=C.p,speed=C.speed,met)
end
"Inventories over the given caches' interiors (Float64): mass, x and y momentum, the vapour species."
function inventory(caches...)
    out=zeros(4)
    for q in caches
        U=Array(q.conserved.U);v=Array(q.solver.vol)
        for k in 1:4,j in axes(v,2),i in axes(v,1);out[k]+=U[i,j,k]*Float64(v[i,j]);end
    end
    out
end
"Every interface face pair of every transported quantity, at the stage observer's call: (pairs seen, pairs whose two sides do not sum to zero)."
function interface_pairs(c)
    x=c.strip;s,ss=c.solver,x.solver;a,b=c.turbulence,x.turbulence;NIW,NIF,NS=s.NIW,s.NIF,ss.NJ;seen=0;open=0
    for (FC,FS) in ((s.Fj_r,ss.Fj_r),(s.Fj_mx,ss.Fj_mx),(s.Fj_my,ss.Fj_my),(s.Fj_e,ss.Fj_e),(c.Fj,x.Fj),(a.Fjk,b.Fjk),(a.Fjw,b.Fjw))
        leg=Array(view(FC,:,1));bottom=Array(view(FS,:,1));top=Array(view(FS,:,NS+1))
        for k in 1:NIW
            seen+=2;open+=(leg[NIW+1-k]+bottom[k]!=0)+(leg[NIW+NIF+k]-top[k]!=0)
        end
    end
    (seen,open)
end
"n deferred steps; the ledger's boundary and source sums, the retries and the interface pairs of every stage."
function march!(c,n;cfl=.3f0,admission=:deferred)
    mass=0.;species=0.;source=0.;retries=0;seen=0;open=0
    try
        P.STRIP_STAGE_OBSERVER[]=(q,stage)->((a,b)=interface_pairs(q);seen+=a;open+=b)
        for _ in 1:n
            P.step_species_ka!(c,cfl;admission)
            mass+=c.last_mass_boundary;species+=c.last_species_boundary;source+=c.last_species_source;retries+=c.last_retries
        end
    finally
        P.STRIP_STAGE_OBSERVER[]=nothing
    end
    (;mass,species,source,retries,seen,open)
end
finite_state(c)=all(q->all(isfinite,Array(q.conserved.U)) && all(isfinite,Array(q.pressure)),(c,c.strip))
const MESH=small_mesh()

@testset "admission: the free-stream C block is the Species free-fluid block" begin
    f=pair(MESH)
    @test f.s.outer_mode==1 && all(==(0x01),Array(f.s.outer_face_kind))      # no wall, no tunnel: every outer face characteristic
    @test f.ss.outer_mode==2 && all(==(0x04),Array(f.ss.outer_face_kind))
    @test f.c.strip!==nothing && f.c.inlet_present && !f.c.strip.inlet_present
    @test f.s.NI==2f.s.NIW+f.s.NIF && f.ss.NI==f.s.NIW && f.ss.NJ==8 && f.ss.NIF==0
    @test occursin("CMeshSpeciesStripTopologyV1",string(f.c.configuration[end]))
    @test f.c.steps==0 && f.c.elapsed==0 && finite_state(f.c)
    # The pair's checkpoint is the pair format, with both blocks' conserved state.
    cp=P.species_checkpoint(f.c)
    @test cp.format=="CMeshSpeciesKA/3" && size(cp.strip.conserved)==(f.ss.NI,f.ss.NJ,4) && cp.geometry_sha256==f.c.geometry_sha256
    # The gas solver's all-far-field block (outer tags 0, outer_mode 0) is not a Species block: the mixture pressure
    # ends are refused on it, so a free stream is the free-fluid block above.
    gas=G.build_sol_ka(CPU(),metrics(MESH;free=false).c;drot=2f0,cxv=CX,cyv=CY,czv=CZ,wv=WT)
    @test gas.outer_mode==0
    gas.cav=Int32(4);G.set_farfield_mixture!(gas;p_ref=f.p,a2m=1.,Tmin=f.t.Tmin,Tmax=f.t.Tmax)
    @test_throws ArgumentError G.set_pressure_outlet!(gas,f.t;p_out=f.p,checked_mixture=true)
end

@testset "a uniform free stream is preserved by the outer contour, both interfaces and the fixed-pressure ends" begin
    # 3 degrees to the frame: the stream crosses the lateral outer faces, the interfaces and the ends obliquely.
    f=pair(MESH;alpha=3.);c=f.c;x=c.strip
    P._stage_residual_pair!(c,Int32(2));P._species_flux_pair!(c)
    scale=Float64(f.s.rho_inf)*f.speed;worst=Dict{String,Float64}()
    # The flux divergence of the uniform state per unit (rho U x cell perimeter), away from the walls (0.05 chord):
    # mass, and momentum against the pressure force p x perimeter that the closed cell must cancel.
    for (id,q,sp,far) in (("C",f.s,c,f.met.c.wdist),("H",f.ss,x,f.met.h.wdist))
        Fi=map(Array,(q.Fi_r,q.Fi_mx,q.Fi_my));Fj=map(Array,(q.Fj_r,q.Fj_mx,q.Fj_my))
        six,siy,sjx,sjy=map(Array,(q.Six,q.Siy,q.Sjx,q.Sjy))
        for j in 1:q.NJ,i in 1:q.NI
            far[i,j]>0.05 || continue
            perimeter=hypot(six[i,j],siy[i,j])+hypot(six[i+1,j],siy[i+1,j])+hypot(sjx[i,j],sjy[i,j])+hypot(sjx[i,j+1],sjy[i,j+1])
            div(k)=Float64(Fi[k][i+1,j])-Float64(Fi[k][i,j])+Float64(Fj[k][i,j+1])-Float64(Fj[k][i,j])
            region=id=="H" ? "H" : j==q.NJ ? "C outer row" : (i==1 || i==q.NI) ? "C ends" :
                (j==1 && (i<=q.NIW || i>q.NIW+q.NIF)) ? "C interface row" : "C interior"
            worst[region*" mass"]=max(get(worst,region*" mass",0.),abs(div(1))/(scale*perimeter))
            worst[region*" momentum"]=max(get(worst,region*" momentum",0.),hypot(div(2),div(3))/(f.p*perimeter))
        end
    end
    for key in sort!(collect(keys(worst)));@printf("FREESTREAM residual %-26s %.3e\n",key,worst[key]);end
    @test Set(keys(worst))==Set(r*q for r in ("C interior","C outer row","C ends","C interface row","H") for q in (" mass"," momentum"))
    @test all(v->v<=2e-6,values(worst))          # Float32 face fluxes: rounding of rho u S and of p S around a closed cell
    # Marched: the state away from the walls stays the free stream (the wall's disturbance has travelled 1e-3 chord).
    r=march!(c,20)
    @test c.steps==20 && r.retries==0 && r.open==0
    deviation=0.
    for (q,far) in ((f.s,f.met.c.wdist),(f.ss,f.met.h.wdist))
        g=q.g;rho,u,v=map(Array,(q.rho,q.ux,q.uy))
        for j in 1:q.NJ,i in 1:q.NI
            far[i,j]>0.05 || continue
            deviation=max(deviation,abs(rho[i+g,j+g]/q.rho_inf-1),hypot(u[i+g,j+g]-q.uxi,v[i+g,j+g]-q.uyi)/f.speed)
        end
    end
    @printf("FREESTREAM deviation after 20 steps, beyond 0.05 chord of the walls: %.3e\n",deviation)
    @test deviation<=1e-5
end

@testset "blunt NACA 0012 in air: 60 steps, both masses conserved over the pair, one interface flux" begin
    for interior_speed in (1.,.8)
        f=pair(MESH;interior_speed);c=f.c
        I0=inventory(c,c.strip);H0=inventory(c.strip)
        r=march!(c,60)
        I1=inventory(c,c.strip);H1=inventory(c.strip)
        @test c.steps==60 && c.strip.steps==60 && c.valid && finite_state(c)
        @test r.retries==0                                  # no rejection at all, from an impulsive start
        @test r.seen==60*3*7*2*f.s.NIW && r.open==0         # every interface face pair of every stage sums to exactly zero
        residual=I1[1]-I0[1]-r.mass
        @printf("AIR interior_speed=%.1f mass: inventory %.6e change %.6e boundary %.6e residual %.3e (relative %.3e); H block's own change %.3e; dt %.3e\n",
            interior_speed,I0[1],I1[1]-I0[1],r.mass,residual,abs(residual)/I1[1],H1[1]-H0[1],f.s.last_global_dt)
        @test abs(residual)<=1e-9*I1[1]                     # the channel pair's clause: 1e-9 of the inventory
        @test abs(H1[1]-H0[1])>1e3*abs(residual)            # mass did cross the interfaces: the balance is over the pair
        @test I1[4]==0 && r.species==0 && r.source==0       # air holds no vapour and makes none
        if interior_speed!=1
            @test abs(r.mass)>100*1e-9*I1[1]                # mass did cross the outer boundary: the clause is not vacuous
            @test abs(residual)<=1e-6*abs(r.mass)           # and the ledger closes against the amount that crossed
        end
        cl,cd=G.loads_ka(f.s;mixture_pressure=c.pressure,wallmode=Int32(2)).+P.loads_strip_ka(c.strip)
        @test isfinite(cl) && isfinite(cd) && all(isfinite,P.loads_strip_ka(c.strip))
    end
end

@testset "both admissions of the free-stream pair publish the same state" begin
    a=pair(MESH).c;b=pair(MESH).c
    for _ in 1:5;P.step_species_ka!(a,.3f0;admission=:deferred);P.step_species_ka!(b,.3f0;admission=:immediate);end
    @test Array(a.conserved.U)==Array(b.conserved.U) && Array(a.strip.conserved.U)==Array(b.strip.conserved.U)
    @test a.elapsed==b.elapsed && a.steps==b.steps==5
end

@testset "cavitating water from a uniform start: 60 steps, vapour appears, both masses conserved" begin
    f=pair(MESH;air=false,sigma=.8,stage_clock_margin=.99);c=f.c
    I0=inventory(c,c.strip)
    r=march!(c,60)
    I1=inventory(c,c.strip)
    vapour=max(maximum(Array(f.t.alpha)),maximum(Array(f.ts.alpha)))
    @test c.steps==60 && c.valid && finite_state(c) && r.open==0
    @test r.retries<=60                                      # at most one halving per step: no retry is exhausted
    @test 0<vapour<=1 && I1[4]>0
    @test all(a->all(x->0<=x<=1,Array(a)),(f.t.alpha,f.ts.alpha))
    mass=I1[1]-I0[1]-r.mass;species=I1[4]-I0[4]-r.species-r.source
    @printf("WATER sigma 0.8: retries %d in 60 steps; max vapour fraction %.3e; mass residual %.3e (relative %.3e); species change %.6e boundary %.3e source %.6e residual %.3e; dt %.3e\n",
        r.retries,vapour,mass,abs(mass)/I1[1],I1[4]-I0[4],r.species,r.source,species,f.s.last_global_dt)
    @test abs(mass)<=1e-9*I1[1]
    @test abs(species)<=max(1e-15,1e-9*I1[4])                # the channel pair's species clause
    @test abs(species)<=1e-6*abs(I1[4]-I0[4])                # and against the amount of vapour made
end
