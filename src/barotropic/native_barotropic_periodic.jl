module NativeBarotropicPeriodic

using SHA, Serialization
using ..NativeBarotropicEOS: Material, phase_state

export PeriodicGrid, Config, FaceState, Solver, face_flux, spatial_operator!,
       step!, checkpoint, restore!, save_checkpoint, load_checkpoint,
       source_receipt, state_arrays, clock_state, AdmissionError

const Scalar = Union{Float32,Float64}
const OPERATOR_PATH = abspath(@__FILE__)
const OPERATOR_SHA256 = bytes2hex(sha256(read(OPERATOR_PATH)))
const EOS_PATH = abspath(String(first(methods(phase_state)).file))
const EOS_SHA256 = bytes2hex(sha256(read(EOS_PATH)))
source_receipt() = (; operator_path=OPERATOR_PATH, operator_sha256=OPERATOR_SHA256,
                       eos_path=EOS_PATH, eos_sha256=EOS_SHA256,
                       arithmetic="native-PC-or-paired-MUSCL-AUSM-periodic-incremental-SSPRK3-v2")

struct AdmissionError <: Exception
    status::Symbol
    message::String
end
Base.showerror(io::IO, e::AdmissionError) = print(io, e.status, ": ", e.message)
require(ok, status, message) = ok || throw(AdmissionError(status, message))

struct PeriodicGrid{T<:Scalar}
    nx::Int
    ny::Int
    Lx::T
    Ly::T
    dx::T
    dy::T
    volume::T
    function PeriodicGrid(nx::Integer, ny::Integer, Lx::T, Ly::T) where {T<:Scalar}
        require(nx >= 1 && ny >= 1, :grid, "positive cell counts required")
        require(all(isfinite, (Lx,Ly)) && Lx > 0 && Ly > 0, :grid,
                "finite positive domain lengths required")
        dx,dy = Lx/T(nx),Ly/T(ny)
        volume = dx*dy
        require(all(isfinite, (dx,dy,volume)) && min(dx,dy,volume)>0, :grid,
                "represented spacing and volume must be finite and positive")
        new{T}(Int(nx),Int(ny),Lx,Ly,dx,dy,volume)
    end
end

struct Config{T<:Scalar}
    cfl::T
    minf2::T
    reconstruction::Symbol
    function Config(cfl::T, minf2::T; reconstruction::Symbol=:pc) where {T<:Scalar}
        require(isfinite(cfl) && 0<cfl<=1, :config, "require finite 0 < CFL <= 1")
        require(isfinite(minf2) && minf2>=0, :config, "require finite minf2 >= 0")
        require(reconstruction in (:pc,:muscl_rho_q), :config, "unsupported reconstruction")
        new{T}(cfl,minf2,reconstruction)
    end
end

"""Primitive PC face state: rho, VELOCITIES u/v (not momenta), p, c, Y."""
struct FaceState{T<:Scalar}
    rho::T
    u::T
    v::T
    p::T
    c::T
    Y::T
end

@inline M1p(M::T) where T = T(0.5)*(M+abs(M))
@inline M1m(M::T) where T = T(0.5)*(M-abs(M))
@inline M2p(M::T) where T = T(0.25)*(M+one(T))*(M+one(T))
@inline M2m(M::T) where T = -T(0.25)*(M-one(T))*(M-one(T))
@inline M4p(M::T) where T = abs(M)>=one(T) ? M1p(M) : M2p(M)*(one(T)-T(2)*M2m(M))
@inline M4m(M::T) where T = abs(M)>=one(T) ? M1m(M) : M2m(M)*(one(T)+T(2)*M2p(M))
@inline P5p(M::T,a::T) where T = abs(M)>=one(T) ? M1p(M)/M : M2p(M)*((T(2)-M)-T(16)*a*M*M2m(M))
@inline P5m(M::T,a::T) where T = abs(M)>=one(T) ? M1m(M)/M : M2m(M)*((-T(2)-M)+T(16)*a*M*M2p(M))

@inline function valid_face(f::FaceState{T}) where T
    all(isfinite,(f.rho,f.u,f.v,f.p,f.c,f.Y)) &&
        f.rho>zero(T) && f.p>zero(T) && f.c>zero(T) && zero(T)<=f.Y<=one(T)
end

"""
    face_flux(L, R, minf2, nx, ny, area)

Area-integrated PC AUSM mass, x/y momentum and species fluxes. L/R contain
velocities. Cartesian oriented normals only. The species donor is selected by
the FINAL area-integrated mass flux, including its pressure correction. Exact
zero reference minf2 selects fa=1 for the whole configuration; positive minf2
uses the existing AUSM arithmetic ordering. This is a new generic operator,
without reconstruction floors, energy, viscosity or a low-Mach accuracy claim.
"""
@inline function face_flux(L::FaceState{T}, R::FaceState{T}, minf2::T,
                           nx::T, ny::T, area::T) where {T<:Scalar}
    z,o=zero(T),one(T)
    require(valid_face(L) && valid_face(R), :face_state, "invalid primitive face state")
    require(isfinite(minf2) && minf2>=z, :cutoff, "finite nonnegative minf2 required")
    require((abs(nx)==o && ny==z)||(nx==z && abs(ny)==o), :normal,
            "only Cartesian unit normals supported")
    require(isfinite(area) && area>z, :area, "finite positive face area required")
    unL=L.u*nx+L.v*ny; unR=R.u*nx+R.v*ny
    utL=-L.u*ny+L.v*nx; utR=-R.u*ny+R.v*nx
    af=T(0.5)*(L.c+R.c)
    Mb2=T(0.5)*(unL*unL+unR*unR)/(af*af)
    Mo2=minf2==z ? o : min(o,max(Mb2,minf2)); Mo=sqrt(Mo2)
    fa=Mo*(T(2)-Mo)
    ML=unL/af; MR=unR/af
    rf=T(0.5)*(L.rho+R.rho)
    Mh=M4p(ML)+M4m(MR)-(T(0.25)/fa)*max(o-Mb2,z)*(R.p-L.p)/(rf*af*af)
    mdot=af*Mh*(Mh>z ? L.rho : R.rho)
    alp=T(0.1875)*(-T(4)+T(5)*fa*fa)
    Pp=P5p(ML,alp); Pm=P5m(MR,alp)
    pf=Pp*L.p+Pm*R.p-T(0.75)*Pp*Pm*(L.rho+R.rho)*(fa*af)*(unR-unL)
    unU=mdot>z ? unL : unR; utU=mdot>z ? utL : utR
    Fmn=mdot*unU+pf; Fmt=mdot*utU
    mass=mdot*area
    mx=(nx*Fmn-ny*Fmt)*area; my=(ny*Fmn+nx*Fmt)*area
    donor=mass>z ? Int8(1) : mass<z ? Int8(-1) : Int8(0)
    q=donor==0 ? z : mass*(donor==1 ? L.Y : R.Y)
    require(all(isfinite,(af,Mb2,fa,ML,MR,rf,Mh,mdot,pf,mass,mx,my,q)) && af>z && fa>z,
            :face_arithmetic, "nonfinite/degenerate AUSM intermediate; no floor applied")
    (; mass,mx,my,q,donor)
end

struct StateBlock{T<:Scalar}
    rho::Matrix{T}
    mx::Matrix{T}
    my::Matrix{T}
    q::Matrix{T}
end
StateBlock(::Type{T},nx,ny) where T = StateBlock(ntuple(_->zeros(T,nx,ny),4)...)
state_arrays(b::StateBlock)=(b.rho,b.mx,b.my,b.q)
function copy_state!(dst::StateBlock, src::StateBlock)
    for (d,s) in zip(state_arrays(dst),state_arrays(src)); copyto!(d,s); end
    dst
end
struct Derived{T<:Scalar}
    u::Matrix{T}; v::Matrix{T}; p::Matrix{T}; c::Matrix{T}; Y::Matrix{T}
    alpha::Matrix{T}; beta::Matrix{T}; raw::Matrix{T}; Al::Matrix{T}
    volume_residual::Matrix{T}
end
Derived(::Type{T},nx,ny) where T = Derived(ntuple(_->zeros(T,nx,ny),10)...)
derived_arrays(d::Derived) = ntuple(k->getfield(d,k),10)
function copy_derived!(dst::Derived,src::Derived)
    for (d,s) in zip(derived_arrays(dst),derived_arrays(src)); copyto!(d,s); end
end

# Primitive increments are explicitly u/v, never momentum increments.
struct DirectionalSlopes{T<:Scalar}
    rho::Matrix{T}; q::Matrix{T}; u::Matrix{T}; v::Matrix{T}; theta::Matrix{T}
end
DirectionalSlopes(::Type{T},nx,ny) where T = DirectionalSlopes(ntuple(_->zeros(T,nx,ny),5)...)
mutable struct ReconstructionDiagnostics{T<:Scalar}
    minmod_reductions::Int
    bound_cells::Int
    faces::Int
    min_theta::T
    min_rho::T
    min_q::T
    min_liquid::T
    min_raw::T
    min_Al::T
    max_c::T
    max_normal_transport::T
    complete::Bool
end
ReconstructionDiagnostics(::Type{T}) where T =
    ReconstructionDiagnostics(0,0,0,one(T),T(Inf),T(Inf),T(Inf),T(Inf),T(Inf),zero(T),zero(T),false)
function reset!(d::ReconstructionDiagnostics{T}) where T
    d.minmod_reductions=0;d.bound_cells=0;d.faces=0;d.min_theta=one(T)
    d.min_rho=T(Inf);d.min_q=T(Inf);d.min_liquid=T(Inf);d.min_raw=T(Inf);d.min_Al=T(Inf)
    d.max_c=zero(T);d.max_normal_transport=zero(T);d.complete=false
    d
end
# Scratch diagnostics are unavailable for PC (reconstruction_scratch === nothing).
# MUSCL reset is per residual: after a successful macrostep these describe
# stage-3 RHS on B, not the committed state or a union over all stages.
# A failed residual may leave partial diagnostics with complete=false.
struct ReconstructionScratch{T<:Scalar}
    x::DirectionalSlopes{T}
    y::DirectionalSlopes{T}
    diagnostics::ReconstructionDiagnostics{T}
end
ReconstructionScratch(::Type{T},nx,ny) where T =
    ReconstructionScratch(DirectionalSlopes(T,nx,ny),DirectionalSlopes(T,nx,ny),ReconstructionDiagnostics(T))

@inline function minmod(a::T,b::T) where {T<:Scalar}
    require(isfinite(a) && isfinite(b),:reconstruction_slope,"nonfinite one-sided increment")
    d = (a>zero(T) && b>zero(T)) ? min(a,b) :
        (a<zero(T) && b<zero(T)) ? max(a,b) : zero(T)
    require(isfinite(d),:reconstruction_slope,"nonfinite minmod increment")
    d
end

"""Common pair bound only. Actual represented endpoints and EOS still decide admission."""
@inline function pair_factor(rho::T,q::T,dr::T,dq::T) where {T<:Scalar}
    dl=dr-dq;liquid=rho-q
    require(all(isfinite,(rho,q,dr,dq,dl,liquid)) && rho>zero(T) && zero(T)<=q<=rho,
            :reconstruction_pair,"invalid center or nonfinite paired increment")
    theta=one(T)
    for (b,d) in ((rho,dr),(q,dq),(liquid,dl))
        half=abs(d)/T(2)
        if half>b
            theta=min(theta,b/half)
        end
    end
    # Never prevfloat(0): a zero factor is an admitted explicit reduction.
    zero(T)<theta<one(T) && (theta=prevfloat(theta))
    theta
end

function reconstruct_slopes!(sl::DirectionalSlopes{T},U::StateBlock{T},D::Derived{T},
                             g::PeriodicGrid{T},axis::Int,diag::ReconstructionDiagnostics{T}) where T
    require(axis in (1,2),:reconstruction_axis,"axis must be 1 or 2")
    for j in 1:g.ny,i in 1:g.nx
        im=i==1 ? g.nx : i-1;ip=i==g.nx ? 1 : i+1
        jm=j==1 ? g.ny : j-1;jp=j==g.ny ? 1 : j+1
        for (src,dst) in ((U.rho,sl.rho),(U.q,sl.q),(D.u,sl.u),(D.v,sl.v))
            left=axis==1 ? src[im,j] : src[i,jm]
            right=axis==1 ? src[ip,j] : src[i,jp]
            a=src[i,j]-left;b=right-src[i,j]
            if !(isfinite(a) && isfinite(b))
                throw(AdmissionError(:reconstruction_slope,"nonfinite one-sided increment at cell ($i,$j), axis $axis"))
            end
            delta=minmod(a,b);dst[i,j]=delta
            abs(delta)<max(abs(a),abs(b)) && (diag.minmod_reductions+=1)
        end
        theta=pair_factor(U.rho[i,j],U.q[i,j],sl.rho[i,j],sl.q[i,j])
        sl.theta[i,j]=theta
        theta<one(T) && (diag.bound_cells+=1)
        diag.min_theta=min(diag.min_theta,theta)
    end
    sl
end

function reconstructed_face(U::StateBlock{T},D::Derived{T},sl::DirectionalSlopes{T},
                            m::Material{T},i,j,side::Int,axis::Int,
                            diag::ReconstructionDiagnostics{T}) where T
    require(side in (-1,1) && axis in (1,2),:reconstruction_side,"invalid side/axis")
    h=T(side)*(sl.theta[i,j]/T(2))
    r=U.rho[i,j]+h*sl.rho[i,j];q=U.q[i,j]+h*sl.q[i,j]
    u=D.u[i,j]+h*sl.u[i,j];v=D.v[i,j]+h*sl.v[i,j]
    if !(all(isfinite,(r,q,u,v)) && r>zero(T) && zero(T)<=q<=r)
        throw(AdmissionError(:reconstruction_state,"invalid represented face at cell ($i,$j), axis $axis, side $side"))
    end
    p=phase_state(m,r,q)
    p.valid || throw(AdmissionError(:reconstruction_eos,"EOS status $(p.status) at cell ($i,$j), axis $axis, side $side"))
    c=sqrt(p.c2)
    diag.faces+=1;diag.min_rho=min(diag.min_rho,r);diag.min_q=min(diag.min_q,q)
    diag.min_liquid=min(diag.min_liquid,r-q);diag.min_raw=min(diag.min_raw,p.raw)
    diag.min_Al=min(diag.min_Al,p.Al);diag.max_c=max(diag.max_c,c)
    diag.max_normal_transport=max(diag.max_normal_transport,(axis==1 ? abs(u) : abs(v))+c)
    FaceState(r,u,v,p.p,c,p.Y)
end

mutable struct Solver{T<:Scalar}
    const grid::PeriodicGrid{T}
    const material::Material{T}
    const config::Config{T}
    const U::StateBlock{T}
    const D::Derived{T}
    const base::StateBlock{T}
    const A::StateBlock{T}
    const B::StateBlock{T}
    const scratchD::Derived{T}
    const Fx::StateBlock{T}
    const Fy::StateBlock{T}
    const rhs::StateBlock{T}
    const reconstruction_scratch::Union{Nothing,ReconstructionScratch{T}}
    const digest::NamedTuple
    time::Float64
    time_compensation::Float64
    steps::Int
    last_dt::T
    last_path::Symbol
    failed::Bool
    history::Vector{NamedTuple}
end
state_arrays(s::Solver)=state_arrays(s.U)
clock_state(s::Solver)=(;time=s.time,compensation=s.time_compensation,steps=s.steps,
                         last_dt=s.last_dt,last_path=s.last_path)

function derive!(d::Derived{T},u::StateBlock{T},m::Material{T}) where T
    for i in eachindex(u.rho)
        rho,q=u.rho[i],u.q[i]
        if !(isfinite(u.mx[i]) && isfinite(u.my[i]))
            throw(AdmissionError(:momentum,"nonfinite momentum at $i"))
        end
        p=phase_state(m,rho,q)
        p.valid || throw(AdmissionError(:eos,"EOS status $(p.status) at cell $i"))
        vx,vy=u.mx[i]/rho,u.my[i]/rho
        if !(isfinite(vx) && isfinite(vy))
            throw(AdmissionError(:velocity,"nonfinite velocity at $i"))
        end
        d.u[i]=vx;d.v[i]=vy;d.p[i]=p.p;d.c[i]=sqrt(p.c2);d.Y[i]=p.Y
        d.alpha[i]=p.alpha;d.beta[i]=p.beta;d.raw[i]=p.raw;d.Al[i]=p.Al
        d.volume_residual[i]=p.volume_residual
    end
    d
end

function Solver(g::PeriodicGrid{T},m::Material{T},c::Config{T},
                rho::Matrix{T},mx::Matrix{T},my::Matrix{T},q::Matrix{T}) where {T<:Scalar}
    require(all(a->size(a)==(g.nx,g.ny),(rho,mx,my,q)),:shape,"all state shapes must match grid")
    U=StateBlock(copy(rho),copy(mx),copy(my),copy(q)); D=Derived(T,g.nx,g.ny)
    derive!(D,U,m)
    blocks=ntuple(_->StateBlock(T,g.nx,g.ny),6)
    Solver(g,m,c,U,D,blocks[1],blocks[2],blocks[3],Derived(T,g.nx,g.ny),
           blocks[4],blocks[5],blocks[6],
           c.reconstruction==:pc ? nothing : ReconstructionScratch(T,g.nx,g.ny),
           source_receipt(),0.0,0.0,0,zero(T),:none,
           false,NamedTuple[])
end

@inline face_state(U::StateBlock{T},D::Derived{T},i,j) where T =
    FaceState(U.rho[i,j],D.u[i,j],D.v[i,j],D.p[i,j],D.c[i,j],D.Y[i,j])
@inline function store_face!(F::StateBlock,i,j,f)
    F.rho[i,j]=f.mass;F.mx[i,j]=f.mx;F.my[i,j]=f.my;F.q[i,j]=f.q
end

# Shared divergence of the once-computed canonical face arrays. Keep arithmetic
# ordering identical to the original PC and experimental MUSCL residual loops.
function divergence!(s::Solver{T}) where T
    g=s.grid
    for (R,Fx,Fy) in zip(state_arrays(s.rhs),state_arrays(s.Fx),state_arrays(s.Fy))
        for j in 1:g.ny, i in 1:g.nx
            il=i==1 ? g.nx : i-1; jb=j==1 ? g.ny : j-1
            R[i,j]=((Fx[i,j]-Fx[il,j])+(Fy[i,j]-Fy[i,jb]))/g.volume
            require(isfinite(R[i,j]),:residual,"nonfinite conservative residual")
        end
    end
    s.rhs
end

"""Derive current state, compute each right/top periodic face once, then div(F).
Mutates scratch/face/residual storage only; never authoritative arrays or clock.
Fx[nx,j] is the nx->1 seam, Fy[i,ny] the ny->1 seam. Both use positive normals.
"""
function spatial_operator_pc!(s::Solver{T},U::StateBlock{T}=s.U) where T
    g=s.grid;d=s.scratchD;derive!(d,U,s.material)
    for j in 1:g.ny, i in 1:g.nx
        ir=i==g.nx ? 1 : i+1; jt=j==g.ny ? 1 : j+1
        L=face_state(U,d,i,j)
        store_face!(s.Fx,i,j,face_flux(L,face_state(U,d,ir,j),s.config.minf2,one(T),zero(T),g.dy))
        store_face!(s.Fy,i,j,face_flux(L,face_state(U,d,i,jt),s.config.minf2,zero(T),one(T),g.dx))
    end
    divergence!(s)
end

function spatial_operator_muscl!(s::Solver{T},U::StateBlock{T}) where T
    g=s.grid;d=s.scratchD;rec=s.reconstruction_scratch::ReconstructionScratch{T};diag=reset!(rec.diagnostics)
    derive!(d,U,s.material)
    reconstruct_slopes!(rec.x,U,d,g,1,diag)
    reconstruct_slopes!(rec.y,U,d,g,2,diag)
    for j in 1:g.ny,i in 1:g.nx
        ir=i==g.nx ? 1 : i+1;jt=j==g.ny ? 1 : j+1
        L=reconstructed_face(U,d,rec.x,s.material,i,j,1,1,diag)
        R=reconstructed_face(U,d,rec.x,s.material,ir,j,-1,1,diag)
        store_face!(s.Fx,i,j,face_flux(L,R,s.config.minf2,one(T),zero(T),g.dy))
        B=reconstructed_face(U,d,rec.y,s.material,i,j,1,2,diag)
        A=reconstructed_face(U,d,rec.y,s.material,i,jt,-1,2,diag)
        store_face!(s.Fy,i,j,face_flux(B,A,s.config.minf2,zero(T),one(T),g.dx))
    end
    divergence!(s)
    diag.complete=true
    s.rhs
end

function spatial_operator!(s::Solver{T},U::StateBlock{T}=s.U) where T
    s.config.reconstruction==:pc ? spatial_operator_pc!(s,U) : spatial_operator_muscl!(s,U)
end


function applied_dt(s::Solver{T},max_dt::Union{Nothing,T}) where T
    max_dt===nothing || require(isfinite(max_dt) && max_dt>zero(T),:dt,"positive finite max_dt required")
    d=s.scratchD;g=s.grid;rate=zero(T)
    for i in eachindex(d.u)
        localrate=(abs(d.u[i])+d.c[i])/g.dx+(abs(d.v[i])+d.c[i])/g.dy
        require(isfinite(localrate) && localrate>zero(T),:dt,"invalid spectral-rate estimate")
        rate=max(rate,localrate)
    end
    dt=s.config.cfl/rate
    max_dt===nothing || (dt=min(dt,max_dt))
    require(isfinite(dt) && dt>zero(T),:dt,"represented step must be positive finite")
    dt
end

# This exact-zero branch preserves even signed-zero uniform conservative arrays.
# It applies to a computed exactly-zero increment, without a magnitude threshold.
@inline function increment(base::T,current::T,rhs::T,dt::T,w::T) where T
    delta=w*((current-base)-dt*rhs)
    delta==zero(T) ? base : base+delta
end
function update_stage!(dest::StateBlock{T},base::StateBlock{T},cur::StateBlock{T},
                       rhs::StateBlock{T},dt::T,w::T) where T
    for (o,b,c,r) in zip(state_arrays(dest),state_arrays(base),state_arrays(cur),state_arrays(rhs))
        for i in eachindex(o); o[i]=increment(b[i],c[i],r[i],dt,w); end
    end
end

function failure!(s::Solver,stage,status,message,path)
    s.failed=true
    push!(s.history,(;kind=:failure,stage,status,message,path,at_step=s.steps,at_time=s.time))
    (;success=false,stage,status,message,dt=zero(s.last_dt),time=s.time,step=s.steps,path)
end

"""CFL-estimated global step; an optional positive same-T cap shortens this dt.
This spectral-rate rule is recorded explicitly; it is not an a priori stability proof.
"""
step!(s::Solver{T};max_dt::Union{Nothing,T}=nothing) where T =
    run_step!(s,nothing,max_dt,:cfl)
"""Explicit positive finite same-T controlled/test step. NO CFL admission is implied."""
step!(s::Solver{T},dt::T) where T = run_step!(s,dt,nothing,:explicit_controlled)

function run_step!(s::Solver{T},explicit_dt::Union{Nothing,T},max_dt::Union{Nothing,T},path) where T
    s.failed && return (;success=false,stage=0,status=:failed_run,
                          message="restore an admitted checkpoint to recover explicitly",
                          dt=zero(T),time=s.time,step=s.steps,path)
    stage=0
    try
        validate_clock(clock_state(s),T,:clock)
        derive!(s.scratchD,s.U,s.material)
        dt=if explicit_dt===nothing
            applied_dt(s,max_dt)
        else
            require(isfinite(explicit_dt) && explicit_dt>zero(T),:dt,"positive finite explicit dt required")
            explicit_dt
        end
        copy_state!(s.base,s.U)
        stage=1; spatial_operator!(s,s.base)
        update_stage!(s.A,s.base,s.base,s.rhs,dt,one(T));derive!(s.scratchD,s.A,s.material)
        stage=2; spatial_operator!(s,s.A)
        update_stage!(s.B,s.base,s.A,s.rhs,dt,one(T)/T(4));derive!(s.scratchD,s.B,s.material)
        stage=3; spatial_operator!(s,s.B)
        update_stage!(s.A,s.base,s.B,s.rhs,dt,T(2)/T(3));derive!(s.scratchD,s.A,s.material)
        # Compensated Float64 clock over actually applied representable T steps.
        y=Float64(dt)-s.time_compensation;time=s.time+y
        comp=(time-s.time)-y
        require(isfinite(time) && time>s.time,:clock,"step is not representable on accumulated clock")
        require(s.steps<typemax(Int),:clock,"step counter overflow")
        copy_state!(s.U,s.A);copy_derived!(s.D,s.scratchD)
        s.time=time;s.time_compensation=comp;s.steps+=1;s.last_dt=dt;s.last_path=path
        (;success=true,stage=3,status=:ok,message="",dt,time=s.time,step=s.steps,path)
    catch e
        e isa AdmissionError || rethrow()
        failure!(s,stage,e.status,e.message,path)
    end
end

const CHECKPOINT_SCHEMA="native-periodic-PC-or-paired-MUSCL-checkpoint-v2"
struct Checkpoint{T<:Scalar}
    data::NamedTuple
    checksum::String
end
function serialized_hash(data)
    io=IOBuffer();serialize(io,data);bytes2hex(sha256(take!(io)))
end
grid_values(g)=Tuple(getfield(g,k) for k in 1:fieldcount(typeof(g)))
material_values(m)=Tuple(getfield(m,k) for k in 1:fieldcount(typeof(m)))
config_values(c)=Tuple(getfield(c,k) for k in 1:fieldcount(typeof(c)))

function validate_clock(c,::Type{T},status) where T
    require(c isa NamedTuple && keys(c)==(:time,:compensation,:steps,:last_dt,:last_path),
            status,"missing or malformed clock fields")
    require(c.time isa Float64 && c.compensation isa Float64 && isfinite(c.time) &&
            isfinite(c.compensation) && c.time>=0 && c.steps isa Int && c.steps>=0 &&
            c.last_dt isa T && isfinite(c.last_dt) && c.last_dt>=zero(T) &&
            c.last_path in (:none,:cfl,:explicit_controlled),status,"invalid clock metadata")
    require((c.steps==0 && c.time==0 && c.compensation==0 && c.last_dt==0 && c.last_path==:none) ||
            (c.steps>0 && c.time>0 && c.last_dt>0 && c.last_path!=:none &&
             abs(c.compensation)<=eps(c.time)/2 && Float64(c.last_dt)<=c.time+eps(c.time)),
            status,"inconsistent clock metadata or implausible compensation")
end
function source_binding(d)
    require(d isa NamedTuple && keys(d)==(:operator_path,:operator_sha256,:eos_path,:eos_sha256,:arithmetic) &&
            all(v->v isa String,values(d)),:checkpoint_source,"malformed source receipt")
    (d.operator_sha256,d.eos_sha256,d.arithmetic)
end
function valid_history(event)
    event isa NamedTuple && hasproperty(event,:kind) || return false
    if event.kind==:failure
        keys(event)==(:kind,:stage,:status,:message,:path,:at_step,:at_time) || return false
        return event.stage isa Int && 0<=event.stage<=3 && event.status isa Symbol &&
            event.message isa String && event.path in (:cfl,:explicit_controlled,:restore) &&
            # Attempted clock values are diagnostics, including the invalid
            # values that caused rejection. The committed clock remains strict.
            event.at_step isa Int && event.at_time isa Float64
    elseif event.kind==:restore
        keys(event)==(:kind,:restored_step,:restored_time,:prior_failed,:checkpoint_checksum) || return false
        return event.restored_step isa Int && event.restored_step>=0 &&
            event.restored_time isa Float64 && isfinite(event.restored_time) && event.restored_time>=0 &&
            event.prior_failed isa Bool && event.checkpoint_checksum isa String
    end
    false
end

"""Complete committed-state checkpoint, including clock and failure metadata.
Scratch and derived arrays are reconstructed. This is a trusted local Julia
Serialization format, version-bound; its SHA is an integrity check, not a signature.
"""
function checkpoint(s::Solver{T}) where T
    validate_clock(clock_state(s),T,:clock)
    derive!(s.scratchD,s.U,s.material)
    data=(;schema=CHECKPOINT_SCHEMA,julia_version=string(VERSION),scalar_type=string(T),
        digest=s.digest,grid=grid_values(s.grid),material=material_values(s.material),
        config=config_values(s.config),arrays=map(copy,state_arrays(s)),clock=clock_state(s),
        failed=s.failed,history=deepcopy(s.history))
    Checkpoint{T}(data,serialized_hash(data))
end

function restore!(s::Solver{T},cp::Checkpoint{C}) where {T,C}
    T===C || return failure!(s,0,:checkpoint_type,"exact checkpoint scalar type mismatch",:restore)
    d=cp.data
    try
        require(serialized_hash(d)==cp.checksum,:checkpoint_checksum,"checkpoint bytes do not match digest")
        require(keys(d)==(:schema,:julia_version,:scalar_type,:digest,:grid,:material,:config,:arrays,:clock,:failed,:history),
                :checkpoint_schema,"missing or malformed checkpoint fields")
        require(d.schema==CHECKPOINT_SCHEMA && d.julia_version==string(VERSION),:checkpoint_version,"checkpoint schema/Julia version mismatch")
        require(d.scalar_type==string(T),:checkpoint_type,"exact scalar type mismatch")
        require(isequal(source_binding(d.digest),source_binding(s.digest)),:checkpoint_source,"source/operator digest mismatch")
        require(isequal(d.grid,grid_values(s.grid)) && isequal(d.material,material_values(s.material)) &&
                isequal(d.config,config_values(s.config)),:checkpoint_config,"material/grid/config mismatch")
        require(d.arrays isa Tuple && length(d.arrays)==4 && all(a->a isa Matrix{T} && size(a)==(s.grid.nx,s.grid.ny),d.arrays),
                :checkpoint_shape,"checkpoint array type/shape mismatch")
        c=d.clock
        validate_clock(c,T,:checkpoint_clock)
        require(d.failed isa Bool && d.history isa Vector{NamedTuple} && all(valid_history,d.history),
                :checkpoint_history,"invalid failure metadata")
        candidate=StateBlock(d.arrays...);derive!(s.scratchD,candidate,s.material)
        # Keep local failures that occurred after the saved state as audit history.
        local_history=deepcopy(s.history)
        merged=deepcopy(d.history)
        for event in local_history; event in merged || push!(merged,event); end
        push!(merged,(;kind=:restore,restored_step=c.steps,restored_time=c.time,
                       prior_failed=s.failed,checkpoint_checksum=cp.checksum))
        copy_state!(s.U,candidate);copy_derived!(s.D,s.scratchD)
        s.time=c.time;s.time_compensation=c.compensation;s.steps=c.steps
        s.last_dt=c.last_dt;s.last_path=c.last_path;s.failed=d.failed;s.history=merged
        # A restored state has no associated completed reconstruction residual.
        s.reconstruction_scratch===nothing || reset!(s.reconstruction_scratch.diagnostics)
        (;success=true,status=:restored)
    catch e
        e isa AdmissionError || rethrow()
        failure!(s,0,e.status,e.message,:restore)
    end
end
save_checkpoint(path::AbstractString,cp::Checkpoint)=open(io->serialize(io,cp),path,"w")
load_checkpoint(path::AbstractString)=open(deserialize,path,"r")

end # module
