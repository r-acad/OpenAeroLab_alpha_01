# ---------------------------------------------------------------------------------------------
# PATH STATUS: **PRODUCTION** turbulence model. It carries a Reboud eddy-viscosity correction, applied at the nut
# write only when the cache's `reboud_n > 0`, in one of three forms (`reboud_form`): the void-fraction form V3,
# `_reboud_void`, Reboud's function of the cell's represented vapour volume fraction at its own pressure, **the default
# since the review of 13 Oct 2026** (`DEFAULT_NUT_DENSITY` is `:reboud_v3`, tag `CMeshSpeciesReboudNutV3/n=<n>`); the
# composition form V2 of 10 Oct, `_reboud_ratio` of the stored reference composition (`:reboud_v2`, tag
# `CMeshSpeciesReboudNutV2/n=<n>`, opt-in); or the density form V1 of 20 Sep, `_reboud_factor` of the mixture density
# (`:reboud_v1`, tag `CMeshSpeciesReboudNutV1/n=<n>`, arm T25's). `:none` stores `reboud_n = 0.0f0`, so the factor is
# never formed: bitwise the pre-20-September path, which reproduces every archived arm. The SST k production uses the
# uncorrected nu_t (the 20 Sep convention, as on the legacy path) unless the opt-in production-consistent form is set
# (`reboud_prod`, tag `CMeshSpeciesReboudProductionV1`). The driver refuses a species run that names neither
# TOPO_SPECIES_NUT_DENSITY nor the 2026-10-10 defaults epoch (memory/project_docs/2026-10-10_REBOUD_DEFAULT_AND_SCHNERR_SAUER_DESIGN.md
# sections 1.4 and 6). Each treatment is bound in the configuration fingerprint, so checkpoints of different settings
# cannot be mixed.
# A SECOND and INDEPENDENT Reboud implementation lives on the legacy branch (`sst_step_ka!` in
# cmesh_solver_ka.jl), which the production path never calls: do not cite a turbulence verdict obtained
# on the legacy path as covering this one, and never infer from this file alone whether a given result
# carries the correction -- read that run's realised `SPECIES NUT DENSITY:` line.
# Map: memory/project_docs/CAVITATION_CODE_MAP.md (checked by tools/check_code_map.jl).
# ---------------------------------------------------------------------------------------------
# Included inside CMeshCavSpecies: fully turbulent conservative SST-2003m.
# Model equations: https://tmbwg.github.io/turbmodels/sst.html . Legacy
# sst_step_ka! remains unchanged. The m variant uses strain-squared production
# and omits isotropic rho*k stress; the shared deviatoric stress is unchanged.
# The prescribed geometry trip is retained as owned metadata, not a source factor.
mutable struct SpeciesSSTKA{A}
    const rk::A
    const rw::A
    const k0::A
    const w0::A
    const ktrial::A
    const wtrial::A
    const dck::A
    const dcw::A
    const sk::A
    const sw::A
    const sigk::A
    const sigw::A
    const rate::A
    const kfloor::A # per-stage k floor correction times cell volume; ledger scratch, never rollback state
    const wpin::A   # per-stage wall-function omega pin (rho*omega change times volume, 2026-09-26); ledger scratch
    const Fik::A
    const Fiw::A
    const Fjk::A
    const Fjw::A
    const wdist::A
    const gammat::A
    const geometry_sha256::String
    last_k_boundary::Float64
    last_w_boundary::Float64
    last_k_source::Float64
    last_w_source::Float64
end

function _sst_geometry_hash(s)
    io=IOBuffer()
    for a in (s.wdist,s.gammat)
        h=Array(a); print(io,eltype(h),':',size(h),'\n'); write(io,reinterpret(UInt8,vec(h)))
    end
    bytes2hex(sha256(take!(io)))
end

function _admit_sst(s)
    s.nu_lam>0.0f0 && s.fullvisc==Int32(1) &&
        all(iszero,(s.keps,s.lag,s.earsm,s.rhok,s.fbm_cdelta,s.pans_fk)) ||
        throw(ArgumentError("conservative SST requires positive laminar viscosity, full stress and no turbulence extensions"))
    isfinite(s.k_inf) && s.k_inf>=1.0f-12 && isfinite(s.w_inf) && s.w_inf>=1.0f-6 ||
        throw(ArgumentError("conservative SST requires finite positive inflow k and omega"))
    all(size(getfield(s,k))==size(s.rho) && typeof(getfield(s,k))==typeof(s.rho)
        for k in (:tk,:tom,:nut)) || throw(ArgumentError("SST state array shape/backend mismatch"))
    all(size(getfield(s,k))==(s.NI,s.NJ) && typeof(getfield(s,k))==typeof(s.rho)
        for k in (:wdist,:gammat)) || throw(ArgumentError("SST geometry shape/backend mismatch"))
end

@kernel function _k_sst_conserved_init!(rk,rw,@Const(rho),@Const(tk),@Const(tom),
                                        NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        rk[i,j]=rho[i+g,j+g]*tk[i+g,j+g]
        rw[i,j]=rho[i+g,j+g]*tom[i+g,j+g]
    end
end

function _build_sst(s,checkpoint)
    s.sst || return nothing
    # Setup/restore admits the static ranges; _k_sst_owned_check! subsequently
    # rejects any changed bits alongside its dynamic state checks.
    all(isfinite,s.wdist) && minimum(s.wdist)>0.0f0 && all(isfinite,s.gammat) &&
        minimum(s.gammat)>=0.0f0 && maximum(s.gammat)<=1.0f0 ||
        throw(ArgumentError("invalid SST wall distance or prescribed transition factor"))
    z(m,n)=fill!(similar(s.rho,m,n),0.0f0)
    a=SpeciesSSTKA((z(s.NI,s.NJ) for _ in 1:15)...,
        z(s.NI+1,s.NJ),z(s.NI+1,s.NJ),z(s.NI,s.NJ+1),z(s.NI,s.NJ+1),
        copy(s.wdist),copy(s.gammat),_sst_geometry_hash(s),0.0,0.0,0.0,0.0)
    if checkpoint===nothing
        _k_sst_conserved_init!(s.backend)(a.rk,a.rw,s.rho,s.tk,s.tom,
            Int32(s.NI),Int32(s.NJ),Int32(s.g);ndrange=(s.NI,s.NJ))
    else
        hasproperty(checkpoint,:turbulence) && checkpoint.turbulence!==nothing ||
            throw(ArgumentError("SST checkpoint lacks authoritative turbulence state"))
        cp=checkpoint.turbulence
        cp.format=="CMeshSpeciesSST/1" && cp.geometry_sha256==a.geometry_sha256 &&
            size(cp.rk)==size(a.rk) && size(cp.rw)==size(a.rw) &&
            eltype(cp.rk)==Float32 && eltype(cp.rw)==Float32 ||
            throw(ArgumentError("SST checkpoint geometry/state mismatch"))
        copyto!(a.rk,cp.rk); copyto!(a.rw,cp.rw)
    end
    a
end

@kernel function _k_sst_owned_check!(status,@Const(rk),@Const(rw),@Const(rho),@Const(tk),@Const(tom),@Const(nut),
                                     @Const(wdist),@Const(gammat),@Const(oldwd),@Const(oldgt),
                                     NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        ia=i+g;ja=j+g;r=rho[ia,ja];k=tk[ia,ja];w=tom[ia,ja];n=nut[ia,ja]
        K=rk[i,j];W=rw[i,j];ek=r*k;ew=r*w
        good=isfinite(r) && r>0.0f0 && isfinite(k) && k>=1.0f-12 && isfinite(w) && w>=1.0f-6 &&
            isfinite(n) && n>=0.0f0 && isfinite(K) && K>0.0f0 && isfinite(W) && W>0.0f0 &&
            abs(K-ek)<=32.0f0*eps(Float32)*max(abs(K),abs(ek)) &&
            abs(W-ew)<=32.0f0*eps(Float32)*max(abs(W),abs(ew)) &&
            reinterpret(UInt32,wdist[i,j])==reinterpret(UInt32,oldwd[i,j]) &&
            reinterpret(UInt32,gammat[i,j])==reinterpret(UInt32,oldgt[i,j])
        status[i,j]=max(status[i,j],good ? 0.0f0 : 1.0f0)
    end
end

function _check_sst_layout!(c)
    a=c.turbulence; a===nothing && return
    s=c.solver
    for (names,shape) in (((:rk,:rw,:k0,:w0,:ktrial,:wtrial,:dck,:dcw,:sk,:sw,:sigk,:sigw,:rate,:kfloor,:wpin,:wdist,:gammat),(s.NI,s.NJ)),
                          ((:Fik,:Fiw),(s.NI+1,s.NJ)),((:Fjk,:Fjw),(s.NI,s.NJ+1)))
        all(size(getfield(a,k))==shape && typeof(getfield(a,k))==typeof(s.rho) for k in names) ||
            throw(ArgumentError("SST cache array shape/backend changed"))
    end
    nothing
end

function _sst_owned_status!(c;status=c.status)
    a=c.turbulence;s=c.solver
    # The phase kernel initialized every full-ghost status entry. Preserve its
    # flags while adding SST failures; the caller performs one shared maximum.
    _k_sst_owned_check!(s.backend)(status,a.rk,a.rw,s.rho,s.tk,s.tom,s.nut,s.wdist,s.gammat,a.wdist,a.gammat,
        Int32(s.NI),Int32(s.NJ),Int32(s.g);ndrange=(s.NI,s.NJ))
end

_sst_checkpoint(a::Nothing)=nothing
_sst_checkpoint(a::SpeciesSSTKA)=(format="CMeshSpeciesSST/1",geometry_sha256=a.geometry_sha256,rk=Array(a.rk),rw=Array(a.rw))

# Finite wall-face SST boundary value, with first-cell distance d1. The interior
# asymptotic profile is 6nu/(beta1*y^2); the prescribed wall value has factor 60.
@inline _sst_wall_omega(nu,d1)=60.0f0*nu/(0.075f0*d1*d1)

@kernel function _k_sst_species_bc_j!(tk,tom,nut,@Const(ux),@Const(uy),@Const(Sjx),@Const(Sjy),@Const(vol),
                                      @Const(kinds),ki::Float32,wi::Float32,nu::Float32,wallmode::Int32,
                                      NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32)
    i,k=@index(Global,NTuple)
    @inbounds if i<=NI && k<=g
        ia=i+g; js=g+k; jd=g+1-k
        if i<=NIW || i>NIW+NIF
            ip=NI+1-i+g
            tk[ia,jd]=tk[ip,js];tom[ia,jd]=tom[ip,js];nut[ia,jd]=nut[ip,js]
        elseif wallmode==Int32(2)
            S=sqrt(Sjx[i,1]^2+Sjy[i,1]^2);d1=0.5f0*vol[i,1]/S+1.0f-30
            tk[ia,jd]=-tk[ia,js]
            tom[ia,jd]=2.0f0*_sst_wall_omega(nu,d1)-tom[ia,js]
            nut[ia,jd]=0.0f0
        else
            tk[ia,jd]=tk[ia,js];tom[ia,jd]=tom[ia,js];nut[ia,jd]=nut[ia,js]
        end
        js=NJ+g+1-k;jd=NJ+g+k
        incoming=kinds[i]==0x01 && ux[ia,NJ+g+1]*Sjx[i,NJ+1]+uy[ia,NJ+g+1]*Sjy[i,NJ+1]<0.0f0
        tk[ia,jd]=incoming ? ki : tk[ia,js]
        tom[ia,jd]=incoming ? wi : tom[ia,js]
        nut[ia,jd]=incoming ? ki/wi : nut[ia,js]
    end
end

@kernel function _k_sst_species_bc_i!(tk,tom,nut,@Const(ux),@Const(uy),@Const(Six),@Const(Siy),
                                      ki::Float32,wi::Float32,NI::Int32,NJ::Int32,g::Int32)
    jj,k=@index(Global,NTuple)
    @inbounds if jj<=NJ+2g && k<=g
        j=clamp(jj-g,Int32(1),NJ)
        for side in Int32(1):Int32(2)
            face=side==Int32(1) ? Int32(1) : NI+Int32(1)
            source=side==Int32(1) ? g+Int32(1) : NI+g
            ghost=side==Int32(1) ? g : NI+g+Int32(1)
            dest=side==Int32(1) ? g+Int32(1)-k : NI+g+k
            direction=side==Int32(1) ? -1.0f0 : 1.0f0
            incoming=direction*(ux[ghost,jj]*Six[face,j]+uy[ghost,jj]*Siy[face,j])<0.0f0
            tk[dest,jj]=incoming ? ki : tk[source,jj]
            tom[dest,jj]=incoming ? wi : tom[source,jj]
            nut[dest,jj]=incoming ? ki/wi : nut[source,jj]
        end
    end
end

# The pre-rebuild row/column pair, kept as the reference for the parity test of the fused fill below.
function _sst_bc_legacy!(c)
    s=c.solver;ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g)
    _k_sst_species_bc_j!(s.backend)(s.tk,s.tom,s.nut,s.ux,s.uy,s.Sjx,s.Sjy,s.vol,s.outer_face_kind,
        s.k_inf,s.w_inf,s.nu_lam,c.wallmode,ni,nj,Int32(s.NIW),Int32(s.NIF),g;ndrange=(s.NI,s.g))
    _k_sst_species_bc_i!(s.backend)(s.tk,s.tom,s.nut,s.ux,s.uy,s.Six,s.Siy,
        s.k_inf,s.w_inf,ni,nj,g;ndrange=(s.NJ+2s.g,s.g))
end

# Fused SST ghost fill (rebuild, 22 Sep 2026). One launch replaces the row/column pair above. Thread (n,k),
# n in 1..NI+NJ+2g, k in 1..g: n <= NI is column i = n writing its bottom and top ghost rows of layer k; n > NI
# is row jj = n-NI writing its left and right ghost columns of layer k. The pair's only read-after-write was
# the column kernel reading the row kernel's output at the source column in a ghost row (the corners); the
# fused kernel recomputes that value in-thread from the same interior cells with the same rules, so no thread
# reads a cell another thread of the launch writes, on either backend. `mode` 0 writes tk and tom, 1 writes
# nut: the coefficient kernel reads no nut, so the fill before it need not touch nut and the fill after it need
# touch nothing else.
@inline function _sst_ghost_bottom(tk,tom,nut,Sjx,Sjy,vol,i,k,nu,wallmode,NI,NIW,NIF,g,wf=Int32(0))
    ia=i+g;js=g+k
    if i<=NIW || i>NIW+NIF
        ip=NI+1-i+g
        (tk[ip,js],tom[ip,js],nut[ip,js])
    elseif wallmode==Int32(2) && wf==Int32(1)
        # All-y+ wall function (2026-09-26): zero-gradient k and omega (no wall flux of k; the wall cell's
        # omega is pinned at commit), zero wall eddy viscosity as before.
        (tk[ia,js],tom[ia,js],0.0f0)
    elseif wallmode==Int32(2)
        S=sqrt(Sjx[i,1]^2+Sjy[i,1]^2);d1=0.5f0*vol[i,1]/S+1.0f-30
        (-tk[ia,js],2.0f0*_sst_wall_omega(nu,d1)-tom[ia,js],0.0f0)
    else
        (tk[ia,js],tom[ia,js],nut[ia,js])
    end
end
@inline function _sst_ghost_top(tk,tom,nut,ux,uy,Sjx,Sjy,kinds,i,k,ki,wi,NJ,g)
    ia=i+g;js=NJ+g+1-k
    incoming=kinds[i]==0x01 && ux[ia,NJ+g+1]*Sjx[i,NJ+1]+uy[ia,NJ+g+1]*Sjy[i,NJ+1]<0.0f0
    incoming ? (ki,wi,ki/wi) : (tk[ia,js],tom[ia,js],nut[ia,js])
end
@kernel function _k_sst_ghosts!(tk,tom,nut,@Const(ux),@Const(uy),@Const(Six),@Const(Siy),@Const(Sjx),@Const(Sjy),
                                @Const(vol),@Const(kinds),ki::Float32,wi::Float32,nu::Float32,wallmode::Int32,mode::Int32,
                                NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32,wf::Int32,ctx)
    n,k=@index(Global,NTuple)
    @inbounds if k<=g
        if n<=NI
            i=Int32(n);ia=i+g
            # Strip topology (plan 4.1-4.2): with a strip pair's context the interface rows (the C block's leg columns'
            # bottom rows, both rows of the strip) belong to the SST exchange; the fill writes every other row as today.
            wb=ctx===nothing || (ctx.block==Int32(0) && NIW<i<=NIW+NIF)
            wt=ctx===nothing || ctx.block==Int32(0)
            b=_sst_ghost_bottom(tk,tom,nut,Sjx,Sjy,vol,i,Int32(k),nu,wallmode,NI,NIW,NIF,g,wf);jd=g+Int32(1)-Int32(k)
            t=_sst_ghost_top(tk,tom,nut,ux,uy,Sjx,Sjy,kinds,i,Int32(k),ki,wi,NJ,g);ju=NJ+g+Int32(k)
            if mode==Int32(0)
                wb && (tk[ia,jd]=b[1];tom[ia,jd]=b[2]);wt && (tk[ia,ju]=t[1];tom[ia,ju]=t[2])
            else
                wb && (nut[ia,jd]=b[3]);wt && (nut[ia,ju]=t[3])
            end
        elseif n<=NI+NJ+2g
            jj=Int32(n)-NI;j=clamp(jj-g,Int32(1),NJ)
            for side in Int32(1):Int32(2)
                face=side==Int32(1) ? Int32(1) : NI+Int32(1)
                isrc=side==Int32(1) ? Int32(1) : NI
                source=isrc+g
                ghost=side==Int32(1) ? g : NI+g+Int32(1)
                dest=side==Int32(1) ? g+Int32(1)-Int32(k) : NI+g+Int32(k)
                direction=side==Int32(1) ? -1.0f0 : 1.0f0
                incoming=direction*(ux[ghost,jj]*Six[face,j]+uy[ghost,jj]*Siy[face,j])<0.0f0
                v=if ctx!==nothing && ctx.block==Int32(1) && (jj<=g || jj>NJ+g)
                    # the strip's corners (critique A 3): not _sst_ghost_bottom's cut branch, which would read the strip's
                    # own far column; the end column's interface-row value instead (finite; corners are read by nothing)
                    (tk[source,jj],tom[source,jj],nut[source,jj])
                elseif ctx!==nothing && ctx.iwall==Int32(1) && side==Int32(1)
                    # the strip's base wall on its i = 1 face (plan 4.1): the foil wall's ghost of _sst_ghost_bottom on an i
                    # face, layer k from cell k, d1 = 0.5 vol/|S_i1|: tk = -k, tom = 2 w_wall - w, nut = 0
                    ks=g+Int32(k)
                    S=sqrt(Six[1,j]^2+Siy[1,j]^2);d1=0.5f0*vol[1,j]/S+1.0f-30
                    (-tk[ks,jj],2.0f0*_sst_wall_omega(nu,d1)-tom[ks,jj],0.0f0)
                elseif incoming
                    (ki,wi,ki/wi)
                elseif jj<=g
                    _sst_ghost_bottom(tk,tom,nut,Sjx,Sjy,vol,isrc,g+Int32(1)-jj,nu,wallmode,NI,NIW,NIF,g,wf)
                elseif jj>NJ+g
                    _sst_ghost_top(tk,tom,nut,ux,uy,Sjx,Sjy,kinds,isrc,jj-NJ-g,ki,wi,NJ,g)
                else
                    (tk[source,jj],tom[source,jj],nut[source,jj])
                end
                if mode==Int32(0)
                    tk[dest,jj]=v[1];tom[dest,jj]=v[2]
                else
                    nut[dest,jj]=v[3]
                end
            end
        end
    end
end
function _sst_bc!(c;nut_only::Bool=false)
    s=c.solver;ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g)
    _k_sst_ghosts!(s.backend)(s.tk,s.tom,s.nut,s.ux,s.uy,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,s.outer_face_kind,
        s.k_inf,s.w_inf,s.nu_lam,c.wallmode,Int32(nut_only ? 1 : 0),ni,nj,Int32(s.NIW),Int32(s.NIF),g,_wall_function(s),
        _strip_ctx(c);ndrange=(s.NI+s.NJ+2s.g,s.g))
end

@inline function _sst_gradient(a,i,j,g,Six,Siy,Sjx,Sjy,vol)
    ia=i+g;ja=j+g;v=a[ia,ja]
    # Difference form of the existing Green-Gauss helper preserves constants even
    # when stored metric closure has roundoff.
    CMeshSolverKA._gg_grad(a[ia-1,ja]-v,a[ia+1,ja]-v,a[ia,ja-1]-v,a[ia,ja+1]-v,0.0f0,
        Six[i,j],Siy[i,j],Six[i+1,j],Siy[i+1,j],Sjx[i,j],Sjy[i,j],Sjx[i,j+1],Sjy[i,j+1],1.0f0/vol[i,j])
end
# The strip pair's twin (strip topology, 5 Oct 2026, plan section 4.4; critique A 4): the same difference form with the
# distance-weighted face value at the interface faces of a cell beside an interface (cmesh_solver_ka.jl,
# _strip_gg_weights): (h_nb phi_c + h_c phi_nb)/(h_c + h_nb) - phi_c = h_c (phi_nb - phi_c)/(h_c + h_nb). Returns the
# four gradients the coefficient kernel forms (u, v, k, omega), out of line; reached only with a strip pair's context.
@noinline function _strip_sst_gradients(ux,uy,tk,tom,Six,Siy,Sjx,Sjy,vol,i::Int32,j::Int32,NJ::Int32,g::Int32,ctx)
    @inbounds begin
        ws,wn,hS,nS,hN,nN=CMeshSolverKA._strip_gg_weights(Sjx,Sjy,vol,i,j,NJ,ctx)
        SxW=Six[i,j];SyW=Siy[i,j];SxE=Six[i+Int32(1),j];SyE=Siy[i+Int32(1),j]
        SxS=Sjx[i,j];SyS=Sjy[i,j];SxN=Sjx[i,j+Int32(1)];SyN=Sjy[i,j+Int32(1)];iv=1.0f0/vol[i,j]
        ia=i+g;ja=j+g
        v=ux[ia,ja]
        gu=CMeshSolverKA._strip_gg_one(ux[ia-1,ja]-v,ux[ia+1,ja]-v,ux[ia,ja-1]-v,ux[ia,ja+1]-v,0.0f0,
            ws,wn,hS,nS,hN,nN,SxW,SyW,SxE,SyE,SxS,SyS,SxN,SyN,iv)
        v=uy[ia,ja]
        gv=CMeshSolverKA._strip_gg_one(uy[ia-1,ja]-v,uy[ia+1,ja]-v,uy[ia,ja-1]-v,uy[ia,ja+1]-v,0.0f0,
            ws,wn,hS,nS,hN,nN,SxW,SyW,SxE,SyE,SxS,SyS,SxN,SyN,iv)
        v=tk[ia,ja]
        gk=CMeshSolverKA._strip_gg_one(tk[ia-1,ja]-v,tk[ia+1,ja]-v,tk[ia,ja-1]-v,tk[ia,ja+1]-v,0.0f0,
            ws,wn,hS,nS,hN,nN,SxW,SyW,SxE,SyE,SxS,SyS,SxN,SyN,iv)
        v=tom[ia,ja]
        gw=CMeshSolverKA._strip_gg_one(tom[ia-1,ja]-v,tom[ia+1,ja]-v,tom[ia,ja-1]-v,tom[ia,ja+1]-v,0.0f0,
            ws,wn,hS,nS,hN,nN,SxW,SyW,SxE,SyE,SxS,SyS,SxN,SyN,iv)
        (gu[1],gu[2],gv[1],gv[2],gk[1],gk[2],gw[1],gw[2])
    end
end

# Scalar SST-2003m coefficients. All inputs are already admitted positive/finite
# where required. The displayed conservative CD includes a density factor:
# cancelling rho changes its fixed nondimensional 1e-10 floor to 1e-10/rho.
# fp (13 Oct 2026): the factor of the k production's eddy viscosity, P = min(fp nu_t S^2, 10 beta* k omega). 1 (the default and
# every existing call) multiplies nu_t S^2 by 1.0f0, which is exact: the production, the sources and the clock are bitwise the
# previous ones. The production-consistent Reboud form passes the Reboud factor (P_k from the corrected nu_t, as OpenFOAM's
# kOmegaSST forms G = nut GbyNu0 when its nut is corrected); the omega production gamma min(S^2, ...) does not read nu_t.
@inline function _sst_coefficients(k,w,d,nu,rho,dudx,dudy,dvdx,dvdy,gkw,fp=1.0f0)
    Sxy=0.5f0*(dudy+dvdx)
    S2=2.0f0*(dudx*dudx+dvdy*dvdy+2.0f0*Sxy*Sxy);S=sqrt(S2)
    CD=max(2.0f0*0.856f0/w*gkw,1.0f-10/rho);sqk=sqrt(k)
    arg1=min(min(max(sqk/(0.09f0*w*d),500.0f0*nu/(d*d*w)),4.0f0*0.856f0*k/(CD*d*d)),4.0f0)
    arg2=min(max(2.0f0*sqk/(0.09f0*w*d),500.0f0*nu/(d*d*w)),4.0f0)
    F1=tanh(arg1^4);F2=tanh(arg2^2)
    D=max(0.31f0*w,S*F2);nut=0.31f0*k/D
    P=min(fp*(nut*S2),10.0f0*0.09f0*k*w)
    sigk=F1*0.85f0+(1.0f0-F1);sigw=F1*0.5f0+(1.0f0-F1)*0.856f0
    beta=F1*0.075f0+(1.0f0-F1)*0.0828f0;gam=F1*(5.0f0/9.0f0)+(1.0f0-F1)*0.44f0
    # P/nut cancels k analytically. Do not suppress omega production when nut
    # is small but positive by flooring its denominator.
    Pw=gam*min(S2,(10.0f0*0.09f0/0.31f0)*w*D);cross=2.0f0*(1.0f0-F1)*0.856f0/w*gkw
    sk=P-0.09f0*w*k;sw=Pw-beta*w*w+cross
    rate=max(P/k+0.09f0*w,(Pw+abs(cross))/w+beta*w)
    (nut,sigk,sigw,sk,sw,rate)
end

# Reboud density correction, DENSITY form (V1, 2026-09-20; kept verbatim, selected by nut_density=:reboud_v1 since
# 10 Oct 2026; Reboud, Stutz and Coutier-Delgosha 1998): the eddy viscosity of the mixture is formed with f(rho) = rho_v + (rho - rho_v)^n / (rho_l - rho_v)^(n-1) in
# place of rho, i.e. the published kinematic nu_t is multiplied by f(rho)/rho. The factor is 1 in liquid
# at or above saturation and in pure vapour, and falls steeply in the mixture (n = 10: about 0.022 at
# alpha 0.5 for a density ratio of 100, where rho_v/rho dominates). As on the legacy path it is a post-hoc rescale of the
# PUBLISHED nu_t, which the momentum stresses, the viscous clock and the SST diffusion consume; the
# SST production and its clock use the unscaled nu_t. reboud_n == 0 leaves nu_t untouched, bitwise.
@inline function _reboud_factor(r::Float32,rl::Float32,rv::Float32,n::Float32)
    (r>=rl || r<=rv) && return 1.0f0
    (rv+(r-rv)^n/(rl-rv)^(n-1.0f0))/r
end
# Reboud correction, COMPOSITION form (V2, 10 Oct 2026; the default from 10 to 13 Oct, opt-in as nut_density=:reboud_v2 since the
# review of 13 Oct; memory/project_docs/2026-10-10_REBOUD_DEFAULT_AND_SCHNERR_SAUER_DESIGN.md sections 1.3 and 6). Reboud's function of a volume
# fraction x at the saturation phase densities, rho_s(x) = rho_l (1-x) + rho_v x:
#     R(x) = [rho_v + (1-x)^n (rho_l - rho_v)] / [rho_v + (1-x)(rho_l - rho_v)]
# (Ducoin, Huang and Young 2012, Int. J. Rotating Machinery 2012:215678, eq. 11, with x the vapour volume fraction). V2 evaluates
# it at the stored reference composition x = a (t.alpha, the published Float32 cache of _alpha(q/r)), the vapour's volume fraction
# at SATURATION densities, which equals the cell's actual vapour volume fraction only at p = p_v: this solver's vapour is
# compressible (v_v = p_v/(rho_v p)), and the review of 13 Oct measured V2 departing from R(beta) of the represented void beta by
# more than V1 does in N60R's developed states. V3 below is the default. The function is exactly 1 at x <= 0 and x >= 1 by the
# early return, before any arithmetic: a cell with q = 0 has alpha = 0.0f0 exactly.
@inline function _reboud_ratio(a::Float32,rl::Float32,rv::Float32,n::Float32)
    (a<=0.0f0 || a>=1.0f0) && return 1.0f0
    w=1.0f0-a; d=rl-rv
    (rv+w^n*d)/(rv+w*d)
end
# Reboud correction, VOID-FRACTION form (V3, the review of 13 Oct 2026; the default since then; design note section 6): R(beta) of
# the cell's represented vapour volume fraction beta at its own pressure, the void fraction the literature's form takes (Ducoin
# et al. 2012 eq. 11; Goncalves 2011; Decaix and Goncalves 2013 eq. 36; with incompressible phases it is Coutier-Delgosha's
# f(rho)/rho of the mixture density). beta is formed as tools/cmesh_species_diagnostics.jl forms it, from the published state: the
# raw root under the guarded pressure cache p (raw = p - psm^2/(4p), the inversion of p = (raw + sqrt(raw^2 + psm^2))/2), the
# reference composition a, and each phase's volume carried from saturation to raw: the vapour's by p_v/raw (isothermal), the
# liquid's, its free gas included, by L = 1 - (raw - p_v)/(rho_l a2l) - lq (raw - p_v)/raw:
#     beta = a (p_v/raw) / [(1-a) L + a (p_v/raw)]      (= a at raw = p_v; the density r cancels)
# The free gas is counted with the liquid, as the readers' beta counts it. Exactly 1 at a <= 0 (no vapour: pure liquid at any
# pressure, tension included) and a >= 1, before any arithmetic; a cell whose raw root rounds to zero or below at the guard floor
# (near vacuum: beta -> 1) and any state whose beta is not in (0, 1) take 1, R's own limit. The mixture terms carry explicit
# fma, so the CPU and the GPU round them alike.
@inline function _reboud_void(a::Float32,p::Float32,rl::Float32,rv::Float32,pv::Float32,a2l::Float32,psm::Float32,lq::Float32,n::Float32)
    (a<=0.0f0 || a>=1.0f0) && return 1.0f0
    raw=p-psm*psm/(4.0f0*p)
    raw>0.0f0 || return 1.0f0
    Vr=pv/raw
    L=1.0f0-(raw-pv)/(rl*a2l)-lq*((raw-pv)/raw)
    vap=a*Vr
    b=vap/fma(1.0f0-a,L,vap)
    0.0f0<b<1.0f0 || return 1.0f0
    _reboud_ratio(b,rl,rv,n)
end
# The Reboud factor of a cell for the form rform (read only when reboud_n > 0): 3 the void-fraction form V3, 2 the composition
# form V2, any other value the density form V1 of 20 Sep. veos = (p_v, a2l, psm, lq) of the block's EOS (V3 only).
@inline function _reboud_select(rform::Int32,r::Float32,a::Float32,p::Float32,rl::Float32,rv::Float32,veos,n::Float32)
    rform==Int32(3) ? _reboud_void(a,p,rl,rv,veos[1],veos[2],veos[3],veos[4],n) :
    rform==Int32(2) ? _reboud_ratio(a,rl,rv,n) : _reboud_factor(r,rl,rv,n)
end

# rform (10 Oct 2026; V3 since 13 Oct): the form of the Reboud correction, read only when reboud_n > 0 (_reboud_select). rprod
# (13 Oct 2026): 1 forms the SST k production from the corrected nu_t (fp = the factor), 0 from the uncorrected one (fp = 1, the
# 20 Sep convention, bitwise). reboud_n == 0 leaves nut and the production untouched, bitwise, whatever rform and rprod say.
# pressure: the block's guarded pressure cache, which the projection publishes before every SST preparation (V3 only).
@kernel function _k_sst_species_coeff!(nut,sigk,sigw,sk,sw,rate,status,@Const(rho),@Const(alpha),@Const(pressure),@Const(tk),@Const(tom),
                                       @Const(ux),@Const(uy),@Const(wdist),@Const(gammat),
                                       @Const(Six),@Const(Siy),@Const(Sjx),@Const(Sjy),@Const(vol),
                                       nu::Float32,reboud_n::Float32,rform::Int32,rprod::Int32,rho_l::Float32,rho_v::Float32,
                                       veos::NTuple{4,Float32},NI::Int32,NJ::Int32,g::Int32,
                                       NIW::Int32,NIF::Int32,wf::Int32,ctx)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        ia=i+g;ja=j+g
        uxg,uyg=_sst_gradient(ux,i,j,g,Six,Siy,Sjx,Sjy,vol)
        vxg,vyg=_sst_gradient(uy,i,j,g,Six,Siy,Sjx,Sjy,vol)
        if wf==Int32(1) && j==Int32(1) && NIW<i<=NIW+NIF
            # All-y+ wall function (2026-09-26): the wall cell's strain is the wall law's, not the Green-Gauss
            # gradient through the wall ghost. S^2 = 2 S_ij S_ij is invariant, so a pure shear of the wall-law
            # magnitude carries it exactly; production, the eddy-viscosity limiter and F2 use it.
            Sw=sqrt(Sjx[i,1]*Sjx[i,1]+Sjy[i,1]*Sjy[i,1]);d1=0.5f0*vol[i,1]/Sw+1.0f-30
            nx=Sjx[i,1]/Sw;ny=Sjy[i,1]/Sw;uw=ux[ia,ja];vw=uy[ia,ja];un=uw*nx+vw*ny
            utx=uw-un*nx;uty=vw-un*ny
            _,_,dudy=CMeshSolverKA.wf_species_state(sqrt(utx*utx+uty*uty),d1,nu)
            uxg=0.0f0;uyg=dudy;vxg=0.0f0;vyg=0.0f0
        end
        kx,ky=_sst_gradient(tk,i,j,g,Six,Siy,Sjx,Sjy,vol)
        wx,wy=_sst_gradient(tom,i,j,g,Six,Siy,Sjx,Sjy,vol)
        if ctx!==nothing && CMeshSolverKA._strip_gg_cell(i,j,NJ,ctx)
            # Strip topology (plan section 4.4): beside an interface of a strip pair, the distance-weighted face value
            # in the strain, production and cross-diffusion gradients (the wall function is refused with a strip, so the
            # branch above never ran on these cells); ctx === nothing compiles this away (today's kernel)
            sg8=_strip_sst_gradients(ux,uy,tk,tom,Six,Siy,Sjx,Sjy,vol,Int32(i),Int32(j),NJ,g,ctx)
            uxg=sg8[1];uyg=sg8[2];vxg=sg8[3];vyg=sg8[4];kx=sg8[5];ky=sg8[6];wx=sg8[7];wy=sg8[8]
        end
        # the Reboud factor first (13 Oct 2026), from the published state only, so that the production-consistent form can hand it
        # to the k production; the saturation densities are the named arguments rho_l, rho_v (the kernel's local `rv` below is
        # rho*vol -- the first version of 20 Sep passed `rv` and the identity test caught the factor at 0.006 instead of 0.022)
        fac=reboud_n>0.0f0 ? _reboud_select(rform,rho[ia,ja],alpha[ia,ja],pressure[ia,ja],rho_l,rho_v,veos,reboud_n) : 1.0f0
        n,ak,aw,Sk,Sw,R=_sst_coefficients(tk[ia,ja],tom[ia,ja],max(wdist[i,j],1.0f-8),nu,
            rho[ia,ja],uxg,uyg,vxg,vyg,kx*wx+ky*wy,(reboud_n>0.0f0 && rprod==Int32(1)) ? fac : 1.0f0)
        rv=rho[ia,ja]*vol[i,j];Sk*=rv;Sw*=rv
        good=isfinite(n) && n>=0.0f0 && isfinite(ak) && isfinite(aw) &&
            isfinite(Sk) && isfinite(Sw) && isfinite(R) && R>=0.0f0
        nut[ia,ja]=reboud_n>0.0f0 ? n*fac : n
        sigk[i,j]=ak;sigw[i,j]=aw;sk[i,j]=Sk;sw[i,j]=Sw;rate[i,j]=R
        status[i,j]=good ? 0.0f0 : 1.0f0
    end
end

@kernel function _k_sst_recover!(tk,tom,@Const(rk),@Const(rw),@Const(rho),NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        # The conserved value is floored at commit; its published primitive may
        # round one ulp below the 1e-12 floor and is republished at the floor.
        k=rk[i,j]/rho[i+g,j+g]
        tk[i+g,j+g]=k<1.0f-12 ? 1.0f-12 : k
        tom[i+g,j+g]=rw[i,j]/rho[i+g,j+g]
    end
end

# (13 Oct 2026) The EOS constants of the void-fraction Reboud form, (p_v, a2l, psm, lq): those of the block's own outlet EOS tuple,
# with which its pressure cache is projected (s.outlet_eos = (p_v, a2l, rho_l, rho_v, psm, lq, afloor)).
_reboud_eos(s)=(s.outlet_eos[1],s.outlet_eos[2],s.outlet_eos[5],s.outlet_eos[6])
function _sst_prepare!(c;slot=nothing)
    a=c.turbulence;s=c.solver
    _k_sst_recover!(s.backend)(s.tk,s.tom,a.rk,a.rw,s.rho,
        Int32(s.NI),Int32(s.NJ),Int32(s.g);ndrange=(s.NI,s.NJ))
    _sst_bc!(c)                                   # tk/tom ghosts; the coefficient kernel reads no nut
    status=slot===nothing ? c.status : slot        # deferred: the flag slot's bank slice
    slot===nothing && fill!(c.status,0.0f0)
    _k_sst_species_coeff!(s.backend)(s.nut,a.sigk,a.sigw,a.sk,a.sw,a.rate,status,s.rho,c.tem.alpha,c.pressure,s.tk,s.tom,
        s.ux,s.uy,s.wdist,s.gammat,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,s.nu_lam,c.reboud_n,c.reboud_form,c.reboud_prod,
        c.tem.rho_l_sat,c.tem.rho_v_sat,_reboud_eos(s),
        Int32(s.NI),Int32(s.NJ),Int32(s.g),Int32(s.NIW),Int32(s.NIF),_wall_function(s),_strip_ctx(c);ndrange=(s.NI,s.NJ))
    slot===nothing && (maximum(c.status)==0 || throw(ArgumentError("nonfinite conservative SST coefficient or source")))
    _sst_bc!(c;nut_only=true)                     # only nut changed
end

@inline function _sst_i_diff(rho,nut,sigk,sigw,xc,yc,Six,Siy,nu,i,j,NI,g)
    if 1<i<=NI
        ia=i+g;ja=j+g
        dx=xc[i,j]-xc[i-1,j];dy=yc[i,j]-yc[i-1,j]
        dn=sqrt(dx*dx+dy*dy)+1.0f-30;S=sqrt(Six[i,j]^2+Siy[i,j]^2)
        factor=0.5f0*(rho[ia-1,ja]+rho[ia,ja])*S/dn
        ck=factor*(nu+0.5f0*(sigk[i-1,j]*nut[ia-1,ja]+sigk[i,j]*nut[ia,ja]))
        cw=factor*(nu+0.5f0*(sigw[i-1,j]*nut[ia-1,ja]+sigw[i,j]*nut[ia,ja]))
        return (ck,cw)
    end
    (0.0f0,0.0f0) # no diffusive flux at the open i ends
end

@inline function _sst_j_diff(rho,nut,sigk,sigw,xc,yc,Sjx,Sjy,vol,nu,wallmode,i,j,NI,NJ,NIW,NIF,g,wf=Int32(0))
    ia=i+g;ja=j+g;S=sqrt(Sjx[i,j]^2+Sjy[i,j]^2)
    if 1<j<=NJ
        dx=xc[i,j]-xc[i,j-1];dy=yc[i,j]-yc[i,j-1]
        dn=sqrt(dx*dx+dy*dy)+1.0f-30; factor=0.5f0*(rho[ia,ja-1]+rho[ia,ja])*S/dn
        ck=factor*(nu+0.5f0*(sigk[i,j-1]*nut[ia,ja-1]+sigk[i,j]*nut[ia,ja]))
        cw=factor*(nu+0.5f0*(sigw[i,j-1]*nut[ia,ja-1]+sigw[i,j]*nut[ia,ja]))
        return (ck,cw)
    elseif j==1 && (i<=NIW || i>NIW+NIF)
        ip=NI+1-i;dx=xc[i,1]-xc[ip,1];dy=yc[i,1]-yc[ip,1]
        dn=sqrt(dx*dx+dy*dy)+1.0f-30
        factor=0.5f0*(rho[ia,g]+rho[ia,g+1])*S/dn
        ck=factor*(nu+0.5f0*(sigk[ip,1]*nut[ia,g]+sigk[i,1]*nut[ia,g+1]))
        cw=factor*(nu+0.5f0*(sigw[ip,1]*nut[ia,g]+sigw[i,1]*nut[ia,g+1]))
        return (ck,cw)
    elseif j==1 && wallmode==Int32(2) && wf==Int32(1)
        return (0.0f0,0.0f0) # all-y+ wall function (2026-09-26): no wall diffusion of k; the wall cell's omega is pinned
    elseif j==1 && wallmode==Int32(2)
        dn=0.5f0*vol[i,1]/S+1.0f-30;v=rho[ia,g+1]*nu*S/dn
        return (v,v) # wall eddy viscosity is zero
    end
    (0.0f0,0.0f0) # transparent/slip foil and open/slip outer boundary
end

# The two-point SST conductances with a strip pair's block context (strip topology, 5 Oct 2026, plan section 4.1), used
# only when the context is bound; the two helpers above are the default path's, untouched. i faces: the strip's i = 1 face
# is the base wall (the j = 1 wall conductance above on an i face, d1 = 0.5 vol/|S_i1|); every other i face as above.
# j faces: an interface face (the C block's leg columns and the strip at j = 1, the strip at j = NJ+1) takes the other
# block's cell through the static centroid table ctx.c_nb and its sigma_k, sigma_w through ctx.sig_nb, the exchange's
# copy of the neighbour's coefficients; its ghost rho and nut are the exchange's copies of the neighbour's. Every operand
# enters in the order the neighbour block's own representation of the face uses (lower cell first), so the upper
# interface's two conductances are the same Float32 operations on the same values. Every other j face as above.
@inline function _sst_i_diff_strip(rho,nut,sigk,sigw,xc,yc,Six,Siy,nu,i,j,NI,g,ctx)
    if ctx.iwall==Int32(1) && i==1
        ia=g+Int32(1);ja=j+g;S=sqrt(Six[1,j]^2+Siy[1,j]^2)
        dn=0.5f0*ctx.vol[1,j]/S+1.0f-30;v=rho[ia,ja]*nu*S/dn
        return (v,v) # the base wall: wall eddy viscosity is zero
    end
    _sst_i_diff(rho,nut,sigk,sigw,xc,yc,Six,Siy,nu,i,j,NI,g)
end
@inline function _sst_j_diff_strip(rho,nut,sigk,sigw,xc,yc,Sjx,Sjy,vol,nu,wallmode,i,j,NI,NJ,NIW,NIF,g,wf,ctx)
    if j==1 && (i<=NIW || i>NIW+NIF)
        ia=i+g;S=sqrt(Sjx[i,1]^2+Sjy[i,1]^2)
        st,fc=CMeshSolverKA._strip_station(i,NIW,NIF,ctx.block)
        dx=xc[i,1]-ctx.c_nb[st,1,fc];dy=yc[i,1]-ctx.c_nb[st,2,fc]
        dn=sqrt(dx*dx+dy*dy)+1.0f-30
        factor=0.5f0*(rho[ia,g]+rho[ia,g+1])*S/dn
        ck=factor*(nu+0.5f0*(ctx.sig_nb[st,1,fc]*nut[ia,g]+sigk[i,1]*nut[ia,g+1]))
        cw=factor*(nu+0.5f0*(ctx.sig_nb[st,2,fc]*nut[ia,g]+sigw[i,1]*nut[ia,g+1]))
        return (ck,cw)
    elseif ctx.block==Int32(1) && j==NJ+1
        ia=i+g;S=sqrt(Sjx[i,j]^2+Sjy[i,j]^2)
        dx=ctx.c_nb[i,1,2]-xc[i,NJ];dy=ctx.c_nb[i,2,2]-yc[i,NJ]
        dn=sqrt(dx*dx+dy*dy)+1.0f-30
        factor=0.5f0*(rho[ia,NJ+g]+rho[ia,NJ+g+1])*S/dn
        ck=factor*(nu+0.5f0*(sigk[i,NJ]*nut[ia,NJ+g]+ctx.sig_nb[i,1,2]*nut[ia,NJ+g+1]))
        cw=factor*(nu+0.5f0*(sigw[i,NJ]*nut[ia,NJ+g]+ctx.sig_nb[i,2,2]*nut[ia,NJ+g+1]))
        return (ck,cw)
    end
    _sst_j_diff(rho,nut,sigk,sigw,xc,yc,Sjx,Sjy,vol,nu,wallmode,i,j,NI,NJ,NIW,NIF,g,wf)
end

# Reconstructed SST advection shares the admitted flow geometry, not its phase
# limiter. tk/tom were recovered from authoritative rk/rw and rho by _sst_prepare!.
# The raw LS gradients are saved before limiting for interior diffusion. Closure
# gradients, production/destruction and the RK trial are unchanged.
@inline function _sst_reconstruct_component!(faces,gradients,primitive,geometry,i,j,g,k,NI,NJ,floor)
    center=primitive[i+g,j+g];lo=center;hi=center;gx=0f0;gy=0f0
    good=isfinite(center) && center>=floor
    if geometry.active[i,j]!=0x00
        for f in 1:4
            ii,jj=_neighbor(i,j,f)
            if 1<=ii<=NI && 1<=jj<=NJ
                v=primitive[ii+g,jj+g]
                good=good && isfinite(v) && v>=floor
                lo=min(lo,v);hi=max(hi,v)
                gx+=geometry.weights[i,j,1,f]*(v-center)
                gy+=geometry.weights[i,j,2,f]*(v-center)
            end
        end
    end
    theta=1f0
    good=good && isfinite(gx) && isfinite(gy)
    gradients[i,j,1,k]=good ? gx : NaN32
    gradients[i,j,2,k]=good ? gy : NaN32
    for f in 1:4
        d=gx*geometry.offsets[i,j,1,f]+gy*geometry.offsets[i,j,2,f]
        good=good && isfinite(d)
        if d>0f0
            theta=min(theta,(hi-center)/d)
        elseif d<0f0
            theta=min(theta,(lo-center)/d)
        end
    end
    theta=clamp(theta,0f0,1f0)
    if !good || !isfinite(theta)
        for f in 1:4;faces[i,j,k,f]=NaN32;end
        return false
    end
    # Check represented face values, not just their exact-arithmetic bounds.
    # Backoff is scalar-specific; the final admitted constant value is not a floor.
    for attempt in 0:8
        attempt==8 && (theta=0f0)
        good=true
        for f in 1:4
            d=gx*geometry.offsets[i,j,1,f]+gy*geometry.offsets[i,j,2,f]
            v=center+theta*d
            good=good && isfinite(v) && lo<=v<=hi && v>=floor
            faces[i,j,k,f]=v
        end
        good && return true
        theta*=.5f0
    end
    false
end

@kernel function _k_sst_reconstruct!(faces,gradients,status,@Const(tk),@Const(tom),geometry,
                                     NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        a=_sst_reconstruct_component!(faces,gradients,tk,geometry,i,j,g,1,NI,NJ,1f-12)
        b=_sst_reconstruct_component!(faces,gradients,tom,geometry,i,j,g,2,NI,NJ,1f-6)
        status[i,j]=a && b ? 0f0 : 1f0
    end
end

function _sst_reconstruct_faces!(c;slot=nothing)
    r=c.reconstruction;s=c.solver
    r===nothing && return nothing
    faces=r.sst_states
    faces===nothing && throw(ArgumentError("reconstructed SST face workspace missing"))
    all(typeof(x)==typeof(faces) for x in (r.sst_diffusion,r.sst_gradients,r.sst_diffusion_terms)) &&
        size(r.sst_diffusion)==(s.NI,s.NJ,3,2) && size(r.sst_gradients)==(s.NI,s.NJ,2,2) &&
        size(r.sst_diffusion_terms)==(s.NI+1,s.NJ+1,4,2) ||
        throw(ArgumentError("reconstructed SST diffusion workspace shape/backend mismatch"))
    status=slot===nothing ? c.status : slot        # deferred: the flag slot's bank slice
    _k_sst_reconstruct!(s.backend)(faces,r.sst_gradients,status,s.tk,s.tom,r.geometry,
        Int32(s.NI),Int32(s.NJ),Int32(s.g);ndrange=(s.NI,s.NJ))
    slot===nothing && (maximum(c.status)==0f0 || throw(ArgumentError("SST reconstruction rejected scalar bounds or nonfinite slope")))
    faces
end

# Outflow through a reconstructed interior face removes the reconstructed
# primitive value. At physical/wake faces the existing cell-donor rule remains.
@inline function _sst_own_face_value(R,rho,faces,i,j,f,k,NI,NJ,g)
    inside=(f==1 && i>1) || (f==2 && i<NI) || (f==3 && j>1) || (f==4 && j<NJ)
    inside ? faces[i,j,k,f] : R[i,j]/rho[i+g,j+g]
end

@inline function _sst_donor(R,primitive,rho,i,j,NI,NJ,g)
    g<i<=NI+g && g<j<=NJ+g ? R[i-g,j-g]/rho[i,j] : primitive[i,j]
end

@inline function _sst_diffusion_terms(kappa,lambda,kx,ky,gxL,gyL,gxR,gyR)
    C=kappa*lambda
    delta=-kappa*(0.5f0*(gxL+gxR)*kx+0.5f0*(gyL+gyR)*ky)
    (C,delta)
end
@inline _sst_correction_outflow(e,w,n,s)=max(e,0f0)+max(-w,0f0)+max(n,0f0)+max(-s,0f0)
# Limited non-orthogonal correction (opt-in, 2 Oct 2026; climit >= 0): the face correction is bounded by climit times the
# two-point flux magnitude, so the face total keeps the two-point sign and the clock bounds its outflow by (1+climit)
# times the two-point coefficient (no separate correction term). climit < 0: the full correction, unchanged.
@inline _sst_limit_correction(d,c,dphi,climit)=(b=climit*c*abs(dphi);clamp(d,-b,b))
# The faces that carry a non-orthogonal correction (full or limited): exactly those on which the two flux kernels below
# take the interior-diffusion branch, i faces 1 < i <= NI and j faces 1 < j <= NJ (with the reconstruction's diffusion
# geometry present). Every other face -- the open i ends, the j = 1 row (wake cut and foil wall alike), the j = NJ+1 outer
# boundary -- takes the plain two-point flux of _sst_i_diff/_sst_j_diff, whose correction is identically zero. The flux
# kernels and the limited clock read these same two predicates, so the clock's face set cannot drift from the fluxes'
# (CMeshSpeciesSSTCorrectionLimitV2, 3 Oct 2026; memory/project_docs/_records/2026-10-02_blunt_te/sst_clock/FIX.md).
@inline _sst_corrected_i(i,NI)=1<i<=NI
@inline _sst_corrected_j(j,NJ)=1<j<=NJ

# The same actual C/delta are stored by the face kernel for its two adjacent
# cells' cap. No cancellation against the advective or two-point flux is used.
@inline function _sst_interior_diffusion(geometry,gradients,rho,nut,sigk,sigw,nu,il,jl,ir,jr,direction,g)
    l=il+g;m=jl+g;r=ir+g;n=jr+g
    rf=0.5f0*(rho[l,m]+rho[r,n])
    kk=rf*(nu+0.5f0*(sigk[il,jl]*nut[l,m]+sigk[ir,jr]*nut[r,n]))
    kw=rf*(nu+0.5f0*(sigw[il,jl]*nut[l,m]+sigw[ir,jr]*nut[r,n]))
    lambda=geometry[il,jl,1,direction];kx=geometry[il,jl,2,direction];ky=geometry[il,jl,3,direction]
    ck,dk=_sst_diffusion_terms(kk,lambda,kx,ky,gradients[il,jl,1,1],gradients[il,jl,2,1],
                             gradients[ir,jr,1,1],gradients[ir,jr,2,1])
    cw,dw=_sst_diffusion_terms(kw,lambda,kx,ky,gradients[il,jl,1,2],gradients[il,jl,2,2],
                             gradients[ir,jr,1,2],gradients[ir,jr,2,2])
    (ck,cw,dk,dw)
end

@kernel function _k_sst_species_flux_i!(Fk,Fw,@Const(Fmass),@Const(rk),@Const(rw),@Const(rho),
                                        @Const(tk),@Const(tom),@Const(nut),@Const(sigk),@Const(sigw),
                                        @Const(xc),@Const(yc),@Const(Six),@Const(Siy),
                                        nu::Float32,ki::Float32,wi::Float32,faces,diffusion,gradients,terms,
                                        NI::Int32,NJ::Int32,g::Int32,climit::Float32,ctx)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI+1 && j<=NJ
        ia=i+g;ja=j+g;f=Fmass[i,j];id=f>=0.0f0 ? ia-1 : ia
        # strip topology (plan 4.1): the strip's i = 1 face is the base wall, with the wall values k = 0 and omega_wall
        # of the foil wall's j = 1 face instead of the inflow test (its carrier mass flux is exactly zero)
        wall=ctx!==nothing && ctx.iwall==Int32(1) && i==1
        incoming=!wall && ((i==1 && f>0.0f0) || (i==NI+1 && f<0.0f0))
        if faces!==nothing && 1<i<=NI
            cell=f>=0f0 ? i-1 : i;side=f>=0f0 ? 2 : 1
            kd=faces[cell,j,1,side];wd=faces[cell,j,2,side]
        else
            kd=incoming ? ki : _sst_donor(rk,tk,rho,id,ja,NI,NJ,g)
            wd=incoming ? wi : _sst_donor(rw,tom,rho,id,ja,NI,NJ,g)
        end
        if diffusion!==nothing && _sst_corrected_i(i,NI)
            ck,cw,dk,dw=_sst_interior_diffusion(diffusion,gradients,rho,nut,sigk,sigw,nu,i-1,j,i,j,1,g)
            if climit>=0f0
                dk=_sst_limit_correction(dk,ck,tk[ia,ja]-tk[ia-1,ja],climit)
                dw=_sst_limit_correction(dw,cw,tom[ia,ja]-tom[ia-1,ja],climit)
            end
            Fk[i,j]=f*kd-ck*(tk[ia,ja]-tk[ia-1,ja])+dk
            Fw[i,j]=f*wd-cw*(tom[ia,ja]-tom[ia-1,ja])+dw
        else
            ck,cw=ctx===nothing ? _sst_i_diff(rho,nut,sigk,sigw,xc,yc,Six,Siy,nu,i,j,NI,g) :
                                  _sst_i_diff_strip(rho,nut,sigk,sigw,xc,yc,Six,Siy,nu,i,j,NI,g,ctx)
            kl=tk[ia-1,ja];wl=tom[ia-1,ja]
            if ctx!==nothing && wall
                S=sqrt(Six[1,j]^2+Siy[1,j]^2);dn=0.5f0*ctx.vol[1,j]/S+1.0f-30
                kl=0.0f0;wl=_sst_wall_omega(nu,dn)
            end
            Fk[i,j]=f*kd-ck*(tk[ia,ja]-kl)
            Fw[i,j]=f*wd-cw*(tom[ia,ja]-wl)
            dk=0f0;dw=0f0
        end
        if terms!==nothing
            terms[i,j,1,1]=ck;terms[i,j,2,1]=cw;terms[i,j,3,1]=dk;terms[i,j,4,1]=dw
        end
    end
end

@kernel function _k_sst_species_flux_j!(Fk,Fw,@Const(Fmass),@Const(rk),@Const(rw),@Const(rho),
                                        @Const(tk),@Const(tom),@Const(nut),@Const(sigk),@Const(sigw),
                                        @Const(xc),@Const(yc),@Const(Sjx),@Const(Sjy),@Const(vol),@Const(kinds),
                                        nu::Float32,ki::Float32,wi::Float32,wallmode::Int32,faces,diffusion,gradients,terms,
                                        NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32,wf::Int32,climit::Float32,ctx)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ+1
        ia=i+g;ja=j+g;f=Fmass[i,j];jd=f>=0.0f0 ? ja-1 : ja
        incoming=j==NJ+1 && kinds[i]==0x01 && f<0.0f0
        if faces!==nothing && 1<j<=NJ
            cell=f>=0f0 ? j-1 : j;side=f>=0f0 ? 4 : 3
            kd=faces[i,cell,1,side];wd=faces[i,cell,2,side]
        else
            kd=incoming ? ki : _sst_donor(rk,tk,rho,ia,jd,NI,NJ,g)
            wd=incoming ? wi : _sst_donor(rw,tom,rho,ia,jd,NI,NJ,g)
        end
        kl=tk[ia,ja-1];wl=tom[ia,ja-1]
        if j==1 && NIW<i<=NIW+NIF && wallmode==Int32(2)
            S=sqrt(Sjx[i,1]^2+Sjy[i,1]^2);dn=0.5f0*vol[i,1]/S+1.0f-30
            kl=0.0f0;wl=_sst_wall_omega(nu,dn)
        end
        if diffusion!==nothing && _sst_corrected_j(j,NJ)
            ck,cw,dk,dw=_sst_interior_diffusion(diffusion,gradients,rho,nut,sigk,sigw,nu,i,j-1,i,j,2,g)
            if climit>=0f0
                dk=_sst_limit_correction(dk,ck,tk[ia,ja]-kl,climit)
                dw=_sst_limit_correction(dw,cw,tom[ia,ja]-wl,climit)
            end
            Fk[i,j]=f*kd-ck*(tk[ia,ja]-kl)+dk
            Fw[i,j]=f*wd-cw*(tom[ia,ja]-wl)+dw
        else
            # ctx: a strip pair's block context (the interface conductances, plan 4.1), or nothing (today's)
            ck,cw=ctx===nothing ? _sst_j_diff(rho,nut,sigk,sigw,xc,yc,Sjx,Sjy,vol,nu,wallmode,i,j,NI,NJ,NIW,NIF,g,wf) :
                                  _sst_j_diff_strip(rho,nut,sigk,sigw,xc,yc,Sjx,Sjy,vol,nu,wallmode,i,j,NI,NJ,NIW,NIF,g,wf,ctx)
            Fk[i,j]=f*kd-ck*(tk[ia,ja]-kl)
            Fw[i,j]=f*wd-cw*(tom[ia,ja]-wl)
            dk=0f0;dw=0f0
        end
        if terms!==nothing
            terms[i,j,1,2]=ck;terms[i,j,2,2]=cw;terms[i,j,3,2]=dk;terms[i,j,4,2]=dw
        end
    end
end

@kernel function _k_sst_species_residual!(dck,dcw,cap,@Const(Fik),@Const(Fiw),@Const(Fjk),@Const(Fjw),
                                          @Const(Fmi),@Const(Fmj),@Const(sk),@Const(sw),@Const(rate),@Const(rk),@Const(rw),faces,terms,
                                          @Const(rho),@Const(nut),@Const(sigk),@Const(sigw),
                                          @Const(xc),@Const(yc),@Const(Six),@Const(Siy),@Const(Sjx),@Const(Sjy),@Const(vol),
                                          nu::Float32,wallmode::Int32,NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32,wf::Int32,
                                          climit::Float32,ctx)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        dck[i,j]=Fik[i+1,j]-Fik[i,j]+Fjk[i,j+1]-Fjk[i,j]-sk[i,j]
        dcw[i,j]=Fiw[i+1,j]-Fiw[i,j]+Fjw[i,j+1]-Fjw[i,j]-sw[i,j]
        if terms===nothing && ctx===nothing
            ek,ew=_sst_i_diff(rho,nut,sigk,sigw,xc,yc,Six,Siy,nu,i+1,j,NI,g)
            wk,ww=_sst_i_diff(rho,nut,sigk,sigw,xc,yc,Six,Siy,nu,i,j,NI,g)
            nk,nw=_sst_j_diff(rho,nut,sigk,sigw,xc,yc,Sjx,Sjy,vol,nu,wallmode,i,j+1,NI,NJ,NIW,NIF,g,wf)
            skc,swc=_sst_j_diff(rho,nut,sigk,sigw,xc,yc,Sjx,Sjy,vol,nu,wallmode,i,j,NI,NJ,NIW,NIF,g,wf)
        elseif terms===nothing
            # a strip pair's block (plan 4.1): the same face conductances as its flux kernels, through the strip twins
            ek,ew=_sst_i_diff_strip(rho,nut,sigk,sigw,xc,yc,Six,Siy,nu,i+1,j,NI,g,ctx)
            wk,ww=_sst_i_diff_strip(rho,nut,sigk,sigw,xc,yc,Six,Siy,nu,i,j,NI,g,ctx)
            nk,nw=_sst_j_diff_strip(rho,nut,sigk,sigw,xc,yc,Sjx,Sjy,vol,nu,wallmode,i,j+1,NI,NJ,NIW,NIF,g,wf,ctx)
            skc,swc=_sst_j_diff_strip(rho,nut,sigk,sigw,xc,yc,Sjx,Sjy,vol,nu,wallmode,i,j,NI,NJ,NIW,NIF,g,wf,ctx)
        else
            ek=terms[i+1,j,1,1];ew=terms[i+1,j,2,1];wk=terms[i,j,1,1];ww=terms[i,j,2,1]
            nk=terms[i,j+1,1,2];nw=terms[i,j+1,2,2];skc=terms[i,j,1,2];swc=terms[i,j,2,2]
        end
        out=max(Fmi[i+1,j],0.0f0)+max(-Fmi[i,j],0.0f0)+max(Fmj[i,j+1],0.0f0)+max(-Fmj[i,j],0.0f0)
        mass=rho[i+g,j+g]*vol[i,j]
        if faces===nothing
            R=(out+max(ek+wk+nk+skc,ew+ww+nw+swc))/mass+rate[i,j]
            good=isfinite(mass) && mass>0.0f0 && isfinite(R) && R>=0.0f0 && isfinite(dck[i,j]) && isfinite(dcw[i,j])
        else
            # Use the same carrier flux and scalar donor values as advection;
            # Source closures are unchanged; the actual positive correction
            # removal is counted separately from advection and two-point diffusion.
            fe=max(Fmi[i+1,j],0f0);fw=max(-Fmi[i,j],0f0)
            fn=max(Fmj[i,j+1],0f0);fs=max(-Fmj[i,j],0f0)
            ok=fe*_sst_own_face_value(rk,rho,faces,i,j,2,1,NI,NJ,g)+
               fw*_sst_own_face_value(rk,rho,faces,i,j,1,1,NI,NJ,g)+
               fn*_sst_own_face_value(rk,rho,faces,i,j,4,1,NI,NJ,g)+
               fs*_sst_own_face_value(rk,rho,faces,i,j,3,1,NI,NJ,g)
            ow=fe*_sst_own_face_value(rw,rho,faces,i,j,2,2,NI,NJ,g)+
               fw*_sst_own_face_value(rw,rho,faces,i,j,1,2,NI,NJ,g)+
               fn*_sst_own_face_value(rw,rho,faces,i,j,4,2,NI,NJ,g)+
               fs*_sst_own_face_value(rw,rho,faces,i,j,3,2,NI,NJ,g)
            # Divide before forming a tiny scalar cell inventory: rk*vol may
            # underflow even with positive admitted rk and carrier mass.
            K=rk[i,j];W=rw[i,j];V=vol[i,j]
            # k floor and the clock (2026-09-12): a cell whose k sits on its 1e-12
            # admission floor cannot fall below it (the commit restores the floor
            # with accounting), so the removal of k from that cell bounds nothing
            # and is excluded from its clock; dividing that removal by the floor
            # inventory locked the global step at 6e-8 on the matched foil.
            # Diffusion, the source rate and the omega clock still bound dt.
            atfloor=K<=1.0f-12*rho[i+g,j+g]*(1.0f0+4.0f0*eps(Float32))
            Rk=(atfloor ? 0.0f0 : (ok/K)/V)+(ek+wk+nk+skc)/mass+rate[i,j]
            Rw=(ow/W)/V+(ew+ww+nw+swc)/mass+rate[i,j]
            if terms!==nothing && climit>=0f0
                # limited correction: each face total keeps the two-point sign, so a corrected face removes at most
                # (1+climit) times its two-point coefficient times the cell value and an uncorrected face at most 1 times
                # it (the two-point term above carries the 1). The allowance climit x c_f is therefore summed over the
                # corrected faces only (V2, 3 Oct 2026). V1 summed all four faces and so charged the j = 1 wake-cut and
                # wall faces for a correction they never carry: 22% of BB8's binding rate at 0.9 CTU (BB8_STORM.md, C1).
                # Same association as V1's sum, so a cell whose four faces are all corrected is bitwise unchanged.
                ak=(_sst_corrected_i(i+1,NI) ? ek : 0f0)+(_sst_corrected_i(i,NI) ? wk : 0f0)+
                   (_sst_corrected_j(j+1,NJ) ? nk : 0f0)+(_sst_corrected_j(j,NJ) ? skc : 0f0)
                aw=(_sst_corrected_i(i+1,NI) ? ew : 0f0)+(_sst_corrected_i(i,NI) ? ww : 0f0)+
                   (_sst_corrected_j(j+1,NJ) ? nw : 0f0)+(_sst_corrected_j(j,NJ) ? swc : 0f0)
                Rk+=atfloor ? 0.0f0 : climit*ak/mass;Rw+=climit*aw/mass
            elseif terms!==nothing
                dk=_sst_correction_outflow(terms[i+1,j,3,1],terms[i,j,3,1],terms[i,j+1,3,2],terms[i,j,3,2])
                dw=_sst_correction_outflow(terms[i+1,j,4,1],terms[i,j,4,1],terms[i,j+1,4,2],terms[i,j,4,2])
                Rk+=(atfloor ? 0.0f0 : (dk/K)/V);Rw+=(dw/W)/V
            end
            R=max(Rk,Rw)
            good=isfinite(mass) && mass>0f0 && isfinite(K) && K>0f0 && isfinite(W) && W>0f0 &&
                isfinite(V) && V>0f0 &&
                isfinite(ok) && ok>=0f0 && isfinite(ow) && ow>=0f0 && isfinite(R) && R>=0f0 &&
                isfinite(dck[i,j]) && isfinite(dcw[i,j])
        end
        cap[i,j]=good ? (R>0.0f0 ? 0.25f0/R : Inf32) : 0.0f0
    end
end

function _sst_residual!(c;slot=nothing,faces_slot=nothing)
    a=c.turbulence;s=c.solver;ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g)
    faces=_sst_reconstruct_faces!(c;slot=faces_slot)
    r=c.reconstruction
    diffusion=r===nothing ? nothing : r.sst_diffusion
    gradients=r===nothing ? nothing : r.sst_gradients
    terms=r===nothing ? nothing : r.sst_diffusion_terms
    ctx=_strip_ctx(c)                               # nothing without a strip pair (plan 4.1)
    _k_sst_species_flux_i!(s.backend)(a.Fik,a.Fiw,s.Fi_r,a.rk,a.rw,s.rho,s.tk,s.tom,s.nut,a.sigk,a.sigw,
        s.xc,s.yc,s.Six,s.Siy,s.nu_lam,s.k_inf,s.w_inf,faces,diffusion,gradients,terms,ni,nj,g,c.sst_climit,ctx;ndrange=(s.NI+1,s.NJ))
    _k_sst_species_flux_j!(s.backend)(a.Fjk,a.Fjw,s.Fj_r,a.rk,a.rw,s.rho,s.tk,s.tom,s.nut,a.sigk,a.sigw,
        s.xc,s.yc,s.Sjx,s.Sjy,s.vol,s.outer_face_kind,s.nu_lam,s.k_inf,s.w_inf,c.wallmode,faces,diffusion,gradients,terms,
        ni,nj,Int32(s.NIW),Int32(s.NIF),g,_wall_function(s),c.sst_climit,ctx;ndrange=(s.NI,s.NJ+1))
    target=slot===nothing ? c.source_dt : slot     # deferred: the clock slot's bank slice
    _k_sst_species_residual!(s.backend)(a.dck,a.dcw,target,a.Fik,a.Fiw,a.Fjk,a.Fjw,s.Fi_r,s.Fj_r,
        a.sk,a.sw,a.rate,a.rk,a.rw,faces,terms,s.rho,s.nut,a.sigk,a.sigw,s.xc,s.yc,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,s.nu_lam,c.wallmode,
        ni,nj,Int32(s.NIW),Int32(s.NIF),g,_wall_function(s),c.sst_climit,ctx;ndrange=(s.NI,s.NJ))
    slot===nothing || return nothing
    cap=minimum(c.source_dt)
    cap>0.0f0 || throw(ArgumentError("invalid or unresolved conservative SST transport/source clock"))
    cap
end

@kernel function _k_sst_species_trial!(ktrial,wtrial,status,@Const(rk),@Const(rw),@Const(k0),@Const(w0),
                                       @Const(dck),@Const(dcw),@Const(carrier_trial),@Const(dtf),@Const(vol),@Const(rho),
                                       weight::Float32,NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        ia=i+g;ja=j+g;fac=dtf[i,j]/vol[i,j]
        # Same rounded density that the admitted carrier commit publishes to SST.
        rn=Float32(carrier_trial[i,j,1])
        K=k0[i,j]+weight*((rk[i,j]-k0[i,j])-fac*dck[i,j])
        W=w0[i,j]+weight*((rw[i,j]-w0[i,j])-fac*dcw[i,j])
        k=K/rn;w=W/rn
        # k may fall below its 1e-12 admission floor within a stage: the commit
        # restores the floor with explicit accounting (a decaying near-wall k
        # otherwise locks the global clock at every step size). Negative K means
        # the explicit cap was violated and remains a retryable rejection, except
        # for a cell already on the floor (2026-09-12): its clock no longer counts
        # the removal of k, so its trial value may pass below the floor and the
        # commit restores the floor with the same accounting. omega keeps its
        # strict floor; wall and inflow omega never approach it.
        atfloor=rk[i,j]<=1.0f-12*rho[ia,ja]*(1.0f0+4.0f0*eps(Float32))
        good=isfinite(rn) && rn>0.0f0 && isfinite(K) && (K>=0.0f0 || atfloor) && isfinite(W) &&
            isfinite(k) && isfinite(w) && w>=1.0f-6
        ktrial[i,j]=K;wtrial[i,j]=W;status[i,j]=good ? 0.0f0 : 1.0f0
    end
end

function _sst_trial!(c,w;retryable::Bool=false,stage::Int=0,slot=nothing)
    a=c.turbulence;s=c.solver
    status=slot===nothing ? c.status : slot        # deferred: the flag slot's bank slice
    slot===nothing && fill!(c.status,0.0f0)
    _k_sst_species_trial!(s.backend)(a.ktrial,a.wtrial,status,a.rk,a.rw,a.k0,a.w0,a.dck,a.dcw,
        c.conserved.Utrial,s.dtf,s.vol,s.rho,w,Int32(s.NI),Int32(s.NJ),Int32(s.g);ndrange=(s.NI,s.NJ))
    slot===nothing || return nothing
    maximum(c.status)==0.0f0 || _reject_stage(c,stage,:sst_trial,
        "SST stage reached a positivity floor/nonfinite state; restore and reduce dt",retryable)
end

# Prescribed trip (2026-09-30, opt-in; CMeshCavSpecies.TRIP_YPLUS_MAX): the inner-layer turbulent k of cell (i,j) in a
# trip column, from the column's resolved wall shear u_tau^2 = nu |u_t| / d1 at its wall cell: u_tau^2/sqrt(beta*) in the
# log layer, falling as (y+/10)^2 below y+ = 10; zero above the inner layer (y+ > ymax), where the laminar layer's small
# omega would turn an imposed k into a large eddy viscosity.
@inline function _sst_trip_k(ux,uy,Sjx,Sjy,vol,wd,i,g,nu,ymax)
    Sw=sqrt(Sjx[i,1]*Sjx[i,1]+Sjy[i,1]*Sjy[i,1]);d1=0.5f0*vol[i,1]/Sw+1.0f-30
    nx=Sjx[i,1]/Sw;ny=Sjy[i,1]/Sw;uw=ux[i+g,1+g];vw=uy[i+g,1+g];un=uw*nx+vw*ny
    utx=uw-un*nx;uty=vw-un*ny
    ut2=nu*sqrt(utx*utx+uty*uty)/d1
    yp=wd*sqrt(ut2)/nu
    yp<=ymax ? (ut2/0.3f0)*min(1.0f0,(0.1f0*yp)*(0.1f0*yp)) : 0.0f0
end

@kernel function _k_sst_species_commit!(rk,rw,tk,tom,kfloor,wpin,@Const(ktrial),@Const(wtrial),@Const(rho),@Const(vol),
                                        @Const(ux),@Const(uy),@Const(Sjx),@Const(Sjy),nu::Float32,
                                        NI::Int32,NJ::Int32,g::Int32,NIW::Int32,NIF::Int32,wf::Int32,
                                        @Const(xc),@Const(wdist),tx0::Float32,tx1::Float32,ymax::Float32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        r=rho[i+g,j+g];K=ktrial[i,j];Kf=1.0f-12*r
        # Prescribed trip (2026-09-30): in the strip the floor is raised to the inner-layer turbulent k and the added
        # amount is ledgered with the k floor below. Off (tx1 == 0) this leaves Kf, bitwise the default path.
        if tx1>0.0f0 && NIW<i<=NIW+NIF && tx0<=xc[i,1]<=tx1
            Kf=max(Kf,r*_sst_trip_k(ux,uy,Sjx,Sjy,vol,wdist[i,j],i,g,nu,ymax))
        end
        if K<Kf
            # k floor correction with accounting (2026-09-11): the committed
            # conserved value is the exact floor product and the added amount
            # times the cell volume is ledgered as a k source contribution.
            kfloor[i,j]=Float32((Float64(Kf)-Float64(K))*Float64(vol[i,j]));K=Kf
        else
            kfloor[i,j]=0.0f0
        end
        rk[i,j]=K
        k=K/r
        tk[i+g,j+g]=k<1.0f-12 ? 1.0f-12 : k
        if wf==Int32(1) && j==Int32(1) && NIW<i<=NIW+NIF
            # All-y+ wall function (2026-09-26): the wall cell's omega is Menter's automatic-wall-treatment value
            # sqrt(omega_vis^2 + omega_log^2) from this stage's committed velocity; the change is ledgered like the
            # k floor (weights 1/6, 2/3, 1) and reported in the omega source.
            Sw=sqrt(Sjx[i,1]*Sjx[i,1]+Sjy[i,1]*Sjy[i,1]);d1=0.5f0*vol[i,1]/Sw+1.0f-30
            nx=Sjx[i,1]/Sw;ny=Sjy[i,1]/Sw;uw=ux[i+g,j+g];vw=uy[i+g,j+g];un=uw*nx+vw*ny
            utx=uw-un*nx;uty=vw-un*ny
            ut,_,_=CMeshSolverKA.wf_species_state(sqrt(utx*utx+uty*uty),d1,nu)
            Wp=r*CMeshSolverKA.wf_species_omega(ut,d1,nu)
            wpin[i,j]=Float32((Float64(Wp)-Float64(wtrial[i,j]))*Float64(vol[i,j]))
            rw[i,j]=Wp;tom[i+g,j+g]=Wp/r
        else
            wpin[i,j]=0.0f0
            rw[i,j]=wtrial[i,j];tom[i+g,j+g]=wtrial[i,j]/r
        end
    end
end

function _sst_commit!(c)
    a=c.turbulence;s=c.solver
    tx0,w=c.trip
    _k_sst_species_commit!(s.backend)(a.rk,a.rw,s.tk,s.tom,a.kfloor,a.wpin,a.ktrial,a.wtrial,s.rho,s.vol,
        s.ux,s.uy,s.Sjx,s.Sjy,s.nu_lam,Int32(s.NI),Int32(s.NJ),Int32(s.g),Int32(s.NIW),Int32(s.NIF),_wall_function(s),
        s.xc,a.wdist,tx0,w>0.0f0 ? tx0+w : 0.0f0,TRIP_YPLUS_MAX;ndrange=(s.NI,s.NJ))
end

function _sst_snapshot!(c)
    a=c.turbulence;copyto!(a.k0,a.rk);copyto!(a.w0,a.rw)
end
