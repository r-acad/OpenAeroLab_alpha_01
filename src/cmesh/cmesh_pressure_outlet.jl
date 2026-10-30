# Opt-in, fixed-static-pressure boundary for the barotropic TEM channel.
# Characteristic derivation, limits, and validation: docs/2026-09-07_PRESSURE_OUTLET.md.
module CMeshPressureOutlet

using KernelAbstractions
using ..CMeshEOSMix: _pmix_vl, _rhomix_vl
export set_pressure_outlet!, clear_pressure_outlet!, pressure_outlet_bc!, pressure_outlet_flux!

const CS2 = 0.6666667f0

function _outlet_eos(tem)
    tem.eos == Int32(1) && tem.wallis == Int32(1) && tem.vaploc == 1.0f0 ||
        throw(ArgumentError("pressure outlet requires eos=1, wallis=1, vaploc=1"))
    all(iszero, (tem.ncg, tem.dk_cmin2, tem.vapbound, tem.tmix, tem.tvp)) ||
        throw(ArgumentError("pressure outlet does not support NCG/DK/bounded/thermal EOS variants"))
    all(iszero, (tem.p3, tem.rp, tem.slip)) ||
        throw(ArgumentError("pressure outlet currently requires p3=0, rp=0 and slip=0"))
    (tem.p_v, tem.a2l, tem.rho_l_sat, tem.rho_v_sat, tem.psm, tem.lq, tem.am_floor_frac)
end

"""Enable fixed exit static pressure on both downstream channel faces.

Supported: KA segmented channel, cav=4, the local-pressure TEM EOS, and subsonic
outflow (plus laminar backflow). Exiting composition and tangential velocity are extrapolated;
backflow supplies pure liquid and the freestream tangential velocity. Supersonic
outflow extrapolates; supersonic inflow or a transonic boundary correction fails.
This fixed-pressure condition reflects acoustic waves; it is not an NRBC.
Re-seed the channel reference after setup. Pressure is in solver units.
`checked_mixture=true` admits any positive finite EOS endpoint density for the
conservative caller; the default retains the legacy 1e-6 setup threshold.
"""
function set_pressure_outlet!(s, tem; p_out::Real, checked_mixture::Bool=false)
    # outer_mode 2 is a strip topology's strip carrier (5 Oct 2026, plan section 3.2): its outlet is its i = NI+1 side,
    # bound here exactly as the C block's (the same p_out; `outlet_alpha` aliases the strip's own TEM alpha).
    s.outer_mode == Int32(1) || s.outer_mode == Int32(2) || throw(ArgumentError("pressure outlet requires a segmented channel"))
    s.cav == Int32(4) || throw(ArgumentError("pressure outlet requires cav=4"))
    s.ff_mix == Int32(1) || throw(ArgumentError("configure mixture far field before pressure outlet"))
    size(tem.alpha) == size(s.rho) && typeof(tem.alpha) == typeof(s.rho) &&
        KernelAbstractions.get_backend(tem.alpha) == KernelAbstractions.get_backend(s.rho) ||
        throw(ArgumentError("pressure outlet TEM alpha must match the flow array shape and backend"))
    s.emix == 0.0f0 || throw(ArgumentError("pressure outlet requires the barotropic carrier (emix=0)"))
    s.ff_Tmin == tem.Tmin && s.ff_Tmax == tem.Tmax ||
        throw(ArgumentError("pressure outlet requires matching TEM and far-field pedestal bands"))
    isfinite(s.ff_Tmin) && isfinite(s.ff_Tmax) && 0 < s.ff_Tmin <= s.ff_Tmax ||
        throw(ArgumentError("pressure outlet requires a positive finite pedestal band"))
    ep = _outlet_eos(tem)
    all(isfinite, ep) && ep[1] > 0 && ep[2] > 0 && ep[3] > ep[4] > 0 &&
        ep[5] >= 0 && ep[6] >= 0 && ep[7] >= 0 ||
        throw(ArgumentError("pressure outlet received invalid EOS parameters"))
    p = Float32(p_out)
    isfinite(p) && p > ep[5]/2 ||
        throw(ArgumentError("p_out must be finite and exceed the EOS smooth-max minimum psm/2"))
    # The inverse denominator is affine in alpha; checking both endpoints proves
    # positivity across [0,1]. Reject values outside the EOS finite-density branch.
    for a in (0.0f0, 1.0f0)
        r = _rhomix_vl(p, a, ep[1], ep[2], ep[3], ep[4], ep[5], ep[6])
        isfinite(r) && (checked_mixture ? r > 0.0f0 : r > 1.0f-6) ||
            throw(ArgumentError("p_out is outside the positive finite-density EOS branch"))
    end
    s.outlet_p = p; s.outlet_eos = ep; s.outlet_alpha = tem.alpha
    s.outlet_mode = Int32(1)
    s.outer_ref_ready = Int32(0); s.outer_ref_signature = UInt64(0)
    nothing
end

function clear_pressure_outlet!(s)
    s.outlet_mode = Int32(0)
    s.outer_ref_ready = Int32(0); s.outer_ref_signature = UInt64(0)
    nothing
end

"""Check that the attached TEM still supplies the EOS and composition used by the outlet."""
function check_pressure_outlet_tem(s, tem)
    s.outlet_mode == Int32(0) && return nothing
    s.outlet_alpha === tem.alpha && s.outlet_eos == _outlet_eos(tem) &&
        s.ff_Tmin == tem.Tmin && s.ff_Tmax == tem.Tmax ||
        throw(ArgumentError("pressure outlet TEM/EOS changed; call set_pressure_outlet! and re-seed"))
    nothing
end

# Local linear characteristic compatibility: du_n = -dp/(rho*c).
# The outgoing acoustic invariant is held at the interior value. Using delta-p
# avoids subtracting large absolute invariants. Composition is a convective mode.
# Return status 1 for unsupported supersonic inflow, 2 for invalid/transonic data.
@inline function outlet_state(rd, ud, vd, Td, pd, cd, ad, nx, ny, ui, vi, pb, ep, Tmin, Tmax,
                              checked_mixture::Bool=false)
    un = ud*nx + vd*ny
    if !(isfinite(rd) && rd > 0 && isfinite(cd) && cd > 0 &&
         isfinite(ud) && isfinite(vd) && isfinite(Td) &&
         isfinite(un) && isfinite(pd) && pd > 0 && isfinite(ad) && 0 <= ad <= 1)
        return (rd, ud, vd, Td, pd, cd, ad, 2.0f0)
    elseif un >= cd
        return (rd, ud, vd, Td, pd, cd, ad, 0.0f0)
    elseif un <= -cd
        return (rd, ud, vd, Td, pd, cd, ad, 1.0f0)
    end
    ub = un + (pd-pb)/(rd*cd)
    ab = ub >= 0 ? ad : 0.0f0
    ut = ub >= 0 ? (-ud*ny + vd*nx) : (-ui*ny + vi*nx)
    pv, a2l, rl, rv, psm, lq, afloor = ep
    rb = _rhomix_vl(pb, ab, pv, a2l, rl, rv, psm, lq)
    rs = rl*(1.0f0-ab) + rv*ab
    pm, dp, _ = _pmix_vl(pv, ab, rb, rs, a2l, rl, rv, lq)
    sq = sqrt(pm*pm + psm*psm)
    cb = sqrt(max(0.5f0*dp*(1.0f0+pm/sq), afloor*a2l))
    Tb = clamp(pb/(rb*CS2), Tmin, Tmax)
    uxb = ub*nx-ut*ny; uyb = ub*ny+ut*nx
    status = isfinite(rb) && (checked_mixture ? rb > 0.0f0 : rb > 1.0f-6) && isfinite(cb) && abs(ub) < cb &&
             isfinite(Tb) && isfinite(uxb) && isfinite(uyb) ? 0.0f0 : 2.0f0
    (rb, uxb, uyb, Tb, pb, cb, ab, status)
end

@kernel function _k_pressure_outlet_bc!(rho, ux, uy, T, pexc, am, alpha, status,
                                        mixture_pressure, direct_pressure::Int32,
                                        @Const(Six), @Const(Siy), ui::Float32, vi::Float32,
                                        pb::Float32, ep::NTuple{7,Float32},
                                        Tmin::Float32, Tmax::Float32, sst_on::Int32,
                                        NI::Int32, NJ::Int32, g::Int32, sides::Int32)
    jj, side = @index(Global, NTuple)
    # sides: bit 1 the i = 1 end, bit 2 the i = NI+1 end (strip topology, 5 Oct 2026, plan 4.1: a strip carrier's outlet
    # is its i = NI+1 side only, its i = 1 side the base wall); every other caller passes 3, both ends, as before
    @inbounds if jj <= NJ+2g && side <= 2 && ((sides>>(Int32(side)-Int32(1)))&Int32(1))==Int32(1)
        j = clamp(jj-g, Int32(1), NJ)
        i = side == 1 ? Int32(1) : NI+Int32(1)
        is = side == 1 ? g+Int32(1) : NI+g
        sg = side == 1 ? -1.0f0 : 1.0f0
        Sx = sg*Six[i,j]; Sy = sg*Siy[i,j]; Sm = sqrt(Sx*Sx+Sy*Sy)
        rd = rho[is,jj]; td = T[is,jj]
        pd = direct_pressure == Int32(0) ? rd*CS2*td+pexc[is,jj] : mixture_pressure[is,jj]
        vals = outlet_state(rd, ux[is,jj], uy[is,jj], td, pd,
                            am[is,jj], alpha[is,jj], Sx/Sm, Sy/Sm, ui, vi, pb, ep, Tmin, Tmax,
                            direct_pressure == Int32(1))
        rb, ub, vb, tb, p, cb, ab, err = vals
        if sst_on != Int32(0) && ub*Sx+vb*Sy < 0.0f0
            # The existing SST/k-epsilon scalar operator has no matching inlet
            # data or boundary face advection for this reversal. Do not invent it.
            err = 3.0f0
        end
        status[side,jj] = err
        for k in 1:g
            id = side == 1 ? g+Int32(1)-k : NI+g+k
            rho[id,jj] = rb; ux[id,jj] = ub; uy[id,jj] = vb; T[id,jj] = tb
            pexc[id,jj] = p-rb*CS2*tb; am[id,jj] = cb; alpha[id,jj] = ab
            if direct_pressure != Int32(0)
                mixture_pressure[id,jj] = p
            end
        end
    end
end

"Validate a separately owned pressure array before any outlet or shared-residual writes."
function check_pressure_array(s, mixture_pressure)
    mixture_pressure isa AbstractMatrix{Float32} &&
        size(mixture_pressure) == size(s.rho) == (s.NI+2s.g, s.NJ+2s.g) &&
        KernelAbstractions.get_backend(mixture_pressure) == s.backend ||
        throw(ArgumentError("mixture_pressure must be a full-ghost Float32 array on the flow backend"))
    all(a -> !Base.mightalias(mixture_pressure, a),
        (s.rho, s.ux, s.uy, s.T, s.pexc, s.am, s.outlet_alpha, s.outlet_status)) ||
        throw(ArgumentError("mixture_pressure must not alias pressure-outlet BC output arrays"))
    nothing
end

"""Apply outlet ghosts, optionally using a separately owned full-ghost mixture-pressure array.

With `mixture_pressure=nothing`, pressure retains its legacy pedestal reconstruction.
Otherwise the supplied interior pressure is used and the pressure returned by the
characteristic state (including supersonic extrapolation) is stored in every ghost layer.
"""
# Deferred admission (2026-09-12): reduce one status or clock array into a single
# device slot without a host transfer. The owner reads every slot once and, on any
# flagged failure, replays the step with the immediate checks for exact diagnostics.
# Launch fusion (2026-09-12, speed-first on FP32 GPUs): a slot is (A, k) with A a UInt32
# device vector of 2*N entries: flags at k (maximum of a status array, 0 = admitted),
# clocks at N+k (minimum of a cap array as the bit pattern of the positive Float32,
# which orders like the value; NaN or negative caps map to 0, exactly what the
# immediate path's `isfinite(cap) && cap > 0` rejects). One workgroup-reduced kernel
# per slot, one atomic per workgroup, no allocation, no copy; min/max are exact, so
# the flags and clocks are bitwise those of the host reductions they replace.
# The KA CPU backend requires the barrier ladder to be static and the local index to
# be re-read after every barrier.
const _SLOT_WG=256
const _SLOT_HALF=64
@inline _slot_cap_bits(x::Float32)=(isnan(x) || x<0.0f0) ? 0x00000000 : reinterpret(UInt32,x)
@kernel function _k_slot_min!(slot,k::Int32,@Const(a),n::Int32)
    li=@index(Local,Linear);gi=@index(Global,Linear);ng=@ndrange()[1]
    buf=@localmem UInt32 (_SLOT_WG,)
    v=0xffffffff
    i=gi
    @inbounds while i<=n
        v=min(v,_slot_cap_bits(a[i])); i+=ng
    end
    @inbounds buf[li]=v
    @synchronize
    li=@index(Local,Linear); @inbounds li<=128 && (buf[li]=min(buf[li],buf[li+128])); @synchronize
    li=@index(Local,Linear); @inbounds li<=64 && (buf[li]=min(buf[li],buf[li+64])); @synchronize
    li=@index(Local,Linear); @inbounds li<=32 && (buf[li]=min(buf[li],buf[li+32])); @synchronize
    li=@index(Local,Linear); @inbounds li<=16 && (buf[li]=min(buf[li],buf[li+16])); @synchronize
    li=@index(Local,Linear); @inbounds li<=8 && (buf[li]=min(buf[li],buf[li+8])); @synchronize
    li=@index(Local,Linear); @inbounds li<=4 && (buf[li]=min(buf[li],buf[li+4])); @synchronize
    li=@index(Local,Linear); @inbounds li<=2 && (buf[li]=min(buf[li],buf[li+2])); @synchronize
    li=@index(Local,Linear); @inbounds li==1 && (KernelAbstractions.@atomic min(slot[k],min(buf[1],buf[2])))
end
@kernel function _k_slot_max!(slot,k::Int32,@Const(a),n::Int32)
    li=@index(Local,Linear);gi=@index(Global,Linear);ng=@ndrange()[1]
    buf=@localmem UInt32 (_SLOT_WG,)
    v=0x00000000
    i=gi
    @inbounds while i<=n
        v=max(v,a[i]!=0.0f0 ? 0x00000001 : 0x00000000); i+=ng
    end
    @inbounds buf[li]=v
    @synchronize
    li=@index(Local,Linear); @inbounds li<=128 && (buf[li]=max(buf[li],buf[li+128])); @synchronize
    li=@index(Local,Linear); @inbounds li<=64 && (buf[li]=max(buf[li],buf[li+64])); @synchronize
    li=@index(Local,Linear); @inbounds li<=32 && (buf[li]=max(buf[li],buf[li+32])); @synchronize
    li=@index(Local,Linear); @inbounds li<=16 && (buf[li]=max(buf[li],buf[li+16])); @synchronize
    li=@index(Local,Linear); @inbounds li<=8 && (buf[li]=max(buf[li],buf[li+8])); @synchronize
    li=@index(Local,Linear); @inbounds li<=4 && (buf[li]=max(buf[li],buf[li+4])); @synchronize
    li=@index(Local,Linear); @inbounds li<=2 && (buf[li]=max(buf[li],buf[li+2])); @synchronize
    li=@index(Local,Linear); @inbounds li==1 && (KernelAbstractions.@atomic max(slot[k],max(buf[1],buf[2])))
end
function _slot_max!(slot::Tuple,a)
    A,k=slot; n=length(a); groups=min(64,cld(n,_SLOT_WG))
    _k_slot_max!(KernelAbstractions.get_backend(A),_SLOT_WG)(A,Int32(k),vec(a),Int32(n);ndrange=groups*_SLOT_WG)
    nothing
end
function _slot_min!(slot::Tuple,a)
    A,k=slot; n=length(a); groups=min(64,cld(n,_SLOT_WG))
    _k_slot_min!(KernelAbstractions.get_backend(A),_SLOT_WG)(A,Int32(_SLOT_HALF+k),vec(a),Int32(n);ndrange=groups*_SLOT_WG)
    nothing
end

function pressure_outlet_bc!(s; mixture_pressure=nothing, conservative_sst::Bool=false, status_slot=nothing,
                             sides::Int32=Int32(3))
    s.outlet_mode == Int32(0) && return nothing
    sides in (Int32(1), Int32(2), Int32(3)) || throw(ArgumentError("pressure outlet sides must be 1, 2 or 3 (got $sides)"))
    # The strip carrier (outer_mode 2, strip topology, plan 4.1): its outlet is its i = NI+1 side only (sides = 2); its
    # i = 1 side is the base wall, whose ghosts are its column fill's. The species step fills the strip's outlet through
    # the fused pressure trio with the same mask; this unfused fill is its reference.
    s.outer_mode == Int32(2) && sides != Int32(2) && throw(ArgumentError(
        "pressure_outlet_bc! on the strip carrier (outer_mode 2, CMeshSpeciesStripTopologyV1) fills its i = NI+1 outlet side only (sides = 2): its i = 1 side is the base wall"))
    (s.outer_mode == Int32(1) || s.outer_mode == Int32(2)) && s.cav == Int32(4) && s.ff_mix == Int32(1) && s.emix == 0.0f0 ||
        throw(ArgumentError("pressure outlet configuration changed outside its supported regime"))
    direct_pressure = mixture_pressure === nothing ? Int32(0) : Int32(1)
    conservative_sst && direct_pressure == Int32(0) &&
        throw(ArgumentError("conservative SST boundary transport requires direct mixture pressure"))
    direct_pressure == Int32(0) || check_pressure_array(s, mixture_pressure)
    pressure = direct_pressure == Int32(0) ? s.pexc : mixture_pressure
    _k_pressure_outlet_bc!(s.backend)(s.rho, s.ux, s.uy, s.T, s.pexc, s.am,
        s.outlet_alpha, s.outlet_status, pressure, direct_pressure,
        s.Six, s.Siy, s.uxi, s.uyi, s.outlet_p,
        s.outlet_eos, s.ff_Tmin, s.ff_Tmax, s.sst && !conservative_sst ? Int32(1) : Int32(0),
        Int32(s.NI), Int32(s.NJ), Int32(s.g), sides;
        ndrange=(s.NJ+2s.g, 2))
    # Fail before a flux can consume a state outside this boundary's derivation.
    # This opt-in correctness check synchronizes; remove only with equivalent error propagation.
    # A deferred owner (status_slot) keeps the flag on the device and replays this
    # immediate check before any flagged state can be published.
    if status_slot === nothing
        maximum(s.outlet_status) == 0.0f0 ||
            throw(ArgumentError("pressure outlet encountered invalid state, supersonic inflow, transonic correction, or unsupported turbulence backflow"))
    elseif status_slot !== :batched   # :batched (2026-09-12): the owner reduces s.outlet_status in its next batch
        _slot_max!(status_slot, s.outlet_status)
    end
    nothing
end

# Prescribe traction at the NUMERICAL face as well as in the reconstruction ghosts.
# A Riemann flux between an interior state and a pb ghost alone does not impose pb
# exactly at that face. The characteristic boundary state owns all advective fluxes.
# Cell-to-i-face displacement from the SAME vertex-mean centers used by the
# existing full-stress operator. No absolute Float32 center subtraction or new
# geometry storage. sg=-1 at west, +1 at east; sx,sy retain stored +i orientation.
@inline function _pressure_end_geometry(sx,sy,sjx_s,sjy_s,sjx_n,sjy_n,sg)
    dx=sg*0.25f0*(sjy_s+sjy_n);dy=-sg*0.25f0*(sjx_s+sjx_n)
    S2=sx*sx+sy*sy;d2=dx*dx+dy*dy
    Sm=sqrt(S2);dn=sqrt(d2);outdot=sg*(dx*sx+dy*sy)
    all(isfinite,(dx,dy,S2,d2,Sm,dn,outdot)) && S2>0f0 && d2>0f0 && outdot>0f0 ||
        return (NaN32,NaN32,NaN32,NaN32)
    (dx,dy,Sm,dn)
end

# The characteristic ghost is a FACE value, not an exterior-cell value. Use the
# cell-to-face displacement with the existing compact gradient correction.
# Repeated cell gradients retain transverse stress; boundary work uses u_b.
@inline function _pressure_end_traction(viscous_face,grad,ui,vi,Ti,ub,vb,Tb,dx,dy,sx,sy,mu)
    Sm=sqrt(sx*sx+sy*sy)
    all(isfinite,(grad...,ui,vi,Ti,ub,vb,Tb,dx,dy,sx,sy,mu,Sm)) &&
        Sm>0f0 && dx*dx+dy*dy>0f0 && mu>=0f0 || return (NaN32,NaN32,NaN32)
    fx,fy,_=viscous_face(grad...,grad...,ui,vi,Ti,ub,vb,Tb,dx,dy,sx/Sm,sy/Sm,Sm,mu,0f0)
    (fx,fy,ub*fx+vb*fy) # no conduction of the artificial barotropic pedestal
end

@inline function _pressure_end_conductance(r,nu,sx,sy,dx,dy)
    Sm=sqrt(sx*sx+sy*sy);dn=sqrt(dx*dx+dy*dy)
    all(isfinite,(r,nu,Sm,dn)) && r>0f0 && nu>=0f0 && Sm>0f0 && dn>0f0 || return Inf32
    (4f0/3f0)*(r*nu*Sm/dn)
end

# Once-only Species setup admission; its existing static metric binding protects
# subsequent calls. Direct low-level callers have the same geometry precondition.
function check_pressure_end_geometry(s;sides::Int32=Int32(3))
    s.nu_lam>0f0 && s.fullvisc==Int32(1) || return nothing
    six,siy,sjx,sjy=map(Array,(s.Six,s.Siy,s.Sjx,s.Sjy))
    for j in 1:s.NJ,side in 1:2
        # sides: bit 1 the i = 1 end, bit 2 the i = NI+1 end (strip topology, plan 4.1: the strip's outlet is side 2 only)
        (sides>>(side-1))&1==1 || continue
        f=side==1 ? 1 : s.NI+1;i=side==1 ? 1 : s.NI;sg=side==1 ? -1f0 : 1f0
        z=_pressure_end_geometry(six[f,j],siy[f,j],sjx[i,j],sjy[i,j],sjx[i,j+1],sjy[i,j+1],sg)
        all(isfinite,z) || throw(ArgumentError("direct full-stress pressure end requires finite positive cell-to-face geometry"))
    end
    nothing
end

@kernel function _k_pressure_outlet_flux!(Fr, Fx, Fy, Fe, @Const(rho), @Const(ux),
                                          @Const(uy), @Const(T), @Const(pexc), @Const(am),
                                          @Const(mixture_pressure), direct_pressure::Int32,
                                          @Const(Six), @Const(Siy), eint::Float32,
                                          @Const(Sjx), @Const(Sjy), @Const(gux), @Const(guy),
                                          @Const(gvx), @Const(gvy), @Const(nut),
                                          nu_lam::Float32, sst_on::Int32, viscous_face, half,
                                          NI::Int32, NJ::Int32, g::Int32, sides::Int32)
    j, side = @index(Global, NTuple)
    # sides: bit 1 the i = 1 end, bit 2 the i = NI+1 end; every caller but a strip topology's strip carrier passes 3
    @inbounds if j <= NJ && side <= 2 && ((sides>>(Int32(side)-Int32(1)))&Int32(1))==Int32(1)
        i = side == 1 ? Int32(1) : NI+Int32(1)
        is = side == 1 ? g+Int32(1) : NI+g
        ig = side == 1 ? g : NI+g+Int32(1)
        Sx = Six[i,j]; Sy = Siy[i,j]; Sm = sqrt(Sx*Sx+Sy*Sy)
        sg = side == 1 ? -1.0f0 : 1.0f0
        un = sg*(ux[is,j+g]*Sx+uy[is,j+g]*Sy)/Sm
        if un < am[is,j+g]   # supersonic outflow retains the interior numerical flux
            r = rho[ig,j+g]; u = ux[ig,j+g]; v = uy[ig,j+g]; t = T[ig,j+g]
            p = direct_pressure == Int32(0) ? r*CS2*t + pexc[ig,j+g] : mixture_pressure[ig,j+g]
            mdot = r*(u*Sx+v*Sy)
            Fr[i,j] = mdot
            Fx[i,j] = mdot*u+p*Sx; Fy[i,j] = mdot*v+p*Sy
            Fe[i,j] = mdot*(eint*t+0.5f0*(u*u+v*v)+p/r)
            if half !== nothing   # the characteristic state owns the whole kinetic half split
                half[i,j,1] = max(mdot,0.0f0); half[i,j,2] = min(mdot,0.0f0)
            end
        end
        if viscous_face !== nothing
            ic=side==1 ? Int32(1) : NI;jj=j+g
            dx,dy,_,_=_pressure_end_geometry(Sx,Sy,Sjx[ic,j],Sjy[ic,j],Sjx[ic,j+1],Sjy[ic,j+1],sg)
            grad=(gux[ic,j],guy[ic,j],gvx[ic,j],gvy[ic,j],0f0,0f0)
            nub=sst_on==Int32(1) ? nut[ig,jj] : 0f0
            mu=rho[ig,jj]*(nu_lam+nub)
            vx,vy,ve=_pressure_end_traction(viscous_face,grad,
                ux[is,jj],uy[is,jj],T[is,jj],ux[ig,jj],uy[ig,jj],T[ig,jj],dx,dy,Sx,Sy,mu)
            # Both subsonic override and retained supersonic numerical flux get
            # exactly one viscous contribution, in the stored +i orientation.
            Fx[i,j]-=vx;Fy[i,j]-=vy;Fe[i,j]-=ve
        end
    end
end

"Use the pressure already written by `pressure_outlet_bc!` when `mixture_pressure` is supplied."
function pressure_outlet_flux!(s; mixture_pressure=nothing, half=nothing, sides::Int32=Int32(3))
    s.outlet_mode == Int32(0) && return nothing
    sides in (Int32(1), Int32(2), Int32(3)) || throw(ArgumentError("pressure outlet sides must be 1, 2 or 3 (got $sides)"))
    half === nothing || mixture_pressure !== nothing ||
        throw(ArgumentError("kinetic half masses at the pressure ends require direct mixture pressure"))
    direct_pressure = mixture_pressure === nothing ? Int32(0) : Int32(1)
    direct_pressure == Int32(0) || check_pressure_array(s, mixture_pressure)
    pressure = direct_pressure == Int32(0) ? s.pexc : mixture_pressure
    # This module is also included under TEM. Resolve the one existing shared
    # helper from the solver type's module; do not include a second flux core.
    viscous_face = direct_pressure==Int32(1) && s.nu_lam>0f0 && s.fullvisc==Int32(1) ?
        getfield(parentmodule(typeof(s)),:_viscous_face_full) : nothing
    _k_pressure_outlet_flux!(s.backend)(s.Fi_r, s.Fi_mx, s.Fi_my, s.Fi_e,
        s.rho, s.ux, s.uy, s.T, s.pexc, s.am, pressure, direct_pressure,
        s.Six, s.Siy, s.eint,s.Sjx,s.Sjy,s.gux,s.guy,s.gvx,s.gvy,s.nut,
        s.nu_lam,Int32(s.sst),viscous_face,half,
        Int32(s.NI), Int32(s.NJ), Int32(s.g), sides; ndrange=(s.NJ, 2))
    nothing
end

@inline _acoustic_branch(un, c) = un <= -c ? Int32(-1) : un >= c ? Int32(1) : Int32(0)

# The impedance law is the existing segmented far-field law. Only its pressure
# storage and EOS recovery differ from the legacy thermal-pedestal implementation.
@inline function inlet_state(rd, ud, vd, Td, pd, cd, ad, nx, ny,
                             ri, ui, vi, pi, ci, ep, Tmin, Tmax)
    un = ud*nx + vd*ny; uni = ui*nx + vi*ny
    bad = (rd, ud, vd, Td, pd, cd, ad, 1.0f0)
    all(isfinite, (rd,ud,vd,Td,pd,cd,ad,nx,ny,un,uni)) &&
        rd > 0.0f0 && cd > 0 && pd > ep[5]/2 && 0 <= ad <= 1 || return bad
    branch = _acoustic_branch(un,cd)
    branch == _acoustic_branch(un,ci) || return bad
    branch == Int32(1) && return (rd,ud,vd,Td,pd,cd,ad,0.0f0)
    if branch == Int32(-1)
        ub = uni; pb = pi; ab = 0.0f0; ut = -ui*ny+vi*nx
    else
        Z = ri*ci
        ub = 0.5f0*(un+uni+(pd-pi)/Z)
        pb = pi+0.5f0*((pd-pi)+Z*(un-uni))
        ab = ub < 0 ? 0.0f0 : ad
        ut = ub < 0 ? (-ui*ny+vi*nx) : (-ud*ny+vd*nx)
    end
    isfinite(pb) && pb > ep[5]/2 && isfinite(ub) && isfinite(ut) || return bad
    pv,A,rl,rv,psm,lq,afloor = ep
    rb = _rhomix_vl(pb,ab,pv,A,rl,rv,psm,lq)
    isfinite(rb) && rb > 0.0f0 || return bad
    raw,dp,_ = _pmix_vl(pv,ab,rb,rl*(1.0f0-ab)+rv*ab,A,rl,rv,lq)
    sq = sqrt(raw*raw+psm*psm)
    c2 = 0.5f0*dp*(1.0f0+raw/sq)
    isfinite(raw) && raw > 0 && isfinite(c2) && c2 > 0 || return bad
    cb = sqrt(max(c2,afloor*A))
    # A corrected characteristic state may not cross either acoustic branch.
    _acoustic_branch(ub,cb) == branch == _acoustic_branch(ub,ci) || return bad
    tb = clamp(pb/(rb*CS2),Tmin,Tmax) # existing carrier pedestal, not a physical temperature
    ux = ub*nx-ut*ny; uy = ub*ny+ut*nx
    all(isfinite,(cb,tb,ux,uy,pb-rb*CS2*tb)) && cb > 0 && tb > 0 || return bad
    (rb,ux,uy,tb,pb,cb,ab,0.0f0)
end

function _check_inlet_reference(s)
    s.outer_mode == Int32(1) && s.outlet_mode == Int32(1) && s.cav == Int32(4) &&
        s.ff_mix == Int32(1) && s.emix == 0.0f0 ||
        throw(ArgumentError("direct mixture inlet requires the supported segmented barotropic outlet configuration"))
    ep = s.outlet_eos
    all(isfinite,ep) && ep[1] > 0 && ep[2] > 0 && ep[3] > ep[4] > 0 &&
        ep[5] >= 0 && ep[6] >= 0 && ep[7] >= 0 &&
        all(isfinite,(s.rho_inf,s.uxi,s.uyi,s.ff_p_ref,s.ff_a2m,s.ff_Tmin,s.ff_Tmax)) &&
        s.rho_inf > 0.0f0 && s.ff_p_ref > ep[5]/2 && s.ff_a2m > 0 &&
        0 < s.ff_Tmin <= s.ff_Tmax ||
        throw(ArgumentError("invalid direct mixture inlet reference/EOS"))
    pv,A,rl,rv,psm,lq,_ = ep
    r = _rhomix_vl(s.ff_p_ref,0.0f0,pv,A,rl,rv,psm,lq)
    raw,_,_ = _pmix_vl(pv,0.0f0,s.rho_inf,rl,A,rl,rv,lq)
    p = 0.5f0*(raw+sqrt(raw*raw+psm*psm))
    # Explicit Float32 operational consistency bound, not a rounding certificate.
    tol = 32.0f0*eps(Float32)
    isfinite(r) && r > 0.0f0 && isfinite(p) && raw > 0 &&
        abs(r-s.rho_inf) <= tol*max(abs(r),abs(s.rho_inf)) &&
        abs(p-s.ff_p_ref) <= tol*max(abs(p),abs(s.ff_p_ref)) ||
        throw(ArgumentError("mixture inlet rho_inf and ff_p_ref are inconsistent with the attached EOS"))
    ci = sqrt(s.ff_a2m)
    isfinite(ci) && isfinite(s.rho_inf*ci) && s.rho_inf*ci > 0 ||
        throw(ArgumentError("invalid mixture inlet acoustic impedance"))
    nothing
end

@kernel function _k_mixture_inlet_bc!(rho,ux,uy,T,pexc,am,alpha,p,
                                      @Const(Sjx),@Const(Sjy),@Const(kinds),
                                      ri::Float32,ui::Float32,vi::Float32,pi::Float32,ci::Float32,
                                      ep::NTuple{7,Float32},Tmin::Float32,Tmax::Float32,
                                      sst_on::Int32,NI::Int32,NJ::Int32,g::Int32)
    i = @index(Global,Linear)
    @inbounds if i <= NI && kinds[i] == 0x01
        ii = i+g; jj = NJ+g
        Sx = Sjx[i,NJ+Int32(1)]; Sy = Sjy[i,NJ+Int32(1)]; Sm = sqrt(Sx*Sx+Sy*Sy)
        rb,ub,vb,tb,pb,cb,ab,err = inlet_state(rho[ii,jj],ux[ii,jj],uy[ii,jj],T[ii,jj],
            p[ii,jj],am[ii,jj],alpha[ii,jj],Sx/Sm,Sy/Sm,ri,ui,vi,pi,ci,ep,Tmin,Tmax)
        if err != 0.0f0 || (sst_on != Int32(0) && ub*Sx+vb*Sy < 0.0f0)
            p[ii,jj+Int32(1)] = -1.0f0 # scratch sentinel; host rejects before any flux
        else
            for k in 1:g
                jd = jj+k
                rho[ii,jd]=rb; ux[ii,jd]=ub; uy[ii,jd]=vb; T[ii,jd]=tb
                pexc[ii,jd]=pb-rb*CS2*tb; am[ii,jd]=cb; alpha[ii,jd]=ab; p[ii,jd]=pb
            end
        end
    end
end

"""Refresh only segmented 0x01 ghosts from freshly EOS-filled interior pressure.

Uses the existing ambient-impedance law with actual EOS density/sound recovery.
An invalid state marks pressure scratch negative and throws before flux use; the
caller must restore/rebuild after failure. Other face kinds and legacy paths are untouched.
"""
function mixture_inlet_bc!(s,p;conservative_sst::Bool=false,inlet_present=nothing,pressure_slot=nothing)
    check_pressure_array(s,p)
    (inlet_present===nothing ? any(==(0x01),s.outer_face_kind) : inlet_present) || return nothing
    _check_inlet_reference(s)
    _k_mixture_inlet_bc!(s.backend)(s.rho,s.ux,s.uy,s.T,s.pexc,s.am,s.outlet_alpha,p,
        s.Sjx,s.Sjy,s.outer_face_kind,s.rho_inf,s.uxi,s.uyi,s.ff_p_ref,sqrt(s.ff_a2m),
        s.outlet_eos,s.ff_Tmin,s.ff_Tmax,s.sst && !conservative_sst ? Int32(1) : Int32(0),
        Int32(s.NI),Int32(s.NJ),Int32(s.g);ndrange=s.NI)
    if pressure_slot===nothing
        minimum(p) > 0.0f0 ||
            throw(ArgumentError("mixture inlet rejected invalid/transonic state or unsupported turbulence inflow; restore pressure/state scratch"))
    elseif pressure_slot!==:batched   # :batched (2026-09-12): the owner reduces p in its next batch
        _slot_min!(pressure_slot,p)
    end
    nothing
end

@kernel function _k_mixture_inlet_flux!(Fr,Fx,Fy,Fe,@Const(rho),@Const(ux),@Const(uy),
                                        @Const(T),@Const(am),@Const(p),@Const(Sjx),@Const(Sjy),
                                        @Const(kinds),eint::Float32,half,NI::Int32,NJ::Int32,g::Int32)
    i = @index(Global,Linear)
    @inbounds if i <= NI && kinds[i] == 0x01
        ii=i+g; ji=NJ+g; j=NJ+Int32(1); jg=ji+Int32(1)
        Sx=Sjx[i,j]; Sy=Sjy[i,j]; Sm=sqrt(Sx*Sx+Sy*Sy)
        un=(ux[ii,ji]*Sx+uy[ii,ji]*Sy)/Sm
        if un < am[ii,ji] # valid supersonic outflow retains the interior numerical flux
            r=rho[ii,jg]; u=ux[ii,jg]; v=uy[ii,jg]; t=T[ii,jg]; pb=p[ii,jg]
            mdot=r*(u*Sx+v*Sy)
            Fr[i,j]=mdot; Fx[i,j]=mdot*u+pb*Sx; Fy[i,j]=mdot*v+pb*Sy
            Fe[i,j]=mdot*(eint*t+0.5f0*(u*u+v*v)+pb/r)
            if half !== nothing   # the characteristic inlet state owns the whole kinetic half split
                half[i,j,1]=max(mdot,0.0f0); half[i,j,2]=min(mdot,0.0f0)
            end
        end
    end
end

"Use the admitted characteristic inlet ghost for mass, traction and enthalpy. Call after mixture_inlet_bc!."
function mixture_inlet_flux!(s,p;half=nothing,inlet_present=nothing,pressure_slot=nothing)
    check_pressure_array(s,p)
    (inlet_present===nothing ? any(==(0x01),s.outer_face_kind) : inlet_present) || return nothing
    _check_inlet_reference(s)
    if pressure_slot===nothing
        minimum(p) > 0.0f0 || throw(ArgumentError("mixture inlet pressure scratch is invalid"))
    elseif pressure_slot!==:batched   # :batched (2026-09-12): the owner reduces p in its next batch
        _slot_min!(pressure_slot,p)
    end
    _k_mixture_inlet_flux!(s.backend)(s.Fj_r,s.Fj_mx,s.Fj_my,s.Fj_e,s.rho,s.ux,s.uy,s.T,s.am,p,
        s.Sjx,s.Sjy,s.outer_face_kind,s.eint,half,Int32(s.NI),Int32(s.NJ),Int32(s.g);ndrange=s.NI)
    nothing
end

end
