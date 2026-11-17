# ============================================================================================
# CMESH2D shared flux core — the rotated-frame KFVS math used by BOTH the plain-Julia solver
# (cmesh_solver.jl) and the KernelAbstractions GPU/CPU solver (cmesh_solver_ka.jl). Extracted
# 2026-07-14 (overnight GPU port) so there is exactly ONE copy of the physics.
# ============================================================================================
const CS2   = 2.0f0/3.0f0
const TGAS_D = 3.0f0

@inline _gas_cs2(::Type{Float32})=CS2
@inline _gas_cs2(::Type{Float64})=2.0/3.0

# Area-integrated Stokes traction for a stationary no-slip wall and the local
# linear gradient grad(u)=u_cell tensor n/d1. muS_over_d1 includes density,
# molecular viscosity, area and inverse wall distance. The normal eigenvalue
# is 4/3; tangential shear is unchanged. Generic arithmetic also serves loads.
@inline function _wall_stokes_traction(u,v,nx,ny,muS_over_d1)
    un=u*nx+v*ny
    normal=un/oftype(un,3)
    (muS_over_d1*(u+normal*nx),muS_over_d1*(v+normal*ny))
end


@inline _mm(a, b) = (a*b <= 0.0f0) ? 0.0f0 : (abs(a) < abs(b) ? a : b)
@inline _faceL_mm(fLL, fL, fR) = fL + 0.5f0*_mm(fL-fLL, fR-fL)     # pond _faceL, s=1 (minmod) branch
@inline _faceR_mm(fL, fR, fRR) = fR - 0.5f0*_mm(fR-fL, fRR-fR)

# WENO5-Z (Borges 2008) — pond thermal.jl:84-95 verbatim. Feed DECREASING index order for the
# right-biased state (mirror symmetry). _wenoblend: WENO5-Z where smooth → minmod at jumps (sensor s).
@inline function _weno5z2(a,b,c,d,e)
    p0=(2.0f0*a-7.0f0*b+11.0f0*c)*0.16666667f0; p1=(-b+5.0f0*c+2.0f0*d)*0.16666667f0; p2=(2.0f0*c+5.0f0*d-e)*0.16666667f0
    t0=a-2.0f0*b+c; s0=a-4.0f0*b+3.0f0*c; b0=1.0833333f0*t0*t0+0.25f0*s0*s0
    t1=b-2.0f0*c+d; b1=1.0833333f0*t1*t1+0.25f0*(b-d)*(b-d)
    t2=c-2.0f0*d+e; s2=3.0f0*c-4.0f0*d+e; b2=1.0833333f0*t2*t2+0.25f0*s2*s2
    # eps 1e-6 (not the pond 1e-12): FP32 steady-RANS needs the larger floor — with 1e-12 the
    # indicator ratios amplify roundoff in near-smooth regions (weight chatter -> Cp wiggles +
    # residual limit cycle at 1.5e-4). At shocks b ~ O(jump^2) >> 1e-6, so shock behavior unchanged.
    eps=1.0f-6; tau=abs(b0-b2); r0=tau/(b0+eps); r1=tau/(b1+eps); r2=tau/(b2+eps)
    a0=0.1f0*(1.0f0+r0*r0); a1=0.6f0*(1.0f0+r1*r1); a2=0.3f0*(1.0f0+r2*r2)
    return (a0*p0+a1*p1+a2*p2)/(a0+a1+a2)
end
@inline _wenoblend(s,a,b,c,d,e) = (1.0f0-s)*_weno5z2(a,b,c,d,e) + s*(c+0.5f0*_mm(c-b,d-c))

# pond _feqT2_gpu VERBATIM (fixed frame: isq=invth=1): 3rd-order thermal Hermite equilibrium
# in the co-moving face frame. Velocities passed here are already FACE-FRAME (un, ut).
@inline function _feqT2(rho, un, ut, T, ufn, uft, cxk, cyk, czk, wk)
    CS2=_gas_cs2(typeof(rho))
    px = un - ufn; py = ut - uft
    inv_cs2 = 1.0f0/CS2; inv_2cs4 = 0.5f0/(CS2*CS2); inv_6cs6 = 1.0f0/(6.0f0*CS2*CS2*CS2)
    cp = px*cxk + py*cyk
    p2 = px*px + py*py
    csq = cxk*cxk + cyk*cyk + czk*czk
    th = T - 1.0f0
    f  = rho
    f += rho*cp*inv_cs2
    f += rho*(cp*cp - CS2*p2)*inv_2cs4
    f += rho*th*0.5f0*(csq*inv_cs2 - TGAS_D)
    f += rho*(cp*cp*cp - 3.0f0*CS2*cp*p2)*inv_6cs6
    f += rho*th*cp*(csq - (TGAS_D+2.0f0)*CS2)*inv_2cs4
    return max(wk*f, 1.0f-12)
end

# ROTATED-frame KFVS flux, 6-STATE stencil version (WENO5-Z capable). recon: 0 = minmod (bitwise-
# identical to the validated v1 — LLL/RRR unused), 1 = WENO5-Z blended to minmod at jumps via the
# pond pressure+density sensor (thermal.jl:211-214). GLOBAL-frame velocities in; (Frho,Fmx,Fmy,FE)·S out.
@inline function _fvflux_normal6(rLLL,uxLLL,uyLLL,TLLL, rLL,uxLL,uyLL,TLL, rL,uxL,uyL,TL,
                                 rR,uxR,uyR,TR, rRR,uxRR,uyRR,TRR, rRRR,uxRRR,uyRRR,TRRR,
                                 nx, ny, S,
                                 cx_arr, cy_arr, cz_arr, w_arr, drot, recon::Int32)
    CS2=_gas_cs2(typeof(rL))
    # rotate stencil velocities into the face frame: un along +n, ut along the tangent
    unLLL = uxLLL*nx + uyLLL*ny; utLLL = -uxLLL*ny + uyLLL*nx
    unLL  = uxLL*nx + uyLL*ny;   utLL  = -uxLL*ny + uyLL*nx
    unL   = uxL *nx + uyL *ny;   utL   = -uxL *ny + uyL *nx
    unR   = uxR *nx + uyR *ny;   utR   = -uxR *ny + uyR *nx
    unRR  = uxRR*nx + uyRR*ny;   utRR  = -uxRR*ny + uyRR*nx
    unRRR = uxRRR*nx + uyRRR*ny; utRRR = -uxRRR*ny + uyRRR*nx
    local rLf, rRf, unLf, unRf, utLf, utRf, TLf, TRf
    if recon == Int32(1)
        # shock/contact sensor (pond :211-214): pressure+density jump ramp [0.1,0.4] -> minmod at jumps
        pLm = rL*CS2*TL; pRm = rR*CS2*TR
        spj = abs(pRm-pLm)/min(pLm,pRm)
        sj  = max(spj, abs(rR-rL)/min(rL,rR))
        sb  = sj <= 0.1f0 ? 0.0f0 : (sj >= 0.4f0 ? 1.0f0 : (sj-0.1f0)*(1.0f0/0.3f0))
        rLf = max(_wenoblend(sb, rLLL,rLL,rL,rR,rRR), 1.0f-4);  rRf = max(_wenoblend(sb, rRRR,rRR,rR,rL,rLL), 1.0f-4)
        unLf = _wenoblend(sb, unLLL,unLL,unL,unR,unRR);          unRf = _wenoblend(sb, unRRR,unRR,unR,unL,unLL)
        utLf = _wenoblend(sb, utLLL,utLL,utL,utR,utRR);          utRf = _wenoblend(sb, utRRR,utRR,utR,utL,utLL)
        TLf = max(_wenoblend(sb, TLLL,TLL,TL,TR,TRR), 1.0f-3);   TRf = max(_wenoblend(sb, TRRR,TRR,TR,TL,TLL), 1.0f-3)
    else
        # minmod (pond :226-229 with s=1) + floors — the validated default
        rLf = max(_faceL_mm(rLL, rL, rR), 1.0f-4);  rRf = max(_faceR_mm(rL, rR, rRR), 1.0f-4)
        unLf = _faceL_mm(unLL, unL, unR);           unRf = _faceR_mm(unL, unR, unRR)
        utLf = _faceL_mm(utLL, utL, utR);           utRf = _faceR_mm(utL, utR, utRR)
        TLf = max(_faceL_mm(TLL, TL, TR), 1.0f-3);  TRf = max(_faceR_mm(TL, TR, TRR), 1.0f-3)
    end
    # arithmetic face frame (pond :201) + rotational internal energy
    ufn = 0.5f0*(unLf+unRf); uft = 0.5f0*(utLf+utRf); erc = 0.5f0*drot*CS2
    # 39-velocity upwind quadrature (pond :251-258, ax=1 semantics in the face frame)
    Frho=zero(nx); Fmn=zero(nx); Fmt=zero(nx); FE=zero(nx)
    @inbounds for k in 1:39
        cxk=cx_arr[k]; cyk=cy_arr[k]; czk=cz_arr[k]; wk=w_arr[k]
        vn = cxk + ufn; vt = cyk + uft; vz = czk
        if vn >= 0.0f0
            fk = _feqT2(rLf, unLf, utLf, TLf, ufn, uft, cxk, cyk, czk, wk); Ts = TLf
        else
            fk = _feqT2(rRf, unRf, utRf, TRf, ufn, uft, cxk, cyk, czk, wk); Ts = TRf
        end
        ke = 0.5f0*(vn*vn + vt*vt + vz*vz) + erc*Ts
        Frho += vn*fk; Fmn += vn*vn*fk; Fmt += vn*vt*fk; FE += vn*ke*fk
    end
    # rotate momentum flux back to the global frame; scale by the face length
    Fmx = nx*Fmn - ny*Fmt; Fmy = ny*Fmn + nx*Fmt
    return Frho*S, Fmx*S, Fmy*S, FE*S
end

"""
The 39-velocity half-range flux of `_fvflux_normal6` (on given face states, no reconstruction) with the lattice scaled to
the face temperature T_f = (T_L + T_R)/2: discrete velocities sqrt(T_f) c_k + u_f, and each state's third-order Hermite
equilibrium taken about T_f (peculiar velocity / sqrt(T_f), reduced temperature T/T_f - 1), as Particles on Demand scales
its lattice with the local temperature. The fixed-reference flux expands about T = 1, so its equilibrium is positive only
for 0.81 < T < 1.67 at zero peculiar velocity; here the bound applies to T/T_f, i.e. to the temperature ratio across one
face (below about 1.47), at any absolute temperature. Equal states give the exact flux at any T. Global-frame velocities
in, (Frho, Fmx, Fmy, FE) S out.
"""
@inline function _fvflux_lattice_tscaled(rL,uxL,uyL,TL, rR,uxR,uyR,TR, nx, ny, S, cx_arr, cy_arr, cz_arr, w_arr, drot)
    CS2=_gas_cs2(typeof(rL))
    unL = uxL*nx + uyL*ny; utL = -uxL*ny + uyL*nx
    unR = uxR*nx + uyR*ny; utR = -uxR*ny + uyR*nx
    rLf = max(rL, 1.0f-4); rRf = max(rR, 1.0f-4); TLf = max(TL, 1.0f-3); TRf = max(TR, 1.0f-3)
    Tf = 0.5f0*(TLf + TRf); sq = sqrt(Tf); isq = 1/sq
    ufn = 0.5f0*(unL + unR); uft = 0.5f0*(utL + utR); erc = 0.5f0*drot*CS2
    Frho=zero(nx); Fmn=zero(nx); Fmt=zero(nx); FE=zero(nx)
    @inbounds for k in 1:39
        cxk=cx_arr[k]; cyk=cy_arr[k]; czk=cz_arr[k]; wk=w_arr[k]
        vn = sq*cxk + ufn; vt = sq*cyk + uft; vz = sq*czk
        if vn >= 0.0f0
            fk = _feqT2(rLf, unL*isq, utL*isq, TLf/Tf, ufn*isq, uft*isq, cxk, cyk, czk, wk); Ts = TLf
        else
            fk = _feqT2(rRf, unR*isq, utR*isq, TRf/Tf, ufn*isq, uft*isq, cxk, cyk, czk, wk); Ts = TRf
        end
        ke = 0.5f0*(vn*vn + vt*vt + vz*vz) + erc*Ts
        Frho += vn*fk; Fmn += vn*vn*fk; Fmt += vn*vt*fk; FE += vn*ke*fk
    end
    Fmx = nx*Fmn - ny*Fmt; Fmy = ny*Fmn + nx*Fmt
    return Frho*S, Fmx*S, Fmy*S, FE*S
end

# Development switch (environment LUDWIG_KINETIC_FLUX at load) for the face flux of the airfoil reconstruction:
#   convected  `_fvflux_convected` instead of `_fvflux_normal6`;
#   lowmach    `_fvflux_normal6` on the face states of `_lowmach_face_states`;
#   efm        `_fvflux_efm` (closed-form half-range Maxwellian, no lattice) instead of `_fvflux_normal6`;
#   efm_lowmach `_fvflux_efm` on the face states of `_lowmach_face_states`;
#   lowmach_shear `_fvflux_normal6` on the face states of `_lowmach_shear_face_states` (tangential jump scaled by the
#              face-normal Mach number, with the floor LUDWIG_KINETIC_SHEAR_FLOOR times the local Mach number; floor 1
#              is `lowmach`, floor 0 leaves no shear dissipation on faces along the flow and diverged on the airfoil);
#   roe        `_fvflux_roe` (Roe's approximate Riemann solver), a non-kinetic reference for flux comparisons;
#   ausmup     `_fvflux_ausmup` (AUSM+-up of Liou 2006, reference Mach number LUDWIG_AUSM_MINF), likewise;
#   roe_shear  `_fvflux_roe` with the shear floor LUDWIG_KINETIC_SHEAR_FLOOR on its shear wave (`_shear_floor_speed`:
#              on a face along the flow the same shear dissipation as `lowmach_shear` with that floor);
#   roe_lowmach_shear  as roe_shear, on face states whose normal velocity jump is scaled by the local Mach number
#              (`_lowmach_normal_face_states`, the normal half of `_lowmach_shear_face_states`): both of
#              `lowmach_shear`'s corrections on Roe's flux;
#   ausmup_shear  `_fvflux_ausmup` with the same shear floor on its tangential momentum flux (AUSM+-up keeps its own
#              low-Mach scaling, LUDWIG_AUSM_MINF);
#   roe_lowmach_shear_entropy  as roe_lowmach_shear, with the entropy wave floored at LUDWIG_ROE_ENTROPY_FLOOR (default 1)
#              times the speed at which the 39-velocity flux diffuses an entropy jump on a face with no normal velocity:
#              Roe's flux with all the dissipation `lowmach_shear` keeps on faces along the flow;
#   ausmup_shear_entropy  ausmup_shear with the same entropy floor (also LUDWIG_ROE_ENTROPY_FLOOR);
#   lowmach_shear_tscaled  `_fvflux_lattice_tscaled` (the 39-velocity flux with its lattice scaled to the face
#              temperature) on the face states of `_lowmach_shear_face_states`: `lowmach_shear`
#              without its fixed temperature window.
# Unset: the half-range flux as before. A constant, so the default code is unchanged.
const KINETIC_FLUX_VARIANT = (v = get(ENV, "LUDWIG_KINETIC_FLUX", "");
    v == "convected" ? 1 : v == "lowmach" ? 2 : v == "efm" ? 3 : v == "efm_lowmach" ? 4 : v == "lowmach_shear" ? 5 :
    v == "roe" ? 6 : v == "ausmup" ? 7 : v == "roe_shear" ? 8 : v == "roe_lowmach_shear" ? 9 : v == "ausmup_shear" ? 10 :
    v == "roe_lowmach_shear_entropy" ? 11 : v == "ausmup_shear_entropy" ? 12 : v == "lowmach_shear_tscaled" ? 13 : 0)
const ROE_ENTROPY_FLOOR = clamp(something(tryparse(Float64, get(ENV, "LUDWIG_ROE_ENTROPY_FLOOR", "1.0")), 1.0), 0.0, 2.0)
const KINETIC_SHEAR_FLOOR = clamp(something(tryparse(Float64, get(ENV, "LUDWIG_KINETIC_SHEAR_FLOOR", "0.3")), 0.3), 0.0, 1.0)
# Optional floor per level of the Newton march's mesh sequencing (17 Nov 2026), environment LUDWIG_KINETIC_SHEAR_FLOOR_LEVELS
# at load, one value per level in the order the levels are marched, e.g. "1,1,0.15" for levels (4,2,1): `newton_cold_run`
# and `newton_pair_cold_run` set the floor of each level before building it (`_set_level_shear_floor!`). The coarse levels
# only provide the start of the next one, so they can take a larger floor, which converges them in fewer steps; the finest
# level's value sets the answer. Unset (default): `_kinetic_shear_floor()` folds to the constant above, so the code is
# unchanged. Set, the face flux reads a global Ref: CPU kernels only. Invalid entries are refused at load.
const KINETIC_SHEAR_FLOOR_LEVELS = let v = strip(get(ENV, "LUDWIG_KINETIC_SHEAR_FLOOR_LEVELS", ""))
    isempty(v) ? Float64[] : [(x = parse(Float64, strip(t)); 0 <= x <= 1 ||
        throw(ArgumentError("LUDWIG_KINETIC_SHEAR_FLOOR_LEVELS: each floor must lie in [0, 1], got $x")); x) for t in split(v, ',')]
end
const KINETIC_SHEAR_FLOOR_SCHEDULED = !isempty(KINETIC_SHEAR_FLOOR_LEVELS)
const KINETIC_SHEAR_FLOOR_NOW = Ref(KINETIC_SHEAR_FLOOR)
@inline _kinetic_shear_floor() = KINETIC_SHEAR_FLOOR_SCHEDULED ? KINETIC_SHEAR_FLOOR_NOW[] : KINETIC_SHEAR_FLOOR
"Set the shear floor of level number `k` of `nlevels` from LUDWIG_KINETIC_SHEAR_FLOOR_LEVELS; no-op when it is unset."
function _set_level_shear_floor!(k::Integer, nlevels::Integer)
    KINETIC_SHEAR_FLOOR_SCHEDULED || return KINETIC_SHEAR_FLOOR
    length(KINETIC_SHEAR_FLOOR_LEVELS) == nlevels || throw(ArgumentError(
        "LUDWIG_KINETIC_SHEAR_FLOOR_LEVELS has $(length(KINETIC_SHEAR_FLOOR_LEVELS)) values for $nlevels levels"))
    KINETIC_SHEAR_FLOOR_NOW[] = KINETIC_SHEAR_FLOOR_LEVELS[k]
end
# Squared reference (freestream) Mach number of the `ausmup` variant's low-Mach scaling, from LUDWIG_AUSM_MINF. Unset
# or invalid: 0, which `_fvflux_mix_face` takes as acoustic scaling on every face (fa = 1, no low-Mach scaling).
const AUSM_MINF2 = (m = something(tryparse(Float64, get(ENV, "LUDWIG_AUSM_MINF", "")), 0.0); isfinite(m) && m > 0 ? m*m : 0.0)

# Development switch (environment LUDWIG_SST_VARIANT at load) for the implicit SST of the Newton march
# (cmesh_steady_newton_sst_ka.jl, cmesh_steady_newton_pair_ka.jl) and the explicit plain-SST sweep of
# cmesh_solver_ka.jl, which the steady diagnostics use to evaluate the turbulence residuals:
#   tmr1994  the NASA Turbulence Modeling Resource "SST" (Menter 1994, as CFL3D with ikoprod=1): production
#            nu_t S^2, production limiter 20 beta* k omega, eddy-viscosity limiter a1 k / max(a1 omega, Omega F2).
# Unset: production nu_t S Omega, limiter 10 beta* k omega, eddy-viscosity limiter with the strain S (as before).
const SST_VARIANT = get(ENV, "LUDWIG_SST_VARIANT", "") == "tmr1994" ? 1 : 0

# erf to about 1e-7 (Abramowitz and Stegun 7.1.26), branch-light so it compiles in kernels. In the split flux below
# W+ + W- = 1 and the exponential terms cancel identically, so the error changes only the dissipation, never the
# consistency (equal states give the exact flux).
@inline function _erf_as(x)
    a = abs(x); t = 1/(1 + oftype(a, 0.3275911)*a)
    y = 1 - t*(oftype(a, 0.254829592) + t*(oftype(a, -0.284496736) + t*(oftype(a, 1.421413741) +
        t*(oftype(a, -1.453152027) + t*oftype(a, 1.061405429)))))*exp(-a*a)
    copysign(y, x)
end

"""
Equilibrium flux method (Pullin 1980; the kinetic flux-vector splitting of Deshpande): the half-range moments of a
continuous Maxwellian of each face state, in closed form, with 3 translational and `drot` rotational degrees of
freedom (R = CS2, so p = rho CS2 T, as in the lattice flux). Global-frame velocities in, (Frho, Fmx, Fmy, FE) S out.
The lattice flux `_fvflux_normal6` approximates the same integrals by the 39-velocity quadrature in a frame moving
with the face; this one needs neither.
"""
@inline function _fvflux_efm(rL,uxL,uyL,TL, rR,uxR,uyR,TR, nx, ny, S, drot)
    CS2=_gas_cs2(typeof(rL))
    unL = uxL*nx + uyL*ny; utL = -uxL*ny + uyL*nx
    unR = uxR*nx + uyR*ny; utR = -uxR*ny + uyR*nx
    rLf = max(rL, 1.0f-4); rRf = max(rR, 1.0f-4); TLf = max(TL, 1.0f-3); TRf = max(TR, 1.0f-3)
    eint = 1.5f0 + 0.5f0*drot                      # internal-plus-translational energy per unit (CS2 T)
    sqpi = sqrt(oftype(rL, pi))
    # one side: s = +1 left (molecules moving along +n), -1 right
    @inline function half(r, un, ut, T, sgn)
        th = CS2*T; sb = sqrt(1/(2*th)); q = un*sb
        W = (1 + sgn*_erf_as(q))/2
        D = sgn*exp(-q*q)/(2*sqpi*sb)
        p = r*th; rE = r*(0.5f0*(un*un + ut*ut) + eint*th)
        Fm = r*(un*W + D)
        (Fm, (r*un*un + p)*W + r*un*D, ut*Fm, (rE + p)*un*W + (rE + 0.5f0*p)*D)
    end
    a = half(rLf, unL, utL, TLf, 1); b = half(rRf, unR, utR, TRf, -1)
    Frho = a[1] + b[1]; Fmn = a[2] + b[2]; Fmt = a[3] + b[3]; FE = a[4] + b[4]
    Fmx = nx*Fmn - ny*Fmt; Fmy = ny*Fmn + nx*Fmt
    return Frho*S, Fmx*S, Fmy*S, FE*S
end

"""
Roe's approximate Riemann solver (Roe 1981) for the same perfect gas as the kinetic fluxes (p = rho CS2 T, gamma - 1 =
1/(3/2 + drot/2)), a non-kinetic reference for flux comparisons: the four waves of the Roe-averaged state, with Harten's
entropy fix of width 0.1 times the averaged sound speed on the two acoustic waves only. The entropy and shear waves keep
|u_n| with no fix, so a shear jump across a face along the flow is not diffused. Global-frame velocities in,
(Frho, Fmx, Fmy, FE) S out; equal states give the exact flux, and faces supersonic from one side give that side's flux.
With a shear floor `cf` > 0 (variants roe_shear, roe_lowmach_shear) the shear wave's speed is sqrt(u_n^2 + s^2) with
s = `_shear_floor_speed`; the entropy wave keeps |u_n|. With an entropy floor `ce` > 0 (variant roe_lowmach_shear_entropy)
the entropy wave's speed is sqrt(u_n^2 + (ce k_e c)^2), k_e = (103/540) sqrt(3/(2 gamma)) = 0.197 at gamma 1.4: with
ce = 1 the speed at which the 39-velocity half-range flux diffuses an entropy jump (density and temperature at equal
pressure) across a face with no normal velocity (the sum over its velocities with c_x > 0 of w c_x (1 - (|c|^2/cs2 - 3)/2)
is 103/1080 at T = 1), which `lowmach_shear` leaves unscaled. `cf` = `ce` = 0 (the defaults) is the unchanged flux.
"""
@inline function _fvflux_roe(rL,uxL,uyL,TL, rR,uxR,uyR,TR, nx, ny, S, drot, cf=0, ce=0)
    CS2=_gas_cs2(typeof(rL))
    eint = 1.5f0 + 0.5f0*drot; gm1 = 1/eint                       # gamma - 1
    unL = uxL*nx + uyL*ny; utL = -uxL*ny + uyL*nx
    unR = uxR*nx + uyR*ny; utR = -uxR*ny + uyR*nx
    rLf = max(rL, 1.0f-4); rRf = max(rR, 1.0f-4); TLf = max(TL, 1.0f-3); TRf = max(TR, 1.0f-3)
    pL = rLf*CS2*TLf; pR = rRf*CS2*TRf
    HL = (eint + 1)*CS2*TLf + 0.5f0*(unL*unL + utL*utL); HR = (eint + 1)*CS2*TRf + 0.5f0*(unR*unR + utR*utR)
    sL = sqrt(rLf); sR = sqrt(rRf); ws = 1/(sL + sR)
    un = (sL*unL + sR*unR)*ws; ut = (sL*utL + sR*utR)*ws; H = (sL*HL + sR*HR)*ws; rr = sL*sR
    q2 = un*un + ut*ut
    c2 = max(gm1*(H - 0.5f0*q2), oftype(H, 1e-12)); c = sqrt(c2)
    dp = pR - pL; dun = unR - unL
    a1 = (dp - rr*c*dun)/(2*c2); a2 = (rRf - rLf) - dp/c2; a3 = rr*(utR - utL); a4 = (dp + rr*c*dun)/(2*c2)
    del = oftype(c, 0.1)*c
    l1 = abs(un - c); l1 = l1 < del ? (l1*l1 + del*del)/(2*del) : l1
    l4 = abs(un + c); l4 = l4 < del ? (l4*l4 + del*del)/(2*del) : l4
    l2 = abs(un)
    l3 = l2
    if cf > 0
        sf = _shear_floor_speed(cf, c, gm1, unL, utL, unR, utR)
        l3 = sqrt(un*un + sf*sf)
    end
    if ce > 0
        se = oftype(c, ce)*(oftype(c, 103)/540)*sqrt(3/(2*(1 + gm1)))*c
        l2 = sqrt(un*un + se*se)
    end
    d1 = l1*a1; d2 = l2*a2; d3 = l3*a3; d4 = l4*a4
    Dr = d1 + d2 + d4
    Dn = d1*(un - c) + d2*un + d4*(un + c)
    Dt = Dr*ut + d3
    DE = d1*(H - un*c) + d2*0.5f0*q2 + d3*ut + d4*(H + un*c)
    mL = rLf*unL; mR = rRf*unR
    Frho = 0.5f0*(mL + mR - Dr)
    Fmn  = 0.5f0*(mL*unL + pL + mR*unR + pR - Dn)
    Fmt  = 0.5f0*(mL*utL + mR*utR - Dt)
    FE   = 0.5f0*(mL*HL + mR*HR - DE)
    Fmx = nx*Fmn - ny*Fmt; Fmy = ny*Fmn + nx*Fmt
    return Frho*S, Fmx*S, Fmy*S, FE*S
end

"""
AUSM+-up (Liou, J. Comput. Phys. 214, 2006) for the same perfect gas, a non-kinetic reference for flux comparisons: the
split of `_fvflux_mix_face` (Kp 0.25, Ku 0.75, sigma 1, beta 1/8, the low-Mach scaling with the squared reference Mach
number `minf2`, by default LUDWIG_AUSM_MINF squared) on the pressure p = rho CS2 T and the sound speed sqrt(gamma CS2 T)
of each face state. Global-frame velocities in, (Frho, Fmx, Fmy, FE) S out.
AUSM+-up convects the tangential velocity with the mass flux from its upwind side, a shear dissipation |mdot| that
vanishes on faces along the flow. With a shear floor `cf` > 0 (variant ausmup_shear) that coefficient becomes
sqrt(mdot^2 + (rho_m s)^2), s = `_shear_floor_speed` and rho_m the mean face density, and the energy flux takes the
work of the added shear stress at the mean tangential velocity (as the shear wave of Roe's flux). With an entropy floor
`ce` > 0 (variant ausmup_shear_entropy) an entropy jump a2 = drho - dp/a^2 is also diffused as by a wave of speed
sqrt(u^2 + (ce k_e a)^2), u = mdot/rho_m, k_e as in `_fvflux_roe`: only the excess over AUSM+-up's own upwind dissipation
|u| is added, along the entropy eigenvector (1, u_n, u_t, |u|^2/2) at the mean face velocity. `cf` = `ce` = 0 (the
defaults) is the unchanged flux.
"""
@inline function _fvflux_ausmup(rL,uxL,uyL,TL, rR,uxR,uyR,TR, nx, ny, S, drot, minf2=AUSM_MINF2, cf=0, ce=0)
    CS2=_gas_cs2(typeof(rL))
    eint = 1.5f0 + 0.5f0*drot
    unL = uxL*nx + uyL*ny; utL = -uxL*ny + uyL*nx
    unR = uxR*nx + uyR*ny; utR = -uxR*ny + uyR*nx
    rLf = max(rL, 1.0f-4); rRf = max(rR, 1.0f-4); TLf = max(TL, 1.0f-3); TRf = max(TR, 1.0f-3)
    gcs2 = (1 + 1/eint)*CS2
    F = _fvflux_mix_face(rLf, unL, utL, TLf, rLf*CS2*TLf, rRf, unR, utR, TRf, rRf*CS2*TRf,
                         sqrt(gcs2*TLf), sqrt(gcs2*TRf), eint*CS2, oftype(rL, minf2), nx, ny, S)
    (cf > 0 || ce > 0) || return F
    af = 0.5f0*(sqrt(gcs2*TLf) + sqrt(gcs2*TRf))
    rm = 0.5f0*(rLf + rRf); mdot = F[1]/S
    dFr = zero(af); dFn = zero(af); dFt = zero(af); dFE = zero(af)
    if cf > 0
        sf = _shear_floor_speed(cf, af, 1/eint, unL, utL, unR, utR)
        dFt = -0.5f0*(sqrt(mdot*mdot + (rm*sf)*(rm*sf)) - abs(mdot))*(utR - utL)   # added tangential momentum flux
        dFE = 0.5f0*(utL + utR)*dFt
    end
    if ce > 0
        un = mdot/rm; unm = 0.5f0*(unL + unR); utm = 0.5f0*(utL + utR)
        se = oftype(af, ce)*(oftype(af, 103)/540)*sqrt(3/(2*(1 + 1/eint)))*af
        a2 = (rRf - rLf) - (rRf*CS2*TRf - rLf*CS2*TLf)/(af*af)
        d2 = -0.5f0*(sqrt(un*un + se*se) - abs(un))*a2                               # added entropy-wave flux
        dFr += d2; dFn += d2*unm; dFt += d2*utm; dFE += d2*0.5f0*(unm*unm + utm*utm)
    end
    return (F[1] + dFr*S, F[2] + (nx*dFn - ny*dFt)*S, F[3] + (ny*dFn + nx*dFt)*S, F[4] + dFE*S)
end

"""
The speed of the shear floor of the Roe and AUSM+-up variants: s = k cf min(c, max(|u_L|, |u_R|)), with c a face sound
speed, gm1 = gamma - 1 and k = (5/18) sqrt(6/gamma) (0.575 at gamma 1.4). On a face with no normal velocity the
39-velocity half-range flux `_fvflux_normal6` at T = 1 diffuses a tangential velocity jump as a shear wave of speed k c
(the sum over its velocities with c_x > 0 of w c_x c_y^2 / cs2 is 5/18; the continuous half-range Maxwellian gives
sqrt(2/(pi gamma)) c = 0.674 c, a ratio of (5/18) sqrt(3 pi) = 0.853), and `lowmach_shear` scales that jump by about
cf z, z = min(1, max M_L, M_R). A shear wave of speed s therefore gives Roe's and AUSM+-up's fluxes the shear dissipation
of `lowmach_shear` on faces along the flow, and |u_n| where the flow crosses the face.
"""
@inline function _shear_floor_speed(cf, c, gm1, unL, utL, unR, utR)
    q = sqrt(max(unL*unL + utL*utL, unR*unR + utR*utR))
    oftype(c, cf)*(oftype(c, 5)/18)*sqrt(6/(1 + gm1))*min(c, q)
end

"""
The two face states (rho, ux, uy, T) with the jump of the face-normal velocity scaled by z = min(1, max(M_L, M_R))
about its mean, the tangential velocity, density and temperature unchanged: the normal half of
`_lowmach_shear_face_states` (Thornber et al. 2008), used with Roe's flux in the variant roe_lowmach_shear, whose shear
wave carries the floor instead of a scaled tangential jump.
"""
@inline function _lowmach_normal_face_states(l, r, drot, nx, ny)
    CS2=_gas_cs2(typeof(l[1]))
    gcs2 = (1 + 1/(1.5f0 + 0.5f0*drot))*CS2
    m2 = max((l[2]*l[2] + l[3]*l[3])/(gcs2*l[4]), (r[2]*r[2] + r[3]*r[3])/(gcs2*r[4]))
    z = min(one(m2), sqrt(m2 + oftype(m2, 1.0f-12)))
    unL = l[2]*nx + l[3]*ny; unR = r[2]*nx + r[3]*ny
    utL = -l[2]*ny + l[3]*nx; utR = -r[2]*ny + r[3]*nx
    unm = 0.5f0*(unL+unR); dn = 0.5f0*z*(unR-unL); unl = unm-dn; unr = unm+dn
    (l[1], unl*nx - utL*ny, unl*ny + utL*nx, l[4]), (r[1], unr*nx - utR*ny, unr*ny + utR*nx, r[4])
end

"""
The two face states (rho, ux, uy, T) with their velocity difference scaled by the local Mach number,
z = min(1, max(M_L, M_R)), about their mean (Thornber et al., J. Comput. Phys. 227, 2008). The half-range kinetic
flux diffuses a velocity difference across a face at the thermal speed whatever the flow speed; with the scaled
difference that diffusion is proportional to the flow speed, vanishes toward a wall, and is unchanged at and above
Mach 1. Density and temperature are not changed.
"""
@inline function _lowmach_face_states(l, r, drot)
    CS2=_gas_cs2(typeof(l[1]))
    gcs2 = (1 + 1/(1.5f0 + 0.5f0*drot))*CS2                 # gamma cs2: the sound speed squared is this times T
    m2 = max((l[2]*l[2] + l[3]*l[3])/(gcs2*l[4]), (r[2]*r[2] + r[3]*r[3])/(gcs2*r[4]))
    z = min(one(m2), sqrt(m2 + oftype(m2, 1.0f-12)))
    um = 0.5f0*(l[2]+r[2]); vm = 0.5f0*(l[3]+r[3]); du = 0.5f0*z*(r[2]-l[2]); dv = 0.5f0*z*(r[3]-l[3])
    (l[1], um-du, vm-dv, l[4]), (r[1], um+du, vm+dv, r[4])
end

"""
As `_lowmach_face_states`, but the jump of the velocity component along the face (unit normal (nx, ny)) is scaled by
the face-normal Mach number with a floor, zt = min(z, sqrt(max(Mn_L^2, Mn_R^2) + (c z)^2 + 1e-6)), c = the floor
(unless given, `KINETIC_SHEAR_FLOOR`, or the current level's value of LUDWIG_KINETIC_SHEAR_FLOOR_LEVELS), while the
normal component keeps z. With `_lowmach_face_states` the shear of a boundary layer is still diffused at about 0.68 of the flow speed |u| across the wall-parallel faces; here at about
0.68 max(|u_n|, c |u|) (the shear-wave dissipation of a Roe flux is |u_n| with an entropy-fix floor). c = 1 is
`_lowmach_face_states`; c = 0 left the Newton linear solves at their cap and the residuals growing on NACA 0012 at
15 deg (15 Nov 2026). Both components are unchanged where the face-normal Mach number is at least 1; the normal one is
unchanged wherever the flow is at or above Mach 1, while the tangential jump of a supersonic flow along the face is
still scaled (as the Roe shear wave, whose speed is |u_n| at any Mach number). Density and temperature are not changed;
the means of both velocity components are kept, so equal states are returned unchanged.
"""
@inline function _lowmach_shear_face_states(l, r, drot, nx, ny, c=_kinetic_shear_floor())
    CS2=_gas_cs2(typeof(l[1]))
    gcs2 = (1 + 1/(1.5f0 + 0.5f0*drot))*CS2
    m2 = max((l[2]*l[2] + l[3]*l[3])/(gcs2*l[4]), (r[2]*r[2] + r[3]*r[3])/(gcs2*r[4]))
    z = min(one(m2), sqrt(m2 + oftype(m2, 1.0f-12)))
    unL = l[2]*nx + l[3]*ny; unR = r[2]*nx + r[3]*ny
    mn2 = max(unL*unL/(gcs2*l[4]), unR*unR/(gcs2*r[4]))
    cz = oftype(m2, c)*z
    zt = min(z, sqrt(mn2 + cz*cz + oftype(m2, 1.0f-6)))
    utL = -l[2]*ny + l[3]*nx; utR = -r[2]*ny + r[3]*nx
    unm = 0.5f0*(unL+unR); utm = 0.5f0*(utL+utR); dn = 0.5f0*z*(unR-unL); dt = 0.5f0*zt*(utR-utL)
    unl = unm-dn; unr = unm+dn; utl = utm-dt; utr = utm+dt
    (l[1], unl*nx - utl*ny, unl*ny + utl*nx, l[4]), (r[1], unr*nx - utr*ny, unr*ny + utr*nx, r[4])
end

"""
Kinetic face flux with convected shear and enthalpy, from the two face states (global-frame velocities in,
(Frho, Fmx, Fmy, FE) S out). Mass and normal momentum are the half-range lattice quadrature of `_fvflux_normal6`
in the same face frame. Tangential momentum and total enthalpy are carried by the net mass flux from its upwind
side, the correction of Haenel, Schwane and Seider (1987) to flux-vector splitting: the half-range form diffuses a
shear layer and a temperature layer lying along the face at the thermal speed, whatever the flow speed; here that
diffusion scales with the mass flux through the face and vanishes where the flow is along the face (boundary
layers, wakes). For equal states both forms are the exact flux.
"""
@inline function _fvflux_convected(rL,uxL,uyL,TL, rR,uxR,uyR,TR, nx, ny, S, cx_arr, cy_arr, cz_arr, w_arr, drot)
    CS2=_gas_cs2(typeof(rL))
    unL = uxL*nx + uyL*ny; utL = -uxL*ny + uyL*nx
    unR = uxR*nx + uyR*ny; utR = -uxR*ny + uyR*nx
    rLf = max(rL, 1.0f-4); rRf = max(rR, 1.0f-4); TLf = max(TL, 1.0f-3); TRf = max(TR, 1.0f-3)
    ufn = 0.5f0*(unL+unR); uft = 0.5f0*(utL+utR)
    Frho=zero(nx); Fmn=zero(nx)
    @inbounds for k in 1:39
        cxk=cx_arr[k]; cyk=cy_arr[k]; czk=cz_arr[k]; wk=w_arr[k]
        vn = cxk + ufn
        fk = vn >= 0.0f0 ? _feqT2(rLf, unL, utL, TLf, ufn, uft, cxk, cyk, czk, wk) :
                           _feqT2(rRf, unR, utR, TRf, ufn, uft, cxk, cyk, czk, wk)
        Frho += vn*fk; Fmn += vn*vn*fk
    end
    # total enthalpy per unit mass: translational (3/2 CS2 T) and rotational (drot/2 CS2 T) energy, p / rho = CS2 T
    hT = (1.5f0 + 0.5f0*drot + 1.0f0)*CS2
    left = Frho >= zero(Frho)
    utu = left ? utL : utR
    Hu  = left ? hT*TLf + 0.5f0*(unL*unL + utL*utL) : hT*TRf + 0.5f0*(unR*unR + utR*utR)
    Fmt = Frho*utu; FE = Frho*Hu
    Fmx = nx*Fmn - ny*Fmt; Fmy = ny*Fmn + nx*Fmt
    return Frho*S, Fmx*S, Fmy*S, FE*S
end

# ---- FULL VISCOUS STRESS (A2c) -----------------------------------------------------------------
# Green-Gauss cell gradient of one scalar from the 4 face-averaged values and the outward metric
# convention of the residual (dc = Fi(i+1)-Fi(i)+Fj(j+1)-Fj(j) => E/N faces +S⃗, W/S faces -S⃗).
# GCL closure (max 2e-7) makes a constant field's gradient vanish identically.
@inline function _gg_grad(phiW, phiE, phiS, phiN, phiC,
                          SxW, SyW, SxE, SyE,
                          SxS, SyS, SxN, SyN, invvol)
    fE = 0.5f0*(phiC+phiE); fW = 0.5f0*(phiW+phiC); fN = 0.5f0*(phiC+phiN); fS = 0.5f0*(phiS+phiC)
    ((fE*SxE - fW*SxW + fN*SxN - fS*SxS)*invvol,
     (fE*SyE - fW*SyW + fN*SyN - fS*SyS)*invvol)
end

# Index-space metric basis used by the SST and EARSM gradient stencils. The
# state stencil includes ghost cells, so every metric increment must span the
# same physical locations as the corresponding state increment.
@inline function _sst_index_metrics(xc, yc, Sjx, Sjy, wdist,
                                    i, j, NI, NJ, NIW, NIF,
                                    corrected_wall_metric::Int32)
    im = max(i - 1, 1); ip = min(i + 1, NI)
    jm = max(j - 1, 1); jp = min(j + 1, NJ)
    xxi = 0.5f0 * (xc[ip,j] - xc[im,j])
    yxi = 0.5f0 * (yc[ip,j] - yc[im,j])
    xet = 0.5f0 * (xc[i,jp] - xc[i,jm])
    yet = 0.5f0 * (yc[i,jp] - yc[i,jm])
    if corrected_wall_metric == Int32(1) && j == 1 && NJ > 1
        if NIW < i <= NIW + NIF
            # The no-slip state ghost is reflected across the foil face. Use
            # its reflected center too; clamping the center at j=1 halves the
            # metric span while the state span still reaches the ghost.
            sx = Sjx[i,1]; sy = Sjy[i,1]
            invs = 1.0f0 / max(sqrt(sx*sx + sy*sy), 1.0f-30)
            twice_d = 2.0f0 * wdist[i,1]
            xghost = xc[i,1] - twice_d * sx * invs
            yghost = yc[i,1] - twice_d * sy * invs
        else
            # A wake-cut ghost is the paired leg's first physical cell.
            partner = NI + 1 - i
            xghost = xc[partner,1]
            yghost = yc[partner,1]
        end
        xet = 0.5f0 * (xc[i,2] - xghost)
        yet = 0.5f0 * (yc[i,2] - yghost)
    end
    return xxi, yxi, xet, yet
end

# Full Navier-Stokes viscous face flux: tau = mu·(grad u + grad uT - 2/3 div·I), q = -k·grad T.
# Face gradient = average of the two adjacent cell Green-Gauss gradients + compact normal-difference
# correction along the center-to-center vector d (kills odd-even decoupling; in the orthogonal-mesh
# limit the normal part reduces EXACTLY to the thin-layer Δφ/dn, so this is a strict superset).
# gk order: (dudx,dudy,dvdx,dvdy,dTdx,dTdy). Returns OUTWARD (Fvx,Fvy,Fve)·S to SUBTRACT.
@inline function _viscous_face_full(g1L,g2L,g3L,g4L,g5L,g6L, g1R,g2R,g3R,g4R,g5R,g6R,
                                    uL,vL,TL, uR,vR,TR, dxc, dyc,
                                    nx, ny, S,
                                    mueff, keff)
    dn2 = dxc*dxc + dyc*dyc + 1.0f-30
    a1 = 0.5f0*(g1L+g1R); a2 = 0.5f0*(g2L+g2R); a3 = 0.5f0*(g3L+g3R)
    a4 = 0.5f0*(g4L+g4R); a5 = 0.5f0*(g5L+g5R); a6 = 0.5f0*(g6L+g6R)
    cu = ((uR-uL) - (a1*dxc + a2*dyc))/dn2
    cv = ((vR-vL) - (a3*dxc + a4*dyc))/dn2
    cT = ((TR-TL) - (a5*dxc + a6*dyc))/dn2
    dudx = a1 + cu*dxc; dudy = a2 + cu*dyc
    dvdx = a3 + cv*dxc; dvdy = a4 + cv*dyc
    dTdx = a5 + cT*dxc; dTdy = a6 + cT*dyc
    dv23 = 0.6666667f0*(dudx + dvdy)                      # Stokes hypothesis (zero bulk viscosity)
    txx = mueff*(2.0f0*dudx - dv23); tyy = mueff*(2.0f0*dvdy - dv23); txy = mueff*(dudy + dvdx)
    Fvx = (txx*nx + txy*ny)*S; Fvy = (txy*nx + tyy*ny)*S
    ufx = 0.5f0*(uL+uR); ufy = 0.5f0*(vL+vR)
    Fve = ufx*Fvx + ufy*Fvy + keff*(dTdx*nx + dTdy*ny)*S
    return Fvx, Fvy, Fve
end

# ---- 3D full Navier-Stokes viscous face flux (F2 2026-07-26: the 3D twin of _viscous_face_full) ------
# The thin-layer 3D viscosity keeps only the FACE-NORMAL derivative; it drops every cross-derivative
# stress. The re-entrant-jet closure that drives sheet->cloud shedding is carried by exactly those
# cross terms (the spanwise shear tau_xz/tau_yz and the in-plane tau_xy). This restores the full stress
# tensor tau = mu(grad u + grad u^T - 2/3 div I), q = -k grad T, from the precomputed Green-Gauss cell
# gradients + a compact center-to-center normal correction (odd-even decoupling killer; reduces EXACTLY
# to the thin-layer normal difference on an orthogonal mesh, so this is a strict superset).
#
# I/J FACES (curvilinear, normal in the x-y plane => nz=0). grad order per cell: velocity (dudx,dudy,
# dudz, dvdx,dvdy,dvdz, dwdx,dwdy,dwdz) then temperature (dTdx,dTdy) — dTdz unused (nz=0). Returns
# (Fvx,Fvy,Fvz,Fve)*S to SUBTRACT from the face momentum/energy flux.
@inline function _visc_face_ij_3d(
        uxL,uyL,uzL, vxL,vyL,vzL, wxL,wyL,wzL, TxL,TyL,
        uxR,uyR,uzR, vxR,vyR,vzR, wxR,wyR,wzR, TxR,TyR,
        uL,vL,wL,TL, uR,vR,wR,TR, dxc, dyc,
        nx, ny, S, mueff, keff)
    dn2 = dxc*dxc + dyc*dyc + 1.0f-30
    a_ux=0.5f0*(uxL+uxR); a_uy=0.5f0*(uyL+uyR); a_uz=0.5f0*(uzL+uzR)
    a_vx=0.5f0*(vxL+vxR); a_vy=0.5f0*(vyL+vyR); a_vz=0.5f0*(vzL+vzR)
    a_wx=0.5f0*(wxL+wxR); a_wy=0.5f0*(wyL+wyR); a_wz=0.5f0*(wzL+wzR)
    a_Tx=0.5f0*(TxL+TxR); a_Ty=0.5f0*(TyL+TyR)
    cu=((uR-uL)-(a_ux*dxc+a_uy*dyc))/dn2; dudx=a_ux+cu*dxc; dudy=a_uy+cu*dyc
    cv=((vR-vL)-(a_vx*dxc+a_vy*dyc))/dn2; dvdx=a_vx+cv*dxc; dvdy=a_vy+cv*dyc
    cw=((wR-wL)-(a_wx*dxc+a_wy*dyc))/dn2; dwdx=a_wx+cw*dxc; dwdy=a_wy+cw*dyc
    cT=((TR-TL)-(a_Tx*dxc+a_Ty*dyc))/dn2; dTdx=a_Tx+cT*dxc; dTdy=a_Ty+cT*dyc
    dudz=a_uz; dvdz=a_vz; dwdz=a_wz                        # z-grads: no in-plane correction (nz=0)
    div23 = 0.6666667f0*(dudx + dvdy + dwdz)
    txx=mueff*(2.0f0*dudx-div23); tyy=mueff*(2.0f0*dvdy-div23)
    txy=mueff*(dudy+dvdx); txz=mueff*(dudz+dwdx); tyz=mueff*(dvdz+dwdy)
    Fvx=(txx*nx+txy*ny)*S; Fvy=(txy*nx+tyy*ny)*S; Fvz=(txz*nx+tyz*ny)*S
    ufx=0.5f0*(uL+uR); ufy=0.5f0*(vL+vR); ufz=0.5f0*(wL+wR)
    Fve=ufx*Fvx+ufy*Fvy+ufz*Fvz+keff*(dTdx*nx+dTdy*ny)*S
    return Fvx, Fvy, Fvz, Fve
end

# K FACES (spanwise, normal = +z, uniform Cartesian => nx=ny=0, center-to-center = (0,0,dz)). The k-flux
# is PER-UNIT-AREA (the update divides by dz, not vol), so NO S factor. grads used per cell: dudx,dvdy
# (for div), dwdx,dwdy (for tau_xz,tau_yz), dudz,dvdz,dwdz (z-shear), dTdz. Returns (Fvx,Fvy,Fvz,Fve).
@inline function _visc_face_k_3d(
        uxL,vyL,wxL,wyL, uzL,vzL,wzL,TzL,
        uxR,vyR,wxR,wyR, uzR,vzR,wzR,TzR,
        uL,vL,wL,TL, uR,vR,wR,TR, dz, mueff, keff)
    dz2 = dz*dz + 1.0f-30
    a_uz=0.5f0*(uzL+uzR); a_vz=0.5f0*(vzL+vzR); a_wz=0.5f0*(wzL+wzR); a_Tz=0.5f0*(TzL+TzR)
    cu=((uR-uL)-a_uz*dz)/dz2; dudz=a_uz+cu*dz
    cv=((vR-vL)-a_vz*dz)/dz2; dvdz=a_vz+cv*dz
    cw=((wR-wL)-a_wz*dz)/dz2; dwdz=a_wz+cw*dz
    cT=((TR-TL)-a_Tz*dz)/dz2; dTdz=a_Tz+cT*dz
    a_dudx=0.5f0*(uxL+uxR); a_dvdy=0.5f0*(vyL+vyR); a_dwdx=0.5f0*(wxL+wxR); a_dwdy=0.5f0*(wyL+wyR)
    div23=0.6666667f0*(a_dudx + a_dvdy + dwdz)
    txz=mueff*(dudz+a_dwdx); tyz=mueff*(dvdz+a_dwdy); tzz=mueff*(2.0f0*dwdz-div23)
    Fvx=txz; Fvy=tyz; Fvz=tzz
    ufx=0.5f0*(uL+uR); ufy=0.5f0*(vL+vR); ufz=0.5f0*(wL+wR)
    Fve=ufx*Fvx+ufy*Fvy+ufz*Fvz+keff*dTdz
    return Fvx, Fvy, Fvz, Fve
end

# Sutherland viscosity factor mu(T)/mu(T_inf) with Sr = S/T_inf (air ambient: 0.404; NOTE cryogenic
# tunnels change Sr strongly — check the test's static temperature). Sr<=0 => 1 (bitwise legacy).
@inline _mufac(Tf, Sr) = Sr <= 0.0f0 ? 1.0f0 : Tf*sqrt(Tf)*(1.0f0+Sr)/(Tf+Sr)

# ---- Wallin-Johansson 2D EARSM (JFM 403, 2000; Menter BSL-EARSM combination) --------------------
# Closed-form 2D solution. Inputs: velocity-gradient tensor components and the turbulence timescale
# tau_t. Outputs: (beta1, beta4, IIS, IIOm) with the anisotropy a = beta1*S* + beta4*(S*Om* - Om*S*),
# S*/Om* the tau_t-normalized symmetric-traceless/antisymmetric parts. Effective eddy viscosity
# nu_t_eff = -(beta1/2)*k*tau_t; the beta4 term is the EXTRA (non-Boussinesq) anisotropy.
# Validated by tools/cmesh_earsm_unittest.jl: the closed form must satisfy the IMPLICIT ARSM
# N*a = -(6/5)S* + (a*Om* - Om**a), N = c1p - (9/4)tr(a S*), for random tensors (self-consistency).
@inline function _wj_earsm_2d(dudx, dudy, dvdx, dvdy, tau_t)
    c1p = 1.8f0                                       # (9/4)(c1-1), c1 = 1.8
    # traceless symmetric strain + rotation, normalized by tau_t
    div2 = 0.5f0*(dudx + dvdy)
    sxx = tau_t*(dudx - div2); syy = tau_t*(dvdy - div2); sxy = tau_t*0.5f0*(dudy + dvdx)
    oxy = tau_t*0.5f0*(dudy - dvdx)                   # Om_xy = -Om_yx
    IIS  = sxx*sxx + syy*syy + 2.0f0*sxy*sxy          # tr(S*S*)
    IIOm = -2.0f0*oxy*oxy                             # tr(Om*Om*) <= 0
    P1 = c1p*(c1p*c1p/27.0f0 + 0.45f0*IIS - (2.0f0/3.0f0)*IIOm)
    P2 = P1*P1 - (c1p*c1p/9.0f0 + 0.9f0*IIS + (2.0f0/3.0f0)*IIOm)^3
    local N
    if P2 >= 0.0f0
        s1 = P1 + sqrt(P2); s2 = P1 - sqrt(P2)
        N = c1p/3.0f0 + cbrt(abs(s1))*sign(s1) + cbrt(abs(s2))*sign(s2)
    else
        pp = sqrt(P1*P1 - P2)
        N = c1p/3.0f0 + 2.0f0*pp^(1.0f0/3.0f0)*cos((1.0f0/3.0f0)*acos(clamp(P1/pp, -1.0f0, 1.0f0)))
    end
    den = N*N - 2.0f0*IIOm
    den = max(den, 1.0f-10)
    b1 = -1.2f0*N/den                                 # beta1 = -(6/5) N / (N^2 - 2 IIOm)
    b4 = -1.2f0/den                                   # beta4 = -(6/5)   / (N^2 - 2 IIOm)
    return b1, b4, sxx, syy, sxy, oxy
end

# PG point-vortex far-field velocity increment (the _vortex_pg form, device-inlinable) — used by the
# far-field SPONGE so its target carries the circulation correction (blending to raw freestream would
# fight the interior circulation and pump the very waves the sponge exists to absorb).
@inline function _pg_vortex_uv(xcc, ycc, gam, xv, yv,
                               m2, bet, ca, sa)
    gam == 0.0f0 && return (0.0f0, 0.0f0)
    dx = xcc - xv; dy = ycc - yv
    dxs = dx*ca + dy*sa; dys = -dx*sa + dy*ca
    r2 = dxs*dxs + dys*dys + 1.0f-6
    Fc = bet/(1.0f0 - m2*(dys*dys/r2))
    c = gam*Fc/(6.2831853f0*r2)
    dus = c*dys; dvs = -c*dxs
    (dus*ca - dvs*sa, dus*sa + dvs*ca)
end

# 4-state compatibility wrapper (existing callers/gates): minmod path, LLL/RRR duplicated (unused).
@inline _fvflux_normal(rLL,uxLL,uyLL,TLL, rL,uxL,uyL,TL, rR,uxR,uyR,TR, rRR,uxRR,uyRR,TRR,
                       nx, ny, S, cx_arr, cy_arr, cz_arr, w_arr, drot) =
    _fvflux_normal6(rLL,uxLL,uyLL,TLL, rLL,uxLL,uyLL,TLL, rL,uxL,uyL,TL, rR,uxR,uyR,TR,
                    rRR,uxRR,uyRR,TRR, rRR,uxRR,uyRR,TRR, nx, ny, S, cx_arr, cy_arr, cz_arr, w_arr, drot, Int32(0))


# AUSM+-up-style mixture flux for cav>=4: reconstruct rho,u,T and represented
# mixture pressure rho*CS2*T+pexc, then split mass flux and interface pressure
# using the supplied mixture sound speeds. recon=1 is the WENO/minmod blend;
# all other existing recon integers take minmod. The checked direct-pressure
# species path supplies its admitted states through _fvflux_mix_pc below.
#
# At equal reconstructed face pressures and zero face velocities, the flux has zero
# mass transfer. This face identity does not establish contact preservation by
# the coupled flow/composition march. Shared species fluxes and EOS projection
# at every stage are tested separately; see the September 8 contact/port audits.
# Finite-grid acoustic diffusion and phase error also require independent tests.
# The auxiliary energy flux supports the carrier's primitive recovery; it is
# not a validated thermodynamic mixture-energy or latent-heat model.
@inline _ausm_M1p(M) = 0.5f0*(M + abs(M))
@inline _ausm_M1m(M) = 0.5f0*(M - abs(M))
@inline _ausm_M2p(M) = 0.25f0*(M + 1.0f0)*(M + 1.0f0)
@inline _ausm_M2m(M) = -0.25f0*(M - 1.0f0)*(M - 1.0f0)
@inline _ausm_M4p(M) = abs(M) >= 1.0f0 ? _ausm_M1p(M) : _ausm_M2p(M)*(1.0f0 - 2.0f0*_ausm_M2m(M))   # beta = 1/8
@inline _ausm_M4m(M) = abs(M) >= 1.0f0 ? _ausm_M1m(M) : _ausm_M2m(M)*(1.0f0 + 2.0f0*_ausm_M2p(M))
@inline _ausm_P5p(M, alp) = abs(M) >= 1.0f0 ? _ausm_M1p(M)/M : _ausm_M2p(M)*((2.0f0 - M) - 16.0f0*alp*M*_ausm_M2m(M))
@inline _ausm_P5m(M, alp) = abs(M) >= 1.0f0 ? _ausm_M1m(M)/M : _ausm_M2m(M)*((-2.0f0 - M) + 16.0f0*alp*M*_ausm_M2p(M))

@inline function _fvflux_mix6(rLLL,uxLLL,uyLLL,TLLL,pLLL, rLL,uxLL,uyLL,TLL,pLL, rL,uxL,uyL,TL,pL,
                              rR,uxR,uyR,TR,pR, rRR,uxRR,uyRR,TRR,pRR, rRRR,uxRRR,uyRRR,TRRR,pRRR,
                              amL, amR, eint, minf2,
                              nx, ny, S, recon::Int32)
    # The reference Mach squared must be finite and nonnegative. Propagate invalid
    # configuration through a nonfinite flux; device code must not throw exceptions.
    if !isfinite(minf2) || minf2 < 0.0f0
        return (NaN32, NaN32, NaN32, NaN32)
    end
    unLLL = uxLLL*nx + uyLLL*ny; utLLL = -uxLLL*ny + uyLLL*nx
    unLL  = uxLL*nx + uyLL*ny;   utLL  = -uxLL*ny + uyLL*nx
    unL   = uxL *nx + uyL *ny;   utL   = -uxL *ny + uyL *nx
    unR   = uxR *nx + uyR *ny;   utR   = -uxR *ny + uyR *nx
    unRR  = uxRR*nx + uyRR*ny;   utRR  = -uxRR*ny + uyRR*nx
    unRRR = uxRRR*nx + uyRRR*ny; utRRR = -uxRRR*ny + uyRRR*nx
    local rLf, rRf, unLf, unRf, utLf, utRf
    local TLf, TRf, pLf, pRf
    if recon == Int32(1)
        # shock/contact sensor on the MIXTURE pressure and the density (same ramp as _fvflux_normal6)
        spj = abs(pR-pL)/max(min(pL,pR), 1.0f-12)
        sj  = max(spj, abs(rR-rL)/min(rL,rR))
        sb  = sj <= 0.1f0 ? 0.0f0 : (sj >= 0.4f0 ? 1.0f0 : (sj-0.1f0)*(1.0f0/0.3f0))
        rLf = max(_wenoblend(sb, rLLL,rLL,rL,rR,rRR), 1.0f-4);  rRf = max(_wenoblend(sb, rRRR,rRR,rR,rL,rLL), 1.0f-4)
        unLf = _wenoblend(sb, unLLL,unLL,unL,unR,unRR);          unRf = _wenoblend(sb, unRRR,unRR,unR,unL,unLL)
        utLf = _wenoblend(sb, utLLL,utLL,utL,utR,utRR);          utRf = _wenoblend(sb, utRRR,utRR,utR,utL,utLL)
        TLf = max(_wenoblend(sb, TLLL,TLL,TL,TR,TRR), 1.0f-3);   TRf = max(_wenoblend(sb, TRRR,TRR,TR,TL,TLL), 1.0f-3)
        pLf = max(_wenoblend(sb, pLLL,pLL,pL,pR,pRR), 1.0f-8);   pRf = max(_wenoblend(sb, pRRR,pRR,pR,pL,pLL), 1.0f-8)
    else
        rLf = max(_faceL_mm(rLL, rL, rR), 1.0f-4);  rRf = max(_faceR_mm(rL, rR, rRR), 1.0f-4)
        unLf = _faceL_mm(unLL, unL, unR);           unRf = _faceR_mm(unL, unR, unRR)
        utLf = _faceL_mm(utLL, utL, utR);           utRf = _faceR_mm(utL, utR, utRR)
        TLf = max(_faceL_mm(TLL, TL, TR), 1.0f-3);  TRf = max(_faceR_mm(TL, TR, TRR), 1.0f-3)
        pLf = max(_faceL_mm(pLL, pL, pR), 1.0f-8);  pRf = max(_faceR_mm(pL, pR, pRR), 1.0f-8)
    end
    _fvflux_mix_face(rLf,unLf,utLf,TLf,pLf,rRf,unRf,utRf,TRf,pRf,
                     amL,amR,eint,minf2,nx,ny,S)
end

# Shared AUSM split. Velocities are already rotated into this face's normal/
# tangent basis. The caller supplies reconstructed or admitted constant states.
# Keep this arithmetic identical for legacy reconstruction and direct pressure.
@inline function _fvflux_mix_face(rLf,unLf,utLf,TLf,pLf,rRf,unRf,utRf,TRf,pRf,
                                  amL,amR,eint,minf2,
                                  nx,ny,S)
    af  = 0.5f0*(amL + amR)
    Mb2 = 0.5f0*(unLf*unLf + unRf*unRf)/(af*af)
    # Exactly zero means no convective reference scale: use acoustic scaling on
    # every face (fa=1), including after a pressure jump induces local velocity.
    # This is a separate zero-reference convention, not the minf2 -> 0+ limit.
    # Positive references retain the original low-Mach expression and arithmetic.
    Mo2 = minf2 == 0.0f0 ? 1.0f0 : min(1.0f0, max(Mb2, minf2)); Mo = sqrt(Mo2)
    fa  = Mo*(2.0f0 - Mo)
    ML = unLf/af; MR = unRf/af
    rf = 0.5f0*(rLf + rRf)
    # Kp = 0.25, sigma = 1: pressure-diffusion in the interface Mach number
    Mh = _ausm_M4p(ML) + _ausm_M4m(MR) - (0.25f0/fa)*max(1.0f0 - Mb2, 0.0f0)*(pRf - pLf)/(rf*af*af)
    mdot = af*Mh*(Mh > 0.0f0 ? rLf : rRf)
    alp = 0.1875f0*(-4.0f0 + 5.0f0*fa*fa)
    Pp = _ausm_P5p(ML, alp); Pm = _ausm_P5m(MR, alp)
    # Ku = 0.75: velocity-diffusion in the interface pressure
    pf = Pp*pLf + Pm*pRf - 0.75f0*Pp*Pm*(rLf + rRf)*(fa*af)*(unRf - unLf)
    # upwinded convected quantities
    local unU, utU, HU
    if mdot > 0.0f0
        unU = unLf; utU = utLf; HU = eint*TLf + 0.5f0*(unLf*unLf + utLf*utLf) + pLf/rLf
    else
        unU = unRf; utU = utRf; HU = eint*TRf + 0.5f0*(unRf*unRf + utRf*utRf) + pRf/rRf
    end
    Frho = mdot
    Fmn  = mdot*unU + pf
    Fmt  = mdot*utU
    FE   = mdot*HU
    Fmx = nx*Fmn - ny*Fmt; Fmy = ny*Fmn + nx*Fmt
    return Frho*S, Fmx*S, Fmy*S, FE*S
end

# Constant admitted mixture states in Cartesian velocity components. This entry
# performs no density/pressure/temperature floor or reconstruction. The species
# stage owns EOS admission; malformed face inputs still produce nonfinite fluxes.
@inline function _fvflux_mix_pc(rL,uL,vL,TL,pL,rR,uR,vR,TR,pR,
                                amL,amR,eint,minf2,
                                nx,ny,S)
    if !(isfinite(rL) && rL>0.0f0 && isfinite(rR) && rR>0.0f0 &&
         isfinite(pL) && pL>0.0f0 && isfinite(pR) && pR>0.0f0 &&
         isfinite(TL) && TL>0.0f0 && isfinite(TR) && TR>0.0f0 &&
         isfinite(amL) && amL>0.0f0 && isfinite(amR) && amR>0.0f0 &&
         isfinite(uL) && isfinite(vL) && isfinite(uR) && isfinite(vR) &&
         isfinite(eint) && isfinite(minf2) && minf2>=0.0f0 &&
         isfinite(nx) && isfinite(ny) && isfinite(S) && S>0.0f0)
        return (NaN32,NaN32,NaN32,NaN32)
    end
    unLf=uL*nx+vL*ny;utLf=-uL*ny+vL*nx
    unRf=uR*nx+vR*ny;utRf=-uR*ny+vR*nx
    _fvflux_mix_face(rL,unLf,utLf,TL,pL,rR,unRf,utRf,TR,pR,
                     amL,amR,eint,minf2,nx,ny,S)
end

# ---- Common-gauge kinetic (KFVS) face for the direct-pressure Species path (2026-09-10) ----------
# The kinetic alternative to the AUSM split above, for the same admitted constant
# face states. Both sides use the T*=1 terms of the cubic Hermite equilibrium
# `_feqT2` before its population floor; T*=1 is a common NUMERICAL face gauge, not
# a physical temperature, so the half moments are linear in each side's density on
# a fixed-velocity contact. Physical EOS pressure enters directly through the
# discrete pressure weights B+/B- (no thermal pedestal is formed or subtracted).
# Vapor is transported by the same half streams (`Fq = Y_L*M+ + Y_R*M-`, formed by
# the species kernels from the returned half masses), which preserves the
# constant-pressure phase-volume identity `v_l*(F-Fq)+v_v*Fq = u_n*S` up to
# Float32 rounding. Negative or underflowed raw populations and invalid pressure
# weights are REFUSED (nonfinite return), never clipped, because clipping breaks
# the moment identities silently. The last return value is the operational face
# clock rate (speed*S): fixed D3Q39 abscissae reach |c_x| = 3, plus the mean face
# velocity, the density-contrast acoustic allowance and the velocity sensitivity of
# both pressure weights. It is an operational bound, not a nonlinear stability proof.
# Population arithmetic is byte-identical to the checked external V2 candidate
# (S0903_RUNS/species_kfvs_face_candidate_20260909_v2, 1440/1440 fixed CPU checks).
@inline function _feq_gauge1(rho, un, ut, ufn, uft, cxk, cyk, wk)
    px = un - ufn; py = ut - uft
    inv_cs2 = 1.0f0/CS2; inv_2cs4 = 0.5f0/(CS2*CS2); inv_6cs6 = 1.0f0/(6.0f0*CS2*CS2*CS2)
    cp = px*cxk + py*cyk
    p2 = px*px + py*py
    f  = rho
    f += rho*cp*inv_cs2
    f += rho*(cp*cp - CS2*p2)*inv_2cs4
    f += rho*(cp*cp*cp - 3.0f0*CS2*cp*p2)*inv_6cs6
    # The dimensionless polynomial and its derivatives with respect to the normal and
    # tangential velocity offsets feed only the operational clock's sensitivity bound.
    inv_cs4 = 1.0f0/(CS2*CS2); inv_2cs6 = 0.5f0/(CS2*CS2*CS2)
    P  = 1.0f0 + cp*inv_cs2 + (cp*cp - CS2*p2)*inv_2cs4 + (cp*cp*cp - 3.0f0*CS2*cp*p2)*inv_6cs6
    Pn = cxk*inv_cs2 + (cp*cxk - CS2*px)*inv_cs4 + ((cp*cp - CS2*p2)*cxk - 2.0f0*CS2*cp*px)*inv_2cs6
    Pt = cyk*inv_cs2 + (cp*cyk - CS2*py)*inv_cs4 + ((cp*cp - CS2*p2)*cyk - 2.0f0*CS2*cp*py)*inv_2cs6
    return wk*f, P, Pn, Pt
end

const KFVS_CMAX = 3.0f0   # largest D3Q39 abscissa magnitude along any axis (shell 5)

@inline function _fvflux_kfvs_pc(rL,uL,vL,pL,cL, rR,uR,vR,pR,cR,
                                 cx_arr, cy_arr, w_arr, nx, ny, S)
    bad = (NaN32,NaN32,NaN32,NaN32,NaN32,NaN32,NaN32)
    if !(isfinite(rL) && rL>0.0f0 && isfinite(rR) && rR>0.0f0 &&
         isfinite(pL) && pL>0.0f0 && isfinite(pR) && pR>0.0f0 &&
         isfinite(cL) && cL>0.0f0 && isfinite(cR) && cR>0.0f0 &&
         isfinite(uL) && isfinite(vL) && isfinite(uR) && isfinite(vR) &&
         isfinite(nx) && isfinite(ny) && isfinite(S) && S>0.0f0 &&
         abs(nx*nx+ny*ny-1.0f0)<=8.0f0*eps(Float32))
        return bad
    end
    unL=uL*nx+vL*ny; utL=-uL*ny+vL*nx
    unR=uR*nx+vR*ny; utR=-uR*ny+vR*nx
    ufn=0.5f0*(unL+unR); uft=0.5f0*(utL+utR)
    mp=0.0f0; mm=0.0f0; bp=0.0f0; bm=0.0f0; mt=0.0f0; gL=0.0f0; gR=0.0f0
    @inbounds for k in 1:39
        cxk=cx_arr[k]; cyk=cy_arr[k]; wk=w_arr[k]
        fL,PL,PLn,PLt=_feq_gauge1(rL,unL,utL,ufn,uft,cxk,cyk,wk)
        fR,PR,PRn,PRt=_feq_gauge1(rR,unR,utR,ufn,uft,cxk,cyk,wk)
        (isfinite(fL) && isfinite(fR) && fL>0.0f0 && fR>0.0f0) || return bad
        vn=cxk+ufn; vt=cyk+uft
        if vn>=0.0f0
            mp+=vn*fL; bp+=vn*(vn-unL)*fL; mt+=vn*vt*fL
        else
            mm+=vn*fR; bm+=vn*(vn-unR)*fR; mt+=vn*vt*fR
        end
        # 1-norm bound of each pressure weight's derivative with respect to the four
        # side velocities, summed over all 39 abscissae so either side of vn=0 is covered.
        gL+=wk*((abs(vn-unL)+abs(vn))*abs(PL)+abs(vn*(vn-unL))*(abs(PLn)+abs(PLt)))
        gR+=wk*((abs(vn-unR)+abs(vn))*abs(PR)+abs(vn*(vn-unR))*(abs(PRn)+abs(PRt)))
    end
    bp/=rL*CS2; bm/=rR*CS2
    # Algebraic pressure replacement with the actual discrete half moments: the
    # thermal pedestal rho*CS2 is never formed or subtracted from physical p.
    fr=mp+mm; fn=unL*mp+unR*mm+bp*pL+bm*pR
    fmx=(nx*fn-ny*mt)*S; fmy=(ny*fn+nx*mt)*S
    (isfinite(fr) && isfinite(fn) && isfinite(fmx) && isfinite(fmy) &&
     isfinite(mp) && isfinite(mm) && isfinite(bp) && isfinite(bm) &&
     mp>=0.0f0 && mm<=0.0f0 && bp>=0.0f0 && bm>=0.0f0) || return bad
    rmin=min(rL,rR); rmax=max(rL,rR)
    rate=(KFVS_CMAX+abs(ufn)+max(cL,cR)*sqrt(rmax/rmin)+(pL*gL+pR*gR)/(CS2*rmin))*S
    isfinite(rate) && rate>0.0f0 || return bad
    # No auxiliary energy flux: the Species path evolves no independent energy state.
    return (fr*S, fmx, fmy, 0.0f0, mp*S, mm*S, rate)
end
