# Experimental Float64 transcription of the production Species AUSM+-up split.
# No energy equation, phase clipping, modified sound speed, or production route.
export AUSMUpFlux, ausm_up_flux, ausm_face_speed

"""All-speed face with a recorded physical convective reference Mach number.

The reference is a flux scale, not a replacement acoustic wave speed. Exactly
zero selects the production acoustic convention fa=1. A positive low-Mach
reference requires the much stricter actual-face explicit diffusion clock.
"""
struct AUSMUpFlux
    reference_mach_squared::Float64
    function AUSMUpFlux(reference_mach_squared::Real)
        isfinite(reference_mach_squared) && reference_mach_squared>=0 ||
            throw(ArgumentError("AUSM reference Mach squared must be finite and nonnegative"))
        new(Float64(reference_mach_squared))
    end
end

@inline _fs_M1p(M)=.5(M+abs(M))
@inline _fs_M1m(M)=.5(M-abs(M))
@inline _fs_M2p(M)=.25(M+1)^2
@inline _fs_M2m(M)=-.25(M-1)^2
@inline _fs_M4p(M)=abs(M)>=1 ? _fs_M1p(M) : _fs_M2p(M)*(1-2_fs_M2m(M))
@inline _fs_M4m(M)=abs(M)>=1 ? _fs_M1m(M) : _fs_M2m(M)*(1+2_fs_M2p(M))
@inline _fs_P5p(M,a)=abs(M)>=1 ? _fs_M1p(M)/M : _fs_M2p(M)*((2-M)-16a*M*_fs_M2m(M))
@inline _fs_P5m(M,a)=abs(M)>=1 ? _fs_M1m(M)/M : _fs_M2m(M)*((-2-M)+16a*M*_fs_M2p(M))

@inline function _fs_ausm_parameters(qL,qR,nx,ny,Mref2)
    unL=qL.u*nx+qL.v*ny;unR=qR.u*nx+qR.v*ny
    af=.5(qL.c+qR.c);rf=.5(qL.rho+qR.rho)
    Mb2=.5(unL^2+unR^2)/af^2
    Mo2=Mref2==0 ? 1. : min(1.,max(Mb2,Mref2))
    Mo=sqrt(Mo2);fa=Mo*(2-Mo);a=.1875(-4+5fa^2)
    (;unL,unR,af,rf,Mb2,fa,Pp=_fs_P5p(unL/af,a),Pm=_fs_P5m(unR/af,a))
end

function ausm_up_flux(left,right,nx::Real,ny::Real,model;reference_mach_squared::Real=0.)
    AUSMUpFlux(reference_mach_squared)(left,right,nx,ny,model)
end

function (flux::AUSMUpFlux)(left,right,nx::Real,ny::Real,model)
    isfinite(nx) && isfinite(ny) && abs(hypot(nx,ny)-1)<=1e-10 ||
        throw(ArgumentError("free-surface AUSM requires a finite unit normal"))
    qL=primitive(left,model);qR=primitive(right,model)
    a=_fs_ausm_parameters(qL,qR,nx,ny,flux.reference_mach_squared)
    Mh=_fs_M4p(a.unL/a.af)+_fs_M4m(a.unR/a.af)-
        (.25/a.fa)*max(1-a.Mb2,0.)*(qR.p-qL.p)/(a.rf*a.af^2)
    donor=Mh>0 ? left : right;qd=Mh>0 ? qL : qR
    mass=a.af*Mh*qd.rho
    pf=a.Pp*qL.p+a.Pm*qR.p-.75a.Pp*a.Pm*(qL.rho+qR.rho)*a.fa*a.af*(a.unR-a.unL)
    (mass*(donor[1]/qd.rho),mass*qd.u+pf*nx,mass*qd.v+pf*ny,
     mass*(donor[4]/qd.rho),mass*(donor[5]/qd.rho))
end

"""Frozen-coefficient actual-face speed bound used by the Species clock.

This is an operational estimate, not a nonlinear stability proof. It includes
Kp/fa pressure transport, pressure split velocity sensitivity, velocity
diffusion and acoustic density contrast. No artificial acoustic floor is used.
The complete step separately admits every phase and the EOS and retries.
"""
function ausm_face_speed(flux::AUSMUpFlux,left,right,nx,ny,model;cell_density)
    qL=primitive(left,model);qR=primitive(right,model)
    cell_density>0 || throw(DomainError(cell_density,"positive cell density required"))
    a=_fs_ausm_parameters(qL,qR,nx,ny,flux.reference_mach_squared)
    pressure_diffusion=(.25/a.fa)*max(1-a.Mb2,0.)*
        (max(qL.rho,qR.rho)/a.rf)*(max(qL.c^2,qR.c^2)/a.af)
    velocity_diffusion=.75abs(a.Pp*a.Pm)*(qL.rho+qR.rho)*a.fa*a.af/cell_density
    pressure_split=(15/16)*(qL.p+qR.p)/(a.af*cell_density)
    acoustic=max(qL.c,qR.c)*sqrt(max(qL.rho,qR.rho)/min(qL.rho,qR.rho))
    result=max(abs(a.unL),abs(a.unR))+acoustic+pressure_diffusion+velocity_diffusion+pressure_split
    isfinite(result) && result>0 || error("invalid five-phase AUSM face clock")
    result
end
