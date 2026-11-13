include(joinpath(@__DIR__,"cmesh_muscl_preconditioner.jl"))

@testset "Frozen component-limiter LS chain uses both neighboring cells" begin
    s,mesh,_=rectangle_case(;scalar_type=Float64)
    affine_state!(s,mesh);s.wall_flux=true;s.nu_lam=0.
    mw=M.set_airfoil_muscl_ka!(s,mesh.X,mesh.Y;limiter=:venkat)
    M._raw_residual_ka!(s,Int32(2),Int32(1))
    rhs=map(copy,M._airfoil_residual(s));saved=map(copy,(s.rho,s.ux,s.uy,s.T))
    w=M.build_jfnk_airfoil_workspace(s;m=8,preconditioner=:kinetic_muscl_ls_fullvisc_block_adi,
        block_scalar_type=Float64)
    fill!(w.line.d0,1.);M._airfoil_kinetic_build!(s,w.kinetic,w.line.d0)
    theta=w.kinetic.muscl_theta;a=mw.geometry;g=s.g
    # Independent reference reconstructs the actual physical LS stencil,
    # keeping the measured component factors fixed during perturbations.
    function frozen_trace(i,j,f,overrides=Dict{Tuple{Int,Int},SVector{4,Float64}}())
        primitive(ii,jj)=haskey(overrides,(ii,jj)) ? M._airfoil_primitive4(overrides[(ii,jj)],s.eint) :
            M._airfoil_air_z(s.rho,s.ux,s.uy,s.T,ii+g,jj+g)
        z=primitive(i,j)
        ntuple(Val(4)) do k
            gradient_x=0.;gradient_y=0.
            for n in 1:4
                ii=a.neighbors[i,j,1,n]-g;jj=a.neighbors[i,j,2,n]-g
                difference=primitive(ii,jj)[k]-z[k]
                gradient_x+=a.weights[i,j,1,n]*difference
                gradient_y+=a.weights[i,j,2,n]*difference
            end
            z[k]+theta[i,j,k]*(gradient_x*a.offsets[i,j,1,f]+gradient_y*a.offsets[i,j,2,f])
        end
    end
    trace_error=maximum(abs(frozen_trace(i,j,f)[k]-mw.states[i,j,k,f])
        for i in 1:s.NI,j in 1:s.NJ,f in 1:4,k in 1:4)
    @test trace_error<2e-14
    @test minimum(theta)>=0&&maximum(theta)<=1
    @test all(isequal(x,y) for (x,y) in zip(rhs,M._airfoil_residual(s)))
    @test all(isequal(x,y) for (x,y) in zip(saved,(s.rho,s.ux,s.uy,s.T)))
    i,j=8,6
    ql=M._airfoil_conserved4(M._airfoil_air_z(s.rho,s.ux,s.uy,s.T,i-1+g,j+g)...,s.eint,Float64)
    qr=M._airfoil_conserved4(M._airfoil_air_z(s.rho,s.ux,s.uy,s.T,i+g,j+g)...,s.eint,Float64)
    zl=frozen_trace(i-1,j,2);zr=frozen_trace(i,j,1)
    sl,nl=M._airfoil_muscl_ls_coefficients(theta,a.weights,a.offsets,i-1,j,2)
    sr,nr=M._airfoil_muscl_ls_coefficients(theta,a.weights,a.offsets,i,j,1)
    @test norm(SVector(nl))>0&&norm(SVector(nr))>0
    nx,ny=s.Six[i,j],s.Siy[i,j];area=hypot(nx,ny)
    function flux(left,right)
        changes=Dict((i-1,j)=>left,(i,j)=>right)
        l=frozen_trace(i-1,j,2,changes);r=frozen_trace(i,j,1,changes)
        SVector(M._airfoil_thermal_face_flux(l,r,nx/area,ny/area,area,s.cxa,s.cya,s.cza,s.wa,s.drot))
    end
    left,right=M._airfoil_muscl_ls_face_jacobians(ql,qr,zl,zr,sl,nl,sr,nr,nx,ny,
        s.cxa,s.cya,s.cza,s.wa,s.drot,s.eint)
    h=1e-6
    for k in 1:4
        vl=SVector{4,Float64}(ntuple(n->n==k ? 1. : 0.,4))
        dl=(flux(ql+h*vl,qr)-flux(ql-h*vl,qr))/(2h)
        dr=(flux(ql,qr+h*vl)-flux(ql,qr-h*vl))/(2h)
        @test isapprox(left[:,k],dl;rtol=5e-5,atol=1e-10)
        @test isapprox(right[:,k],dr;rtol=5e-5,atol=1e-10)
    end
    @test isapprox(M._airfoil_block4(w.kinetic.face_i_left,i,j),left;rtol=2e-12,atol=1e-14)
    @test isapprox(M._airfoil_block4(w.kinetic.face_i_right,i,j),right;rtol=2e-12,atol=1e-14)
    reversed_left,reversed_right=M._airfoil_muscl_ls_face_jacobians(qr,ql,zr,zl,sr,nr,sl,nl,-nx,-ny,
        s.cxa,s.cya,s.cza,s.wa,s.drot,s.eint)
    @test isapprox(reversed_left,-right;rtol=1e-9,atol=1e-10)
    @test isapprox(reversed_right,-left;rtol=1e-9,atol=1e-10)
    result=M.step_jfnk_airfoil_ka!(s,w,1f0;globalization=:pseudo_transient,linear_maxiter=8)
    @test isfinite(result.linear_relative)&&result.linear_iterations<=8
    @test M._airfoil_physical_state(s.rho,s.ux,s.uy,s.T)
    @test_throws ArgumentError M.build_airfoil_kinetic_block_workspace(s;muscl_ls_block=true)
end

const LS_PC_REPORT=joinpath(PRIVATE,"validation","airfoil_mesh_solver_20261112","compressible",
    "muscl_ls_preconditioner_"*Dates.format(now(UTC),"yyyymmdd_HHMMSS"))
mkpath(LS_PC_REPORT)
ls_receipt=Dict("generated_utc"=>string(now(UTC)),"julia_version"=>string(VERSION),"passed"=>true,
    "scope"=>"CPU exact component-limiter replay, independently reconstructed physical-LS derivatives for all4conserved columns on both cells, shared-face orientation, unchanged R0/state and physicality. Nearest face-pair reconstruction derivatives only; wider LS stencil, limiter and full-viscous gradient derivatives remain omitted. No physical airfoil convergence or GPU qualification.",
    "source_sha256"=>Dict(p=>bytes2hex(open(sha256,joinpath(ROOT,p))) for p in
        ("src/compressible/cmesh_steady_kinetic_blocks_ka.jl","src/compressible/cmesh_steady_jfnk_ka.jl",
         "src/compressible/cmesh_geometric_reconstruction_ka.jl","tools/tests/cmesh_muscl_ls_preconditioner.jl")))
open(io->TOML.print(io,ls_receipt;sorted=true),joinpath(LS_PC_REPORT,"result.toml"),"w")
println("MUSCL_LS_PC_REPORT ",LS_PC_REPORT)
