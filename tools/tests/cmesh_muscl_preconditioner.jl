include(joinpath(@__DIR__,"cmesh_geometric_muscl.jl"))

@testset "Frozen primitive-slope MUSCL face derivative" begin
    s,_,_=rectangle_case(;scalar_type=Float64)
    zl=(1.02,.23,.04,1.03);zr=(.97,.19,.02,.96)
    ql=M._airfoil_conserved4(1.,.2,.03,1.,s.eint,Float64)
    qr=M._airfoil_conserved4(.99,.21,.025,.98,s.eint,Float64)
    dl=zl.-M._airfoil_primitive4(ql,s.eint)
    dr=zr.-M._airfoil_primitive4(qr,s.eint)
    function frozen_flux(a,b,nx=.06,ny=.08)
        l=M._airfoil_primitive4(a,s.eint).+dl
        r=M._airfoil_primitive4(b,s.eint).+dr
        M._airfoil_kinetic_flux2(M._airfoil_conserved4(l...,s.eint,Float64),
            M._airfoil_conserved4(r...,s.eint,Float64),nx,ny,
            s.cxa,s.cya,s.cza,s.wa,s.drot,s.eint)
    end
    a,b=M._airfoil_muscl_face_jacobians(ql,qr,zl,zr,.06,.08,
        s.cxa,s.cya,s.cza,s.wa,s.drot,s.eint)
    vl=SVector(.02,-.01,.015,.03);vr=SVector(-.01,.02,.01,-.02);h=1e-6
    reference=(frozen_flux(ql+h*vl,qr+h*vr)-frozen_flux(ql-h*vl,qr-h*vr))/(2h)
    @test norm(a*vl+b*vr-reference)<3e-5*norm(reference)
    @test eltype(a)===eltype(b)===Float64
    # Reversing a shared face swaps the physical cells and reverses its flux.
    ar,br=M._airfoil_muscl_face_jacobians(qr,ql,zr,zl,-.06,-.08,
        s.cxa,s.cya,s.cza,s.wa,s.drot,s.eint)
    @test isapprox(ar,-b;rtol=1e-9,atol=1e-10)
    @test isapprox(br,-a;rtol=1e-9,atol=1e-10)
    q=M._airfoil_primitive4(ql,s.eint)
    @test isapprox(M._airfoil_muscl_trace_chain(ql,q,s.eint),Matrix{Float64}(I,4,4);atol=1e-15)
end

@testset "MUSCL block model leaves the exact residual and physical state intact" begin
    s,mesh,_=rectangle_case(;scalar_type=Float64)
    @test_throws ArgumentError M.build_jfnk_airfoil_workspace(s;preconditioner=:kinetic_muscl_fullvisc_block_adi)
    affine_state!(s,mesh);s.wall_flux=true;s.nu_lam=1e-5
    mw=M.set_airfoil_muscl_ka!(s,mesh.X,mesh.Y;limiter=:venkat)
    M._raw_residual_ka!(s,Int32(2),Int32(1))
    rhs=map(copy,M._airfoil_residual(s));state=map(copy,(s.rho,s.ux,s.uy,s.T))
    w=M.build_jfnk_airfoil_workspace(s;m=8,preconditioner=:kinetic_muscl_fullvisc_block_adi,
        block_scalar_type=Float64)
    fill!(w.line.d0,1.)
    M._airfoil_kinetic_build!(s,w.kinetic,w.line.d0)
    @test w.kinetic.muscl_block&&!w.kinetic.weno_block&&w.kinetic.fullvisc_block
    @test all(all(isfinite,getproperty(w.kinetic,k)) for k in (:face_i_left,:face_i_right,:face_j_left,:face_j_right,:diagonal))
    @test all(isequal(a,b) for (a,b) in zip(rhs,M._airfoil_residual(s)))
    @test all(isequal(a,b) for (a,b) in zip(state,(s.rho,s.ux,s.uy,s.T)))
    # This checks the assembled kernel at an actual limited, skew-grid trace.
    i,j=8,6;g=s.g
    ql=M._airfoil_conserved4(s.rho[i-1+g,j+g],s.ux[i-1+g,j+g],s.uy[i-1+g,j+g],s.T[i-1+g,j+g],s.eint,Float64)
    qr=M._airfoil_conserved4(s.rho[i+g,j+g],s.ux[i+g,j+g],s.uy[i+g,j+g],s.T[i+g,j+g],s.eint,Float64)
    zl=M._airfoil_trace4(mw.states,i-1,j,2);zr=M._airfoil_trace4(mw.states,i,j,1)
    a,b=M._airfoil_muscl_face_jacobians(ql,qr,zl,zr,s.Six[i,j],s.Siy[i,j],s.cxa,s.cya,s.cza,s.wa,s.drot,s.eint)
    # Isolate convection; the full-stress correction is independently tested.
    s.nu_lam=0.;M._airfoil_kinetic_build!(s,w.kinetic,w.line.d0)
    @test isapprox(M._airfoil_block4(w.kinetic.face_i_left,i,j),a;rtol=8eps(Float64),atol=0)
    @test isapprox(M._airfoil_block4(w.kinetic.face_i_right,i,j),b;rtol=8eps(Float64),atol=0)
    M.clear_airfoil_muscl_ka!(s)
    @test_throws ArgumentError M._airfoil_kinetic_build!(s,w.kinetic,w.line.d0)
end

@testset "MUSCL wake block represents one shared physical face" begin
    xs=[.5*(1-cospi(k/24)) for k in 0:24]
    yt(x)=x==1. ? 0. : 5*.12*(.2969sqrt(x)-.126x-.3516x^2+.2843x^3-.1036x^4)
    foil=CMeshGen.AirfoilPts(vcat(reverse(xs),xs[2:end]),vcat(-reverse(yt.(xs)),yt.(xs[2:end])))
    mesh=CMeshGen.generate_cmesh(foil;n_foil=48,n_wake=12,n_normal=20,
        dn1=.003,growth_max=1.8,R_ff=20.,L_wake=20.,smooth_iters=60,bl_freeze=5)
    met=CMeshMetrics.build_metrics(mesh;scalar_type=Float64)
    cx,cy,cz,qw,_,_=build_d3q39_static(Float64)
    s=M.build_sol_ka(CPU(),met;cxv=cx,cyv=cy,czv=cz,wv=qw,scalar_type=Float64)
    M.set_freestream_ka!(s,.3,4.);s.wall_flux=true;s.nu_lam=1e-5;s.fullvisc=Int32(1)
    M.set_airfoil_muscl_ka!(s,mesh.X,mesh.Y;limiter=:venkat)
    M._raw_residual_ka!(s,Int32(2),Int32(1))
    w=M.build_jfnk_airfoil_workspace(s;m=4,preconditioner=:kinetic_muscl_fullvisc_block_adi,block_scalar_type=Float64)
    fill!(w.line.d0,1.);M._airfoil_kinetic_build!(s,w.kinetic,w.line.d0)
    for i in 1:s.NIW
        ip=s.NI+1-i
        @test M._airfoil_block4(w.kinetic.face_j_left,ip,1)==-M._airfoil_block4(w.kinetic.face_j_right,i,1)
        @test M._airfoil_block4(w.kinetic.face_j_right,ip,1)==-M._airfoil_block4(w.kinetic.face_j_left,i,1)
    end
    i=s.NIW+div(s.NIF,2);g=s.g
    pressure=M._airfoil_wall_pressure_jacobian(s.ux[i+g,g+1],s.uy[i+g,g+1],s.Sjx[i,1],s.Sjy[i,1],s.eint,Float64)
    @test M._airfoil_block4(w.kinetic.face_j_right,i,1)==pressure
    @test iszero(M._airfoil_block4(w.kinetic.face_j_left,i,1))
    result=M.step_jfnk_airfoil_ka!(s,w,1f0;globalization=:pseudo_transient,linear_maxiter=8)
    @test isfinite(result.linear_relative)&&result.linear_iterations<=8
    @test M._airfoil_physical_state(s.rho,s.ux,s.uy,s.T)
end

const PC_REPORT=joinpath(PRIVATE,"validation","airfoil_mesh_solver_20261112","compressible",
    "muscl_preconditioner_"*Dates.format(now(UTC),"yyyymmdd_HHMMSS"))
mkpath(PC_REPORT)
receipt=Dict("generated_utc"=>string(now(UTC)),"julia_version"=>string(VERSION),
    "checks"=>82,"passed"=>true,
    "scope"=>"CPU affine/positive thermal MUSCL and frozen primitive-slope face Jacobian, assembled-kernel consistency, exact unchanged residual/state, analytic wall and exact paired wake-block conservation. No physical airfoil convergence or GPU/timing qualification.",
    "source_sha256"=>Dict(p=>bytes2hex(open(sha256,joinpath(ROOT,p))) for p in
        ("src/compressible/cmesh_steady_kinetic_blocks_ka.jl","src/compressible/cmesh_steady_jfnk_ka.jl",
         "src/compressible/cmesh_geometric_reconstruction_ka.jl","tools/tests/cmesh_muscl_preconditioner.jl")))
open(io->TOML.print(io,receipt;sorted=true),joinpath(PC_REPORT,"result.toml"),"w")
println("MUSCL_PC_REPORT ",PC_REPORT)
