include(joinpath(@__DIR__,"cmesh_geometric_muscl.jl"))

@testset "Float64 thermal state, geometry, affine traces and uniform residual" begin
    s,mesh,met=rectangle_case(;scalar_type=Float64)
    @test eltype(s.rho)===eltype(s.Fi_r)===eltype(s.vol)===Float64
    @test typeof(s.eint)===typeof(s.uxi)===typeof(s.nu_lam)===Float64
    @test s.eint==.5*(3+s.drot)*(2/3)
    @test abs(sum(s.wa)-1)<2e-15
    coeff=affine_state!(s,mesh)
    w=M.set_airfoil_muscl_ka!(s,mesh.X,mesh.Y)
    M._airfoil_muscl_prepare!(s,w)
    @test eltype(w.states)===eltype(w.geometry.weights)===Float64
    errors=zeros(4)
    for j in 1:s.NJ,i in 1:s.NI,f in 1:4
        ii,jj=f==1 ? (i,j) : f==2 ? (i+1,j) : f==3 ? (i,j) : (i,j+1)
        fx=f<=2 ? (mesh.X[ii,j]+mesh.X[ii,j+1])/2 : (mesh.X[i,jj]+mesh.X[i+1,jj])/2
        fy=f<=2 ? (mesh.Y[ii,j]+mesh.Y[ii,j+1])/2 : (mesh.Y[i,jj]+mesh.Y[i+1,jj])/2
        for k in 1:4
            z,a,b=coeff[k];errors[k]=max(errors[k],abs(w.states[i,j,k,f]-(z+a*fx+b*fy)))
        end
    end
    @test maximum(errors)<2e-13
    M.set_freestream_ka!(s,.3,4.);s.wall_flux=false
    M.refresh_raw_residual_ka!(s;wallmode=Int32(0))
    errors64=[maximum(abs.(a)./s.vol) for a in (s.dc_r,s.dc_mx,s.dc_my,s.dc_e)]
    @test maximum(errors64)<1e-9
    @test all(isfinite,errors64)
    println("Float64 constant-state volume residual maxima: ",errors64)
    # Float32 metrics or rounded quadrature cannot masquerade as Float64 input.
    cx,cy,cz,qw,_,_=build_d3q39_static(Float64)
    met32=CMeshMetrics.build_metrics(mesh.X,mesh.Y,0,s.NI)
    @test_throws ArgumentError M.build_sol_ka(CPU(),met32;cxv=cx,cyv=cy,czv=cz,wv=qw,scalar_type=Float64)
    @test_throws ArgumentError M.build_sol_ka(CPU(),met;cxv=cx,cyv=cy,czv=cz,wv=Float32.(qw),scalar_type=Float64)
end

@testset "Float64 finite-difference Jacobian and native/Krylov storage" begin
    s,mesh,_=rectangle_case(;scalar_type=Float64)
    M.set_airfoil_muscl_ka!(s,mesh.X,mesh.Y;limiter=:venkat)
    s.wall_flux=true
    w=M.build_jfnk_airfoil_workspace(s;m=8)
    for (dst,src) in zip((w.line.rho0,w.line.ux0,w.line.uy0,w.line.T0),(s.rho,s.ux,s.uy,s.T))
        copyto!(dst,src)
    end
    M._raw_residual_ka!(s,Int32(2),Int32(1))
    M._av_copy!(w.base_residual,M._airfoil_residual(s));fill!(w.line.d0,1.)
    for k in 1:4,j in 1:s.NJ,i in 1:s.NI
        w.delta[k][i,j]=sin(.17i+.23j+.11k)
    end
    h=M._airfoil_fd_step(w.delta,M._av_norm(w.delta),length(s.vol),1.,.3,2.,1f0)
    @test h isa Float64&&h<1e-6
    M._airfoil_jacobian_vec!(w.work,w.delta,s,w;fd_scale=1f0)
    a=map(copy,w.work)
    M._airfoil_jacobian_vec!(w.work,w.delta,s,w;fd_scale=.5f0)
    difference=sqrt(sum(sum(abs2,a[k].-w.work[k]) for k in 1:4))/M._av_norm(a)
    @test isfinite(difference)&&difference<1e-4
    @test s.rho==w.line.rho0&&s.ux==w.line.ux0&&s.T==w.line.T0
    kw=M.build_krylov_airfoil_workspace(s,w)
    @test eltype(kw.rhs)===Float64
    @test eltype(kw.jacobian)===Float64
    result=M.step_jfnk_airfoil_ka!(s,w,1f0;globalization=:pseudo_transient,linear_maxiter=12)
    @test result.linear_iterations<=12&&isfinite(result.linear_relative)
    @test M._airfoil_physical_state(s.rho,s.ux,s.uy,s.T)
    println("Float64 directional Jv scale sensitivity: ",difference,"; linear residual: ",result.linear_relative)
end

@testset "Float64 SST equations and prescribed omega constraints" begin
    s,mesh,_=rectangle_case(;scalar_type=Float64)
    s.wall_flux=true;s.nu_lam=hypot(s.uxi,s.uyi)/3e6
    M.set_sst_ka!(s);M.set_airfoil_muscl_ka!(s,mesh.X,mesh.Y)
    M.applyBC_ka!(s,Int32(2));M.prepare_dt_ka!(s,.45f0);M.sst_step_ka!(s)
    saved=map(copy,(s.tk,s.tom,s.nut,s.dtf));clock=s.last_global_dt
    d=M.steady_airfoil_diagnostics_ka(s)
    @test d.all_six_equation_residuals_available
    @test all(isfinite,(d.turb_k_equation_l2,d.turb_omega_equation_l2))
    @test d.turb_omega_wall_constraint_relative_linf<=1e-12
    @test d.turb_omega_wall_constraint_cells==2s.NIF
    @test all(isequal(a,b) for (a,b) in zip(saved,(s.tk,s.tom,s.nut,s.dtf)))
    @test s.last_global_dt==clock
end

@testset "Optional full Float64 kinetic block preconditioner" begin
    s,mesh,_=rectangle_case(;scalar_type=Float64)
    s.wall_flux=true;s.nu_lam=1e-5
    M.set_airfoil_muscl_ka!(s,mesh.X,mesh.Y;limiter=:venkat)
    w=M.build_jfnk_airfoil_workspace(s;m=8,preconditioner=:kinetic_fullvisc_block_adi,
        block_scalar_type=Float64)
    @test eltype(w.kinetic.diagonal)===eltype(w.kinetic.cp)===Float64
    @test eltype(M._airfoil_block4(w.kinetic.diagonal,1,1))===Float64
    ql=M._airfoil_conserved4(1.,.2,.03,1.,s.eint,Float64)
    qr=M._airfoil_conserved4(.9,.15,.02,.95,s.eint,Float64)
    @test eltype(ql)===Float64
    a,b=M._airfoil_kinetic_face_jacobians(ql,qr,.06,.08,s.cxa,s.cya,s.cza,s.wa,s.drot,s.eint)
    @test eltype(a)===eltype(b)===Float64
    vl=SVector(.02,-.01,.015,.03);vr=SVector(-.01,.02,.01,-.02);h=1e-6
    reference=(M._airfoil_kinetic_flux2(ql+h*vl,qr+h*vr,.06,.08,s.cxa,s.cya,s.cza,s.wa,s.drot,s.eint)-
        M._airfoil_kinetic_flux2(ql-h*vl,qr-h*vr,.06,.08,s.cxa,s.cya,s.cza,s.wa,s.drot,s.eint))/(2h)
    @test norm(a*vl+b*vr-reference)<=3e-5*norm(reference)
    result=M.step_jfnk_airfoil_ka!(s,w,1f0;globalization=:pseudo_transient,linear_maxiter=12)
    @test isfinite(result.linear_relative)&&result.linear_iterations<=12
    @test M._airfoil_physical_state(s.rho,s.ux,s.uy,s.T)
    @test_throws ArgumentError M.build_jfnk_airfoil_workspace(s;block_scalar_type=Float16)
end

@testset "Actual Float64 Krylov solve reports its true manufactured-system residual" begin
    s,mesh,_=rectangle_case(;scalar_type=Float64)
    M.set_airfoil_muscl_ka!(s,mesh.X,mesh.Y;limiter=:venkat)
    s.wall_flux=true
    w=M.build_jfnk_airfoil_workspace(s;m=8)
    for (dst,src) in zip((w.line.rho0,w.line.ux0,w.line.uy0,w.line.T0),(s.rho,s.ux,s.uy,s.T))
        copyto!(dst,src)
    end
    M._raw_residual_ka!(s,Int32(2),Int32(1))
    M._av_copy!(w.base_residual,M._airfoil_residual(s))
    # A deliberately diagonal-dominant frozen system isolates actual Float64
    # packing/operator/preconditioner/GMRES execution from PC quality.
    @. w.line.d0=1e4*s.vol
    fill!(w.line.lamS,0.);fill!(w.line.lamN,0.)
    known=ntuple(k->[.01sin(.17i+.23j+.11k) for i in 1:s.NI,j in 1:s.NJ],4)
    M._airfoil_jacobian_vec!(w.work,known,s,w)
    M._av_copy!(w.rhs,w.work)
    kw=M.build_krylov_airfoil_workspace(s,w)
    saved=map(copy,(s.rho,s.ux,s.uy,s.T))
    iterations,relative=M._airfoil_krylov_gmres!(s,w,kw;linear_tol=1e-5,linear_maxiter=24)
    @test 0<iterations<=24&&relative<1e-5
    @test eltype(kw.solver.x)===Float64
    mul!(kw.residual,kw.jacobian,kw.solver.x)
    explicit=norm(kw.rhs-kw.residual)/norm(kw.rhs)
    @test isapprox(relative,explicit;rtol=1e-9,atol=1e-12)
    error=M._av_norm(ntuple(k->w.delta[k].-known[k],4))/M._av_norm(known)
    @test error<2e-5
    @test all(isequal(a,b) for (a,b) in zip(saved,(s.rho,s.ux,s.uy,s.T)))
end
