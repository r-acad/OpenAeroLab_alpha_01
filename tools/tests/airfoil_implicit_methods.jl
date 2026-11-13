using Test
include(joinpath(@__DIR__,"../../apps/airfoil_web/src/backends/rans_acceleration.jl"))
using .RANSAcceleration
@testset "Named Airfoil iteration methods and saved configuration" begin
    for method in RANSAcceleration.METHODS
        cfg=Dict{String,Any}("steps"=>5000,"time_mode"=>"steady")
        configure_compressible!(cfg,Dict("rans_method"=>method))
        @test cfg["rans_method"]==method
        @test cfg["irsj"]==(method=="irsj")
        if implicit_method(method)
            @test cfg["sfd_start"]>cfg["steps"]
            @test cfg["equation_probe"] && !cfg["rk_weight_closure"]
            @test cfg["implicit_recovery_steps"]==0
            @test_throws ArgumentError configure_compressible!(copy(cfg),Dict("rans_method"=>method);coupled=true)
        end
        method=="standard" || @test_throws ArgumentError configure_compressible!(
            Dict{String,Any}("steps"=>5000,"time_mode"=>"physical"),Dict("rans_method"=>method))
    end
    for (key,value) in (("cfd_implicit_cfl",NaN),("cfd_implicit_basis",97),
        ("cfd_implicit_warmup_steps",5000),("cfd_implicit_sst_substeps",0),
        ("cfd_implicit_nonmonotone_window",true),("cfd_implicit_preconditioner","unknown"))
        @test_throws ArgumentError configure_compressible!(Dict{String,Any}("steps"=>5000),
            Dict{String,Any}("rans_method"=>"jfnk_native",key=>value))
    end
    cfg=Dict{String,Any}("steps"=>5000,"irsj"=>true)
    configure_compressible!(cfg,Dict())
    @test cfg["rans_method"]=="irsj" # old study compatibility
end

@testset "Experimental Newton controls preserve defaults and reject invalid budgets" begin
    fresh()=Dict{String,Any}("steps"=>5000,"time_mode"=>"steady")
    defaults=configure_compressible!(fresh(),Dict("rans_method"=>"jfnk_native"))
    @test defaults["implicit_globalization"]=="steady"
    @test defaults["implicit_basis"]==16
    @test defaults["implicit_linear_maxiter"]==16
    @test defaults["implicit_cfl"]==1.
    @test defaults["implicit_max_cfl"]==2.
    @test defaults["implicit_preconditioner"]=="line"
    @test !isempty(defaults["implicit_source_sha256"])
    request=Dict{String,Any}("rans_method"=>"jfnk_native","cfd_implicit_globalization"=>"pseudo_transient",
        "cfd_implicit_cfl"=>20.,"cfd_implicit_max_cfl"=>1e4,"cfd_implicit_basis"=>60,
        "cfd_implicit_linear_maxiter"=>120,"cfd_implicit_linear_tolerance"=>.02,
        "cfd_implicit_physical_fraction"=>.15,"cfd_implicit_preconditioner"=>"kinetic_weno_fullvisc_block_adi")
    selected=configure_compressible!(fresh(),request)
    @test selected["implicit_basis"]==60&&selected["implicit_linear_maxiter"]==120
    @test selected["implicit_max_cfl"]==1e4
    for (key,value) in (("cfd_implicit_globalization","unknown"),("cfd_implicit_basis",60.5),
                        ("cfd_implicit_linear_maxiter",0),("cfd_implicit_linear_maxiter",10001),
                        ("cfd_implicit_linear_maxiter",true),("cfd_implicit_linear_tolerance",NaN),
                        ("cfd_implicit_linear_tolerance",0.),("cfd_implicit_physical_fraction",.51),
                        ("cfd_implicit_max_cfl",10.))
        @test_throws ArgumentError configure_compressible!(fresh(),merge(request,Dict(key=>value)))
    end
    @test_throws ArgumentError configure_compressible!(fresh(),Dict("rans_method"=>"line_implicit",
        "cfd_implicit_globalization"=>"pseudo_transient"))
    @test_throws ArgumentError configure_compressible!(fresh(),Dict("rans_method"=>"jfnk_native","cfd_implicit_cfl"=>20.))
    # Restart storage and total iterations are deliberately independent.
    short=configure_compressible!(fresh(),merge(request,Dict("cfd_implicit_linear_maxiter"=>5)))
    @test short["implicit_basis"]==60&&short["implicit_linear_maxiter"]==5
end

include(joinpath(@__DIR__,"..","..","apps","airfoil_web","src","workers","implicit_methods.jl"))
@testset "Workspace memory admission and numerical option forwarding" begin
    native=implicit_memory_guard(768,343,3,60,"kinetic_weno_fullvisc_block_adi","jfnk_native")
    @test native.estimated_workspace_bytes>695.91*1024^2
    @test native.estimated_workspace_bytes<1024^3
    @test native.workspace_budget_bytes==2*1024^3
    library=implicit_memory_guard(768,343,3,60,"kinetic_weno_fullvisc_block_adi","jfnk_krylov")
    @test library.estimated_workspace_bytes>native.estimated_workspace_bytes
    @test_throws ArgumentError implicit_memory_guard(10000,10000,3,96,"line","jfnk_native")
    @test_throws ArgumentError implicit_memory_guard(768,343,3,60,"line","jfnk_native";available_bytes=100*1024^2)
    @test implicit_memory_guard(32,20,3,8,"line","jfnk_native";available_bytes=1024^3).workspace_budget_bytes==512*1024^2
    mixed=implicit_memory_guard(1548,112,3,96,"kinetic_fullvisc_block_adi","jfnk_native";state_scalar_type=Float64)
    full=implicit_memory_guard(1548,112,3,96,"kinetic_fullvisc_block_adi","jfnk_native";
        state_scalar_type=Float64,block_scalar_type=Float64)
    @test full.estimated_workspace_bytes>mixed.estimated_workspace_bytes
    @test full.estimated_workspace_bytes<2*1024^3
    @test full.state_scalar_type==full.block_scalar_type=="Float64"
    lagged=implicit_memory_guard(1548,112,3,96,"kinetic_fullvisc_block_adi","jfnk_native";
        state_scalar_type=Float64,block_scalar_type=Float64,
        lagged_component_limiter=true,muscl_ls_factors=true)
    @test lagged.estimated_workspace_bytes-full.estimated_workspace_bytes==cld(5*13*sizeof(Float64)*1548*112,4)
    @test lagged.lagged_component_limiter&&lagged.muscl_ls_factors
    @test lagged.estimated_workspace_bytes<lagged.workspace_budget_bytes==2*1024^3
    @test_throws ArgumentError implicit_memory_guard(32,20,3,8,"line","jfnk_native";lagged_component_limiter=true)
    @test_throws ArgumentError implicit_memory_guard(1548,112,3,96,"kinetic_fullvisc_block_adi","jfnk_native";
        state_scalar_type=Float64,block_scalar_type=Float64,lagged_component_limiter=true,muscl_ls_factors=true,
        available_bytes=2lagged.estimated_workspace_bytes-2)
    @test_throws ArgumentError implicit_memory_guard(32,20,3,8,"line","jfnk_native";block_scalar_type=Float16)
    built=Ref{Any}(nothing);stepped=Ref{Any}(nothing)
    fake=(;KA=(;get_backend=x->:cpu,CPU=()->:cpu),
        build_jfnk_airfoil_controller=(s;kwargs...)->(built[]=Dict(kwargs);:controller),
        step_jfnk_rans_ka! = (s,c;kwargs...)->(stepped[]=Dict(kwargs);(;accepted=true)))
    state=(;NI=64,NJ=20,g=3,rho=zeros(Float32,1,1))
    cfg=Dict{String,Any}("rans_method"=>"jfnk_native","implicit_globalization"=>"pseudo_transient",
        "implicit_cfl"=>20.,"implicit_max_cfl"=>1e4,"implicit_basis"=>60,
        "implicit_linear_maxiter"=>120,"implicit_linear_tolerance"=>.02,
        "implicit_physical_fraction"=>.15,"implicit_preconditioner"=>"kinetic_weno_fullvisc_block_adi")
    runtime=_implicit_runtime(fake,state,cfg)
    @test built[][:m]==60&&built[][:globalization]==:pseudo_transient
    @test built[][:max_cfl]==1f4&&built[][:physical_fraction]==.15f0
    @test built[][:preconditioner]==:kinetic_weno_fullvisc_block_adi
    @test haskey(cfg,"implicit_workspace_memory")&&haskey(cfg,"implicit_source_sha256")
    _implicit_step!(fake,state,runtime,cfg)
    @test stepped[][:linear_tol]==.02&&stepped[][:linear_maxiter]==120
    @test stepped[][:corrected_wall_metric]&&!stepped[][:nonorth_diffusion]
    cfg["sst_wall_metric_correction"]=false
    _implicit_step!(fake,state,runtime,cfg)
    @test !stepped[][:corrected_wall_metric]&&!stepped[][:nonorth_diffusion]
end
@testset "Line-implicit SST worker uses the effective metric setting" begin
    settings=Bool[]
    fake=(;step_implicit_line_ka! = (s,w,cfl;kwargs...)->(;accepted=true,merit_before=2.,merit_after=1.),
        prepare_dt_ka! = (s,cfl)->nothing,
        sst_step_ka! = (s;corrected_wall_metric,nonorth_diffusion)->push!(settings,corrected_wall_metric),
        refresh_raw_residual_ka! = s->nothing,_implicit_merit=s->1.)
    runtime=(;controller=nothing,workspace=nothing,cfl=Ref(1f0),max_cfl=2f0)
    _implicit_step!(fake,nothing,runtime,Dict("implicit_sst_substeps"=>2))
    @test settings==[true,true]
    empty!(settings)
    _implicit_step!(fake,nothing,runtime,Dict("implicit_sst_substeps"=>2,"sst_wall_metric_correction"=>false))
    @test settings==[false,false]
end
@testset "Raw SST export availability requires actual finite equation data" begin
    names=_implicit_metric_names()
    @test names[1:11]==("iteration","mass_l2","momentum_x_l2","momentum_y_l2","energy_l2",
        "turb_k_update_defect","turb_omega_update_defect","mass_net_frac","cl","cd","cm")
    @test !_implicit_raw_sst_available(Dict{String,Any}[])
    @test !_implicit_raw_sst_available([Dict("turb_k_update_defect"=>0.0)])
    row=Dict(name=>1.0 for name in names)
    @test _implicit_raw_sst_available([row])
    row["turb_omega_equation_l2"]=NaN
    @test !_implicit_raw_sst_available([row])
    @test "turb_omega_wall_constraint_relative_linf" in names
end
