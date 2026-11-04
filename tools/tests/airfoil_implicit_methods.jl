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
    for (key,value) in (("cfd_implicit_cfl",NaN),("cfd_implicit_basis",25),
        ("cfd_implicit_warmup_steps",5000),("cfd_implicit_sst_substeps",0),
        ("cfd_implicit_nonmonotone_window",true),("cfd_implicit_preconditioner","unknown"))
        @test_throws ArgumentError configure_compressible!(Dict{String,Any}("steps"=>5000),
            Dict{String,Any}("rans_method"=>"jfnk_native",key=>value))
    end
    cfg=Dict{String,Any}("steps"=>5000,"irsj"=>true)
    configure_compressible!(cfg,Dict())
    @test cfg["rans_method"]=="irsj" # old study compatibility
end
