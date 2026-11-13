using Test
const ROOT=dirname(dirname(@__DIR__))
const SCRATCH=joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(dirname(ROOT),"02_PRIVATE_PROJECT_DEVELOPMENT")),"TEMP_DELETEABLE","airfoil_workflow_component")
mkpath(SCRATCH)
@testset "Actual browser component compilation and minimal DOM behavior" begin
    node=Sys.which("node")
    if node===nothing
        @test_skip "Node is optional and only checks browser application components"
    else
        # The user was informed of this narrow browser-JavaScript check; Julia
        # remains the harness and all solver/data/maintenance tools remain Julia.
        for file in ("app.js","studies.js","workflow.js","workflow_help.js","output_controls.js","research_settings.js","benchmarks.js")
            command=Cmd([node,"--check",joinpath(ROOT,"apps","airfoil_web","static","scripts",file)])
            @test success(command)
        end
        dir=mktempdir(SCRATCH)
        fixture=joinpath(dir,"airfoil_workflow_dom.cjs")
        cp(joinpath(@__DIR__,"fixtures","airfoil_workflow_dom.cjs"),fixture)
        output=read(Cmd([node,fixture,ROOT]),String)
        println(output)
        @test occursin(r"PASS \d+ browser-component assertions",output)
    end
end
