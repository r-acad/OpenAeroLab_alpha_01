using Test
include(joinpath(@__DIR__,"..","install","gr_runtime.jl"))
module FakePlots
    module GR
        module GRPreferences
            const libpath=Ref(raw"C:\GR first;C:\GR second;C:\GR first")
        end
    end
end
module NoGR end

@testset "GR plugin search path stays local and idempotent" begin
    original=get(ENV,"PATH",nothing)
    try
        ENV["PATH"]=raw"C:\Existing;C:\GR second"
        configure_gr_windows_path!(FakePlots)
        if Sys.iswindows()
            @test ENV["PATH"]==raw"C:\GR first;C:\GR second;C:\Existing"
            once=ENV["PATH"];configure_gr_windows_path!(FakePlots)
            @test ENV["PATH"]==once
            @test_throws ErrorException configure_gr_windows_path!(NoGR)
        else
            @test ENV["PATH"]==raw"C:\Existing;C:\GR second"
        end
    finally
        original===nothing ? delete!(ENV,"PATH") : (ENV["PATH"]=original)
    end
end
