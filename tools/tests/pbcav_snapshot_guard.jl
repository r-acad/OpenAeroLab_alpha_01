using Test

module SnapshotGuardTest
using SHA, TOML, Printf
module AirfoilGPULock
const LOCK_DIR = "test only"
end
include(joinpath(@__DIR__, "..", "..", "apps", "airfoil_web", "src", "lowspeed", "lowspeed_backend.jl"))
end

const S = SnapshotGuardTest
@testset "Airfoil Lab verifies the vendored PBCav revision" begin
    withenv("AIRFOIL_LOWSPEED_PIN"=>nothing, "AIRFOIL_LOWSPEED_SNAPSHOT"=>nothing) do
        snapshot, revision = S._lowspeed_snapshot_check()
        @test snapshot == S.LOWSPEED_VENDOR_SNAPSHOT
        @test startswith(revision, S.LOWSPEED_COMMIT_PINNED)
        @test occursin("+local-fgmres-true-residual-", revision)
        @test S.lowspeed_capabilities()["revision"] == revision

        # A throwaway worktree-shaped copy verifies that a changed source file
        # remains rejected; the actual checked-in snapshot is never changed.
        workspace_temp = normpath(joinpath(@__DIR__, "..", "..", "..",
                                 "02_PRIVATE_PROJECT_DEVELOPMENT", "TEMP_DELETEABLE"))
        test_parent = isdir(workspace_temp) ? workspace_temp : tempdir()
        mktempdir(test_parent; prefix="pbcav_guard_test_") do dir
            copy = joinpath(dir, "snapshot")
            cp(snapshot, copy)
            withenv("AIRFOIL_LOWSPEED_SNAPSHOT"=>copy) do
                capabilities = S.lowspeed_capabilities()
                @test capabilities["commit"] == "external snapshot"
                @test startswith(capabilities["revision"], "external snapshot")
            end
            manifest = S.TOML.parsefile(joinpath(@__DIR__, "..", "..", "apps", "airfoil_web", "manifests", "lowspeed_snapshot_manifest_s1030.toml"))
            # This copied vendor tree omits historical tests intentionally; use
            # the same vendored-file policy as the actual app guard above.
            @test isempty(S._lowspeed_snapshot_mismatches(copy, manifest; vendored=true))
            open(joinpath(copy, "src", "pbcav", "linsolve_coupled.jl"), "a") do io
                write(io, "\n# deliberately changed for guard test\n")
            end
            @test S._lowspeed_snapshot_mismatches(copy, manifest; vendored=true) ==
                  ["src/pbcav/linsolve_coupled.jl"]
        end
    end
end
