# Run with julia --project=solvers/pbcav_s1030/src/pbcav tools/tests/pbcav_coupled_fgmres.jl
using Test, LinearAlgebra, KernelAbstractions
include(joinpath(@__DIR__, "..", "..", "solvers", "pbcav_s1030", "src", "pbcav", "PBCav.jl"))

function toywork(b; weights=ones(length(b)), restart=4)
    n = length(b)
    (; N=n, restart, dx=zeros(n), T1=zeros(n), T2=zeros(n), Wt=weights, R=b,
       Vb=zeros(n, restart+1), Zb=zeros(n, restart), H=zeros(restart+1, restart),
       g=zeros(restart+1), cs=zeros(restart), sn=zeros(restart), Y=zeros(n))
end

dotfn(cw, x, y; backend) = dot(x, y)
identity_precond!(z, r, cw, w, m, bc, cfg) = copyto!(z, r)
function solve_toy(A, b; maxit=8, weights=ones(length(b)), first_wrong=false)
    cw = toywork(b; weights)
    cfg = (; backend=CPU(), cpl_eta=1e-12, cpl_abs=1e-14, cpl_maxit=maxit, cpl_gs=:mgs)
    count = Ref(0)
    function apply!(y, x, cw, w, m, bc, cfg)
        count[] += 1
        if first_wrong && count[] == 1
            copyto!(y, x) # Emulate a misleading Arnoldi estimate; later products are exact.
        else
            mul!(y, A, x)
        end
        y
    end
    result = PBCav._cpl_fgmres!(cw, nothing, nothing, nothing, cfg, apply!, identity_precond!, dotfn)
    result, cw.dx, norm(weights .* (b - A*cw.dx))
end

@testset "PBCav coupled FGMRES verifies its residual" begin
    A = [4.0 1.0; 1.0 3.0]
    r, x, actual = solve_toy(A, [1.0, 2.0]; weights=[0.5, 2.0])
    @test r.converged
    @test actual <= max(1e-12*r.r0, 1e-14)
    @test r.rest ≈ actual atol=1e-14
    @test x ≈ A \ [1.0, 2.0]

    # The first product makes the recurrence claim convergence immediately. The true
    # residual reveals the bad update, so the method must restart and repair it.
    A = [1.0 0.0; 0.0 2.0]
    r, x, actual = solve_toy(A, [1.0, 1.0]; first_wrong=true)
    @test r.converged && r.iters > 1
    @test actual <= max(1e-12*r.r0, 1e-14)
    @test r.rest ≈ actual atol=1e-14
    r, _, actual = solve_toy(A, [1.0, 1.0]; first_wrong=true, maxit=1)
    @test !r.converged && r.iters == 1
    @test r.rest ≈ actual atol=1e-14
    @test actual > 0.1

    # A zero operator is an Arnoldi breakdown, not convergence or a NaN solution.
    r, x, actual = solve_toy(zeros(2, 2), [1.0, 1.0])
    @test !r.converged && isfinite(r.rest)
    @test r.rest ≈ actual
    @test all(isfinite, x)

    # If the Arnoldi recurrence becomes nonfinite, dx still represents the last
    # verified iterate; report its residual rather than the invalid estimate.
    cw = toywork([1.0, 1.0])
    cfg = (; backend=CPU(), cpl_eta=1e-12, cpl_abs=1e-14, cpl_maxit=4, cpl_gs=:mgs)
    calls = Ref(0)
    function corrupted_dot(cw, x, y; backend)
        calls[] += 1
        calls[] == 2 && (cw.g[1] = Inf)
        dot(x, y)
    end
    identity_apply!(y, x, cw, w, m, bc, cfg) = copyto!(y, x)
    r = PBCav._cpl_fgmres!(cw, nothing, nothing, nothing, cfg,
                           identity_apply!, identity_precond!, corrupted_dot)
    @test !r.converged
    @test r.rest ≈ norm(cw.R - cw.dx)
    @test isfinite(r.rest)
end
