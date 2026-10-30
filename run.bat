@echo off
if exist "%~dp0SOLVER_HALT" (
    echo SOLVER_LAUNCHER_HALTED 1>&2
    exit /b 87
)
REM ==============================================================================
REM Optimized launcher for LUDWIG on Windows
REM ==============================================================================
REM Flags rationale (added 2026-05-28 as part of C4 perf wins):
REM   --check-bounds=no   : ~10-30% speedup on tight inner loops (Newton solver,
REM                          population stream-collide). Bounds checks add 1-2
REM                          extra ops per array access; in a kernel doing 39
REM                          population reads + 39 writes per cell per step that
REM                          adds up to 100s of ops/cell saved.
REM   -O3                 : Aggressive LLVM inlining. Helps cross-`@noinline`
REM                          callees (esp. the 13-mom Newton solver chain).
REM   -t auto             : Thread-auto for CPU-side work (slice rendering,
REM                          AMR-seam diagnostic post-processing, IO).
REM
REM Default launcher (run.bat) — full module precompilation check on every
REM launch. Safe, correct, slow (~6-15 min JIT on first batch).
REM
REM Usage:
REM   run.bat                                 (runs cases_to_run.yaml, full check)
REM   run.bat src/PREPOST/freestream_gate_check.jl CASES/013_freestream_AMR_gate
REM                                           (runs the gate check)
REM
REM Falls back to src/main.jl when no argument is supplied.
REM ==============================================================================
if "%~1"=="" (
    if not exist "%~dp0cases_to_run.yaml" (
        echo Create cases_to_run.yaml from cases_to_run.example.yaml and add a configured case. 1>&2
        exit /b 2
    )
    julia --project=src --check-bounds=no -O3 -t auto src/main.jl
) else (
    julia --project=src --check-bounds=no -O3 -t auto %*
)
