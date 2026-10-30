using StaticArrays

# Local alias `F` historically mapped to Float32. Now mirrors the global
# LBM_F type alias defined in dependencies.jl (selected at module-load
# time from the LUDWIG_PRECISION env var). This means D3Q19/D3Q27/D3Q39
# weight tables and the active lattice constants (W, CS2_LATTICE, etc.)
# automatically follow the chosen precision.
const F = LBM_F

function build_d3q27_static()
    cx_list = Int32[]
    cy_list = Int32[]
    cz_list = Int32[]
    w_list = F[]

    for z in -1:1, y in -1:1, x in -1:1
        push!(cx_list, x)
        push!(cy_list, y)
        push!(cz_list, z)

        d2 = x^2 + y^2 + z^2
        if d2 == 0
            push!(w_list, F(8/27))
        elseif d2 == 1
            push!(w_list, F(2/27))
        elseif d2 == 2
            push!(w_list, F(1/54))
        elseif d2 == 3
            push!(w_list, F(1/216))
        end
    end

    opp_list = zeros(Int32, 27)
    mirror_y_list = zeros(Int32, 27)

    for i in 1:27
        for j in 1:27
            if cx_list[j] == -cx_list[i] &&
               cy_list[j] == -cy_list[i] &&
               cz_list[j] == -cz_list[i]
                opp_list[i] = j
            end
            if cx_list[j] == cx_list[i] &&
               cy_list[j] == -cy_list[i] &&
               cz_list[j] == cz_list[i]
                mirror_y_list[i] = j
            end
        end
    end

    return (
        SVector{27, Int32}(cx_list),
        SVector{27, Int32}(cy_list),
        SVector{27, Int32}(cz_list),
        SVector{27, F}(w_list),
        SVector{27, Int32}(opp_list),
        SVector{27, Int32}(mirror_y_list)
    )
end

"""
    build_d3q19_static()

Build D3Q19 lattice (the standard incompressible-LBM velocity set). 19
directions: 1 rest + 6 face + 12 edge (NO space-diagonals like D3Q27).

  Shell 0: (0,0,0)               ×1    w = 1/3
  Shell 1: (±1,0,0) perms        ×6    w = 1/18 each
  Shell 2: (±1,±1,0) perms       ×12   w = 1/36 each

cs² = 1/3 (same as D3Q27).

Velocity-set ordering convention (so OPP and MIRROR_Y stay simple):
  index 1: (0,0,0)
  indices 2-7: (±1,0,0), (0,±1,0), (0,0,±1)  in {+x, -x, +y, -y, +z, -z} order
  indices 8-19: 12 edge directions in (d1,d2)∈{(x,y),(x,z),(y,z)} × 4 sign combos

D3Q19 is the standard production choice for athermal incompressible LBM
(PowerFLOW, ProLB, etc.). 30% less memory + compute than D3Q27 at the cost
of weaker 4th-order isotropy. The lost isotropy is recovered for hydrodynamic
purposes by Recursive Regularization (Latt & Chopard 2006, Coreixas 2017).
"""
function build_d3q19_static()
    cx_list = Int32[]
    cy_list = Int32[]
    cz_list = Int32[]
    w_list  = F[]

    # Shell 0: rest
    push!(cx_list, Int32(0)); push!(cy_list, Int32(0)); push!(cz_list, Int32(0))
    push!(w_list, F(1/3))

    # Shell 1: 6 axis-aligned (±1,0,0) permutations
    for dim in 1:3
        for s in (-1, 1)
            c = [Int32(0), Int32(0), Int32(0)]
            c[dim] = Int32(s)
            push!(cx_list, c[1]); push!(cy_list, c[2]); push!(cz_list, c[3])
            push!(w_list, F(1/18))
        end
    end

    # Shell 2: 12 plane-diagonal (±1,±1,0) permutations
    # (xy plane, xz plane, yz plane) × 4 sign combinations each
    for d1 in 1:3, d2 in (d1+1):3
        for s1 in (-1, 1), s2 in (-1, 1)
            c = [Int32(0), Int32(0), Int32(0)]
            c[d1] = Int32(s1)
            c[d2] = Int32(s2)
            push!(cx_list, c[1]); push!(cy_list, c[2]); push!(cz_list, c[3])
            push!(w_list, F(1/36))
        end
    end

    @assert length(cx_list) == 19 "D3Q19 must have exactly 19 directions, got $(length(cx_list))"

    # Verify weight sum = 1
    w_sum = sum(w_list)
    @assert abs(w_sum - 1.0f0) < 1.0f-5 "D3Q19 weights must sum to 1, got $w_sum"

    # Verify isotropy: Σ w_i c_ix² = cs² = 1/3
    cs2_check = sum(w_list[k] * F(cx_list[k])^2 for k in 1:19)
    @assert abs(cs2_check - F(1/3)) < 1.0f-5 "D3Q19 cs² must be 1/3, got $cs2_check"
    cross_check = sum(w_list[k] * F(cx_list[k]) * F(cy_list[k]) for k in 1:19)
    @assert abs(cross_check) < 1.0f-7 "D3Q19 cross-moment Σw*cx*cy must be 0, got $cross_check"

    # Build opposite and mirror_y mappings
    opp_list = zeros(Int32, 19)
    mirror_y_list = zeros(Int32, 19)

    for i in 1:19
        for j in 1:19
            if cx_list[j] == -cx_list[i] &&
               cy_list[j] == -cy_list[i] &&
               cz_list[j] == -cz_list[i]
                opp_list[i] = j
            end
            if cx_list[j] == cx_list[i] &&
               cy_list[j] == -cy_list[i] &&
               cz_list[j] == cz_list[i]
                mirror_y_list[i] = j
            end
        end
    end

    for i in 1:19
        @assert opp_list[i] > 0 "D3Q19: no opposite found for direction $i = ($(cx_list[i]),$(cy_list[i]),$(cz_list[i]))"
        @assert mirror_y_list[i] > 0 "D3Q19: no y-mirror found for direction $i = ($(cx_list[i]),$(cy_list[i]),$(cz_list[i]))"
    end

    return (
        SVector{19, Int32}(cx_list),
        SVector{19, Int32}(cy_list),
        SVector{19, Int32}(cz_list),
        SVector{19, F}(w_list),
        SVector{19, Int32}(opp_list),
        SVector{19, Int32}(mirror_y_list)
    )
end

"""
    build_d3q39_static()

Build D3Q39 Variant B (Frapolli/Chikatamarla/Karlin, Phys. Rev. E 2015) lattice.

6 velocity shells (39 directions total):
  Shell 0: (0,0,0)           ×1    ε=0   w=1/12
  Shell 1: (±1,0,0) perms    ×6    ε=1   w=1/12
  Shell 2: (±1,±1,±1)        ×8    ε=3   w=1/27
  Shell 3: (±2,0,0) perms    ×6    ε=4   w=2/135
  Shell 4: (±2,±2,0) perms   ×12   ε=8   w=1/432
  Shell 5: (±3,0,0) perms    ×6    ε=9   w=1/1620

cs² = 2/3 (not 1/3). Drops D3Q27 edge shell (±1,±1,0), adds shells at ε=4,8,9.
Max |c| = 3, fits within BLOCK_SIZE=8 neighbor table.
"""
function build_d3q39_static()
    cx_list = Int32[]
    cy_list = Int32[]
    cz_list = Int32[]
    w_list  = F[]

    # Shell 0: rest (0,0,0) — 1 direction
    push!(cx_list, Int32(0)); push!(cy_list, Int32(0)); push!(cz_list, Int32(0))
    push!(w_list, F(1/12))

    # Shell 1: (±1,0,0) permutations — 6 directions
    for dim in 1:3
        for s in (-1, 1)
            c = [Int32(0), Int32(0), Int32(0)]
            c[dim] = Int32(s)
            push!(cx_list, c[1]); push!(cy_list, c[2]); push!(cz_list, c[3])
            push!(w_list, F(1/12))
        end
    end

    # Shell 2: (±1,±1,±1) — 8 directions
    for sx in (-1, 1), sy in (-1, 1), sz in (-1, 1)
        push!(cx_list, Int32(sx)); push!(cy_list, Int32(sy)); push!(cz_list, Int32(sz))
        push!(w_list, F(1/27))
    end

    # Shell 3: (±2,0,0) permutations — 6 directions
    for dim in 1:3
        for s in (-1, 1)
            c = [Int32(0), Int32(0), Int32(0)]
            c[dim] = Int32(2*s)
            push!(cx_list, c[1]); push!(cy_list, c[2]); push!(cz_list, c[3])
            push!(w_list, F(2/135))
        end
    end

    # Shell 4: (±2,±2,0) permutations — 12 directions
    # All axis pairs (xy, xz, yz), each with 4 sign combos
    for d1 in 1:3, d2 in (d1+1):3
        for s1 in (-1, 1), s2 in (-1, 1)
            c = [Int32(0), Int32(0), Int32(0)]
            c[d1] = Int32(2*s1)
            c[d2] = Int32(2*s2)
            push!(cx_list, c[1]); push!(cy_list, c[2]); push!(cz_list, c[3])
            push!(w_list, F(1/432))
        end
    end

    # Shell 5: (±3,0,0) permutations — 6 directions
    for dim in 1:3
        for s in (-1, 1)
            c = [Int32(0), Int32(0), Int32(0)]
            c[dim] = Int32(3*s)
            push!(cx_list, c[1]); push!(cy_list, c[2]); push!(cz_list, c[3])
            push!(w_list, F(1/1620))
        end
    end

    @assert length(cx_list) == 39 "D3Q39 must have exactly 39 directions, got $(length(cx_list))"

    # Verify weight sum = 1
    w_sum = sum(w_list)
    @assert abs(w_sum - 1.0f0) < 1.0f-5 "D3Q39 weights must sum to 1, got $w_sum"

    # Verify isotropy: Σ w_i c_ix c_jx = cs² δ_ij  (cs² = 2/3)
    cs2_check = sum(w_list[k] * F(cx_list[k])^2 for k in 1:39)
    @assert abs(cs2_check - F(2/3)) < 1.0f-5 "D3Q39 cs² must be 2/3, got $cs2_check"
    cross_check = sum(w_list[k] * F(cx_list[k]) * F(cy_list[k]) for k in 1:39)
    @assert abs(cross_check) < 1.0f-7 "D3Q39 cross-moment Σw*cx*cy must be 0, got $cross_check"

    # Build opposite and mirror mappings
    opp_list = zeros(Int32, 39)
    mirror_y_list = zeros(Int32, 39)

    for i in 1:39
        for j in 1:39
            if cx_list[j] == -cx_list[i] &&
               cy_list[j] == -cy_list[i] &&
               cz_list[j] == -cz_list[i]
                opp_list[i] = j
            end
            if cx_list[j] == cx_list[i] &&
               cy_list[j] == -cy_list[i] &&
               cz_list[j] == cz_list[i]
                mirror_y_list[i] = j
            end
        end
    end

    # Verify all opposites found
    for i in 1:39
        @assert opp_list[i] > 0 "D3Q39: no opposite found for direction $i = ($(cx_list[i]),$(cy_list[i]),$(cz_list[i]))"
        @assert mirror_y_list[i] > 0 "D3Q39: no y-mirror found for direction $i = ($(cx_list[i]),$(cy_list[i]),$(cz_list[i]))"
    end

    return (
        SVector{39, Int32}(cx_list),
        SVector{39, Int32}(cy_list),
        SVector{39, Int32}(cz_list),
        SVector{39, F}(w_list),
        SVector{39, Int32}(opp_list),
        SVector{39, Int32}(mirror_y_list)
    )
end

# ── Build the active lattice ─────────────────────────────────────────────────
# D3Q27 constants are always built (used for thermal DDF, initial defaults)
const C_X_D3Q27, C_Y_D3Q27, C_Z_D3Q27, W_D3Q27, OPP_D3Q27, MIRROR_Y_D3Q27 = build_d3q27_static()

# Active lattice — defaults to D3Q27, can be overridden at startup for D3Q39
# These are used by kernels and must match the lattice selected in config
global C_X      = C_X_D3Q27
global C_Y      = C_Y_D3Q27
global C_Z      = C_Z_D3Q27
global W         = W_D3Q27
global OPP       = OPP_D3Q27
global MIRROR_Y  = MIRROR_Y_D3Q27

# Speed of sound in lattice units — D3Q27: 1/3, D3Q39: 0.6
# Precision-aware globals — type now LBM_F (Float32 by default, Float64 with
# LUDWIG_PRECISION=double). Was ::Float32-annotated previously.
global CS2_LATTICE::LBM_F = F(1/3)
global CS4_LATTICE::LBM_F = CS2_LATTICE * CS2_LATTICE

"""
    initialize_lattice!(lattice_type::Symbol; shift_x=0, shift_y=0, shift_z=0)

Set up the active lattice constants (C_X, C_Y, C_Z, W, OPP, MIRROR_Y, CS2_LATTICE)
based on the selected lattice type. Called once at startup after config loading.

For D3Q39, an optional Galilean shift (Phase 3.2) can be applied to the velocity set.
The shift adds integer vector U = (shift_x, shift_y, shift_z) to all discrete velocities.
Weights, OPP, and mirror mappings are index-based and remain invariant under the shift
(Frapolli/Chikatamarla/Karlin, PRL 2016).
"""
function initialize_lattice!(lattice_type::Symbol;
                             shift_x::Int32=Int32(0), shift_y::Int32=Int32(0), shift_z::Int32=Int32(0))
    # PonD uses the D3Q39 velocity set (cs²=2/3) for its 2nd-order Grad closure, UNSHIFTED —
    # the co-moving frame is per-cell (handled in the FV flux), not a global Galilean shift.
    if lattice_type == :pond
        lattice_type = :d3q39
    end
    if lattice_type == :d3q19
        cx, cy, cz, w, opp, my = build_d3q19_static()
        global C_X      = cx
        global C_Y      = cy
        global C_Z      = cz
        global W         = w
        global OPP       = opp
        global MIRROR_Y  = my
        global CS2_LATTICE = F(1/3)
        global CS4_LATTICE = CS2_LATTICE * CS2_LATTICE
        println("[Lattice] Initialized D3Q19 (standard incompressible): 19 dirs, cs²=1/3")
    elseif lattice_type == :d3q39
        cx, cy, cz, w, opp, my = build_d3q39_static()

        # Apply Galilean shift to velocities (weights, OPP, mirror are invariant)
        if shift_x != 0 || shift_y != 0 || shift_z != 0
            cx = SVector{39, Int32}(cx .+ shift_x)
            cy = SVector{39, Int32}(cy .+ shift_y)
            cz = SVector{39, Int32}(cz .+ shift_z)
            max_c = maximum(abs.([cx..., cy..., cz...]))
            println("[Lattice] Applied Galilean shift: U = ($shift_x, $shift_y, $shift_z)")
            println("[Lattice] Max |c'| = $max_c (BLOCK_SIZE must be > $max_c)")
        end

        global C_X      = cx
        global C_Y      = cy
        global C_Z      = cz
        global W         = w
        global OPP       = opp
        global MIRROR_Y  = my
        global CS2_LATTICE = F(2/3)
        global CS4_LATTICE = CS2_LATTICE * CS2_LATTICE
        println("[Lattice] Initialized D3Q39 Variant B (Frapolli/Karlin): 39 dirs, cs²=2/3")
    else
        # Reset to D3Q27 defaults (already set, but explicit for clarity)
        global C_X      = C_X_D3Q27
        global C_Y      = C_Y_D3Q27
        global C_Z      = C_Z_D3Q27
        global W         = W_D3Q27
        global OPP       = OPP_D3Q27
        global MIRROR_Y  = MIRROR_Y_D3Q27
        global CS2_LATTICE = F(1/3)
        global CS4_LATTICE = CS2_LATTICE * CS2_LATTICE
        println("[Lattice] Initialized D3Q27: 27 dirs, cs²=1/3")
    end
end
