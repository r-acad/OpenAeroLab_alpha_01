"""Snap only Float32-scale roundoff at integer grid boundaries.

Native configuration, geometry and cell sizes can pass through Float32. Two
Float32 ULPs around the nearest integer absorb that conversion noise; meaningful
positive fractions still round outward to the next block.
"""
function _roundoff_snap_grid_integer(value::Real)
    v=Float64(value)
    isfinite(v) || throw(ArgumentError("Grid quotient must be finite"))
    nearest=round(v)
    reference=Float32(max(1.0,abs(nearest)))
    tolerance=isfinite(reference) ? 2*Float64(eps(reference)) : 2*eps(max(1.0,abs(nearest)))
    abs(v-nearest)<=tolerance ? nearest : v
end

function _stable_block_ceil(value::Real)
    v=Float64(value)
    isfinite(v) && v>=0 || throw(ArgumentError("Grid block quotient must be finite and nonnegative"))
    snapped=_roundoff_snap_grid_integer(v)
    snapped<Float64(typemax(Int)) || throw(ArgumentError("Grid dimension exceeds integer capacity; use a coarser cell size"))
    ceil(Int,snapped)
end

function _stable_grid_block_count(length::Real,dx::Real,block_size::Integer;minimum_blocks::Integer=1)
    isfinite(length) && length>=0 && isfinite(dx) && dx>0 && block_size>0 && minimum_blocks>=0 ||
        throw(ArgumentError("Grid length/cell size and block counts must be finite and valid"))
    max(minimum_blocks,_stable_block_ceil(Float64(length)/(Float64(dx)*block_size)))
end
