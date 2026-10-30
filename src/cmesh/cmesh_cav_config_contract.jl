module CMeshCavConfigContract

export validate_vaploc

"""Return a finite binary vaploc selector, or fail before any state is allocated."""
function validate_vaploc(value::Float64)
    isfinite(value) && (value == 0.0 || value == 1.0) ||
        throw(ArgumentError("vaploc must be exactly 0 or 1, got $value"))
    value
end

end # module
