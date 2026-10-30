module PhysicalTimeAccounting
export PhysicalClock, record_step!, remaining_step, reached_time

mutable struct PhysicalClock
    elapsed::Float64
    steps::Int
    dt_min::Float64
    dt_max::Float64
end
PhysicalClock() = PhysicalClock(0.0, 0, Inf, 0.0)

function record_step!(clock::PhysicalClock, dt::Real)
    d = Float64(dt)
    isfinite(d) && d > 0 || throw(ArgumentError("applied timestep must be finite and positive"))
    t = clock.elapsed + d
    isfinite(t) && t > clock.elapsed || throw(ArgumentError("timestep does not advance the physical clock"))
    clock.elapsed = t; clock.steps += 1
    clock.dt_min = min(clock.dt_min, d); clock.dt_max = max(clock.dt_max, d)
    t
end

function reached_time(clock::PhysicalClock, target::Real)
    isfinite(target) && target >= 0 || throw(ArgumentError("target must be finite and nonnegative"))
    clock.elapsed >= target - 8eps(Float64)*max(1.0, abs(target))
end

function remaining_step(clock::PhysicalClock, target::Real)
    reached_time(clock, target) && throw(ArgumentError("target has already been reached"))
    cap = Float32(target-clock.elapsed)
    isfinite(cap) && cap > 0 || throw(ArgumentError("remaining time is outside the Float32 timestep range"))
    cap
end
end
