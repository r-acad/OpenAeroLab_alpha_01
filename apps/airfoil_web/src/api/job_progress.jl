module AirfoilJobProgress

"Preserve a point's latest iteration update when its worker emits a heartbeat."
function preserve_iteration!(latest::AbstractDict, point, event::AbstractDict, update::AbstractDict)
    kind=String(get(event,"event","running"))
    if kind=="running" && !haskey(event,"gpu_wait") && haskey(latest,point)
        return copy(latest[point])
    elseif kind=="iteration"
        latest[point]=Dict{String,Any}(update)
    elseif kind in ("started","gpu_wait","gpu_acquired","point_complete","point_failed") ||
           (kind=="running" && haskey(event,"gpu_wait"))
        delete!(latest,point)
    end
    update
end

end
