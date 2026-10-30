# Read-only visibility into the durable comparison queue. Wire responses still
# use the app's MessagePack/BSON codecs; JSON is only an on-disk queue format.
using JSON3, TOML
const PARITY_CAMPAIGN_DIR=joinpath(PRIVATE_ROOT,"TEMP_DELETEABLE", "build", "airfoil_web_cache","parity20_re3m_20261019")
const CH_RECOVERY_PARENT=joinpath(PRIVATE_ROOT,"TEMP_DELETEABLE", "build", "airfoil_web_cache","gas_strip_operator_20261020")
const CH_RECOVERY_NAME="coupled_baseline_20261020"
const CAMPAIGN_CACHE=Ref{Any}(nothing)
const CAMPAIGN_CACHE_TIME=Ref(0.)
_campaign_json(path)=try JSON3.read(read(path,String),Dict{String,Any}) catch;Dict{String,Any}();end
_campaign_toml(path)=try TOML.parsefile(path) catch;Dict{String,Any}();end

_trial_number(value)=value isa Real && !(value isa Bool) && isfinite(value) ? Float64(value) : nothing
function _trial_count(value)
    n=_trial_number(value)
    n!==nothing && 0<=n<=1e9 && isinteger(n) ? Int(n) : nothing
end
_trial_text(value;limit=180)=value isa AbstractString ? first(String(value),limit) : ""

"Read one explicitly named evidence file; never follow an escaping link/junction."
function _campaign_evidence_path(root,relative;limit=8_000_000)
    isdir(root)&&!islink(root)||return nothing
    path=joinpath(root,relative)
    isfile(path)&&!islink(path)||return nothing
    try
        base=lowercase(realpath(root));actual=lowercase(realpath(path))
        startswith(actual,base*string(Base.Filesystem.path_separator))&&filesize(path)<=limit ? path : nothing
    catch
        nothing
    end
end
function _campaign_evidence_json(root,relative)
    path=_campaign_evidence_path(root,relative)
    path===nothing ? Dict{String,Any}() : _campaign_json(path)
end
function _campaign_evidence_toml(root,relative)
    path=_campaign_evidence_path(root,relative)
    path===nothing ? Dict{String,Any}() : _campaign_toml(path)
end
function _campaign_fixed_link(root,relative,alias)
    _campaign_evidence_path(root,relative)===nothing ? nothing : "/api/campaign/files/"*alias
end

function _campaign_recovery(root)
    directory=joinpath(root,"recovery_pilot_20261020_chi0_250k")
    status=_campaign_evidence_json(directory,"status.json");report=_campaign_evidence_json(directory,"comparison.json")
    candidate=get(report,"candidate",nothing);candidate isa AbstractDict||(candidate=Dict{String,Any}())
    guard=get(candidate,"guard",nothing);guard isa AbstractDict||(guard=Dict{String,Any}())
    equations=get(candidate,"final_conservative_equations",nothing);equations isa AbstractDict||(equations=Dict{String,Any}())
    Dict{String,Any}("available"=>!isempty(status)||!isempty(report),"status"=>_trial_text(get(status,"status","pending")),
        "name"=>"A-plus baseline recovery; SFD=0 control","qualified"=>false,"full_campaign_released"=>false,
        "actual_iterations"=>_trial_count(get(report,"actual_steps",nothing)),"requested_iterations"=>_trial_count(get(report,"requested_steps",nothing)),
        "cl"=>_trial_number(get(candidate,"cl",nothing)),"cd"=>_trial_number(get(candidate,"cd",nothing)),
        "guard_status"=>_trial_text(get(guard,"status","unavailable")),
        "mass_convective_linf"=>_trial_number(get(equations,"mass_linf_convective",nothing)),
        "mass_net_frac"=>_trial_number(get(equations,"mass_signed_flux_fraction",nothing)),
        "assessment"=>"A-plus baseline remains unconverged; full CFD sweeps held. This completed numerical control is not a polar or stall qualification.",
        "report_url"=>_campaign_fixed_link(directory,"RECOVERY_REPORT.txt","recovery_report.txt"),
        "comparison_url"=>_campaign_fixed_link(directory,"comparison.json","recovery_comparison.json"))
end

function _campaign_stall_reference(root)
    directory=joinpath(root,"xfoil_stall_extension","recovery20_20261020")
    report=_campaign_evidence_json(directory,"reference_report.json");counts=get(report,"counts",nothing);counts isa AbstractDict||(counts=Dict{String,Any}())
    records=get(report,"records",Any[])
    codes=records isa AbstractVector ? unique([String(r["naca"]) for r in records if r isa AbstractDict&&get(r,"naca",nothing) isa AbstractString&&occursin(r"^[0-9]{4}$",r["naca"])]) : String[]
    Dict{String,Any}("available"=>!isempty(report),"airfoils"=>length(codes),"qualified"=>false,"cfd_stall_parity"=>false,
        "supplement_requested"=>_trial_count(get(counts,"supplement_requested",nothing)),
        "supplement_converged"=>_trial_count(get(counts,"supplement_converged",nothing)),
        "new_actual_executable_runs"=>_trial_count(get(counts,"new_actual_executable_runs",nothing)),
        "positive_complete_brackets"=>_trial_count(get(counts,"positive_stall_brackets",nothing)),
        "negative_complete_brackets"=>_trial_count(get(counts,"negative_stall_brackets",nothing)),
        "qualification"=>"Actual XFOIL supplement only; complete sampled brackets do not establish CFD or experimental stall parity. Missing/failed angles remain explicit.",
        "report_url"=>_campaign_fixed_link(directory,"STALL_REFERENCE_REPORT.txt","stall_reference_report.txt"),
        "data_url"=>_campaign_fixed_link(directory,"reference_report.json","stall_reference.json"),
        "polar_url"=>_campaign_fixed_link(directory,"original20_stall_reference.png","stall_reference.png"))
end

function _campaign_ch_pilot(parent)
    parent===nothing&&return Dict{String,Any}("available"=>false,"status"=>"pending","qualified"=>false,"full_campaign_released"=>false)
    directory=joinpath(parent,CH_RECOVERY_NAME)
    status=_campaign_evidence_json(parent,CH_RECOVERY_NAME*".status.json")
    receipt=_campaign_evidence_toml(directory,"pilot_receipt.toml")
    raw=_trial_text(get(receipt,"status",get(status,"status","pending")))
    command=get(status,"command",Any[])
    requested=nothing
    if command isa AbstractVector
        i=findfirst(==("--steps"),command)
        i!==nothing&&i<length(command)&&command[i+1] isa AbstractString&&(requested=_trial_count(tryparse(Float64,command[i+1])))
    end
    sample=Dict{String,Any}();history=_campaign_evidence_path(directory,"iteration_history.csv";limit=64_000_000)
    if history!==nothing
        for line in reverse(split(_campaign_tail(history;bytes=8192),'\n'))
            cells=split(line,',');length(cells)>=17||continue
            values=tryparse.(Float64,cells);all(x->x!==nothing&&isfinite(x),values)||continue
            iteration=_trial_count(values[1]);iteration===nothing&&continue
            sample=Dict("iteration"=>iteration,"cl"=>values[3],"cd"=>values[4],"mass_net_frac"=>values[5],"raw_mass_linf"=>values[6]);break
        end
    end
    Dict{String,Any}("available"=>!isempty(status)||!isempty(receipt),"status"=>raw,"qualified"=>false,
        "full_campaign_released"=>false,"stall_validated"=>false,"clock"=>"Unequal local pseudo-time; no elapsed physical time",
        "actual_iterations"=>_trial_count(get(receipt,"iterations",get(sample,"iteration",nothing))),"requested_iterations"=>requested,
        "cl"=>get(sample,"cl",nothing),"cd"=>get(sample,"cd",nothing),"raw_mass_linf"=>get(sample,"raw_mass_linf",nothing),
        "mass_net_frac"=>get(sample,"mass_net_frac",nothing),"cells"=>_trial_count(get(receipt,"cells",nothing)),
        "failure"=>_trial_text(get(receipt,"failure",get(status,"error",""))),
        "qualification"=>"New C+H thermal gas/SST diagnostic; exact interface flux tests and process completion do not qualify aerodynamic convergence or stall parity.",
        "report_url"=>_campaign_fixed_link(directory,"COUPLED_DIAGNOSTIC_REPORT.txt","ch_pilot_report.txt"),
        "receipt_url"=>_campaign_fixed_link(directory,"pilot_receipt.toml","ch_pilot_receipt.toml"),
        "summary_url"=>_campaign_fixed_link(directory,"diagnostic_summary.json","ch_pilot_summary.json"))
end
function _trial_status(status)
    value=lowercase(_trial_text(status))
    occursin("fail",value)||occursin("refus",value) ? "refused" :
        occursin("stop",value)||occursin("interrupt",value) ? "stopped" :
        value=="running" ? "running" : occursin("prepar",value) ? "prepared" :
        occursin("finish",value)||occursin("complet",value) ? "completed" : "unavailable"
end
function _trial_file(path)
    isfile(path) && !islink(path) || return nothing
    try
        filesize(path)<=2_000_000 || return nothing
        TOML.parsefile(path)
    catch
        nothing # atomic replacement or malformed diagnostic must not break the page
    end
end
function _trial_directories(parent;within=parent)
    isdir(parent) && !islink(parent) || return String[]
    try
        base,allowed=lowercase(realpath(parent)),lowercase(realpath(within))
        (base==allowed || startswith(base,allowed*string(Base.Filesystem.path_separator))) || return String[]
    catch
        return String[]
    end
    result=String[]
    for name in readdir(parent)
        occursin(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$",name) || continue
        startswith(lowercase(name),"source") && continue
        path=joinpath(parent,name)
        isdir(path) && !islink(path) || continue
        # Also refuse junctions/reparse points escaping the fixed experiment root.
        try
            startswith(lowercase(realpath(path)),lowercase(realpath(parent))*string(Base.Filesystem.path_separator)) || continue
        catch
            continue
        end
        push!(result,path)
    end
    result
end
function _trial_metadata(directory,names)
    for name in names
        path=joinpath(directory,name)
        isfile(path) && return (_trial_file(path),true)
    end
    (nothing,false)
end
function _trial_sample(directory)
    result=Dict{String,Any}()
    path=joinpath(directory,"solver.log")
    isfile(path) && !islink(path) || return result
    records=filter(line->startswith(line,"ITERATION,"),split(_campaign_tail(path),'\n'))
    for record in reverse(records)
        fields=split(record,',');length(fields) in (8,9) || continue
        values=tryparse.(Float64,fields[2:end])
        all(v->v!==nothing && isfinite(v),values) || continue
        n,total=_trial_count(values[1]),_trial_count(values[2]);n===nothing && continue
        result["iteration"]=n;result["requested_iterations"]=total
        result["mass_residual"]=values[3];result["cl"]=values[6];result["cd"]=values[7]
        length(values)==8 && (result["mass_net_frac"]=values[8])
        break
    end
    result
end
function _development_trial(kind,name,arm,state,metadata;sample=Dict{String,Any}())
    bad=state===nothing
    state=bad ? Dict{String,Any}() : state
    bad && (sample=Dict{String,Any}())
    metadata=metadata===nothing ? Dict{String,Any}() : metadata
    raw=get(state,"status",get(metadata,"status","unavailable"))
    initial=something(_trial_count(get(state,"checkpoint_initial_iteration",0)),0)
    accepted=_trial_count(get(state,"last_fully_accepted_trial_iteration",get(state,"accepted_trial_iterations",nothing)))
    recorded=_trial_count(get(state,"iteration",get(sample,"iteration",nothing)))
    kind=="sfd" && (accepted=recorded)
    actual=accepted===nothing ? recorded : initial+accepted
    requested=_trial_count(get(metadata,"steps_per_arm",get(metadata,"fixed_requested_steps",get(sample,"requested_iterations",nothing))))
    row=Dict{String,Any}("kind"=>kind,"name"=>name,"arm"=>arm,
        "backend"=>kind=="sfd" ? "transonic" : "cavitation",
        "status"=>bad ? "unavailable" : _trial_status(raw),
        "detail"=>bad ? "Unreadable diagnostic file; progress not inferred" : _trial_text(raw),
        "experimental"=>true,"qualified"=>false,"accepted_iterations"=>accepted,
        "requested_iterations"=>requested,"iteration"=>actual,"sample_iteration"=>recorded,
        "heartbeat_utc"=>_trial_text(get(state,"heartbeat_utc",get(metadata,"finished_utc",""));limit=64),
        "mass_residual_definition"=>kind=="sfd" ? "scaled continuity L-infinity" : "convective continuity L-infinity",
        "clock"=>"Steady numerical experiment; no physical averages or stall qualification")
    for key in ("cl","cd");row[key]=_trial_number(get(state,key,get(sample,key,nothing)));end
    equations=get(state,"equation_residuals",nothing)
    mass=equations isa AbstractDict ? get(equations,"mass",nothing) : nothing
    row["mass_residual"]=mass isa AbstractDict ? _trial_number(get(mass,"linf_convective",nothing)) : _trial_number(get(sample,"mass_residual",nothing))
    row["mass_net_frac"]=mass isa AbstractDict ? _trial_number(get(mass,"signed_boundary_over_reference_flux",nothing)) : _trial_number(get(sample,"mass_net_frac",nothing))
    row
end

"Read fixed trial/arm files, with one explicitly named durable chunk level."
function _campaign_development_tests(root)
    rows=Dict{String,Any}[]
    function boundary_rows!(directory,label)
        metadata,present=_trial_metadata(directory,("boundary_trial.toml","experiment.toml"))
        found=false
        for arm in ("fixed_pressure","native_characteristic","extrapolated")
            arm_path=joinpath(directory,arm)
            arm_path in _trial_directories(directory;within=root) || continue
            path=joinpath(arm_path,"trial_state.toml");isfile(path) || continue
            push!(rows,_development_trial("boundary",label,arm,_trial_file(path),metadata));found=true
        end
        !found && present && push!(rows,_development_trial("boundary",label,"awaiting worker",metadata,metadata))
    end
    for directory in _trial_directories(joinpath(root,"boundary_trials");within=root)
        boundary_rows!(directory,basename(directory))
        # Durable controllers split a long run into bounded chunks. Admit only
        # this one named intermediate level; never descend recursively.
        for chunk in _trial_directories(directory;within=root)
            occursin(r"^chunk[0-9]+_to[0-9]+$",basename(chunk)) || continue
            boundary_rows!(chunk,basename(directory)*" / "*basename(chunk))
        end
    end
    for directory in _trial_directories(joinpath(root,"sfd_trials");within=root)
        metadata,present=_trial_metadata(directory,("trial.toml","sfd_trial.toml"));present || continue
        sample=_trial_sample(directory)
        if metadata!==nothing && _trial_status(get(metadata,"status",""))=="completed"
            result=_trial_file(joinpath(directory,"result.toml"))
            if result!==nothing
                for key in ("cl","cd","steps","mass_net_frac")
                    haskey(result,key) && (sample[key=="steps" ? "iteration" : key]=result[key])
                end
            end
        end
        push!(rows,_development_trial("sfd",basename(directory),"SFD",metadata,metadata;sample))
    end
    rank=Dict("running"=>0,"refused"=>1,"stopped"=>2,"prepared"=>3,"completed"=>4,"unavailable"=>5)
    sort!(rows;by=r->(get(rank,r["status"],6),r["kind"],r["name"],r["arm"]))
end

function _campaign_tail(path;bytes=4096)
    isfile(path)||return ""
    open(path,"r") do io
        seekend(io);seek(io,max(0,position(io)-bytes));String(read(io))
    end
end

function _campaign_active(root,queue)
    backend=string(get(queue,"backend",""));backend in ("transonic","cavitation")||return nothing
    state=_campaign_toml(joinpath(root,backend*"_worker_state.toml"))
    code=string(get(state,"naca",get(queue,"naca","")))
    occursin(r"^[0-9]{4}$",code)||return nothing
    alpha=get(state,"alpha_deg",0.)
    alpha isa Real && isfinite(alpha) && abs(alpha)<=90 || return nothing
    pointdir=joinpath(root,backend,"naca"*code,@sprintf("alpha_%+03d",round(Int,alpha)))
    result=Dict{String,Any}("backend"=>backend,"naca"=>code,"alpha_deg"=>alpha,
        "status"=>get(state,"status","starting"),"iteration"=>0,
        "total_iterations"=>get(state,"requested_steps",500000),"phase"=>"Preparing Julia worker")
    records=filter(line->startswith(line,"ITERATION,"),split(_campaign_tail(joinpath(pointdir,"solver.log")),'\n'))
    if !isempty(records)
        values=split(last(records),',')
        if length(values) in (8,9)
            numbers=tryparse.(Float64,values[2:end])
            if all(x->x!==nothing&&isfinite(x),numbers)
                merge!(result,Dict("iteration"=>round(Int,numbers[1]),"total_iterations"=>round(Int,numbers[2]),
                    "residual_scaled"=>numbers[3],"cl"=>numbers[6],"cd"=>numbers[7],"phase"=>"CFD iterations running"))
                length(numbers)==8 && (result["mass_net_frac"]=numbers[8])
            end
        end
    end
    status=string(result["status"])
    if startswith(status,"interrupted")
        result["phase"]="Stopped for numerical investigation; last recorded sample"
    elseif status=="finished"
        result["phase"]="Finished; convergence qualification required"
    elseif status=="failed"
        result["phase"]="Worker failed; last recorded sample"
    end
    result
end

function _campaign_status(root=PARITY_CAMPAIGN_DIR;ch_parent=root==PARITY_CAMPAIGN_DIR ? CH_RECOVERY_PARENT : nothing)
    isdir(root)||return Dict("available"=>false,"message"=>"No reference campaign has been started.")
    queue=_campaign_json(joinpath(root,"queue_status.json"))
    codes=sort!(unique([replace(name,"naca"=>"") for directory in (joinpath(root,"xfoil","tripped"),joinpath(root,"mesh_admission"))
        if isdir(directory) for name in readdir(directory) if occursin(r"^naca[0-9]{4}$",name)]))
    rows=Dict{String,Any}[]
    for code in codes
        row=Dict{String,Any}("naca"=>code,"requested_angles"=>31)
        admission=_campaign_toml(joinpath(root,"mesh_admission","naca"*code,"admission.toml"))
        row["mesh_status"]=get(admission,"status","pending")
        row["mesh_cells"]=get(admission,"mesh_cells",nothing)
        for transition in ("tripped","natural")
            reference=_campaign_json(joinpath(root,"xfoil",transition,"naca"*code,"status.json"))
            row["xfoil_"*transition]=Dict("attempted"=>get(reference,"status","")=="finished" ? 31 : 0,
                "converged"=>get(reference,"converged_points",0))
        end
        for backend in ("transonic","cavitation")
            finished=0;failed=0;interrupted=0;guard_passed=0
            for alpha in -15:15
                point=_campaign_toml(joinpath(root,backend,"naca"*code,@sprintf("alpha_%+03d",alpha),"campaign_point.toml"))
                state=get(point,"status","")
                finished+=state=="finished";failed+=state=="failed"
                interrupted+=startswith(state,"interrupted")
                guard_passed+=state=="finished" && get(point,"convergence_status","")=="early_stop_eligible"
            end
            row[backend]=Dict("finished"=>finished,"failed"=>failed,"interrupted"=>interrupted,"continuity_guard_passed"=>guard_passed)
        end
        row["polar_url"]=isfile(joinpath(root,"reports","naca$(code)_polars.png")) ? "/api/campaign/files/naca$(code)_polars.png" : nothing
        push!(rows,row)
    end
    deliveries=_campaign_json(joinpath(root,"telegram_delivery.json"))
    Dict{String,Any}("available"=>true,"queue_status"=>get(queue,"status","not_started"),
        "mach"=>.10,"reynolds"=>3e6,"alpha_start"=>-15,"alpha_stop"=>15,"alpha_step"=>1,
        "airfoils"=>rows,"active"=>_campaign_active(root,queue),"telegram_delivered_files"=>length(deliveries),
        "development_tests"=>_campaign_development_tests(root),
        "recovery_pilot"=>_campaign_recovery(root),"stall_reference_extension"=>_campaign_stall_reference(root),
        "ch_coupled_pilot"=>_campaign_ch_pilot(ch_parent),
        "full_campaign_admitted"=>get(_campaign_json(joinpath(root,"full_campaign_ready.json")),"ready",false),
        "qualification"=>"Finished CFD points and continuity guard passes are separate from fully qualified aerodynamic parity. Stall matching remains unestablished.")
end

function _campaign_cached_status()
    if CAMPAIGN_CACHE[]===nothing || time()-CAMPAIGN_CACHE_TIME[]>4
        CAMPAIGN_CACHE[]=_campaign_status();CAMPAIGN_CACHE_TIME[]=time()
    end
    CAMPAIGN_CACHE[]
end

function _campaign_report_asset(name;parity_root=PARITY_CAMPAIGN_DIR,ch_parent=CH_RECOVERY_PARENT)
    recovery="recovery_pilot_20261020_chi0_250k";stall=joinpath("xfoil_stall_extension","recovery20_20261020")
    allowed=Dict("recovery_report.txt"=>(parity_root,joinpath(recovery,"RECOVERY_REPORT.txt")),
        "recovery_comparison.json"=>(parity_root,joinpath(recovery,"comparison.json")),
        "stall_reference_report.txt"=>(parity_root,joinpath(stall,"STALL_REFERENCE_REPORT.txt")),
        "stall_reference.json"=>(parity_root,joinpath(stall,"reference_report.json")),
        "stall_reference.png"=>(parity_root,joinpath(stall,"original20_stall_reference.png")),
        "ch_pilot_report.txt"=>(ch_parent,joinpath(CH_RECOVERY_NAME,"COUPLED_DIAGNOSTIC_REPORT.txt")),
        "ch_pilot_receipt.toml"=>(ch_parent,joinpath(CH_RECOVERY_NAME,"pilot_receipt.toml")),
        "ch_pilot_summary.json"=>(ch_parent,joinpath(CH_RECOVERY_NAME,"diagnostic_summary.json")))
    root,relative=if haskey(allowed,name)
        allowed[name]
    elseif occursin(r"^naca[0-9]{4}_(polars\.png|parity\.json)$",name)||name=="all_points.csv"
        (parity_root,joinpath("reports",name))
    else
        return HTTP.Response(404,"Not found")
    end
    root===nothing&&return HTTP.Response(404,"Not found")
    path=_campaign_evidence_path(root,relative;limit=16_000_000)
    path===nothing&&return HTTP.Response(404,"Not found")
    mime=endswith(name,".png") ? "image/png" : endswith(name,".csv") ? "text/csv; charset=utf-8" : endswith(name,".json") ? "application/json" : "text/plain; charset=utf-8"
    HTTP.Response(200,["Content-Type"=>mime,"Cache-Control"=>"no-store"],read(path))
end
