const RUN_MONITOR_LOCK=ReentrantLock()
const RUN_MONITOR_GLOBAL=Ref{Any}(nothing)

struct _ProcessMemoryCounters
    cb::UInt32
    page_fault_count::UInt32
    peak_working_set::Csize_t
    working_set::Csize_t
    quota_peak_paged_pool::Csize_t
    quota_paged_pool::Csize_t
    quota_peak_nonpaged_pool::Csize_t
    quota_nonpaged_pool::Csize_t
    pagefile_usage::Csize_t
    peak_pagefile_usage::Csize_t
    private_usage::Csize_t
end

function _process_memory(pid::Integer)
    result=Dict{String,Any}("pid"=>pid,"available"=>false,"working_set_gb"=>nothing,
        "private_gb"=>nothing,"peak_working_set_gb"=>nothing,"source"=>"live")
    pid>0 || return result
    try
        if Sys.iswindows()
            handle=ccall((:OpenProcess,"kernel32"),Ptr{Cvoid},(UInt32,Int32,UInt32),0x0410,0,UInt32(pid))
            handle==C_NULL && return result
            try
                counters=Ref(_ProcessMemoryCounters(UInt32(sizeof(_ProcessMemoryCounters)),0,0,0,0,0,0,0,0,0,0))
                ok=ccall((:GetProcessMemoryInfo,"psapi"),Int32,
                    (Ptr{Cvoid},Ref{_ProcessMemoryCounters},UInt32),handle,counters,UInt32(sizeof(_ProcessMemoryCounters)))
                ok==0 && return result
                result["working_set_gb"]=Float64(counters[].working_set)/1024^3
                result["private_gb"]=Float64(counters[].private_usage)/1024^3
                result["peak_working_set_gb"]=Float64(counters[].peak_working_set)/1024^3
            finally
                ccall((:CloseHandle,"kernel32"),Int32,(Ptr{Cvoid},),handle)
            end
        elseif isfile("/proc/$pid/status")
            text=read("/proc/$pid/status",String)
            for (key,label) in (("working_set_gb","VmRSS"),("private_gb","VmSize"),("peak_working_set_gb","VmHWM"))
                matchvalue=match(Regex("(?m)^"*label*":\\s*(\\d+)\\s+kB"),text)
                matchvalue===nothing || (result[key]=parse(Float64,matchvalue[1])/1024^2)
            end
        else
            return result
        end
        result["available"]=result["working_set_gb"]!==nothing
    catch
    end
    result
end

function _monitor_global_snapshot()
    lock(RUN_MONITOR_LOCK) do
        previous=RUN_MONITOR_GLOBAL[]
        previous!==nothing && time()-previous.time<2 && return deepcopy(previous.data)
        gpu=Dict{String,Any}(String(key)=>value for (key,value) in pairs(_gpu_memory()))
        gpu["units"]="GiB"
        gpu["used_gb"]=get(gpu,"available",false) ? max(0.,gpu["total_gb"]-gpu["free_gb"]) : nothing
        total=Float64(Sys.total_memory())/1024^3
        free=Float64(Sys.free_memory())/1024^3
        data=Dict{String,Any}("sampled_utc"=>_now_utc(),"gpu"=>gpu,
            "ram"=>Dict{String,Any}("units"=>"GiB","system_total_gb"=>total,"system_available_gb"=>free,
                "system_used_gb"=>max(0.,total-free),"server"=>_process_memory(getpid())))
        RUN_MONITOR_GLOBAL[]=(time=time(),data=data)
        deepcopy(data)
    end
end

"Read the last complete CSV row without repeatedly scanning a long run history."
function _last_csv_row(path)
    isfile(path) || return Dict{String,Any}()
    try
        open(path,"r") do io
            eof(io) && return Dict{String,Any}()
            headers=strip.(split(readline(io),','));header_end=position(io)
            start=max(header_end,filesize(path)-131_072)
            seek(io,start);text=String(read(io));lines=split(text,'\n')
            start>header_end && !isempty(lines) && popfirst!(lines)
            # Native histories use println. An in-flight partial row is ignored.
            !endswith(text,"\n") && !isempty(lines) && pop!(lines)
            for line in Iterators.reverse(lines)
                values=strip.(split(strip(line),','))
                length(values)==length(headers) || continue
                row=Dict{String,Any}()
                for (key,value) in zip(headers,values)
                    number=tryparse(Float64,value)
                    row[String(key)]=number===nothing ? String(value) : isfinite(number) ? number : nothing
                end
                get(row,"Step",nothing) isa Real && return row
            end
            Dict{String,Any}()
        end
    catch
        Dict{String,Any}()
    end
end

function _run_history(id)
    folder=joinpath(_case_dir(id,"run"),"RESULTS","csv")
    (convergence=_last_csv_row(joinpath(folder,"convergence.csv")),
     resources=_last_csv_row(joinpath(folder,"resource_history.csv")),
     forces=_last_csv_row(joinpath(folder,"forces.csv")))
end

function _run_wall_seconds(state,recorded)
    start=get(state,"run_started_utc",get(state,"solver_started_utc",nothing))
    start isa AbstractString || return recorded
    try
        instant(text)=Dates.datetime2unix(DateTime(endswith(text,"Z") ? chop(text;tail=1) : text))
        active=state["status"] in ("waiting_for_gpu","running","queued","starting")
        finish=active ? time() : instant(get(state,"run_finished_utc",state["updated_utc"]))
        max(0.,finish-instant(start))
    catch
        recorded
    end
end

function _monitor_phase(id,state,iteration)
    status=String(state["status"])
    status in ("meshing","sizing","mesh_sizing") && return Dict("key"=>"meshing","label"=>"Building mesh","detail"=>get(get(state,"mesh_progress",Dict()),"detail",state["message"]))
    status=="waiting_for_gpu" && return Dict("key"=>"waiting_for_gpu","label"=>"Waiting for GPU","detail"=>state["message"])
    status in ("failed","mesh_failed","cancelled") && return Dict("key"=>status,"label"=>replace(status,'_'=>' '),"detail"=>state["message"])
    post=get(state,"postprocessing",Dict())
    any(get(entry,"status","")=="running" for entry in values(post)) && return Dict("key"=>"postprocessing","label"=>"Generating reports and renders","detail"=>"Solver data is available while requested outputs are generated")
    status=="completed" && return Dict("key"=>"completed","label"=>"Requested iterations completed","detail"=>"Completion of the requested iteration count does not by itself establish steady convergence")
    status=="mesh_ready" && return Dict("key"=>"mesh_ready","label"=>"Mesh ready","detail"=>"Choose the output folder and run the analysis")
    log=joinpath(_case_dir(id,"run"),"solver.log")
    active_export=nothing
    if isfile(log)
        try
            text=open(log,"r") do io;seek(io,max(0,filesize(log)-65_536));String(read(io));end
            for line in split(text,'\n')
                item=match(r"^\[([A-Za-z0-9_]+)\]\s+(START|DONE|Warning).*?step\s+(\d+)",line)
                item===nothing && continue
                if item[2]=="START";active_export=(stage=item[1],step=item[3]);else;active_export=nothing;end
            end
        catch
        end
    end
    active_export===nothing || return Dict("key"=>"writing_results","label"=>"Writing result fields","detail"=>"$(active_export.stage), iteration $(active_export.step)")
    iteration>0 ? Dict("key"=>"solving","label"=>"Solving on GPU","detail"=>"Iteration $iteration") :
        Dict("key"=>"initializing","label"=>"Initializing solver","detail"=>get(state,"message","Preparing the solver"))
end

function _run_monitoring(id,state;history=_run_history(id))
    result=_monitor_global_snapshot()
    step=maximum(Int(round(get(row,"Step",0))) for row in (history.convergence,history.resources,history.forces))
    total=Int(state["case"]["steps"])
    result["iteration"]=step;result["total_iterations"]=total
    result["progress_percent"]=min(100.,100step/max(total,1))
    result["phase"]=_monitor_phase(id,state,step)
    result["simulation_time_s"]=get(history.convergence,"Time_phys_s",get(history.forces,"Time_phys_s",nothing))
    result["elapsed_s"]=get(history.resources,"Elapsed_s",nothing)
    result["wall_s"]=_run_wall_seconds(state,result["elapsed_s"])
    result["wall_time_source"]=haskey(state,"run_started_utc") ? "wall_clock" : "recorded"
    result["resources_last_sample"]=history.resources
    result["diagnostics"]=history.convergence
    result["performance"]=Dict("mlups"=>get(history.convergence,"MLUPS",nothing),
        "iterations_per_second"=>get(history.resources,"Iter_per_s",nothing),
        "domain_travel_pct"=>get(history.convergence,"Domain_travel_pct",nothing))
    result["coefficients"]=Dict(key=>value for (key,value) in history.forces if startswith(key,"C"))
    residuals=Dict(key=>value for (key,value) in history.convergence if occursin(r"(?i)residual|resid_|l2_norm",key) && value isa Real)
    result["residuals"]=Dict("available"=>!isempty(residuals),"values"=>residuals,
        "reason"=>isempty(residuals) ? "This solver run does not export a residual metric. Iteration, density, performance and force histories are shown instead." : "")
    pid=get(state,"solver_pid",nothing)
    if pid===nothing
        process=lock(JOB_LOCK) do;get(JOB_PROCESSES,id,nothing);end
        process===nothing || (pid=try getpid(process) catch;nothing;end)
    end
    active=state["status"] in ("running","meshing","sizing","mesh_sizing")
    process=active && pid isa Integer ? _process_memory(pid) : Dict{String,Any}("pid"=>pid,"available"=>false,"source"=>"not_running",
        "working_set_gb"=>nothing,"private_gb"=>nothing,"peak_working_set_gb"=>nothing)
    if !get(process,"available",false) && get(history.resources,"RAM_used_MiB",nothing) isa Real
        process["working_set_gb"]=history.resources["RAM_used_MiB"]/1024
        process["source"]="recorded";process["recorded_iteration"]=get(history.resources,"Step",nothing)
    end
    result["ram"]["solver"]=process
    result["gpu"]["recorded_solver_used_gb"]=get(history.resources,"VRAM_used_MiB",nothing) isa Real ? history.resources["VRAM_used_MiB"]/1024 : nothing
    result["gpu"]["recorded_iteration"]=get(history.resources,"Step",nothing)
    result
end
