module AirfoilXfoilBackend

using SHA,UUIDs,Printf,TOML
if !isdefined(parentmodule(@__MODULE__), :AirfoilJobControl)
    Base.include(parentmodule(@__MODULE__), joinpath(@__DIR__, "..", "job_control.jl"))
end
using ..AirfoilJobControl
using ..AirfoilJXBackend

const ROOT=normpath(joinpath(@__DIR__,"..","..","..",".."))
const PRIVATE_ROOT=normpath(get(ENV,"LUDWIG_PRIVATE_ROOT",
    joinpath(ROOT,"..","02_PRIVATE_PROJECT_DEVELOPMENT")))
const DEFAULT_EXE=joinpath(PRIVATE_ROOT,"Xfoil_code_2026_10_18_BASELINE","SCRATCH","build","bin","xfoil.exe")
export run_case,capabilities

executable()=abspath(get(ENV,"AIRFOIL_XFOIL_EXE",DEFAULT_EXE))
capabilities()=Dict("available"=>isfile(executable()),"name"=>"XFOIL executable",
    "mach_min"=>0.0,"mach_max_exclusive"=>1.0,
    "model"=>"XFOIL viscous/inviscid panel and integral-boundary-layer solver with subsonic compressibility correction; no resolved CFD volume field or shock/stall validation claim")

function _value(flow,key,default;lo=-Inf,hi=Inf)
    v=get(flow,key,default)
    v isa Real && !(v isa Bool) && isfinite(v) && lo<=v<=hi || throw(ArgumentError("invalid XFOIL $key"))
    Float64(v)
end

function _bounded_process(command::Cmd,input::String,logpath::String;timeout=45.0,max_log_bytes=16_000_000,heartbeat=_->nothing)
    AirfoilJobControl.check_cancelled()
    began=time();stop_reason="";last_notice=began;process=nothing
    open(logpath,"w") do io
        process=run(pipeline(ignorestatus(AirfoilJobControl.process_command(command)),stdin=input,stdout=io,stderr=io);wait=false)
        owned=AirfoilJobControl.track!(process)
        try
            while process_running(process)
                elapsed=time()-began
                if elapsed>=timeout
                    stop_reason="wall_time_limit";AirfoilJobControl.finish!(owned);break
                elseif filesize(logpath)>max_log_bytes
                    stop_reason="numerical_output_limit";AirfoilJobControl.finish!(owned);break
                end
                if time()-last_notice>=2
                    heartbeat(round(elapsed;digits=1));last_notice=time()
                end
                sleep(.05)
            end
            wait(process)
        finally
            AirfoilJobControl.finish!(owned)
            wait(process)
        end
    end
    AirfoilJobControl.check_cancelled()
    # Keep both ends of a numerical print loop without leaving a giant file.
    bytes=filesize(logpath)
    if bytes>2_000_000
        head,tail=open(logpath,"r") do io
            first=read(io,1_000_000);seek(io,bytes-1_000_000);(first,read(io))
        end
        open(logpath,"w") do io
            write(io,head);write(io,"\n[omitted $(bytes-2_000_000) repetitive log bytes]\n");write(io,tail)
        end
    end
    Dict{String,Any}("exit_code"=>process.exitcode,"stop_reason"=>stop_reason,
        "elapsed_seconds"=>time()-began,"original_log_bytes"=>bytes,
        "stdin_closed_at_eof"=>true)
end

function _polar(path,alpha)
    isfile(path)||return nothing
    result=nothing
    for line in eachline(path)
        columns=split(line);length(columns)>=7||continue
        values=tryparse.(Float64,columns[1:7])
        all(v->v!==nothing&&isfinite(v),values)||continue
        abs(values[1]-alpha)<=1e-5||continue
        values[3]>=0 && all(x->0<=x<=1,values[6:7])||continue
        result=Dict{String,Any}(k=>Float64(v) for (k,v) in zip(
            ("alpha_deg","cl","cd","cdp","cm","top_xtr","bot_xtr"),values))
    end
    result
end

function _cp(path)
    rows=Dict{String,Any}[]
    if isfile(path)
        for line in eachline(path)
            columns=split(line);length(columns)>=2||continue
            x=tryparse(Float64,columns[1]);p=tryparse(Float64,columns[end])
            x!==nothing && p!==nothing && isfinite(x)&&isfinite(p) && -.1<=x<=1.1 || continue
            push!(rows,Dict{String,Any}("x"=>x,"cp"=>p))
        end
    end
    length(rows)>=8 || return Dict("upper"=>Any[],"lower"=>Any[])
    le=argmin([r["x"] for r in rows])
    upper=rows[1:le];lower=rows[le:end]
    sort!(upper;by=r->r["x"]);sort!(lower;by=r->r["x"])
    Dict("upper"=>upper,"lower"=>lower)
end

function _script(flow,alpha;continuation=false,maxiter=500,panels=240)
    re=_value(flow,"reynolds",3e6;lo=1);mach=_value(flow,"mach",0.;lo=0,hi=prevfloat(1.))
    ncrit=_value(flow,"ncrit",9.;lo=.01,hi=100)
    upper=_value(flow,"trip_upper",1.;lo=0,hi=1);lower=_value(flow,"trip_lower",1.;lo=0,hi=1)
    script="PLOP\nG\n\nLOAD geometry.dat\nPPAR\nN $panels\n\n\nPANE\nOPER\nVISC $re\nMACH $mach\nITER $maxiter\nVPAR\nN $ncrit\nXTR $upper $lower\n\n"
    if continuation && abs(alpha)>0
        direction=sign(alpha);a=0.0
        while abs(a)<abs(alpha)-1e-8
            script*="ALFA $a\n";a+=direction*min(1.,abs(alpha)-abs(a))
        end
    end
    script*"PACC\npolar.txt\n\nALFA $alpha\nCPWR cp.txt\nPACC\n\nQUIT\n"
end

function _geometry(case,foil)
    kind=case["airfoil"]["kind"]
    if kind=="naca4"
        f=AirfoilJXBackend.JXfoil.naca4_airfoil(parse(Int,case["airfoil"]["id"]);n=401)
        return f,"NACA four-digit equations,401 cosine-spaced points; actual executable repanels to240 computational panels"
    end
    foil,"native source coordinates; actual executable repanels to240 computational panels"
end

# geometry_override (23 Oct 2026): nothing (every existing use, unchanged) or a NamedTuple (geometry=(name,x,y),
# method, scope, pairing, paired_sha256) when XFOIL must run the section another solver uses (the low-speed CFD's).
function run_case(case,foil,outdir;log=_->nothing,progress=_->nothing,timeout=45.0,max_log_bytes=16_000_000,maxiter=500,
                  geometry_override=nothing)
    exe=executable();isfile(exe)||throw(ArgumentError("XFOIL executable not found. Set AIRFOIL_XFOIL_EXE to the installed xfoil.exe path."))
    flow=case["flow"];_value(flow,"mach",0.;lo=0,hi=prevfloat(1.))
    geometry,method=geometry_override===nothing ? _geometry(case,foil) : (geometry_override.geometry,geometry_override.method)
    exe_sha=bytes2hex(sha256(read(exe)));points=Any[]
    root=joinpath(outdir,"xfoil");mkpath(root)
    for (index,alpha) in enumerate(case["alphas"])
        AirfoilJobControl.check_cancelled()
        directory=joinpath(root,@sprintf("point_%03d_%+.6f_%s",index,Float64(alpha),string(uuid4())[1:8]));mkpath(directory)
        progress(Dict("event"=>"started","alpha_deg"=>alpha,"point_index"=>index))
        log("Actual XFOIL executable: point $index/$(length(case["alphas"])), alpha=$alpha deg; Mach=$(flow["mach"]), Re=$(flow["reynolds"])")
        attempts=Any[];chosen=nothing;cp=Dict("upper"=>Any[],"lower"=>Any[]);selected_dir=directory
        for (attempt,continuation) in enumerate((false,true))
            AirfoilJobControl.check_cancelled()
            continuation && abs(alpha)<1e-8 && break
            work=joinpath(directory,continuation ? "continuation_retry" : "direct");mkpath(work)
            source=joinpath(work,"geometry.dat")
            open(source,"w") do io
                println(io,replace(geometry.name,'\n'=>' ','\r'=>' '))
                for (x,y) in zip(geometry.x,geometry.y);println(io,x," ",y);end
            end
            input=joinpath(work,"input.txt");write(input,_script(flow,Float64(alpha);continuation,maxiter))
            command=Cmd(Cmd([exe]);dir=work,windows_hide=true)
            result=_bounded_process(command,input,joinpath(work,"output.log");timeout,max_log_bytes,
                heartbeat=elapsed->progress(Dict("event"=>"running","alpha_deg"=>alpha,"elapsed_seconds"=>elapsed,"attempt"=>attempt)))
            candidate=_polar(joinpath(work,"polar.txt"),alpha);candidate_cp=_cp(joinpath(work,"cp.txt"))
            accepted=isempty(result["stop_reason"]) && result["exit_code"]==0 && candidate!==nothing &&
                !isempty(candidate_cp["upper"])&&!isempty(candidate_cp["lower"])
            result["accepted_converged_polar_entry"]=accepted;result["attempt"]=attempt
            result["initialization"]=continuation ? "independent zero-to-target continuation with<=1deg increments" : "independent freestream target"
            push!(attempts,result);selected_dir=work
            if accepted;chosen=candidate;cp=candidate_cp;break;end
            log("XFOIL alpha=$alpha attempt$attempt yielded no accepted target polar/Cp entry ($(result["stop_reason"]), exit$(result["exit_code"])); retaining diagnostic files")
        end
        modified=get(flow,"cfd_te_thickness",0.)>0
        provenance=Dict{String,Any}("backend"=>"actual_xfoil_executable","executable_sha256"=>exe_sha,
            "executable_name"=>basename(exe),"requested_mach"=>flow["mach"],"analysis_mach"=>flow["mach"],
            "reynolds"=>flow["reynolds"],"ncrit"=>get(flow,"ncrit",9.),
            "trip_upper"=>get(flow,"trip_upper",1.),"trip_lower"=>get(flow,"trip_lower",1.),
            "max_iterations"=>maxiter,"surface_panels"=>240,"geometry_method"=>method,
            "source_geometry_sha256"=>bytes2hex(sha256(read(joinpath(selected_dir,"geometry.dat")))),
            "cfd_geometry_modified"=>modified,"geometry_match_scope"=>modified ? "CFD trailing-edge override changes CFD geometry; XFOIL retains native source contour, so no matched-geometry parity claim" : "same analytic NACA definition or native source contour; solver computational panelings differ",
            "attempts"=>attempts,"converged"=>chosen!==nothing,
            "convergence_evidence"=>"fresh finite matching-alpha polar row and nonempty Cp from a normally exited actual executable; no aerodynamic or stall validation certificate")
        if geometry_override!==nothing
            provenance["geometry_match_scope"]=geometry_override.scope
            provenance["geometry_pairing"]=geometry_override.pairing
            provenance["paired_cfd_geometry_sha256"]=geometry_override.paired_sha256
        end
        open(joinpath(directory,"provenance.toml"),"w") do io;TOML.print(io,provenance);end
        point=Dict{String,Any}("backend"=>"xfoil","alpha_deg"=>alpha,"analysis_mach"=>flow["mach"],
            "requested_mach"=>flow["mach"],"reynolds"=>flow["reynolds"],"converged"=>chosen!==nothing,
            "status"=>chosen===nothing ? "nonconverged" : "converged",
            "cl"=>nothing,"cd"=>nothing,"cm"=>nothing,"cp"=>cp,"fields"=>Dict{String,Any}(),"available_fields"=>String[],
            "provenance"=>provenance,"native_geometry"=>Dict("name"=>geometry.name,"x"=>copy(geometry.x),"y"=>copy(geometry.y),"role"=>"actual XFOIL input geometry"),
            "message"=>chosen===nothing ? "Actual XFOIL did not produce a finite converged target polar and Cp within bounded attempts. Failed point retained; no interpolation." : "Actual XFOIL executable returned a converged polar entry at the requested Mach. CFD and stall parity remain separate validation tasks.",
            "artifacts"=>Dict("polar"=>joinpath(selected_dir,"polar.txt"),"cp"=>joinpath(selected_dir,"cp.txt"),
                "geometry"=>joinpath(selected_dir,"geometry.dat"),"input"=>joinpath(selected_dir,"input.txt"),"log"=>joinpath(selected_dir,"output.log"),"provenance"=>joinpath(directory,"provenance.toml")))
        modified && (point["message"]*=" CFD uses an explicitly modified trailing edge; these native-XFOIL and CFD geometries differ.")
        chosen===nothing || merge!(point,chosen)
        push!(points,point);progress(Dict("event"=>chosen===nothing ? "point_failed" : "point_complete","alpha_deg"=>alpha,"point"=>point))
    end
    Dict("results"=>points)
end

end
