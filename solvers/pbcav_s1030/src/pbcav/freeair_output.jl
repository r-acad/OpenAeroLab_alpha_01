"Native free-air outputs; portable checkpoints contain arrays and TOML, never serialized Julia objects."
const FREEAIR_RESTART_KIND = "PBCavFreeAir/1"
const FREEAIR_RESTART_MUTABLE_KNOBS = Set(("PB_RESTART", "PB_OUT_DIR", "PB_MAX_STEPS", "PB_WALL_CAP",
    "PB_STEADY_TOL", "PB_TURB_STEADY_TOL", "PB_OUTPUT_FIELDS", "PB_OUTPUT_CHECKPOINT", "PB_FOIL", "PB_STOP"))

function _freeair_disk_source_signature()
    io=IOBuffer()
    for f in sort!(filter(f->endswith(f,".jl"),readdir(@__DIR__)))
        write(io,f);write(io,read(joinpath(@__DIR__,f)))
    end
    bytes2hex(sha256(take!(io)))
end
const FREEAIR_LOADED_SOURCE_SHA256 = _freeair_disk_source_signature()
# A long run may outlive edits on disk. Stamp the code loaded by this module,
# never whatever source happens to be present when the final checkpoint saves.
freeair_source_signature() = FREEAIR_LOADED_SOURCE_SHA256
freeair_restart_knobs(k) = Dict(String(n)=>repr(v) for (n,v) in k if !(n in FREEAIR_RESTART_MUTABLE_KNOBS))
function freeair_mesh_signature(m::PBMesh)
    io=IOBuffer()
    for f in (:xn,:yn,:cnode,:own,:nei,:bown,:btag)
        write(io,reinterpret(UInt8,vec(Array(getfield(m,f)))))
    end
    bytes2hex(sha256(take!(io)))
end

"Save all time levels, current boundary conditions, timestep controller, circulation and conservation ledger."
function write_freeair_checkpoint(path::AbstractString,m::PBMesh,st::PBState,bc::BCSet,k::AbstractDict;
                                  vortex=nothing,control=Dict{String,Any}(),ledger=nothing)
    arrays=Pair{String,Array}[string(f)=>Array(getfield(st,f)) for f in STATE_FIELDS]
    append!(arrays,["bc_"*string(f)=>Array(getfield(bc,f)) for f in (:kind,:ub,:vb,:pb)])
    metadata=Dict{String,Any}("sigma"=>0.0,"p_ref"=>0.0,"q"=>0.0,"p_v"=>0.0,
        "kind"=>FREEAIR_RESTART_KIND,"knobs"=>freeair_restart_knobs(k),
        "mesh_hashes"=>[freeair_mesh_signature(m)],"source_sha256"=>freeair_source_signature(),
        "t"=>st.t,"dt"=>st.dt,"dt_prev"=>st.dt_prev,"nstep"=>st.nstep,"control"=>control,
        "circulation"=>vortex===nothing ? 0.0 : vortex.G,
        "vortex_steps"=>vortex===nothing ? Int[] : [x[1] for x in vortex.hist],
        "vortex_values"=>vortex===nothing ? Float64[] : [x[2] for x in vortex.hist])
    if ledger!==nothing
        metadata["ledger"]=Dict(string(f)=>(f===:rejections ? Dict(string(n)=>v for (n,v) in ledger.rejections) : getfield(ledger,f))
            for f in fieldnames(PBLedger) if f!==:last_rejection)
    end
    PBRestartIO.write_pbrestart(path,metadata,arrays;writer="PBCav native free-air driver")
end

"Validate an entire checkpoint before changing the destination state or boundary arrays."
function read_freeair_checkpoint!(path::AbstractString,m::PBMesh,st::PBState,bc::BCSet,k::AbstractDict;
                                 control=Dict{String,Any}(),ledger=nothing)
    header,arrays=PBRestartIO.read_pbrestart(path);meta=header["meta"]
    get(meta,"kind","")==FREEAIR_RESTART_KIND || throw(ArgumentError("Checkpoint is not a native free-air continuation"))
    meta["source_sha256"]==freeair_source_signature() || throw(ArgumentError("Checkpoint solver source differs from this solver"))
    meta["mesh_hashes"]==[freeair_mesh_signature(m)] || throw(ArgumentError("Checkpoint mesh, connectivity or boundary tags differ"))
    meta["knobs"]==freeair_restart_knobs(k) || throw(ArgumentError("Checkpoint physics or numerical settings differ; only budgets, stop limits and output paths may change"))
    for (prefix,obj,fields) in (("",st,STATE_FIELDS),("bc_",bc,(:kind,:ub,:vb,:pb)))
        for f in fields
            name=prefix*string(f);a=get(arrays,name,nothing);dest=getfield(obj,f)
            a isa Array && size(a)==size(dest) && eltype(a)==eltype(dest) || throw(ArgumentError("Checkpoint array $name has incompatible size or type"))
            all(isfinite,a) || throw(ArgumentError("Checkpoint array $name contains nonfinite values"))
        end
    end
    # The native free-air path has no phase change. Match the same density and
    # SST admissibility rules used before a step, across every stored time level.
    all(>(0),arrays["rho"]) || throw(ArgumentError("Checkpoint density is not positive"))
    maximum(abs,arrays["rho"].-1.)<=SST_THETA_MAX || throw(ArgumentError("Checkpoint free-air density is not uniform"))
    all(iszero,arrays["q"]) || throw(ArgumentError("Checkpoint contains vapour in a single-phase free-air run"))
    for name in ("rk","rw","mut")
        all(>=(0),arrays[name]) || throw(ArgumentError("Checkpoint $name is negative"))
    end
    if get(k,"PB_TURBULENCE",:laminar) === :sst
        all(>(0),arrays["rw"]) || throw(ArgumentError("Checkpoint SST omega is not positive"))
    else
        all(name->all(iszero,arrays[name]),("rk","rw","mut")) ||
            throw(ArgumentError("Checkpoint laminar state contains active turbulence fields"))
    end
    arrays["bc_kind"]==Array(bc.kind) || throw(ArgumentError("Checkpoint boundary condition kinds differ"))
    # Convert every scalar, controller row, ledger row and vortex record before
    # the first copyto!. A malformed integer or table must leave all destinations
    # (including control and ledger) unchanged.
    finite_real(value,label;positive=false,nonnegative=false) = begin
        value isa Real && !(value isa Bool) && isfinite(value) || throw(ArgumentError("Checkpoint $label must be a finite number"))
        converted=Float64(value)
        isfinite(converted) && (!positive || converted>0) && (!nonnegative || converted>=0) ||
            throw(ArgumentError("Checkpoint $label is outside its admitted range"))
        converted
    end
    count_value(value,label) = begin
        value isa Integer && !(value isa Bool) && 0<=value<=typemax(Int) ||
            throw(ArgumentError("Checkpoint $label must be a nonnegative integer"))
        Int(value)
    end
    newtime=finite_real(get(meta,"t",nothing),"t";nonnegative=true)
    newdt=finite_real(get(meta,"dt",nothing),"dt";positive=true)
    newprev=finite_real(get(meta,"dt_prev",nothing),"dt_prev";nonnegative=true)
    newstep=count_value(get(meta,"nstep",nothing),"nstep")
    (newstep==0 || newprev>0) && newtime>=newprev || throw(ArgumentError("Checkpoint time levels are inconsistent"))
    rawcontrol=get(meta,"control",nothing)
    rawcontrol isa AbstractDict && all(name->name in ("streak","nrej","nstall","tally"),keys(rawcontrol)) ||
        throw(ArgumentError("Checkpoint timestep controller is invalid"))
    newcontrol=Dict{String,Any}(name=>count_value(get(rawcontrol,name,0),"controller.$name") for name in ("streak","nrej","nstall"))
    rawtally=get(rawcontrol,"tally",Dict{String,Int}())
    rawtally isa AbstractDict && all(name->Symbol(name) in REJECTION_REASONS,keys(rawtally)) ||
        throw(ArgumentError("Checkpoint controller rejection reasons are invalid"))
    newcontrol["tally"]=Dict{String,Int}(String(name)=>count_value(value,"controller.tally.$name") for (name,value) in rawtally)
    sum(values(newcontrol["tally"]);init=Int128(0))==newcontrol["nrej"] && newcontrol["nstall"]<=newcontrol["nrej"] ||
        throw(ArgumentError("Checkpoint rejection counters are inconsistent"))
    staged_control=copy(control);empty!(staged_control)
    try
        merge!(staged_control,newcontrol)
    catch
        throw(ArgumentError("Checkpoint controller destination has incompatible types"))
    end
    circulation=finite_real(get(meta,"circulation",nothing),"circulation")
    steps=get(meta,"vortex_steps",nothing);values_=get(meta,"vortex_values",nothing)
    steps isa AbstractVector && values_ isa AbstractVector && length(steps)==length(values_) ||
        throw(ArgumentError("Checkpoint vortex history is incomplete"))
    vortex_steps=[count_value(value,"vortex step") for value in steps]
    vortex_values=[finite_real(value,"vortex value") for value in values_]
    all(step->0<step<=newstep,vortex_steps) && all(>(0),diff(vortex_steps)) ||
        throw(ArgumentError("Checkpoint vortex steps are invalid"))
    (isempty(vortex_values) || last(vortex_values)==circulation) || throw(ArgumentError("Checkpoint circulation and vortex history differ"))
    if get(k,"PB_VORTEX",:none) === :none
        iszero(circulation) && isempty(vortex_steps) || throw(ArgumentError("Checkpoint has a vortex that this run does not use"))
    else
        every=Int(k["PB_VORTEX_EVERY"])
        all(step->step%every==0,vortex_steps) || throw(ArgumentError("Checkpoint vortex update steps differ from the configured interval"))
    end
    staged_ledger=nothing
    ledger===nothing || ledger isa PBLedger || throw(ArgumentError("Checkpoint ledger destination is incompatible"))
    if haskey(meta,"ledger")
        rawledger=meta["ledger"]
        names=filter(!=(:last_rejection),fieldnames(PBLedger))
        rawledger isa AbstractDict && Set(String.(keys(rawledger)))==Set(string.(names)) ||
            throw(ArgumentError("Checkpoint conservation ledger has missing or unknown rows"))
        staged_ledger=PBLedger()
        for f in names
            value=rawledger[string(f)]
            converted=if f===:rejections
                value isa AbstractDict && all(name->Symbol(name) in REJECTION_REASONS,keys(value)) ||
                    throw(ArgumentError("Checkpoint ledger rejection reasons are invalid"))
                Dict{Symbol,Int}(Symbol(name)=>count_value(n,"ledger.rejections.$name") for (name,n) in value)
            elseif fieldtype(PBLedger,f)===Int
                count_value(value,"ledger.$f")
            else
                finite_real(value,"ledger.$f")
            end
            setfield!(staged_ledger,f,converted)
        end
        staged_ledger.nsteps==newstep || throw(ArgumentError("Checkpoint ledger and state step counts differ"))
    elseif ledger!==nothing && newstep>0
        throw(ArgumentError("Checkpoint has no conservation ledger for continuation"))
    end
    for (prefix,obj,fields) in (("",st,STATE_FIELDS),("bc_",bc,(:kind,:ub,:vb,:pb))), f in fields
        copyto!(getfield(obj,f),arrays[prefix*string(f)])
    end
    st.t=newtime;st.dt=newdt;st.dt_prev=newprev;st.nstep=newstep
    empty!(control);merge!(control,staged_control)
    if ledger!==nothing && staged_ledger!==nothing
        for f in fieldnames(PBLedger)
            setfield!(ledger,f,getfield(staged_ledger,f))
        end
    end
    meta["vortex_history"]=collect(zip(vortex_steps,vortex_values))
    meta
end

"Save the committed state as an unstructured VTU with actual solver cells and common aerodynamic fields."
function write_freeair_vtu(path::AbstractString,m::PBMesh,st::PBState,cfg,fs::FreeStream)
    ispath(path) && throw(ArgumentError("Refusing to overwrite $path"))
    x=Array(m.xn);y=Array(m.yn);cn=Array(m.cnode)
    u=Array(lev(st.ux,2));v=Array(lev(st.uy,2));p=Array(lev(st.p,2));rho=Array(lev(st.rho,2))
    q=0.5fs.rho*fs.U^2
    fields=("velocity_u"=>u,"velocity_v"=>v,"speed"=>hypot.(u,v),"pressure"=>p,"Cp"=>(p.-fs.p_inf)./q,
        "density"=>rho,"k"=>Array(lev(st.rk,2))./rho,"omega"=>Array(lev(st.rw,2))./rho,
        "eddy_viscosity_ratio"=>Array(lev(st.mut,2))./cfg.mu)
    all(pr->all(isfinite,last(pr)),fields) || throw(ArgumentError("Refusing to export nonfinite committed fields"))
    tmp=path*".partial."*string(getpid())
    try
        open(tmp,"w") do io
            println(io,"<?xml version=\"1.0\"?><VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\"><UnstructuredGrid>")
            println(io,"<Piece NumberOfPoints=\"$(length(x))\" NumberOfCells=\"$(m.nc)\"><Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">")
            for n in eachindex(x);println(io,x[n],' ',y[n]," 0");end
            println(io,"</DataArray></Points><Cells><DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">")
            for c in 1:m.nc;println(io,join(cn[:,c].-1,' '));end
            println(io,"</DataArray><DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">",join(4:4:4m.nc,' '),"</DataArray><DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">",join(fill(9,m.nc),' '),"</DataArray></Cells><CellData>")
            for (name,values) in fields
                println(io,"<DataArray type=\"Float64\" Name=\"$name\" format=\"ascii\">",join(values,' '),"</DataArray>")
            end
            println(io,"</CellData></Piece></UnstructuredGrid></VTKFile>")
        end
        mv(tmp,path)
    finally
        isfile(tmp)&&rm(tmp)
    end
    path
end
