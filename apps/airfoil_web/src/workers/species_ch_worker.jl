# The two-block (C+H) branch of the shared Species worker: Species air and the free-fluid domain on a finite
# (blunt) trailing edge in a free stream. Included at CFDBackend module scope; reached from `_species_worker`
# (species_backend.jl) when the input names the C+H mesh.
#
# The solver is the production strip pair of src/cmesh/cmesh_species_strip.jl, unchanged. Its C block is a
# free-stream block here: the Species carrier's free-fluid contract (every outer face characteristic, the mixture
# far field; fixed static pressure on the downstream ends) instead of the tunnel's walls and inlet. Both blocks are
# configured by the same `_species_carrier` call, the interface flux is the strip's one value on both sides, and
# the pair's conservation ledger is on, so the result carries a measured mass and species balance over both blocks.

"The C+H mesh of the run and both blocks' metrics: the C block's outer faces all characteristic, the strip's all interface."
function _species_ch_blocks(cfg)
    af=haskey(cfg,"naca4_code") ? CMeshGen.naca4_points(cfg["naca4_code"]) : CMeshGen.read_airfoil_dat(cfg["geometry"])
    flow=Dict{String,Any}("reynolds"=>cfg["reynolds"],"cfd_te_thickness"=>cfg["cfd_te_thickness"],
        "cfd_farfield_angle_deg"=>cfg["cfd_farfield_angle_deg"])
    for key in _SPECIES_CH_MESH_KEYS
        haskey(cfg,key) && (flow[key]=cfg[key])
    end
    built=_species_ch_mesh(af,flow,cfg["resolution"]);m=built.mesh
    # The base is a wall of both blocks: the strip's only wall, and a segment the C block's open foil range lacks.
    base=[(m.X[m.NIW+1,1],m.Y[m.NIW+1,1],m.X[m.NIW+m.NIF+1,1],m.Y[m.NIW+m.NIF+1,1])]
    c=_species_external_metrics(CMeshMetrics._build_metrics(m.X,m.Y,m.NIW,m.NIF,fill(UInt8(0),m.NI);
        trip_upper=cfg["trip_upper"],trip_lower=cfg["trip_lower"],extra_segments=base))
    h=CMeshMetrics._build_metrics(m.strip.X,m.strip.Y,m.strip.NI,0,fill(UInt8(4),m.strip.NI);
        channel_provenance=true,wall_segments=base)
    (;mesh=m,metadata=built.metadata,c,h)
end

"""The pair's inventories over both blocks' interiors (Float64): mass, x and y momentum and the vapour species.
Each block is summed pairwise (`sum` of an array), so the sum's own rounding stays below the balance it measures."""
function _species_pair_inventory(c)
    total=zeros(4)
    for q in (c,c.strip)
        U=Array(q.conserved.U);vol=Float64.(Array(q.solver.vol))
        for k in 1:4
            total[k]+=sum(view(U,:,:,k).*vol)
        end
    end
    total
end

"""The force of every wall face of the pair on the body, as rows (block, index, x, y, fx, fy): the C block's foil
faces and the H block's base faces (its column 1), each the published pressure and the no-slip Stokes traction of
the first cell, the terms `loads_ka` and `loads_strip_ka` sum."""
function _species_ch_face_forces(c,m)
    rows=Tuple{String,Int,Float64,Float64,Float64,Float64}[]
    function face!(block,index,xa,ya,xb,yb,sx,sy,vol,rho,u,v,p,nu)
        sf=hypot(sx,sy);d1=0.5vol/sf+1e-30;mu=rho*nu*sf/d1
        tx,ty=CMeshSolverKA._wall_stokes_traction(u,v,sx/sf,sy/sf,mu)
        push!(rows,(block,index,0.5(xa+xb),0.5(ya+yb),-p*sx+tx,-p*sy+ty))
    end
    s=c.solver;g=s.g;foil=m.NIW+1:m.NIW+m.NIF;range=(first(foil)+g):(last(foil)+g)
    rho,u,v,p=map(a->Float64.(Array(view(a,range,g+1))),(s.rho,s.ux,s.uy,c.pressure))
    for (k,i) in enumerate(foil)
        face!("C",i,m.X[i,1],m.Y[i,1],m.X[i+1,1],m.Y[i+1,1],Float64(s.Sjx1_h[i]),Float64(s.Sjy1_h[i]),
            Float64(s.vol1_h[i]),rho[k],u[k],v[k],p[k],Float64(s.nu_lam))
    end
    x=c.strip;ss=x.solver;rows_h=(g+1):(g+ss.NJ)
    rho,u,v,p=map(a->Float64.(Array(view(a,g+1,rows_h))),(ss.rho,ss.ux,ss.uy,x.pressure))
    for j in 1:ss.NJ
        face!("H",j,m.strip.X[1,j],m.strip.Y[1,j],m.strip.X[1,j+1],m.strip.Y[1,j+1],Float64(x.Six1_h[j]),
            Float64(x.Siy1_h[j]),Float64(x.voli1_h[j]),rho[j],u[j],v[j],p[j],Float64(ss.nu_lam))
    end
    rows
end

"The pair's coefficients: the solver's own surface and base loads, and the moment of the wall-face forces."
function _species_ch_loads(c,m)
    s=c.solver
    cl,cd=CMeshSolverKA.loads_ka(s;mixture_pressure=c.pressure,wallmode=Int32(2))
    base_cl,base_cd=CMeshCavSpecies.loads_strip_ka(c.strip)
    q=0.5Float64(s.rho_inf)*(Float64(s.uxi)^2+Float64(s.uyi)^2)
    cm=sum(_aerodynamic_cm(r[3],r[4],r[5],r[6],q) for r in _species_ch_face_forces(c,m))
    (;cl=cl+base_cl,cd=cd+base_cd,cm,base_cl,base_cd)
end

"""Whether every interface face of the committed state has one flux on both sides, in every transported quantity:
the lower leg's entry is the exact negative of the strip's bottom face and the upper leg's entry equals the strip's
top face (the pair's copy makes this structural; this reads it back). Evaluates the pair's fluxes of that state."""
function _species_ch_interface_single_valued(c)
    P=CMeshCavSpecies;x=c.strip;s,ss=c.solver,x.solver
    P._stage_residual_pair!(c,Int32(2));P._species_flux_pair!(c)
    pairs=Any[(s.Fj_r,ss.Fj_r),(s.Fj_mx,ss.Fj_mx),(s.Fj_my,ss.Fj_my),(s.Fj_e,ss.Fj_e),(c.Fj,x.Fj)]
    if s.sst
        P._sst_residual_pair!(c)
        push!(pairs,(c.turbulence.Fjk,x.turbulence.Fjk),(c.turbulence.Fjw,x.turbulence.Fjw))
    end
    all(pairs) do (a,b)
        leg=Array(view(a,:,1));bottom=Array(view(b,:,1));top=Array(view(b,:,ss.NJ+1))
        all(leg[s.NIW+1-k]==-bottom[k] && leg[s.NIW+s.NIF+k]==top[k] for k in 1:s.NIW)
    end
end

"Both blocks' final fields (the two-block field files of the C+H workers) and the foil's surface pressure."
function _export_species_ch_fields(c,m,cfg,iteration,p_inf,speed)
    s=c.solver;x=c.strip;out=cfg["outdir"]
    block=_species_block_fields(s,c.pressure,p_inf,speed)
    field=_write_block_fields([(id="C",ni=s.NI,nj=s.NJ,source=()->block.at),
        (id="H",ni=x.solver.NI,nj=x.solver.NJ,source=()->_species_block_fields(x.solver,x.pressure,p_inf,speed).at)],out,iteration)
    surface=m.NIW+1:m.NIW+m.NIF;ile=surface[argmin(block.xc[surface,1])]
    open(joinpath(out,"surface_cp.csv"),"w") do io
        println(io,"surface,x,cp")
        for i in surface;println(io,i>ile ? "upper" : "lower",",",block.xc[i,1],",",block.cp(i,1));end
    end
    field
end

function _species_ch_worker(cfg,settings)
    material=cfg["cav_fluid"];domain=cfg["cav_domain"]
    refusal=species_finite_edge_refusal(cfg)
    refusal===nothing || throw(ArgumentError(refusal))
    out=cfg["outdir"]
    blocks=_species_ch_blocks(cfg);m=blocks.mesh;metadata=blocks.metadata
    metadata["geometry_support"]=haskey(cfg,"naca4_code") ? "NACA four-digit equations and natural C2 surface evaluation, identical to Create Mesh" :
        "Natural C2 surface evaluation, identical to Create Mesh"
    _write_ch_mesh_artifacts(m,metadata,out,cfg["resolution"])
    digest(a)=bytes2hex(sha256(reinterpret(UInt8,vec(a))))
    xhash,yhash,sxhash,syhash=digest(m.X),digest(m.Y),digest(m.strip.X),digest(m.strip.Y)
    println("MESH_READY,mesh_preview.msgpack");flush(stdout)
    backend=cfg["gpu"] ? CUDA.CUDABackend() : KernelAbstractions.CPU()
    (;s,tem,p_ref,a2m,speed)=_species_carrier(backend,blocks.c,cfg,settings)
    strip=_species_carrier(backend,blocks.h,cfg,settings)
    CMeshSolverKA.GLOBAL_DT[]=1
    # The C block's outer reference; the strip has no outer boundary and takes none.
    CMeshSolverKA.seed_channel_reference_ka!(s)
    CMeshCavSpecies.LEDGER_ENABLED[]=true
    trip_x=settings["cav_trip_x"]
    material=="air" && cfg["trip_upper"]!=cfg["trip_lower"] && trip_x==0 && println(
        "Species asymmetric trip locations unsupported; geometry trip mask retained as metadata only, no k-floor trip applied.")
    c=CMeshCavSpecies.build_species_ka(s,tem;wallmode=Int32(2),reconstruction=settings["cav_species_reconstruction"],
        eos_precision=Symbol(settings["cav_eos_precision"]),nut_density=Symbol(settings["cav_nut_density"]),
        reboud_n=settings["cav_reboud_n"],sst_correction_limit=settings["cav_sst_correction_limit"],trip_x,trip_width=settings["cav_trip_width"],
        stage_clock_margin=cfg["cav_stage_clock_margin"],strip=(solver=strip.s,tem=strip.tem))
    ss=strip.s;cells=s.NI*s.NJ+ss.NI*ss.NJ
    @printf("Conservative Species C+H pair initialized on %s: %s, %s; C block %d x %d, H block %d x %d (%d cells); physical time\n",
        cfg["gpu"] ? "GPU" : "CPU",material,domain,s.NI,s.NJ,ss.NI,ss.NJ,cells)
    println("BOUNDARY: ",metadata["outer_boundary_condition"])
    println("SPECIES MODEL SETTINGS: ",settings,"; actual Reboud form=",c.reboud_form," exponent=",c.reboud_n," trip=",c.trip,
        " SST limit=",c.sst_climit," stage clock margin=",c.stage_margin);flush(stdout)
    material=="air" && println("AIR EOS: isothermal p=c_air^2*rho, no phase change, no total-energy equation")
    println("MESH COORDINATE RECEIPT: x_sha256=$xhash y_sha256=$yhash strip_x_sha256=$sxhash strip_y_sha256=$syhash");flush(stdout)
    # The pair's balance over the march: the inventory change of both blocks against the ledger's boundary and
    # source sums of every accepted step.
    start=_species_pair_inventory(c);mass_in=0.0;species_in=0.0;species_source=0.0;retries=0
    function advance!()
        CMeshCavSpecies.step_species_ka!(c,Float32(cfg["cfl"]);admission=:deferred)
        mass_in+=c.last_mass_boundary;species_in+=c.last_species_boundary;species_source+=c.last_species_source
        retries+=c.last_retries
    end
    function sample()
        CMeshCavSpecies._stage_residual_pair!(c,Int32(2))
        raw,scaled,l2,net=_species_mass_residual((s,ss),speed)
        load=_species_ch_loads(c,m)
        (;raw,scaled,l2,mass_net=net,cl=load.cl,cd=load.cd,cm=load.cm)
    end
    march=_species_march(cfg,c,false,speed;advance!,sample)
    finish=_species_pair_inventory(c)
    single_valued=_species_ch_interface_single_valued(c)
    single_valued || error("Species C+H interface flux is not single-valued")
    load=_species_ch_loads(c,m)
    open(joinpath(out,"wall_force_faces.csv"),"w") do io
        println(io,"block,index,x,y,fx,fy")
        for row in _species_ch_face_forces(c,m);println(io,join(row,','));end
    end
    field=_export_species_ch_fields(c,m,cfg,march.steps,p_ref,speed)
    meta=_species_result_meta(cfg,settings,metadata,c,tem,march,false,false,trip_x,speed,a2m;cm_last=load.cm)
    merge!(meta,Dict{String,Any}("mesh_mode"=>SPECIES_CH_MESH,"mesh_topology"=>SPECIES_CH_MESH,"mesh_blocks"=>2,
        "block_cells"=>Dict("C"=>s.NI*s.NJ,"H"=>ss.NI*ss.NJ),"mesh_cells"=>cells,
        "mesh_node_x_sha256"=>xhash,"mesh_node_y_sha256"=>yhash,"strip_node_x_sha256"=>sxhash,"strip_node_y_sha256"=>syhash,
        "full_field_mesh_sha256"=>field.meshsha,"full_field_cells"=>field.total,
        "fields_blocks_msgpack"=>"fields_blocks.msgpack","field_full_msgpack"=>"fields_blocks_full.msgpack",
        "cp_csv"=>"surface_cp.csv","iteration_csv"=>"iteration_history.csv","wall_force_csv"=>"wall_force_faces.csv",
        "coefficient_scope"=>"foil surface (C block) plus the base (H block's wall)",
        "base_cl"=>load.base_cl,"base_cd"=>load.base_cd,
        "stage_clock_margin"=>Float64(c.stage_margin),"retried_attempts"=>retries,
        "interface_flux_single_valued"=>single_valued,
        "pair_conservation"=>Dict{String,Any}(
            "mass_inventory"=>finish[1],"mass_change"=>finish[1]-start[1],"mass_boundary"=>mass_in,
            "mass_residual_relative"=>abs(finish[1]-start[1]-mass_in)/finish[1],
            "species_inventory"=>finish[4],"species_change"=>finish[4]-start[4],
            "species_boundary"=>species_in,"species_source"=>species_source,
            "species_residual"=>finish[4]-start[4]-species_in-species_source,
            "scope"=>"both blocks' interiors against the pair's conservation ledger summed over the accepted steps"),
        "validation"=>"experimental two-block Species backend in a free stream; aerodynamic, cavitation and stall parity unestablished"))
    open(joinpath(out,"result.toml"),"w") do io;TOML.print(io,meta);end
    println("Conservative Species C+H worker complete");flush(stdout)
end
