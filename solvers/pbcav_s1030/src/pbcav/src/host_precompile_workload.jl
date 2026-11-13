"CPU-only projection, nonzero coupled implicit SST step and read-only diagnostics."
function _cpu_sst_workload()
    P=PBCav
    m=P.box_mesh(8,8;Lx=1.,Ly=.25)
    kinds=Dict(P.BT_W=>P.BC_INLET,P.BT_E=>P.BC_FREESTREAM_OUTLET,
        P.BT_S=>P.BC_NOSLIP,P.BT_N=>P.BC_SLIP)
    values=Dict(P.BT_W=>(1.,0.,0.),P.BT_E=>(1.,0.,0.))
    bc=P.bcset(m;kinds,values)
    cfg=P.PBConfig(mu=1/3e6,dt=.005,nonorth=1.,turbulence=:sst,sst_tu=.01,sst_mut_ratio=10.,
        n_outer_max=40,cpl_au=.95,cpl_restart=60,cpl_maxit=300,
        turb_step_relax=.5,turb_mut_base=:committed)
    st=P.allocate_state(m);w=P.pbwork(m,bc,cfg)
    P.init_state!(st,m,(x,y)->(1-exp(-y/.025),.01*sin(2pi*x)*sin(4pi*y),0.))
    P.init_turbulence!(st,m,bc,cfg;work=w)
    projection=P.project_start!(st,m,bc,cfg;work=w,bernoulli=false)
    # The precompile task is one bounded step, not a flow convergence attempt.
    result=P.pbcav_run!(st,m,bc,cfg;wall_cap=600.,max_steps=1,work=w,
        steady_tol=1e-6,turb_steady_tol=1e-7,max_halvings=2)
    shear=P.wall_shear(st,m,bc,cfg;work=w)
    sample=deepcopy(st);diagnostic_work=P.pbwork(m,bc,cfg)
    turbulence=P.turbulence_steady_residual(sample,m,bc,diagnostic_work,cfg)
    (state=st,mesh=m,bc=bc,config=cfg,projection=projection,result=result,
        wall=shear,turbulence=turbulence)
end
