# Geometry-only paired refinement for a finite-base C+H mesh. Included inside
# CMeshGen after the generator definitions. This never transfers a CFD state.
export paired_uniform_strip_refine

function _paired_strip_source(m::ChannelMesh)
    s=m.strip
    s===nothing && throw(ArgumentError("paired strip refinement requires a C+H mesh with two distinct wake interfaces"))
    m.NI==2m.NIW+m.NIF && size(m.X)==size(m.Y)==(m.NI+1,m.NJ+1) ||
        throw(ArgumentError("paired strip refinement received inconsistent C-block dimensions"))
    s.NI==m.NIW && size(s.X)==size(s.Y)==(m.NIW+1,s.NJ+1) ||
        throw(ArgumentError("paired strip refinement received inconsistent H-block dimensions"))
    length(m.growth)==m.NI+1 && length(m.outer_face_kind)==m.NI ||
        throw(ArgumentError("paired strip refinement received inconsistent growth or boundary tags"))
    all(isfinite,m.X) && all(isfinite,m.Y) && all(isfinite,s.X) && all(isfinite,s.Y) ||
        throw(ArgumentError("paired strip refinement requires finite node coordinates"))
    s
end

"Check both blocks and their actual interfaces; optionally enforce every existing air gate."
function _paired_strip_quality_report(m::ChannelMesh;check_air_gates::Bool=true)
    s=_paired_strip_source(m)
    lower_exact=all(s.X[k,1]===m.X[m.NIW+2-k,1] &&
                    s.Y[k,1]===m.Y[m.NIW+2-k,1] for k in 1:m.NIW+1)
    upper_exact=all(s.X[k,end]===m.X[m.NIW+m.NIF+k,1] &&
                    s.Y[k,end]===m.Y[m.NIW+m.NIF+k,1] for k in 1:m.NIW+1)
    lower_exact && upper_exact || error("paired strip refinement lost bitwise interface node identity")
    _te_post_checks_strip(m.X,m.Y,s.X,s.Y,m.NI,m.NJ,m.NIW,m.NIF,s.NJ,m.channel)
    summary=CMeshQualityMetrics.quality_summary(m)
    gates=CMeshQualityMetrics.quality_gates(summary;cavitating=false,strip_grade=true)
    failed=CMeshQualityMetrics.gate_failures(gates)
    check_air_gates && !isempty(failed) && throw(ArgumentError(
        "paired strip mesh fails existing air quality gates: "*join(failed,"; ")))
    (;summary,gates,failed,passed=isempty(failed),lower_interface_bitwise=lower_exact,
      upper_interface_bitwise=upper_exact)
end

# Exact parent copies at child zero and at the last node preserve every parent
# ray before C2 projection. The four TE-adjacent intervals provide a small
# positive corner-width margin without changing any parent's interval sum.
function _paired_strip_subdivide(A::Matrix{Float64},m::ChannelMesh,factor::Int,
                                  te_margin::Float64)
    out=Matrix{Float64}(undef,m.NI*factor+1,size(A,2))
    for i in 1:m.NI
        widths=ones(factor)
        if factor>1 && i in (m.NIW,m.NIW+m.NIF,m.NIW+1,m.NIW+m.NIF+1)
            widths .= (factor-te_margin)/(factor-1)
            widths[i in (m.NIW,m.NIW+m.NIF) ? factor : 1]=te_margin
        end
        partial=0.0
        for k in 0:factor-1
            row=(i-1)*factor+k+1
            if k==0
                out[row,:].=A[i,:]
            else
                t=partial/factor
                out[row,:].=(1-t).*A[i,:].+t.*A[i+1,:]
            end
            partial+=widths[k+1]
        end
    end
    out[end,:].=A[end,:]
    out
end

"""Project only the open foil surface onto the supplied source C2 curve.

The two finite-base corners, both wake interfaces, and the outer contour remain
fixed. All viscous rows move together, with a quintic fade over 0.15 chord.
"""
function _paired_strip_project_wall!(X,Y,NIW::Int,NIF::Int,curve)
    maximum_shift=0.0
    maximum_first_spacing_relative_change=0.0
    for i in NIW+2:NIW+NIF
        p=_airfoil_closest(curve,X[i,1],Y[i,1])
        dx=p.x-X[i,1];dy=p.y-Y[i,1]
        maximum_shift=max(maximum_shift,hypot(dx,dy))
        before=hypot(X[i,2]-X[i,1],Y[i,2]-Y[i,1])
        x0,y0=X[i,1],Y[i,1]
        for j in axes(X,2)
            t=clamp(hypot(X[i,j]-x0,Y[i,j]-y0)/0.15,0.0,1.0)
            weight=1-t^3*(10-15t+6t^2)
            X[i,j]+=weight*dx;Y[i,j]+=weight*dy
        end
        X[i,1]=p.x;Y[i,1]=p.y
        after=hypot(X[i,2]-X[i,1],Y[i,2]-Y[i,1])
        before>0 || error("paired strip source has a zero first wall spacing")
        maximum_first_spacing_relative_change=max(maximum_first_spacing_relative_change,
                                                   abs(after-before)/before)
    end
    (;maximum_shift,maximum_first_spacing_relative_change)
end

"""
    paired_uniform_strip_refine(m,af;factor=8,te_margin=1.05,
        cross_count=m.strip.NJ,cross_growth_max=1.15,cap=gap/8,
        check_air_gates=true) -> (mesh,metadata)

Refine a finite-base C+H *geometry* with paired streamwise subdivision. The C
block retains its two distinct interfaces. Every parent C ray is copied before
the open airfoil wall is projected to the C2 curve defined by `af`; its native
finite-base endpoints are fixed. The H block is then rebuilt from both final
interfaces using the existing symmetric graded distribution.

`check_air_gates=true` enforces the unchanged CMeshQualityV1 air gates, including
the interface volume-ratio limit. An explicitly unchecked candidate still must
pass the existing strip convexity, corner, base, outlet and connectivity checks;
its metadata says `admitted=false`. This permits a caller to apply an outlet
correction and then call `_paired_strip_quality_report` before admission.

This function never smooths the parent mapping, relaxes a gate, changes an outer
boundary kind, transfers a solution, or selects a solver boundary condition.
"""
function paired_uniform_strip_refine(m::ChannelMesh,af::AirfoilPts;
        factor::Integer=8,te_margin::Real=1.05,
        cross_count::Union{Nothing,Integer}=nothing,
        cross_growth_max::Real=1.15,cap::Union{Nothing,Real}=nothing,
        check_air_gates::Bool=true)
    source=_paired_strip_source(m)
    f=Int(factor);margin=Float64(te_margin);rho=Float64(cross_growth_max)
    1<=f<=64 || throw(ArgumentError("paired refinement factor must be in 1:64"))
    isfinite(margin) && 0<margin<f && f>1 ||
        (f==1 && margin==1.0) || throw(ArgumentError(
            "te_margin must be positive and below factor (factor=1 requires te_margin=1)"))
    isfinite(rho) && 1<=rho<=1.5 || throw(ArgumentError("cross_growth_max must be finite and in [1,1.5]"))
    ns=cross_count===nothing ? source.NJ : Int(cross_count)
    ns>=4 && iseven(ns) || throw(ArgumentError("cross_count must be even and at least four"))
    gap=hypot(m.X[m.NIW+m.NIF+1,1]-m.X[m.NIW+1,1],
              m.Y[m.NIW+m.NIF+1,1]-m.Y[m.NIW+1,1])
    isfinite(gap) && gap>1e-12 || throw(ArgumentError("paired strip refinement requires a resolved finite trailing-edge gap"))
    limit=cap===nothing ? gap/8 : Float64(cap)
    isfinite(limit) && limit>0 || throw(ArgumentError("strip cross-interval cap must be positive and finite"))
    # The caller supplies the exact contour used for the parent, avoiding any
    # implicit replacement of its finite trailing-edge geometry.
    corners=((m.X[m.NIW+1,1],m.Y[m.NIW+1,1]),
             (m.X[m.NIW+m.NIF+1,1],m.Y[m.NIW+m.NIF+1,1]))
    scale=max(1.0,maximum(abs,af.x),maximum(abs,af.y))
    max(hypot(corners[1][1]-af.x[1],corners[1][2]-af.y[1]),
        hypot(corners[2][1]-af.x[end],corners[2][2]-af.y[end]))<=1e-12scale ||
        throw(ArgumentError("source curve endpoints do not match the two finite-base corners"))
    X=_paired_strip_subdivide(m.X,m,f,margin)
    Y=_paired_strip_subdivide(m.Y,m,f,margin)
    parent_indices=1:f:m.NI*f+1
    X[parent_indices,:]==m.X && Y[parent_indices,:]==m.Y ||
        error("paired strip refinement failed to preserve parent rays before projection")
    nw,nf=m.NIW*f,m.NIF*f
    projection=_paired_strip_project_wall!(X,Y,nw,nf,_airfoil_spline(af))
    Xs,Ys,t=_strip_block_q(X,Y,nw,nf,ns,rho,limit)
    dn1_i=[hypot(X[i,2]-X[i,1],Y[i,2]-Y[i,1]) for i in axes(X,1)]
    growth=Vector{Float64}(undef,m.NI*f+1)
    # Growth is a diagnostic of the parent schedule, interpolated with the same
    # child coordinates. The actual first spacing is stored independently above.
    growth_matrix=reshape(copy(m.growth),:,1)
    growth.=_paired_strip_subdivide(growth_matrix,m,f,margin)[:,1]
    strip=StripBlock(nw,ns,ns÷2,Xs,Ys,t,dn1_i)
    refined=ChannelMesh(size(X,1)-1,m.NJ,nw,nf,X,Y,m.dn1,growth,m.channel,
                        repeat(m.outer_face_kind;inner=f),strip)
    report=_paired_strip_quality_report(refined;check_air_gates)
    final_gap=hypot(Xs[1,end]-Xs[1,1],Ys[1,end]-Ys[1,1])
    final_gap==gap || error("paired strip refinement changed finite trailing-edge thickness")
    bx,by=Xs[1,end]-Xs[1,1],Ys[1,end]-Ys[1,1]
    base_normal_spacings=[abs((Xs[2,j]-Xs[1,j])*by/gap-
                              (Ys[2,j]-Ys[1,j])*bx/gap) for j in 1:ns+1]
    metadata=Dict{String,Any}(
        "construction"=>"paired_uniform_strip_refinement_v1",
        "uniform_i_refinement_factor"=>f,"te_child_margin"=>margin,
        "source_c_cells"=>m.NI*m.NJ,"refined_c_cells"=>refined.NI*refined.NJ,
        "refined_h_cells"=>nw*ns,"cross_count"=>ns,
        "cross_growth_max"=>rho,"cross_interval_cap"=>limit,
        "finite_te_gap"=>gap,"parent_rays_preserved_before_curve_projection"=>true,
        "base_first_node_normal_spacing_minimum"=>minimum(base_normal_spacings),
        "base_first_node_normal_spacing_maximum"=>maximum(base_normal_spacings),
        "base_first_normal_spacing_over_foil_dn1"=>maximum(base_normal_spacings)/m.dn1,
        "source_viscous_first_spacing_maximum_relative_change"=>projection.maximum_first_spacing_relative_change,
        "conditioned_curve_projection_maximum_shift"=>projection.maximum_shift,
        "wall_representation"=>"source-interpolating C2 open curve; native finite-base corners fixed",
        "lower_interface_nodes_bitwise_equal"=>report.lower_interface_bitwise,
        "upper_interface_nodes_bitwise_equal"=>report.upper_interface_bitwise,
        "quality_check_requested"=>check_air_gates,
        "air_gates_passed"=>report.passed,"air_gate_failures"=>report.failed,
        "admitted"=>check_air_gates&&report.passed,
        "admission_scope"=>"geometry and existing air quality gates only; not a CFD solution, y-plus study, water/cavitation admission or grid-convergence result",
        "quality_summary"=>Dict(String(k)=>v for (k,v) in pairs(report.summary)))
    refined,metadata
end
