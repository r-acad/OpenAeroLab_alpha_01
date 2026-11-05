# Experimental semicoarsened scalar multilevel inverse for the airfoil JFNK
# preconditioner. It does not change the nonlinear finite-volume equations.
# Consecutive C-mesh columns are aggregated first, then pairs of wall-normal
# rows. Wake-cut partner columns remain paired on every level.
using SparseArrays

struct AirfoilMGLevel{A}
    ni::Int32
    nj::Int32
    niw::Int32
    nif::Int32
    lamW::A
    lamE::A
    lamS::A
    lamN::A
    d0::A
    diagonal::A
    rhs::A
    x::A
    old::A
    work::A
    residual::A
    bp::A
end

struct AirfoilMGWorkspace{L}
    levels::Vector{L}
    axes::Vector{Symbol}
    coarse_lu::Base.RefValue{Any}
    cycles::Int
end

@kernel function _k_airfoil_mg_aggregate!(cw,ce,cs,cn,cd0,
    @Const(fw),@Const(fe),@Const(fs),@Const(fn),@Const(fd0),
    NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        a=Int32(2)*i-Int32(1);z=a+Int32(1)
        cw[i,j]=fw[a,j];ce[i,j]=fe[z,j]
        cs[i,j]=fs[a,j]+fs[z,j]
        cn[i,j]=fn[a,j]+fn[z,j]
        cd0[i,j]=fd0[a,j]+fd0[z,j]
    end
end

@kernel function _k_airfoil_mg_aggregate_j!(cw,ce,cs,cn,cd0,
    @Const(fw),@Const(fe),@Const(fs),@Const(fn),@Const(fd0),
    NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        a=Int32(2)*j-Int32(1);z=a+Int32(1)
        cw[i,j]=fw[i,a]+fw[i,z]
        ce[i,j]=fe[i,a]+fe[i,z]
        cs[i,j]=fs[i,a];cn[i,j]=fn[i,z]
        cd0[i,j]=fd0[i,a]+fd0[i,z]
    end
end

@kernel function _k_airfoil_mg_rhs!(work,@Const(rhs),@Const(old),
    @Const(lw),@Const(le),NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        work[i,j]=rhs[i,j]+(i>1 ? lw[i,j]*old[i-1,j] : 0.0f0)+
            (i<NI ? le[i,j]*old[i+1,j] : 0.0f0)
    end
end

@kernel function _k_airfoil_mg_restrict_j!(coarse,@Const(fine),NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        coarse[i,j]=fine[i,Int32(2)*j-Int32(1)]+fine[i,Int32(2)*j]
    end
end

@kernel function _k_airfoil_mg_residual!(res,@Const(rhs),@Const(x),
    @Const(lw),@Const(le),@Const(ls),@Const(ln),@Const(d0),
    NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        wake=i<=NIW || i>NIW+NIF
        south=j>1 ? x[i,j-1] : (wake ? x[NI+Int32(1)-i,1] : 0.0f0)
        north=j<NJ ? x[i,j+1] : 0.0f0
        west=i>1 ? x[i-1,j] : 0.0f0
        east=i<NI ? x[i+1,j] : 0.0f0
        diag=d0[i,j]+lw[i,j]+le[i,j]+ls[i,j]+ln[i,j]
        res[i,j]=rhs[i,j]-(diag*x[i,j]-lw[i,j]*west-le[i,j]*east-
                           ls[i,j]*south-ln[i,j]*north)
    end
end

@kernel function _k_airfoil_mg_prolong_add_j!(fine,@Const(coarse),NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        fine[i,j]+=coarse[i,(j+Int32(1))÷Int32(2)]
    end
end

@kernel function _k_airfoil_mg_restrict!(coarse,@Const(fine),NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        coarse[i,j]=fine[Int32(2)*i-Int32(1),j]+fine[Int32(2)*i,j]
    end
end

@kernel function _k_airfoil_mg_prolong_add!(fine,@Const(coarse),NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        fine[i,j]+=coarse[(i+Int32(1))÷Int32(2),j]
    end
end

function _airfoil_mg_level(z,ni::Int,nj::Int,niw::Int,nif::Int,
                            lw,le,ls,ln,d0)
    arr()=z(ni,nj)
    AirfoilMGLevel(Int32(ni),Int32(nj),Int32(niw),Int32(nif),
        lw,le,ls,ln,d0,arr(),arr(),arr(),arr(),arr(),arr(),arr())
end

function build_airfoil_mg_workspace(s::KSol,line,adi;max_levels::Int=9,
                                   cycles::Int=1)
    2<=max_levels<=10 || throw(ArgumentError("multilevel depth must be 2:10"))
    1<=cycles<=8 || throw(ArgumentError("multilevel cycles must be 1:8"))
    z(ni,nj)=similar(s.dtf,Float32,(ni,nj))
    first=_airfoil_mg_level(z,s.NI,s.NJ,s.NIW,s.NIF,
        adi.lamW,adi.lamE,line.lamS,line.lamN,line.d0)
    levels=typeof(first)[first]
    axes=Symbol[]
    while length(levels)<max_levels
        parent=levels[end]
        ni=Int(parent.ni);niw=Int(parent.niw);nif=Int(parent.nif)
        (iseven(ni) && iseven(niw) && iseven(nif) && niw>=2 && nif>=2) || break
        n2=ni÷2;w2=niw÷2;f2=nif÷2;nj=Int(parent.nj)
        push!(levels,_airfoil_mg_level(z,n2,nj,w2,f2,
            z(n2,nj),z(n2,nj),z(n2,nj),z(n2,nj),z(n2,nj)))
        push!(axes,:i)
    end
    while length(levels)<max_levels
        parent=levels[end]
        nj=Int(parent.nj)
        (iseven(nj) && nj>=14) || break
        ni=Int(parent.ni);j2=nj÷2
        push!(levels,_airfoil_mg_level(z,ni,j2,Int(parent.niw),Int(parent.nif),
            z(ni,j2),z(ni,j2),z(ni,j2),z(ni,j2),z(ni,j2)))
        push!(axes,:j)
    end
    length(levels)>=2 || throw(ArgumentError("C-mesh columns cannot be semicoarsened"))
    AirfoilMGWorkspace(levels,axes,Ref{Any}(nothing),cycles)
end

function _airfoil_mg_coarse_factor(level::AirfoilMGLevel)
    ni=Int(level.ni);nj=Int(level.nj);niw=Int(level.niw);nif=Int(level.nif)
    n=ni*nj
    n<=5000 || return nothing
    lw=Array(level.lamW);le=Array(level.lamE)
    ls=Array(level.lamS);ln=Array(level.lamN);d0=Array(level.d0)
    rows=Int[];cols=Int[];values=Float64[]
    sizehint!(rows,5n);sizehint!(cols,5n);sizehint!(values,5n)
    add!(row,col,value)=(push!(rows,row);push!(cols,col);push!(values,Float64(value)))
    for j in 1:nj,i in 1:ni
        k=i+(j-1)*ni
        add!(k,k,d0[i,j]+lw[i,j]+le[i,j]+ls[i,j]+ln[i,j])
        i>1 && add!(k,k-1,-lw[i,j])
        i<ni && add!(k,k+1,-le[i,j])
        if j>1
            add!(k,k-ni,-ls[i,j])
        elseif i<=niw || i>niw+nif
            add!(k,ni+1-i,-ls[i,j])
        end
        j<nj && add!(k,k+ni,-ln[i,j])
    end
    lu(sparse(rows,cols,values,n,n))
end

function _prepare_airfoil_mg!(mg::AirfoilMGWorkspace,b)
    levels=mg.levels
    for l in 1:length(levels)-1
        f=levels[l];c=levels[l+1]
        aggregate=mg.axes[l]===:i ? _k_airfoil_mg_aggregate! : _k_airfoil_mg_aggregate_j!
        aggregate(b)(c.lamW,c.lamE,c.lamS,c.lamN,c.d0,
            f.lamW,f.lamE,f.lamS,f.lamN,f.d0,c.ni,c.nj;
            ndrange=(Int(c.ni),Int(c.nj)))
    end
    for level in levels
        @. level.diagonal=level.d0+level.lamW+level.lamE
    end
    KernelAbstractions.synchronize(b)
    mg.coarse_lu[]=_airfoil_mg_coarse_factor(levels[end])
    mg
end

function _airfoil_mg_smooth!(level::AirfoilMGLevel,b,nsweeps::Int)
    for _ in 1:nsweeps
        copyto!(level.old,level.x)
        _k_airfoil_mg_rhs!(b)(level.work,level.rhs,level.old,
            level.lamW,level.lamE,level.ni,level.nj;
            ndrange=(Int(level.ni),Int(level.nj)))
        _k_implicit_cut_line_solve!(b)(level.x,level.bp,level.work,
            level.lamS,level.lamN,level.diagonal,
            level.ni,level.nj,level.niw,level.nif;
            ndrange=Int(level.niw+level.nif))
    end
    level
end

function _airfoil_mg_cycle!(mg::AirfoilMGWorkspace,b,idx::Int)
    levels=mg.levels;f=levels[idx]
    if idx==length(levels)
        factor=mg.coarse_lu[]
        if factor===nothing
            _airfoil_mg_smooth!(f,b,12)
        else
            solution=factor\vec(Array(f.rhs))
            copyto!(f.x,reshape(Float32.(solution),Int(f.ni),Int(f.nj)))
        end
        return f
    end
    _airfoil_mg_smooth!(f,b,2)
    _k_airfoil_mg_residual!(b)(f.residual,f.rhs,f.x,
        f.lamW,f.lamE,f.lamS,f.lamN,f.d0,
        f.ni,f.nj,f.niw,f.nif;ndrange=(Int(f.ni),Int(f.nj)))
    c=levels[idx+1]
    restrict=mg.axes[idx]===:i ? _k_airfoil_mg_restrict! : _k_airfoil_mg_restrict_j!
    restrict(b)(c.rhs,f.residual,c.ni,c.nj;
        ndrange=(Int(c.ni),Int(c.nj)))
    fill!(c.x,0.0f0)
    _airfoil_mg_cycle!(mg,b,idx+1)
    prolong=mg.axes[idx]===:i ? _k_airfoil_mg_prolong_add! : _k_airfoil_mg_prolong_add_j!
    prolong(b)(f.x,c.x,f.ni,f.nj;
        ndrange=(Int(f.ni),Int(f.nj)))
    _airfoil_mg_smooth!(f,b,2)
    f
end

function _airfoil_mg_precondition!(out,input,vol,mg::AirfoilMGWorkspace,b)
    fine=mg.levels[1]
    @. fine.rhs=input*vol
    fill!(fine.x,0.0f0)
    for _ in 1:mg.cycles
        _airfoil_mg_cycle!(mg,b,1)
    end
    copyto!(out,fine.x)
    out
end
