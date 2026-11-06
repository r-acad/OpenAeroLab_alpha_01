module WallSamplePlots
# Palette matches the existing JFEM-Boxer scales; see THIRD_PARTY_NOTICES.md.
using SHA,Statistics,LinearAlgebra,CairoMakie
export read_frame,foreach_sample,rasterize,plot_wall_samples
const MAX_SAMPLES=40_000_000
function read_frame(path)
    open(path,"r") do io
        read(io,8)==collect(codeunits("LBMWFLD\0"))||error("Invalid wall frame")
        ltoh(read(io,UInt32))==1&&ltoh(read(io,UInt32))==1||error("Unsupported wall frame version/fields")
        count=Int(ltoh(read(io,UInt64)));0<count<=MAX_SAMPLES||error("Invalid wall sample count")
        hash=String(read(io,64));occursin(r"^[a-f0-9]{64}$",hash)||error("Invalid wall geometry hash")
        step=ltoh(read(io,Int64));step>0||error("Initialization is not an evolved field")
        scalar()=reinterpret(Float64,ltoh(read(io,UInt64)))
        time,rho,speed,q=scalar(),scalar(),scalar(),scalar()
        all(isfinite,(time,rho,speed,q))&&rho>0&&speed>0&&q>0||error("Invalid pressure reference")
        filesize(path)==128+4count||error("Incomplete wall frame")
        pressure=read!(io,Vector{Float32}(undef,count))
        Base.ENDIAN_BOM==0x04030201||map!(x->reinterpret(Float32,ltoh(reinterpret(UInt32,x))),pressure,pressure)
        (count=count,geometry_sha256=hash,step=step,time=time,rho=rho,speed=speed,q=q,pressure=pressure)
    end
end
function foreach_sample(f,path;expected_count=nothing,expected_sha=nothing)
    expected_sha===nothing||bytes2hex(open(sha256,path))==expected_sha||error("Wall geometry hash mismatch")
    open(path,"r") do io
        read(io,8)==collect(codeunits("LBMWGEO\0"))||error("Invalid wall geometry")
        ltoh(read(io,UInt32))==1&&ltoh(read(io,UInt32))==36||error("Unsupported wall geometry layout")
        count=Int(ltoh(read(io,UInt64)));0<count<=MAX_SAMPLES||error("Invalid wall geometry count")
        expected_count===nothing||expected_count==count||error("Geometry/frame sample counts differ")
        filesize(path)==24+36count||error("Incomplete wall geometry")
        bytes=Vector{UInt8}(undef,36*32768)
        for start in 1:32768:count
            n=min(32768,count-start+1);block=@view bytes[1:36n];read!(io,block);words=reinterpret(UInt32,block)
            @inbounds for j in 1:n
                off=9(j-1);value(k)=reinterpret(Float32,ltoh(words[off+k]))
                x=(value(1),value(2),value(3));normal=(value(4),value(5),value(6));dx=value(7)
                all(isfinite,x)&&all(isfinite,normal)&&isfinite(dx)&&dx>0||error("Invalid wall sample geometry")
                norm2=sum(abs2,normal);.98<=norm2<=1.02||error("Invalid wall normal")
                packed=ltoh(words[off+9]);parent=ltoh(words[off+8]);parent>0||error("Unmapped parent facet")
                level=UInt8(packed&0xff);links=UInt8((packed>>8)&0xff);flags=UInt16(packed>>16)
                1<=level<=32&&links<=18&&(flags&~UInt16(3))==0||error("Invalid wall sample support metadata")
                (flags&1)==0||links>0||error("Supported wall sample has no contributing links")
                f(start+j-1,x,normal,dx,parent,level,links,flags)
            end
        end
        count
    end
end
dot3(a,b)=a[1]*b[1]+a[2]*b[2]+a[3]*b[3]
cross3(a,b)=(a[2]*b[3]-a[3]*b[2],a[3]*b[1]-a[1]*b[3],a[1]*b[2]-a[2]*b[1])
unit(a)=a./sqrt(dot3(a,a))
function legacy_vertices(path)
    path===nothing&&return Float32[]
    open(path,"r") do io
        String(read(io,4))=="LBMP"&&ltoh(read(io,UInt32))==1||error("Invalid optional original surface")
        n=Int(ltoh(read(io,UInt32)));0<n<=1_000_000||error("Original surface is too large")
        objects=Int(ltoh(read(io,UInt32)));0<=objects<=10000||error("Invalid object count")
        for _ in 1:objects;len=Int(ltoh(read(io,UInt32)));0<=len<=4096||error("Invalid object name");skip(io,len);end
        read!(io,Vector{Float32}(undef,9n))
    end
end
"Depth-resolved native samples; no parent-facet averaging or pressure interpolation."
function rasterize(geometry,frame;width=1500,height=850,legacy=nothing)
    200<=width<=3840&&200<=height<=2160||error("Invalid render size")
    lo=fill(Inf,3);hi=fill(-Inf,3)
    foreach_sample(geometry;expected_count=frame.count,expected_sha=frame.geometry_sha256) do _,x,_,_,_,_,_,_
        for a in 1:3;lo[a]=min(lo[a],x[a]);hi[a]=max(hi[a],x[a]);end
    end
    original=legacy_vertices(legacy)
    for i in 1:3:length(original),a in 1:3;lo[a]=min(lo[a],original[i+a-1]);hi[a]=max(hi[a],original[i+a-1]);end
    center=(lo.+hi)./2
    eye=unit((1.3,2.2,1.4));right=unit(cross3((0.,0.,1.),eye));up=cross3(eye,right)
    projected=[(dot3((x,y,z),right),dot3((x,y,z),up)) for x in (lo[1],hi[1]),y in (lo[2],hi[2]),z in (lo[3],hi[3])]
    rx=extrema(first,projected);ry=extrema(last,projected)
    scale=.88min(width/max(rx[2]-rx[1],1e-6),height/max(ry[2]-ry[1],1e-6))
    project(x)=begin p=(x[1]-center[1],x[2]-center[2],x[3]-center[3]);(width/2+scale*dot3(p,right),height/2+scale*dot3(p,up),dot3(p,eye)) end
    depth=fill(-Inf32,width,height);values=fill(NaN32,width,height);covered=falses(width,height)
    # Neutral original geometry gives honest visual context where pressure is absent.
    function triangle!(a,b,c)
        x1,y1,z1=project(a);x2,y2,z2=project(b);x3,y3,z3=project(c)
        den=(y2-y3)*(x1-x3)+(x3-x2)*(y1-y3);abs(den)>1e-12||return
        for j in max(1,floor(Int,min(y1,y2,y3))):min(height,ceil(Int,max(y1,y2,y3))),i in max(1,floor(Int,min(x1,x2,x3))):min(width,ceil(Int,max(x1,x2,x3)))
            u=((y2-y3)*(i-x3)+(x3-x2)*(j-y3))/den;v=((y3-y1)*(i-x3)+(x1-x3)*(j-y3))/den;w=1-u-v
            min(u,v,w)>=-1e-6||continue;z=u*z1+v*z2+w*z3
            if z>depth[i,j];depth[i,j]=z;covered[i,j]=true;end
        end
    end
    for i in 1:9:length(original);triangle!(Tuple(original[i:i+2]),Tuple(original[i+3:i+5]),Tuple(original[i+6:i+8]));end
    visible=Ref(0);finite=Ref(0)
    foreach_sample(geometry;expected_count=frame.count) do index,x,n,dx,_,_,_,flags
        (flags&1)!=0||return
        pressure=frame.pressure[index];isfinite(pressure)||return;finite[]+=1
        dot3(n,eye)>1e-5||return
        px,py,pz=project(x);t=unit(cross3(n,abs(n[3])<.9 ? (0.,0.,1.) : (1.,0.,0.)));u=cross3(n,t)
        radius=.8dx;ax=scale*radius*dot3(t,right);ay=scale*radius*dot3(t,up);bx=scale*radius*dot3(u,right);by=scale*radius*dot3(u,up)
        det=ax*by-bx*ay;rx=max(.65,hypot(ax,bx));ry=max(.65,hypot(ay,by));cp=Float32(pressure/frame.q)
        if max(rx,ry)<1.2||abs(det)<1e-8
            i=round(Int,px);j=round(Int,py)
            # Compare at the pixel centre, rather than at the sample centre:
            # on a sloping original facet those depths can differ by a pixel.
            z=pz-(dot3(n,right)*(i-px)+dot3(n,up)*(j-py))/(scale*dot3(n,eye))
            if 1<=i<=width&&1<=j<=height&&z>=depth[i,j]-2e-4
                depth[i,j]=z;values[i,j]=cp;covered[i,j]=true;visible[]+=1
            end
            return
        end
        tz=radius*dot3(t,eye);uz=radius*dot3(u,eye)
        for j in max(1,floor(Int,py-ry)):min(height,ceil(Int,py+ry)),i in max(1,floor(Int,px-rx)):min(width,ceil(Int,px+rx))
            sx=i-px;sy=j-py;a=(sx*by-bx*sy)/det;b=(ax*sy-sx*ay)/det
            a*a+b*b<=1||continue;z=pz+a*tz+b*uz
            if z>=depth[i,j]-2e-4;depth[i,j]=z;values[i,j]=cp;covered[i,j]=true;visible[]+=1;end
        end
    end
    (values=values,covered=covered,finite_samples=finite[],painted_pixels=visible[],world_bounds=(lo,hi),eye=eye)
end
function plot_wall_samples(geometry,field,output;title="Native wall-sample Cp",legacy=nothing)
    frame=read_frame(field);image=rasterize(geometry,frame;legacy)
    good=filter(isfinite,frame.pressure);isempty(good)&&error("No finite wall pressures")
    limits=Tuple(quantile(good./frame.q,[.02,.98]));limits[1]<limits[2]||(limits=(limits[1]-.001,limits[2]+.001))
    palette=[RGBf(rgb...) for rgb in ((.14,.24,.50),(.16,.56,.74),(.28,.72,.44),(.94,.79,.26),(.86,.24,.22))]
    colors=fill(RGBf(1,1,1),size(image.values));ramp=cgrad(palette)
    for i in eachindex(colors)
        colors[i]=isfinite(image.values[i]) ? ramp[clamp((image.values[i]-limits[1])/(limits[2]-limits[1]),0,1)] : image.covered[i] ? RGBf(.68,.71,.74) : RGBf(1,1,1)
    end
    fig=Figure(size=(1600,1100),fontsize=19,backgroundcolor=:white)
    Label(fig[1,1:2],title,fontsize=27,font=:bold)
    Label(fig[2,1:2],"Recorded step $(frame.step) · $(frame.count) native cell/facet samples · t = $(round(frame.time;sigdigits=5)) s")
    ax=Axis(fig[3,1],aspect=DataAspect());image!(ax,colors);hidedecorations!(ax);hidespines!(ax)
    Colorbar(fig[3,2],colormap=ramp,limits=limits,label="Cp",width=22)
    Label(fig[4,1:2],"Fine native wall samples; gray: pressure unavailable. Adjacent fluid-cell pressure, no facet averaging.",fontsize=16)
    Label(fig[5,1:2],"Native saved-field render · Aero Lab browser capture unavailable · convergence not certified",fontsize=16)
    mkpath(dirname(output));Base.invokelatest(save,output,fig;px_per_unit=1)
    (frame=frame,image=image,limits=limits)
end
end
