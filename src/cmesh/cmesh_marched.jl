# Marched (offset-layer) airfoil mesh. EXPERIMENTAL AND OPT-IN: no existing mesh path calls it.
# Included into module CMeshGen.
#
# Every layer is an offset of the previous one, so the cells are regular by construction:
#   1. first line: the surface, distributed on the C2 airfoil curve by two-sided hyperbolic-tangent stretching per
#      side, and the wake line(s) with a geometric spacing matched to the trailing-edge spacing;
#   2. ONE wall-normal spacing law for all surface columns (growth g1 from dn1 to `cap`, a solved middle growth, g3);
#   3. marching: each node advances along the normal of its layer, the normals smoothed along the layer by one implicit
#      solve over a width proportional to the distance from the wall; creases are rounded and nodes may slide along
#      their layer away from the wall.
# Sharp trailing edge: one C block; the two wake rows coincide. Blunt trailing edge: the C+H topology: the C block is
# marched around the foil and two legs that leave the base corners along the wake direction, and the base's cells are
# extruded downstream between the legs as a second block, the strip.
# `wake_aspect = A`: the first-row height at a wake station is at least its spacing divided by A (instead of the wall
# spacing all the way to the outlet); with a blunt edge the strip follows the same rule and widens downstream.
# The outer boundary is the offset of the first line at distance `far`, not a rectangle.

"Wall-normal layer thicknesses: (dn1, g1) to the cap, a solved middle growth, g3 beyond dmid; the last row at `far`."
function _marched_normal_law(dn1,NJ,far;g1=1.14,cap=0.004,g3=1.18,dmid=1.0,lag=0.35)
    function law(g2)
        h=zeros(NJ);h[1]=dn1;r=g1;d=dn1
        for j in 2:NJ
            target=h[j-1]<cap ? g1 : d<dmid ? g2 : g3
            r+=lag*(target-r);h[j]=h[j-1]*r;d+=h[j]
        end
        h,d
    end
    law(g1)[2]<far&&return _marched_single_ratio(dn1,NJ,far)       # not enough rows for a slow middle: one ratio
    lo,hi=1.0,g1
    for _ in 1:80
        mid=(lo+hi)/2;law(mid)[2]>far ? (hi=mid) : (lo=mid)
    end
    law((lo+hi)/2)[1]
end
function _marched_single_ratio(dn1,NJ,far)
    total(g)=dn1*(g^NJ-1)/(g-1);lo,hi=1.0001,2.0
    for _ in 1:80;mid=(lo+hi)/2;total(mid)>far ? (hi=mid) : (lo=mid);end
    g=(lo+hi)/2;[dn1*g^(j-1) for j in 1:NJ]
end
"Geometric distances 0, d1, ... of n cells with first cell `first` and total `L`."
function _marched_geometric(first,n,L)
    total(g)=first*(g^n-1)/(g-1);lo,hi=1.0000001,3.0
    total(lo)>=L&&return collect(range(0,L;length=n+1))
    for _ in 1:100;mid=(lo+hi)/2;total(mid)>L ? (hi=mid) : (lo=mid);end
    g=(lo+hi)/2;d=zeros(n+1);for k in 1:n;d[k+1]=d[k]+first*g^(k-1);end
    d.*=L/d[end];d
end

"Cross parameter 0..1 of the strip: edge cells `edge` high at both legs, growth `g`, largest cell `cap`; the cell count a multiple of `multiple`."
function _marched_strip_parameter(gap,edge,g,cap,multiple::Int)
    edge<gap/4||throw(ArgumentError("strip edge cell $edge is not small against the base height $gap"))
    heights(r,n)=[min(edge*r^(m-1),cap) for m in 1:n]
    n=1
    while sum(heights(g,n))<gap/2||(2n)%multiple!=0
        n+=1;n<=2048||error("strip needs more than 4096 cross cells")
    end
    # the edge cell keeps its height exactly; the growth is lowered until the n cells fill half the base
    lo,hi=1.0,Float64(g)
    for _ in 1:80;mid=(lo+hi)/2;sum(heights(mid,n))>gap/2 ? (hi=mid) : (lo=mid);end
    h=heights((lo+hi)/2,n);half=cumsum(h).*(0.5/sum(h))
    vcat(0.0,half[1:end-1],0.5,1.0.-reverse(half[1:end-1]),1.0)
end

"""
    marched_airfoil_mesh(af; n_foil, n_wake, n_normal, dn1, alpha_deg, farfield_angle_deg=alpha_deg, far=30, te_ds=5e-4, ...)

`n_foil` surface cells in total (even), `n_wake` cells per wake side, `n_normal` rows. Returns (; mesh, info) with
mesh = (X, Y, NI, NJ, NIW, NIF, dn1, strip) in the C-mesh convention (i: lower outlet -> lower surface -> nose -> upper
surface -> upper outlet; j outward). Sharp trailing edge: the two wake rows coincide at j = 1 and strip is nothing.
Blunt trailing edge: the wake rows are the two legs and strip = (X, Y, NI = n_wake, NJ, t, gap) is the H block
between them (row 1 on the lower leg, column 1 on the base).
"""
function marched_airfoil_mesh(af;n_foil::Int,n_wake::Int,n_normal::Int,dn1::Float64,alpha_deg::Float64,
        farfield_angle_deg::Float64=alpha_deg,far::Float64=30.0,te_ds::Float64=5e-4,le_ds_factor::Float64=0.06,
        strip_edge=nothing,strip_growth=1.2,strip_cap=1/8,strip_multiple::Int=4,te_base_fraction=1/16,
        g1=1.14,cap=0.004,g3=1.18,gamma=2.0,wake_aspect=nothing,cells_min=nothing,shape=0.1,shape_from=0.005,slide=0.3,slide_from=0.02,kwargs...)
    iseven(n_foil)||throw(ArgumentError("n_foil must be even"))
    cond=condition_airfoil(af);a=cond.airfoil
    gap=hypot(a.x[1]-a.x[end],a.y[1]-a.y[end]);blunt=gap>1e-8
    # blunt trailing edge: the surface and leg spacing at the corners is at most a fraction of the base height
    blunt&&(te_ds=min(te_ds,te_base_fraction*gap))
    curve=_airfoil_spline(a);S=curve.s[end]
    ev(s)=(p=_airfoil_eval(curve,s);(p.x,p.y))
    sq=range(0,S;length=40001);sle=sq[argmin([ev(s)[1] for s in sq])]
    xmin=ev(sle)[1];kle=maximum(abs(cond.curvature[k]) for k in eachindex(cond.curvature) if a.x[k]<=xmin+0.02)
    le_ds=clamp(le_ds_factor/max(kle,1.0),8dn1,0.001)
    nh=n_foil÷2
    tl=_vinokur(nh,le_ds/sle,te_ds/sle)               # lower side, measured from the nose
    tu=_vinokur(nh,le_ds/(S-sle),te_ds/(S-sle))       # upper side, measured from the nose
    sfoil=vcat([sle-sle*tl[k] for k in nh+1:-1:1],[sle+(S-sle)*tu[k] for k in 2:nh+1])   # TE lower -> nose -> TE upper
    te=((a.x[1]+a.x[end])/2,(a.y[1]+a.y[end])/2)
    wdir=(cosd(farfield_angle_deg),sind(farfield_angle_deg));up=(-wdir[2],wdir[1])
    # the two points the wake rows leave from: the trailing-edge node (sharp) or the base corners (blunt)
    pl=blunt ? ev(0.0) : te;pu=blunt ? ev(S) : te
    blunt&&((pu[1]-pl[1])*up[1]+(pu[2]-pl[2])*up[2]>0.5gap||error("marched_airfoil_mesh: the airfoil curve does not run lower trailing edge -> nose -> upper trailing edge"))
    dw=_marched_geometric(te_ds,n_wake,far)
    NIW=n_wake;NIF=n_foil;NI=2NIW+NIF;NJ=n_normal;n=NI+1
    X=zeros(n,NJ+1);Y=zeros(n,NJ+1)
    # Legs: straight, along the wake direction. The base is in general not perpendicular to it; the streamwise offset
    # of the two corners is removed gradually (over about ten base heights) so that the strip's cross lines end up
    # perpendicular to the legs.
    offset=(pu[1]-pl[1])*wdir[1]+(pu[2]-pl[2])*wdir[2]
    # First-row height along the wake (wake_aspect, off by default). There is no wall in the wake: with the wall
    # spacing kept to the outlet the first rows reach aspect ratios of 1e5. With wake_aspect = A the first-row height at
    # wake station k is max(dn1, spacing_k / A). `ex[k+1]` is the excess over dn1 (0 near the trailing edge); a cell of
    # the wall's law of height h becomes (ex^4 + h^4)^(1/4) there: uniform cells of height about ex next to the cut,
    # the wall's law further out. Layers of far-wake columns are therefore shifted outwards by a slowly varying amount
    # instead of being re-spaced, so nothing changes in the far rows relative to their neighbours.
    ex=zeros(NIW+1)
    if wake_aspect!==nothing
        for k in 1:NIW;ex[k+1]=max(0.0,(k<NIW ? (dw[k+2]-dw[k])/2 : dw[k+1]-dw[k])/wake_aspect-dn1);end
    end
    grown(e,hh)=e==0 ? hh : (e^4+hh^4)^0.25
    # Blunt edge: the strip's cross cells follow the same rule (edge cells as high as the C block's first row at that
    # station), so the strip widens downstream and the legs open by half of that on each side (a fraction of a degree).
    tb=blunt ? _marched_strip_parameter(gap,strip_edge===nothing ? dn1 : strip_edge,strip_growth,strip_cap*gap,strip_multiple) : Float64[]
    gperp=(pu[1]-pl[1])*up[1]+(pu[2]-pl[2])*up[2]
    spread=[blunt ? (sum(grown(ex[k],(tb[j+1]-tb[j])*gperp) for j in 1:length(tb)-1)-gperp)/2 : 0.0 for k in 1:NIW+1]
    for k in 0:NIW
        b=blunt ? 1/(1+(dw[k+1]/(10gap))^2) : 1.0
        dl=dw[k+1]+(1-b)*offset/2;du=dw[k+1]-(1-b)*offset/2
        X[NIW+1-k,1]=pl[1]+dl*wdir[1]-spread[k+1]*up[1];Y[NIW+1-k,1]=pl[2]+dl*wdir[2]-spread[k+1]*up[2]
        X[NIW+NIF+1+k,1]=pu[1]+du*wdir[1]+spread[k+1]*up[1];Y[NIW+NIF+1+k,1]=pu[2]+du*wdir[2]+spread[k+1]*up[2]
    end
    for (k,s) in enumerate(sfoil)
        k in (1,length(sfoil))&&continue                   # the two end nodes are the points the legs leave from
        X[NIW+k,1],Y[NIW+k,1]=ev(s)
    end
    # Layer thicknesses: the wall's law h in every surface column; in the wake columns the same law with the first-row
    # rule above. (A first attempt re-spaced the finished wake columns to per-column laws that all ended at `far`; that
    # changed the far rows, where cells are chords tall and columns a thousandth of a chord apart, and gave non-convex
    # cells. It is gone.)
    h=_marched_normal_law(dn1,NJ,far;g1,cap,g3)
    exn=zeros(n);for k in 1:NIW;exn[NIW+1-k]=ex[k+1];exn[NIW+NIF+1+k]=ex[k+1];end
    H=[grown(exn[i],h[j]) for i in 1:n,j in 1:NJ]
    dist=zeros(n);w2v=zeros(n)
    px=zeros(n);py=zeros(n);nx=zeros(n);ny=zeros(n);rx=zeros(n);ry=zeros(n);ds=zeros(n);tmpx=zeros(n);tmpy=zeros(n);seg=zeros(n);lo_=zeros(n);up_=zeros(n);di_=zeros(n);cp_=zeros(n)
    # (I - W^2/2 d2/ds2) v = r along the layer (coefficients in lo_, di_, up_), the two end values of r kept: Thomas
    function solve!(v,r)
        v[1]=r[1];v[n]=r[n]
        cp_[2]=up_[2]/di_[2];v[2]=(r[2]-lo_[2]*r[1])/di_[2]
        for i in 3:n-1
            den=di_[i]-lo_[i]*cp_[i-1];cp_[i]=up_[i]/den
            rhs=i==n-1 ? r[i]-up_[i]*r[n] : r[i]
            v[i]=(rhs-lo_[i]*v[i-1])/den
        end
        for i in n-2:-1:2;v[i]-=cp_[i]*v[i+1];end
        v
    end
    # `cells`: a lower bound of the smoothing width in cells (a sharp concave corner, such as the junction of a blunt base
    # and the wake cut, needs its neighbours turned from the first row on)
    function coefficients!(qx,qy,W2;cells=0.0)
        for i in 1:n-1;seg[i]=hypot(qx[i+1]-qx[i],qy[i+1]-qy[i]);end
        for i in 2:n-1
            hb=(seg[i-1]+seg[i])/2;w2=max(W2[i],(cells*hb)^2/2)
            lo_[i]=-w2/(seg[i-1]*hb);up_[i]=-w2/(seg[i]*hb);di_[i]=1-lo_[i]-up_[i]
        end
    end
    for j in 1:NJ
        @views px.=X[:,j];@views py.=Y[:,j]
        # raw normals from the parabola through three points; the outlet ends on their straight lines
        for i in 2:n-1
            ax=px[i]-px[i-1];ay=py[i]-py[i-1];bx=px[i+1]-px[i];by=py[i+1]-py[i]
            hm=hypot(ax,ay);hp=hypot(bx,by)
            tx=(hp*ax/hm+hm*bx/hp)/(hm+hp);ty=(hp*ay/hm+hm*by/hp)/(hm+hp);t=hypot(tx,ty)
            rx[i]=-ty/t;ry[i]=tx/t;ds[i]=min(hm,hp)
        end
        rx[1]=-up[1];ry[1]=-up[2];rx[n]=up[1];ry[n]=up[2];ds[1]=ds[2];ds[n]=ds[n-1]
        nx.=rx;ny.=ry
        # smoothing of the normals along the layer over a physical width gamma x (distance from the wall): one implicit
        # diffusion solve on the layer's own arc length (the outlet normals fixed). A concave turn of angle t in the
        # C-line (the trailing-edge junction) makes neighbouring columns cross at distance spacing/t unless the turn is
        # spread over a width that grows with the distance; convex regions fan out and are hardly changed.
        cmin=cells_min===nothing ? 0.0 : cells_min
        if j>1||cmin>0
            for i in 1:n;w2v[i]=j>1 ? (gamma*(dist[i]+H[i,j]))^2/2 : 0.0;end
            coefficients!(px,py,w2v;cells=cmin)
            solve!(nx,rx);solve!(ny,ry)
            for i in 2:n-1;q=hypot(nx[i],ny[i]);nx[i]/=q;ny[i]/=q;end
        end
        # every node advances by the same length along its smoothed direction. (Dividing by the cosine between the
        # smoothed and the raw normal, to keep the true thickness, feeds position noise back into the step length and
        # makes the layers jagged within a few rows; the thickness lost at a turn is a few percent.)
        for i in 1:n
            X[i,j+1]=px[i]+H[i,j]*nx[i];Y[i,j+1]=py[i]+H[i,j]*ny[i]
        end
        for i in 1:n;dist[i]+=H[i,j];end
        dwall=dist[NIW+1]
        # rounding of creases: the new layer's shape smoothed along its arc over a width shape x distance, applied to the
        # normal component only (nodes keep their tangential position). A concave turn of the C-line is otherwise
        # carried outwards as a sharp crease in every layer.
        if shape>0&&dwall>shape_from
            @views px.=X[:,j+1];@views py.=Y[:,j+1]
            for i in 1:n;w2v[i]=(shape*clamp((dist[i]-shape_from)/(4shape_from),0.0,1.0)*dist[i])^2/2;end
            coefficients!(px,py,w2v);solve!(tmpx,px);solve!(tmpy,py)
            for i in 2:n-1
                dn=(tmpx[i]-px[i])*nx[i]+(tmpy[i]-py[i])*ny[i]
                X[i,j+1]=px[i]+dn*nx[i];Y[i,j+1]=py[i]+dn*ny[i]
            end
        end
        # tangential sliding along the new layer (outside the boundary layer): evens out the spacing ratio
        w=clamp((dwall-slide_from)/(4slide_from),0.0,1.0)*slide
        if w>0
            @views px.=X[:,j+1];@views py.=Y[:,j+1]
            s=zeros(n);for i in 2:n;s[i]=s[i-1]+hypot(px[i]-px[i-1],py[i]-py[i-1]);end
            snew=copy(s)
            for _ in 1:4
                for i in 2:n-1;tmpx[i]=snew[i]+0.25w*(log(snew[i+1]-snew[i])-log(snew[i]-snew[i-1]))*min(snew[i+1]-snew[i],snew[i]-snew[i-1]);end
                for i in 2:n-1;snew[i]=tmpx[i];end
            end
            k=1
            for i in 2:n-1
                while k<n-1&&s[k+1]<snew[i];k+=1;end
                f=(snew[i]-s[k])/(s[k+1]-s[k])
                X[i,j+1]=px[k]+f*(px[k+1]-px[k]);Y[i,j+1]=py[k]+f*(py[k+1]-py[k])
            end
        end
    end
    amin=Inf
    for j in 1:NJ,i in 1:NI
        area=0.5*((X[i+1,j+1]-X[i,j])*(Y[i,j+1]-Y[i+1,j])-(X[i,j+1]-X[i+1,j])*(Y[i+1,j+1]-Y[i,j]))
        amin=min(amin,area)
    end
    amin>0||error("marched_airfoil_mesh: non-positive cell area $amin")
    # The strip (H block) of a blunt edge: nodes (NIW+1) x (NS+1); row 1 is the lower leg and row NS+1 the upper leg
    # (the C block's own nodes, bit for bit), column 1 is the base, column NIW+1 the outlet. Every cross line is
    # straight; at the base the cross cells are strip_edge high at both legs (default: the wall spacing, so the cell
    # height is continuous across the legs), growing by strip_growth to at most strip_cap x base height; downstream
    # they follow the wake's first-row rule. NS is a multiple of strip_multiple (mesh sequencing by node skipping).
    strip=nothing
    if blunt
        NS=length(tb)-1;t=zeros(NS+1)
        Xs=zeros(NIW+1,NS+1);Ys=zeros(NIW+1,NS+1)
        for k in 1:NIW+1
            xl,yl,xu,yu=X[NIW+2-k,1],Y[NIW+2-k,1],X[NIW+NIF+k,1],Y[NIW+NIF+k,1]
            if ex[k]==0
                t.=tb
            else
                for j in 1:NS;t[j+1]=t[j]+grown(ex[k],(tb[j+1]-tb[j])*gperp);end
                t./=t[end]
            end
            for j in 2:NS;Xs[k,j]=xl+t[j]*(xu-xl);Ys[k,j]=yl+t[j]*(yu-yl);end
            Xs[k,1]=xl;Ys[k,1]=yl;Xs[k,NS+1]=xu;Ys[k,NS+1]=yu
        end
        for j in 1:NS,k in 1:NIW
            area=0.5*((Xs[k+1,j+1]-Xs[k,j])*(Ys[k,j+1]-Ys[k+1,j])-(Xs[k,j+1]-Xs[k+1,j])*(Ys[k+1,j+1]-Ys[k,j]))
            amin=min(amin,area)
        end
        amin>0||error("marched_airfoil_mesh: non-positive strip cell area $amin")
        strip=(;X=Xs,Y=Ys,NI=NIW,NJ=NS,t=tb,gap)
    end
    (;mesh=(;X,Y,NI,NJ,NIW,NIF,dn1,strip),info=(;le_ds,te_ds,far,heights=h,first_heights=H[:,1],area_min=amin,
        wake_ratio=dw[3]>0 ? (dw[3]-dw[2])/(dw[2]-dw[1]) : 1.0))
end
