module PondThermalIO
using WriteVTK, Printf, TOML
include(joinpath(@__DIR__,"..","..","io","lbm_slice_format.jl"))
using .LBMSliceFormat
export surface_sample_map, write_surface, write_surface_vtk, write_volume, write_planes, wall_loads

function atomic_write(writer,path)
    mkpath(dirname(path));tmp=path*".partial"
    open(writer,tmp,"w");Base.Filesystem.rename(tmp,path);path
end

"Map triangles to fluid candidates along their original normal, with an explicit opposite-side fallback."
function surface_sample_map(mesh,geometry,offset,findcell)
    ids=zeros(Int32,length(geometry.triangles));signs=ones(Int8,length(ids));h=minimum(mesh.dx)
    for (t,tri) in enumerate(geometry.triangles)
        c=ntuple(a->sum(v[a] for v in tri)/3+offset[a],3);n=geometry.normals[t]
        # A stair-step solid mask can place the nearest outward probe inside a
        # solid voxel. Exhaust the original side before considering the reverse
        # direction; a nearer reverse-side fluid cell is not proof of bad winding.
        for sign in (1,-1)
            for radius in (.25,.5,1.,2.,4.,8.,16.)
                p=ntuple(a->c[a]+sign*radius*h*n[a],3)
                i=findcell(p...)
                if i>0;ids[t]=i;signs[t]=sign;break;end
            end
            ids[t]>0 && break
        end
    end
    (;ids,signs)
end

function _surface_values(geometry,map,P,scales)
    n=length(geometry.triangles);press=fill(Float32(NaN),n);cp=copy(press)
    qinf=.5scales.rho*scales.speed^2;pinf=scales.rho*scales.sound^2/scales.gamma
    for t in 1:n
        i=map.ids[t];i>0 || continue
        press[t]=P[i,5]*scales.rho*scales.sound^2-pinf
        cp[t]=press[t]/qinf
    end
    (;press,cp)
end

function write_surface(path,geometry,offset,map,P,scales;step=0,time_s=0.,moment_origin=(0.,0.,0.),reference_area=1.,reference_chord=1.)
    n=length(geometry.triangles);press,cp=_surface_values(geometry,map,P,scales)
    atomic_write(path) do io
        write(io,codeunits("LBMP"));write(io,UInt32(1),UInt32(n),UInt32(1))
        name="thermal_FV_surface";write(io,UInt32(sizeof(name)),codeunits(name))
        for tri in geometry.triangles,v in tri,a in 1:3;write(io,Float32(v[a]+offset[a]));end
        # Sampling fallback is not an orientation repair. Preserve source normals.
        for t in 1:n,a in 1:3;write(io,Float32(geometry.normals[t][a]));end
        write(io,press)
        # Triangle shear is not reconstructed from voxel tractions. A missing
        # quantity must remain missing rather than falsely report zero friction.
        write(io,fill(Float32(NaN),n));write(io,cp);write(io,ones(Int32,n))
    end
    meta=Dict("format"=>"LBMP","format_version"=>1,"solver"=>"thermal_kinetic_fv",
        "experimental"=>true,"geometry_frame"=>"domain","triangle_order"=>"solver_geometry",
        "mesh_offset_m"=>collect(offset),"triangles"=>n,"missing_samples"=>count(==(0),map.ids),
        "sampling_revision"=>4,"sampling"=>"first fluid candidate along original triangle normal; opposite side only if all original-side probes fail",
        "opposite_side_samples"=>count(==(-1),map.signs),"normal_convention"=>"original geometry normals; sampling does not repair winding",
        "sampling_scope"=>"Voxel membership does not establish the geometric exterior; thin or underresolved components can remain ambiguous.",
        "pressure_units"=>"Pa gauge","shear_available"=>false,"wall_model"=>"staircase voxel faces",
        "moment_origin_m"=>collect(moment_origin),"reference_area_m2"=>reference_area,
        "reference_chord_m"=>reference_chord,"step"=>step,"time_s"=>time_s)
    atomic_write(io->TOML.print(io,meta;sorted=true),path*".toml")
    path
end

"Original triangle surface with the same gauge-pressure/Cp samples as LBMP; path is a filename stem."
function write_surface_vtk(path,geometry,offset,map,P,scales;step=0,time_s=0.)
    n=length(geometry.triangles);press,cp=_surface_values(geometry,map,P,scales)
    points=Matrix{Float32}(undef,3,3n)
    cells=Vector{MeshCell{VTKCellType,Vector{Int}}}()
    for (t,tri) in enumerate(geometry.triangles)
        for k in 1:3,a in 1:3;points[a,3(t-1)+k]=tri[k][a]+offset[a];end
        push!(cells,MeshCell(VTKCellTypes.VTK_TRIANGLE,collect(3(t-1)+1:3t)))
    end
    mkpath(dirname(path))
    vtk_grid(path*".partial.vtu",points,cells;compress=true,append=false) do vtk
        vtk["Pressure"]=press;vtk["Cp"]=cp;vtk["Shear"]=fill(Float32(NaN),n)
        vtk["TimeValue",VTKFieldData()]=[time_s]
        vtk["Iteration",VTKFieldData()]=[step]
    end
    final=path*".vtu";Base.Filesystem.rename(path*".partial.vtu",final)
    meta=Dict("format"=>"VTU","solver"=>"thermal_kinetic_fv","experimental"=>true,
        "geometry_frame"=>"domain","triangle_order"=>"solver_geometry","mesh_offset_m"=>collect(offset),
        "triangles"=>n,"missing_samples"=>count(==(0),map.ids),"pressure_units"=>"Pa gauge",
        "cp_units"=>"1","shear_available"=>false,"wall_model"=>"staircase voxel faces",
        "sampling_revision"=>4,"sampling"=>"first fluid candidate along original triangle normal; opposite side only if all original-side probes fail",
        "opposite_side_samples"=>count(==(-1),map.signs),"normal_convention"=>"original geometry normals; sampling does not repair winding",
        "sampling_scope"=>"Voxel membership does not establish the geometric exterior; thin or underresolved components can remain ambiguous.",
        "step"=>step,"time_s"=>time_s)
    atomic_write(io->TOML.print(io,meta;sorted=true),final*".toml")
    final
end

function write_volume(path,mesh,P,scales;step=0,time_s=0.,gradient=nothing)
    n=length(mesh.x);points=Matrix{Float32}(undef,3,8n)
    gradient===nothing || size(gradient)==(n,5,3) || throw(ArgumentError("Volume gradients must have dimensions (fluid cells, 5 primitive fields, 3 spatial axes)"))
    cells=Vector{MeshCell{VTKCellType,Vector{Int}}}()
    # Standard VTK hexahedron order; vertices are independent per leaf cell.
    corners=((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1))
    for i in 1:n
        c=(mesh.x[i],mesh.y[i],mesh.z[i]);h=mesh.dx[i]/2
        for k in 1:8,a in 1:3;points[a,8(i-1)+k]=c[a]+h*corners[k][a];end
        push!(cells,MeshCell(VTKCellTypes.VTK_HEXAHEDRON,collect(8(i-1)+1:8i)))
    end
    velocity=permutedims(Float32.(P[:,2:4].*scales.sound))
    speed=sqrt.(vec(sum(abs2,velocity;dims=1)))
    temperature=Float32.(P[:,5] ./ P[:,1] .* (scales.sound^2/scales.gas_constant))
    mach=Float32.(speed./sqrt.(scales.gamma*scales.gas_constant.*temperature))
    vorticity=if gradient===nothing
        nothing
    else
        curl=Matrix{Float32}(undef,3,n)
        for i in 1:n
            curl[1,i]=(gradient[i,4,2]-gradient[i,3,3])*scales.sound
            curl[2,i]=(gradient[i,2,3]-gradient[i,4,1])*scales.sound
            curl[3,i]=(gradient[i,3,1]-gradient[i,2,2])*scales.sound
        end
        curl
    end
    mkpath(dirname(path))
    vtk_grid(path*".partial.vtu",points,cells;compress=true,append=false) do vtk
        vtk["Density"]=Float32.(P[:,1].*scales.rho)
        vtk["Velocity"]=velocity;vtk["VelocityMagnitude"]=speed
        vtk["Pressure"]=Float32.(P[:,5].*scales.rho*scales.sound^2)
        vtk["Temperature"]=temperature;vtk["Mach"]=mach;vtk["Level"]=Float32.(mesh.level)
        vorticity===nothing || (vtk["Vorticity"]=vorticity)
        vtk["Obstacle"]=zeros(Float32,n)
        vtk["TimeValue",VTKFieldData()]=[time_s]
    end
    Base.Filesystem.rename(path*".partial.vtu",path*".vtu")
    path*".vtu"
end

function write_planes(folder,mesh,P,scales,planes,findcell;step=0,time_s=0.,offset=(0.,0.,0.),max_cells=262144,gradient=nothing)
    mkpath(folder);files=String[];axisnames=("x","y","z")
    for plane in planes
        tag=plane.tag;kind=plane.plane
        a,b,normal=kind==:xz ? (1,3,2) : kind==:yz ? (2,3,1) : (1,2,3)
        coord=plane.coord===nothing ? mesh.origin[normal]+mesh.extent[normal]/2 : Float64(plane.coord)
        upper=mesh.origin[normal]+mesh.extent[normal]
        # Leaf lookup is half-open at the outer maximum. A cut exactly on that
        # face displays its interior trace, while retaining its physical plane
        # coordinate. The relative-coordinate bound also handles maxima at zero,
        # where prevfloat(0)-origin can otherwise round back to the full extent.
        lookup_coord=coord==upper ? min(prevfloat(upper),mesh.origin[normal]+prevfloat(mesh.extent[normal])) : coord
        h=minimum(mesh.dx)
        while ceil(Int,mesh.extent[a]/h)*ceil(Int,mesh.extent[b]/h)>max_cells;h*=2;end
        n1=ceil(Int,mesh.extent[a]/h);n2=ceil(Int,mesh.extent[b]/h)
        names=("density","pressure","u","v","w","velocity_magnitude","mach","temperature","vorticity_"*axisnames[normal])
        values=[fill(Float32(NaN),n1,n2) for _ in names]
        level=zeros(UInt8,n1,n2);obstacle=ones(UInt8,n1,n2)
        for j in 1:n2,i in 1:n1
            point=ntuple(k->k==normal ? lookup_coord : k==a ? mesh.origin[a]+(i-.5)*h : mesh.origin[b]+(j-.5)*h,3)
            cell=findcell(point...);cell>0 || continue
            r=P[cell,1]*scales.rho;p=P[cell,5]*scales.rho*scales.sound^2
            ux=P[cell,2]*scales.sound;uy=P[cell,3]*scales.sound;uz=P[cell,4]*scales.sound
            speed=sqrt(ux^2+uy^2+uz^2);temp=p/(r*scales.gas_constant)
            vals=(r,p,ux,uy,uz,speed,speed/sqrt(scales.gamma*scales.gas_constant*temp),temp)
            for k in 1:8;values[k][i,j]=vals[k];end
            if gradient!==nothing
                omega=normal==1 ? gradient[cell,4,2]-gradient[cell,3,3] :
                      normal==2 ? gradient[cell,2,3]-gradient[cell,4,1] : gradient[cell,3,1]-gradient[cell,2,2]
                values[9][i,j]=omega*scales.sound
            end
            level[i,j]=UInt8(mesh.level[cell]);obstacle[i,j]=0
        end
        header=Dict{String,Any}("solver"=>"thermal_kinetic_fv","experimental"=>true,
            "plane"=>String(kind),"plane_tag"=>tag,"axis1"=>axisnames[a],"axis2"=>axisnames[b],
            "normal_axis"=>axisnames[normal],"normal_coord_m"=>coord,"frame"=>"domain",
            "normal_sample_coord_m"=>lookup_coord,
            "mesh_offset_m"=>collect(offset),"domain_size_m"=>collect(mesh.extent),
            "origin_m"=>[mesh.origin[a],mesh.origin[b]],"spacing_m"=>h,
            "window_m"=>[mesh.origin[a],mesh.origin[a]+n1*h,mesh.origin[b],mesh.origin[b]+n2*h],
            "step"=>step,"state_step"=>step,"time_s"=>time_s,"with_fields"=>true,
            "pressure_ref_lattice"=>scales.rho*scales.sound^2/scales.gamma,
            "dynamic_pressure_ref_lattice"=>.5scales.rho*scales.speed^2,
            "pressure_note"=>"Absolute physical Pa; legacy reference keys also contain Pa. Cp=(p-p_inf)/q_inf.",
            "rho_ref_kg_m3"=>scales.rho,"u_ref_m_s"=>scales.speed,
            "combine_rule"=>"exact leaf-cell value at sample center; no covered coarse cells",
            "sampling_note"=>"Uniform display sampling may be coarser than the leaf cells; maximum boundary cuts use the one-sided interior leaf. Missing/solid locations are NaN.")
        units=("kg/m3","Pa","m/s","m/s","m/s","m/s","1","K","1/s")
        arrays=Any[(names[k],values[k],units[k],names[k]) for k in eachindex(names)]
        push!(arrays,("level",level,"1","AMR leaf level"),("obstacle",obstacle,"1","solid or outside fluid leaf mesh"))
        file=joinpath(folder,@sprintf("plane_%s_%06d.lbmslice",tag,step))
        write_lbmslice(file,header,arrays);push!(files,file)
    end
    files
end

"Force from the same wall fluxes used by the conservative momentum equation."
function wall_loads(mesh,flux,scales,origin)
    force=zeros(3);moment=zeros(3);factor=scales.rho*scales.sound^2
    pinf=inv(scales.gamma)
    for f in eachindex(mesh.owner)
        mesh.face_tag[f]==7 || continue
        v=[Float64(flux[f,a+1])-(mesh.axis[f]==a ? pinf*mesh.sign[f]*mesh.area[f] : 0.) for a in 1:3].*factor
        r=(mesh.face_x[f]-origin[1],mesh.face_y[f]-origin[2],mesh.face_z[f]-origin[3])
        force.+=v
        moment[1]+=r[2]*v[3]-r[3]*v[2];moment[2]+=r[3]*v[1]-r[1]*v[3];moment[3]+=r[1]*v[2]-r[2]*v[1]
    end
    (;force,moment)
end
end
