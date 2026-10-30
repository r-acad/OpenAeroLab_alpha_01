# Optional Cartesian reference benchmark boundary. A moving vertical gate is
# an external body; its two pressure tractions need not be equal.
export vertical_gate_modifier

"""A zero-thickness vertical gate lifted from the floor at a specified speed.

The face's closed fraction transmits no constituent mass. Slip wall pressure
tractions act separately on the two adjacent cells and are reported as an
external gate force per span. The open fraction uses the conservative face
flux already computed by the integrator. This first-order cut-face benchmark
treatment is not an airfoil immersed-wall model.
"""
function vertical_gate_modifier(face_index::Integer,time::Real;speed::Real=1.5,
                                 initial_bottom::Real=0.)
    all(isfinite,(time,speed,initial_bottom)) && time>=0 && speed>=0 ||
        throw(ArgumentError("invalid moving gate time/speed"))
    receipt=Dict{String,Float64}("force_x_on_fluid_per_span"=>0.,"bottom"=>initial_bottom+speed*time)
    function modify!(U,w,model,dx,dy,ymin)
        nx,ny,_=size(U);2<=face_index<=nx || throw(ArgumentError("gate must be on an internal cell face"))
        left,right=face_index-1,face_index;force=0.
        for j in 1:ny
            bottom=ymin+(j-1)*dy
            open_fraction=clamp((receipt["bottom"]-bottom)/dy,0.,1.)
            closed=1-open_fraction
            for k in 1:5;w.flux_x[face_index,j,k]*=open_fraction;end
            pL=mixture_pressure(_cartesian_state(U,left,j),model)
            pR=mixture_pressure(_cartesian_state(U,right,j),model)
            w.source[left,j,2]-=closed*pL/dx
            w.source[right,j,2]+=closed*pR/dx
            force+=closed*(pR-pL)*dy
        end
        receipt["force_x_on_fluid_per_span"]=force
    end
    (;modify!,receipt)
end
