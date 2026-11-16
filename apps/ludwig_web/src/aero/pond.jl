"Thermal finite-volume backend; the mesh is built by the main Cartesian AMR mesher."
_pond_case(case::AbstractDict) = get(case,"solver","")=="pond_thermal"

# Ideal air. The solver takes both numbers from the case file; the app offers no other gas.
const POND_GAMMA = 1.4
const POND_GAS_CONSTANT = 287.05

"The thermal free stream is set by Mach number and static temperature; speed and sound speed follow from them."
function _pond_flow!(flow::AbstractDict, input::AbstractDict)
    temperature=_number(get(input,"temperature_k",288.15),"flow.temperature_k";lo=100,hi=2000)
    mach=_number(get(input,"mach",nothing),"flow.mach";lo=0.05,hi=1.5)
    sound=sqrt(POND_GAMMA*POND_GAS_CONSTANT*temperature)
    flow["velocity"]=mach*sound
    flow["speed_of_sound"]=sound
    flow["mach"]=mach
    flow["temperature_k"]=temperature;flow["gamma"]=POND_GAMMA
    flow["gas_constant"]=POND_GAS_CONSTANT
    flow["cfl"]=_number(get(input,"cfl",.25),"flow.cfl";lo=.01,hi=.4)
    wall=String(get(input,"wall_model","slip"))
    wall in ("slip","noslip") || throw(ArgumentError("flow.wall_model must be slip or noslip"))
    flow["wall_model"]=wall
    flow
end
