"Thermal finite-volume backend; the mesh is built by the main Cartesian AMR mesher."
_pond_case(case::AbstractDict) = get(case,"solver","")=="pond_thermal"


function _pond_flow!(flow::AbstractDict, input::AbstractDict)
    mode=String(get(input,"input_mode","speed"))
    mode in ("speed","mach") || throw(ArgumentError("flow.input_mode must be speed or mach"))
    temperature=_number(get(input,"temperature_k",288.15),"flow.temperature_k";lo=100,hi=2000)
    gamma=_number(get(input,"gamma",1.4),"flow.gamma";lo=1.4,hi=1.4)
    gas_constant=_number(get(input,"gas_constant",287.05),"flow.gas_constant";lo=287.05,hi=287.05)
    sound=sqrt(gamma*gas_constant*temperature)
    if mode=="mach"
        mach=_number(get(input,"mach",nothing),"flow.mach";lo=0.05,hi=1.5)
        flow["velocity"]=mach*sound
    end
    flow["speed_of_sound"]=sound
    flow["mach"]=flow["velocity"]/sound
    flow["temperature_k"]=temperature;flow["gamma"]=gamma
    flow["gas_constant"]=gas_constant;flow["input_mode"]=mode
    flow["cfl"]=_number(get(input,"cfl",.25),"flow.cfl";lo=.01,hi=.4)
    wall=String(get(input,"wall_model","slip"))
    wall in ("slip","noslip") || throw(ArgumentError("flow.wall_model must be slip or noslip"))
    flow["wall_model"]=wall
    .05 <= flow["mach"] <= 1.5 || throw(ArgumentError("Thermal kinetic FV currently admits freestream Mach from 0.05 to 1.5"))
    flow
end
