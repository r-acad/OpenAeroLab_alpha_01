module WorkbenchStatic
using HTTP
const FILES=Dict("workbench.css"=>"text/css", "workbench.js"=>"application/javascript", "palettes.js"=>"application/javascript")
function response(path::AbstractString)
    name=String(chopprefix(path,"/common-static/"))
    haskey(FILES,name) || return HTTP.Response(404,"Not found")
    HTTP.Response(200,["Content-Type"=>FILES[name]*"; charset=utf-8","Cache-Control"=>"no-store"],read(joinpath(@__DIR__,"static",name)))
end
end
