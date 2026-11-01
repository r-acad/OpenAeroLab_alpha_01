const REQUEST_DIAGNOSTICS_ROOT=joinpath(RUNTIME_ROOT,"diagnostics")

"Retain a local diagnostic without request bodies, upload data or credentials."
function _request_failure(route,method,error,backtrace)
    id=string(uuid4())
    recorded=false
    try
        mkpath(REQUEST_DIAGNOSTICS_ROOT)
        write(joinpath(REQUEST_DIAGNOSTICS_ROOT,"$id.log"),
            "Ludwig Aero Lab request diagnostic\nUTC: $(_now_utc())\nRequest: $method $route\n\n"*
            sprint(showerror,error,backtrace)*"\n")
        recorded=true
    catch diagnostic_error
        @warn "Could not retain request diagnostic" id exception=diagnostic_error
    end
    retryable=error isa Base.IOError || error isa SystemError
    message=retryable ? "The case files are temporarily unavailable. Retrying the refresh." :
        "The case could not be inspected. Open the diagnostic report for details."
    @error "Aero Lab request failed" route request_id=id exception=(error,backtrace)
    _response(retryable ? 503 : 500,Dict("ok"=>false,"error"=>message,"retryable"=>retryable,
        "request_id"=>id,"diagnostic_url"=>recorded ? "/api/aero/diagnostics/$id" : nothing))
end

function request_diagnostic(id)
    file=joinpath(REQUEST_DIAGNOSTICS_ROOT,"$(_id(id)).log")
    isfile(file) || return _bad("Diagnostic report not found",404)
    HTTP.Response(200,["Content-Type"=>"text/plain; charset=utf-8","Cache-Control"=>"no-store",
        "X-Content-Type-Options"=>"nosniff"],read(file))
end
