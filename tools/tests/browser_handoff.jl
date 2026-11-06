using Test
include(joinpath(@__DIR__,"..","..","apps","common","Browser.jl"))
const JULIA=joinpath(Sys.BINDIR,Sys.iswindows() ? "julia.exe" : "julia")
const NO_BROWSER=`$JULIA --startup-file=no -e $("exit(0)")`
@testset "Only safe local workbench destinations may be opened" begin
    for url in ("http://127.0.0.1:8058/","http://127.0.0.1:8058/aero-lab","http://127.0.0.1:8058/aero-lab?job=647d1c8f-98d4-4032-867f-9e11c4765ffc","http://127.0.0.1:8058/aero-lab?job=647d1c8f-98d4-4032-867f-9e11c4765ffc&reporting=1")
        @test DesktopBrowser.open_browser(url;command=NO_BROWSER)
    end
    for url in ("https://example.com/","http://127.0.0.1:0/","http://127.0.0.1:99999/","http://127.0.0.1:8058/'","http://127.0.0.1:8058/\n","http://127.0.0.1:8058/aero-lab?job=x","http://127.0.0.1:8058/aero-lab?job=647d1c8f-98d4-4032-867f-9e11c4765ffc&redirect=https://example.com")
        @test_throws ArgumentError DesktopBrowser.open_browser(url;command=NO_BROWSER)
    end
end
