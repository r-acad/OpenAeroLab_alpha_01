using Test
const ROOT=normpath(joinpath(@__DIR__,"../.."))
@testset "Workbench HTML controls have unique IDs" begin
    for relative in ("apps/airfoil_web/static/index.html","apps/ludwig_web/static/index.html",
                     "apps/ludwig_web/static/aero_lab.html","apps/ludwig_web/static/solver_catalog.html")
        html=read(joinpath(ROOT,relative),String)
        ids=[m.captures[1] for m in eachmatch(r"\bid\s*=\s*\"([^\"]+)\"",html)]
        counts=Dict(id=>count(==(id),ids) for id in unique(ids))
        duplicates=sort!([id for (id,n) in counts if n>1])
        @testset "$relative" begin
            @test !isempty(ids)
            @test isempty(duplicates)
        end
    end
end

@testset "Airfoil setup defaults and control registry" begin
    html=read(joinpath(ROOT,"apps/airfoil_web/static/index.html"),String)
    script=read(joinpath(ROOT,"apps/airfoil_web/static/scripts/app.js"),String)
    function selected_value(id)
        select=match(Regex("(?s)<select\\b[^>]*\\bid=\""*id*"\"[^>]*>(.*?)</select>"),html)
        select===nothing && return nothing
        option=match(r"<option\b(?=[^>]*\bselected\b)[^>]*\bvalue=\"([^\"]+)\"[^>]*>",select.captures[1])
        option===nothing ? nothing : option.captures[1]
    end
    @test selected_value("source-kind")=="dat"
    @test selected_value("backend")=="lowspeed"
    @test selected_value("lowspeed-resolution")=="auto"
    @test selected_value("lowspeed-topology")=="auto"
    registry=match(r"(?s)const el = Object.fromEntries\(\[(.*?)\]\.map",script)
    @test registry!==nothing
    ids=Set(m.captures[1] for m in eachmatch(r"\bid\s*=\s*\"([^\"]+)\"",html))
    registered=[m.captures[1] for m in eachmatch(r"\"([^\"]+)\"",registry.captures[1])]
    @test isempty(setdiff(registered,ids))
    @test length(registered)==length(unique(registered))
    @test "cancel-button" in ids
    server=read(joinpath(ROOT,"apps/airfoil_web/src/server.jl"),String)
    service_revision=parse(Int,match(r"\"api_revision\"\s*=>\s*(\d+)",server).captures[1])
    client_revision=parse(Int,match(r"const minimumServiceRevision\s*=\s*(\d+)",script).captures[1])
    @test service_revision>=27
    @test client_revision==service_revision
end

@testset "Visible Airfoil workflow has direct, unique panel destinations" begin
    html=read(joinpath(ROOT,"apps/airfoil_web/static/index.html"),String)
    nav=match(r"(?s)<nav\b[^>]*workflow-bar[^>]*>(.*?)</nav>",html)
    @test nav!==nothing
    @test !occursin("<details",nav.captures[1])
    @test !occursin("menu-popover",nav.captures[1])
    targets=[m.captures[1] for m in eachmatch(r"data-workflow-target=\"([^\"]+)\"",nav.captures[1])]
    destinations=Set(m.captures[1] for m in eachmatch(r"data-airfoil-panel=\"([^\"]+)\"",html))
    @test length(targets)>=10
    @test length(targets)==length(unique(targets))
    @test isempty(setdiff(targets,destinations))
    @test occursin("/static/scripts/workflow.js",html)
    @test occursin("id=\"study-result-location\"",html)
    @test occursin("id=\"menu-results\"",html)
end

@testset "Airfoil setup panel nesting" begin
    html=read(joinpath(ROOT,"apps/airfoil_web/static/index.html"),String)
    void=Set(["area","base","br","col","embed","hr","img","input","link","meta","param","source","track","wbr"])
    stack=String[];mismatches=String[]
    for tag in eachmatch(r"<(/?)([A-Za-z][\w:-]*)(?:[^>\"']|\"[^\"]*\"|'[^']*')*>",html)
        name=lowercase(tag.captures[2])
        name in void && continue
        endswith(tag.match,"/>") && continue
        if tag.captures[1]=="/"
            expected=isempty(stack) ? "none" : pop!(stack)
            name==expected || push!(mismatches,"expected /$expected, found /$name at $(tag.offset)")
        else
            push!(stack,name)
        end
    end
    @test isempty(mismatches)
    @test isempty(stack)
end
