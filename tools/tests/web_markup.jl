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
    @test service_revision>=29
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

@testset "Airfoil workflow: solver first, numbered steps, Advanced groups, no withdrawn inputs" begin
    html=read(joinpath(ROOT,"apps/airfoil_web/static/index.html"),String)
    aside=match(r"(?s)<aside\b[^>]*setup-column[^>]*>(.*?)</aside>",html)
    @test aside!==nothing
    setup=aside.captures[1]
    options(id)=[m.captures[1] for m in eachmatch(r"<option\b[^>]*\bvalue=\"([^\"]*)\"",
        match(Regex("(?s)<select\\b[^>]*\\bid=\""*id*"\"[^>]*>(.*?)</select>"),html).captures[1])]
    # The solver decides which inputs exist: its selector stands above every step panel and offers runnable solvers only.
    solver=findfirst("id=\"backend\"",setup);first_panel=findfirst("data-airfoil-panel=",setup)
    @test solver!==nothing && first_panel!==nothing && first(solver)<first(first_panel)
    @test occursin(r"<span>Solver</span>\s*<select id=\"backend\"",setup)
    @test options("backend")==["lowspeed","transonic","cavitation","lbm","jxfoil","xfoil","both","cfd_both","xfoil_cfd_both","xfoil_lowspeed"]
    @test !occursin("disabled",match(r"(?s)<select id=\"backend\">(.*?)</select>",setup).captures[1])
    # The tab row reads as numbered steps in working order; the other tabs follow a separator.
    nav=match(r"(?s)<nav\b[^>]*workflow-bar[^>]*>(.*?)</nav>",html).captures[1]
    steps=[(m.captures[1],m.captures[2]) for m in
        eachmatch(r"workflow-step[^>]*data-workflow-target=\"([^\"]+)\"[^>]*><span class=\"step-number\">(\d)</span>",nav)]
    @test steps==[("model","1"),("flow","2"),("mesh","3"),("solver","4"),("outputs","5"),("result-controls","6")]
    separator=findfirst("workflow-separator",nav)
    @test separator!==nothing
    @test all(first(findfirst("data-workflow-target=\"$name\"",nav))>first(separator) for name in ("study","display","validation","help"))
    # One Advanced group per step that has rarely needed settings, closed in the page source; no other disclosure widget.
    details=[m.match for m in eachmatch(r"<details\b[^>]*>",setup)]
    @test [match(r"id=\"([^\"]+)\"",d).captures[1] for d in details]==["advanced-flow","advanced-mesh","advanced-solver","advanced-outputs"]
    @test all(d->occursin("advanced-group",d) && !occursin(r"\bopen\b",d),details)
    # The Mesh and Run steps end with their actions; the existing buttons keep their ids.
    ids=[m.captures[1] for m in eachmatch(r"\bid\s*=\s*\"([^\"]+)\"",html)]
    @test all(in(ids),("create-mesh","create-mesh-step","run-button","run-step-button"))
    # One mesh resolution control; removed controls are absent.
    @test count(==("preview-resolution"),ids)==1
    @test isempty(intersect(ids,["cfd-resolution","refine-mesh","open-saved-airfoil","transport","lbm-wall-model",
                                 "tab-button-campaign","tab-campaign","setup-method"]))
    # Removed options have no control: mesh profile, trailing-edge topology, base faces, low-speed topology, the
    # research implicit settings and the saved-setting list.
    @test isempty(intersect(ids,["cfd-mesh","gas-mesh-field","cfd-te-topology","cfd-te-topology-field","cfd-n-base","base-faces-field",
        "lowspeed-topology","implicit-method-options","saved-flow-overrides","clear-saved-flow-overrides","research-settings-note"]))
    @test !any(startswith("implicit-"),ids)
    @test options("rans-method")==["standard","irsj","newton"]
    @test options("lowspeed-resolution")==["auto","fine"]
    @test [m.captures[1] for m in eachmatch(r"data-validation=\"([^\"]+)\"",html)]==["rae2822-case9","oat15a-2.5","naca0012-stall"]
    @test !occursin("data-open-tab",html) && !occursin("data-command",html) && !occursin("research_settings",html)
    @test !occursin(r"(?i)a-plus|aplus",html)
    @test options("lowspeed-start")==["auto","projection"]
    @test options("lowspeed-mut-base")==["auto","extrapolated"]
    @test options("gurney-refine")==["auto","standard","fine"]
    @test !occursin(r"(?i)>[^<>]*legacy[^<>]*<",html)
    @test !occursin("campaign",html)
    # The server serves exactly the page files that exist, each under its own path only.
    server=read(joinpath(ROOT,"apps/airfoil_web/src/server.jl"),String)
    served=match(r"(?s)const STATIC_FILES = Dict\((.*?)\n\)",server)
    @test served!==nothing
    files=unique(m.captures[1] for m in eachmatch(r"=>\s*\"([^\"]+)\"",served.captures[1]))
    @test !isempty(files) && all(file->isfile(joinpath(ROOT,"apps/airfoil_web/static",file)),files)
    @test [m.captures[1] for m in eachmatch(r"\"([^\"]+)\"\s*=>",served.captures[1])]==files
    @test !occursin("campaign",server) && !occursin("research_settings",server)
    # The shared workbench script binds no hook that no page carries.
    workbench=read(joinpath(ROOT,"apps/common/static/workbench.js"),String)
    @test !occursin("data-open-tab",workbench) && !occursin("data-command",workbench)
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
