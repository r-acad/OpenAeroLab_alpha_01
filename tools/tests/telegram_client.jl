using Test
include(joinpath(@__DIR__,"..","telegram","TelegramClient.jl"))
using .TelegramClient
@testset "Private Telegram client boundary" begin
    @test chunks("")==["(empty response)"]
    @test join(chunks(repeat("abc",3000)))==repeat("abc",3000)
    @test all(length(c)<=3500 for c in chunks(repeat("\U0001f680",7001)))
    @test authorized(Dict("chat"=>Dict("type"=>"private","id"=>12),"from"=>Dict("id"=>12)),12)
    @test !authorized(Dict("chat"=>Dict("type"=>"group","id"=>12),"from"=>Dict("id"=>12)),12)
    @test !authorized(Dict("chat"=>Dict("type"=>"private","id"=>12),"from"=>Dict("id"=>13)),12)
    @test !authorized(Dict("chat"=>Dict("type"=>"private","id"=>12),"from"=>Dict("id"=>12,"is_bot"=>true)),12)
    @test !occursin("SECRET",repr(Client("SECRET",12)))
end
