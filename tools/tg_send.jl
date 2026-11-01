# Optional outbound sender. Credentials stay in the user's environment.
include(joinpath(@__DIR__,"telegram","TelegramClient.jl"))
using .TelegramClient
length(ARGS)>=2 || error("Usage: julia --project=third_party/JXfoil tools/tg_send.jl FILE CAPTION [photo|doc]")
send_file(Client(),ARGS[1],ARGS[2];photo=length(ARGS)<3||ARGS[3]=="photo")
println("Telegram attachment sent")
