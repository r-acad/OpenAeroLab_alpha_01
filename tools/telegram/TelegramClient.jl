module TelegramClient
using HTTP, JSON3
export Client, setting, call, send_text, send_file, download_file, authorized, chunks

function setting(name)
    name in ("TELEGRAM_BOT_TOKEN","TELEGRAM_CHAT_ID") || error("Unsupported credential name")
    value=get(ENV,name,"")
    if isempty(value) && Sys.iswindows()
        # Capture Windows User-environment settings without printing credentials.
        script="[Environment]::GetEnvironmentVariable('$name','User')"
        value=read(Cmd(["powershell","-NoProfile","-NonInteractive","-Command",script]),String)
    end
    isempty(strip(value)) && error("$name is not configured in the user environment")
    strip(value)
end
struct Client
    token::String
    chat::Int64
end
Base.show(io::IO,::Client)=print(io,"TelegramClient.Client(<credentials redacted>)")
Client()=Client(setting("TELEGRAM_BOT_TOKEN"),parse(Int64,setting("TELEGRAM_CHAT_ID")))
function call(client::Client,method::String,fields::AbstractDict;timeout=35)
    response=try
        HTTP.post("https://api.telegram.org/bot$(client.token)/$method",
            ["Content-Type"=>"application/json"],JSON3.write(fields);
            readtimeout=timeout,status_exception=false,retry=false)
    catch
        error("Telegram network error on $method") # HTTP exceptions contain the secret-bearing URL.
    end
    response.status==200 || error("Telegram HTTP $(response.status) on $method")
    data=JSON3.read(response.body,Dict{String,Any})
    get(data,"ok",false) || error("Telegram rejected $method")
    get(data,"result",nothing)
end
function chunks(text;limit=3500)
    chars=collect(strip(text))
    isempty(chars) && return ["(empty response)"]
    [String(chars[i:min(i+limit-1,end)]) for i in 1:limit:length(chars)]
end
function send_text(client::Client,text;reply_to=nothing)
    for part in chunks(text)
        fields=Dict{String,Any}("chat_id"=>client.chat,"text"=>part,"disable_web_page_preview"=>true)
        reply_to===nothing || (fields["reply_parameters"]=Dict("message_id"=>reply_to,"allow_sending_without_reply"=>true))
        call(client,"sendMessage",fields)
        reply_to=nothing
    end
    nothing
end
function send_file(client::Client,path,caption="";photo=true)
    isfile(path)||error("Telegram attachment is missing")
    key=photo ? "photo" : "document";method=photo ? "sendPhoto" : "sendDocument"
    open(path,"r") do io
        form=HTTP.Form(Dict("chat_id"=>string(client.chat),"caption"=>first(caption,1000),
            key=>HTTP.Multipart(basename(path),io,photo ? "image/png" : "application/octet-stream")))
        response=try
            HTTP.post("https://api.telegram.org/bot$(client.token)/$method",[],form;
                status_exception=false,readtimeout=120,retry=false)
        catch
            error("Telegram upload network error")
        end
        response.status==200 && get(JSON3.read(response.body,Dict{String,Any}),"ok",false) || error("Telegram upload rejected")
    end
    nothing
end
function download_file(client::Client,id,path)
    info=call(client,"getFile",Dict("file_id"=>id))
    get(info,"file_size",0)<=20_000_000 || error("Attachment exceeds the 20 MB bridge limit")
    response=try
        HTTP.get("https://api.telegram.org/file/bot$(client.token)/$(info["file_path"])";
            status_exception=false,readtimeout=60,retry=false)
    catch
        error("Telegram attachment download failed")
    end
    response.status==200 && length(response.body)<=20_000_000 || error("Attachment is unavailable or too large")
    mkpath(dirname(path));write(path,response.body)==length(response.body)||error("Incomplete attachment download")
end
function authorized(message,chat)
    peer=get(message,"chat",Dict());sender=get(message,"from",Dict())
    get(peer,"type","")=="private" && get(peer,"id",nothing)==chat &&
        get(sender,"id",nothing)==chat && !get(sender,"is_bot",false)
end
end
