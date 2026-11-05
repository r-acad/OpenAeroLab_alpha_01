# Windows setup after the native .cmd bootstrap has found Julia. No .ps1 is loaded.
using Downloads, UUIDs
include(joinpath(@__DIR__, "setup.jl"))

function visual_cpp_installed()
    command = Cmd(Cmd(["reg.exe", "query", raw"HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64", "/v", "Installed"]);
        ignorestatus=true, windows_hide=true)
    output = try read(pipeline(command; stderr=devnull), String) catch; "" end
    occursin(r"(?im)^\s*Installed\s+REG_DWORD\s+0x0*1\b", output)
end

function microsoft_signature_valid(path)
    # Inline signature query is allowed by AllSigned; it does not load a script file.
    script = raw"$s=Get-AuthenticodeSignature -LiteralPath $env:OA_SIGNATURE_FILE; if ($s.Status -eq 'Valid' -and $s.SignerCertificate.Subject -match 'O=Microsoft Corporation(?:,|$)') { exit 0 }; exit 1"
    command = Cmd(Cmd(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script]);
        ignorestatus=true, windows_hide=true)
    process = try run(pipeline(addenv(command, "OA_SIGNATURE_FILE"=>path); stdout=devnull, stderr=devnull)) catch; return false end
    process.exitcode == 0
end

function install_visual_cpp(; offline=false, installed=visual_cpp_installed,
    verify=microsoft_signature_valid, download=Downloads.download,
    execute=command->run(command), private=OpenAeroLabSetup.private_root())
    installed() && return
    local_installer = get(ENV, "LUDWIG_VCREDIST_INSTALLER", "")
    offline && isempty(local_installer) && error("The Microsoft Visual C++ x64 runtime is missing. Ask IT to install it, or set LUDWIG_VCREDIST_INSTALLER to the signed Microsoft x64 installer.")
    folder = joinpath(private, "TEMP_DELETEABLE", "setup")
    mkpath(folder)
    installer = isempty(local_installer) ? joinpath(folder, "vc_redist_$(uuid4()).exe") : abspath(local_installer)
    if isempty(local_installer)
        logfolder = joinpath(private, "runtime", "install_logs"); mkpath(logfolder)
        logpath = joinpath(logfolder, "visual_cpp_$(uuid4()).log")
        println("Downloading the Microsoft Visual C++ x64 runtime required by CUDA. Log: $logpath")
        open(logpath, "w") do log
            try
                OpenAeroLabSetup.instantiate_with_retry(; log) do
                    download("https://aka.ms/vs/17/release/vc_redist.x64.exe", installer)
                end
            catch err
                OpenAeroLabSetup.network_guidance(log)
                error("Visual C++ download failed: $(OpenAeroLabSetup.safe_error(err)). See $logpath, or provide an IT-approved LUDWIG_VCREDIST_INSTALLER.")
            end
        end
    end
    isfile(installer) || error("LUDWIG_VCREDIST_INSTALLER does not point to an existing installer.")
    verify(installer) || error("The Visual C++ installer's Microsoft signature could not be verified, so it was not run. If company policy blocks the Windows signature query, ask IT to install the Microsoft x64 runtime first; do not weaken security policy.")
    println("Installing the signed runtime. Windows may request administrator authorization.")
    command = Cmd(Cmd([installer, "/install", "/quiet", "/norestart"]); ignorestatus=true, windows_hide=true)
    process = execute(command)
    process.exitcode in (0, 1638, 3010) || error("Visual C++ installation failed ($(process.exitcode)). Install the x64 runtime from Microsoft, then rerun setup.")
    process.exitcode == 3010 && println("The runtime installer requests a Windows restart before GPU use.")
end

function main(args=ARGS)
    options = OpenAeroLabSetup.options(args)
    # Explicit installer proxy applies to this Julia process and its children.
    # Never print proxy values because they can contain credentials.
    if haskey(ENV, "LUDWIG_PROXY") && !isempty(ENV["LUDWIG_PROXY"])
        ENV["HTTPS_PROXY"] = ENV["LUDWIG_PROXY"]
    end
    if !("--dry-run" in options || "--help" in options || "-h" in options)
        install_visual_cpp(; offline="--offline" in options)
    end
    OpenAeroLabSetup.main(args)
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        main()
    catch err
        print(stderr, OpenAeroLabSetup.safe_error(err))
        println(stderr, "\nSetup failed. Fix the stated issue, then rerun setup.cmd.")
        exit(1)
    end
end
