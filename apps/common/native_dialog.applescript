on run argv
    set mode to item 1 of argv
    set initialPath to item 2 of argv
    set promptText to item 3 of argv
    set suggestedName to item 4 of argv
    set initialFolder to path to home folder
    if initialPath is not "" then
        try
            set initialFolder to POSIX file initialPath as alias
        end try
    end if
    try
        if mode is "folder" then
            set chosen to choose folder with prompt promptText default location initialFolder
        else if mode is "save" then
            set chosen to choose file name with prompt promptText default name suggestedName default location initialFolder
        else
            set chosen to choose file with prompt promptText default location initialFolder
        end if
        return POSIX path of chosen
    on error number -128
        return ""
    end try
end run
