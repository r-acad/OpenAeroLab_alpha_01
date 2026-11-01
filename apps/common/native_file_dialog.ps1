$ErrorActionPreference = 'Stop'
try {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -ReferencedAssemblies System.Windows.Forms -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public class LudwigDialogOwner : System.Windows.Forms.IWin32Window {
    [DllImport("user32.dll")] private static extern IntPtr GetForegroundWindow();
    private IntPtr handle = GetForegroundWindow();
    public IntPtr Handle { get { return handle; } }
}
'@
    if ($env:LUDWIG_DIALOG_COMPILE_ONLY -eq '1') { exit 0 }
    $save = $env:LUDWIG_DIALOG_MODE -eq 'save'
    if ($save) { $dialog = New-Object System.Windows.Forms.SaveFileDialog }
    else { $dialog = New-Object System.Windows.Forms.OpenFileDialog }
    try {
        $dialog.Title = $env:LUDWIG_DIALOG_TITLE
        $dialog.Filter = 'Ludwig studies (*.ludwig-study)|*.ludwig-study'
        $dialog.DefaultExt = 'ludwig-study'
        $dialog.AddExtension = $true
        $dialog.ValidateNames = $true
        $dialog.RestoreDirectory = $true
        $initial = $env:LUDWIG_DIALOG_PATH
        if ([System.IO.Directory]::Exists($initial)) { $dialog.InitialDirectory = $initial }
        elseif ($initial) {
            $parent = [System.IO.Path]::GetDirectoryName($initial)
            if ([System.IO.Directory]::Exists($parent)) { $dialog.InitialDirectory = $parent }
            $dialog.FileName = [System.IO.Path]::GetFileName($initial)
        }
        if ($save) { $dialog.OverwritePrompt = $true }
        else { $dialog.CheckFileExists = $true; $dialog.Multiselect = $false }
        $owner = New-Object LudwigDialogOwner
        $accepted = $dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK
        $response = @{ ok = $true; cancelled = !$accepted; path = $(if ($accepted) { $dialog.FileName } else { $null }) }
    } finally { $dialog.Dispose() }
} catch { $response = @{ ok = $false; error = $_.Exception.Message } }
[System.IO.File]::WriteAllText($env:LUDWIG_DIALOG_RESULT, ($response | ConvertTo-Json -Compress), (New-Object System.Text.UTF8Encoding($false)))
