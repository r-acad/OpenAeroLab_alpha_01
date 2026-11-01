$ErrorActionPreference = 'Stop'
try {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class LudwigFolderDialog {
    [ComImport, Guid("DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7")]
    private class FileOpenDialog { }
    [ComImport, Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IShellItem {
        void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
        void GetParent(out IShellItem parent);
        void GetDisplayName(uint type, out IntPtr name);
        void GetAttributes(uint mask, out uint attributes);
        void Compare(IShellItem item, uint hint, out int order);
    }
    [ComImport, Guid("42F85136-DB7E-439C-85F1-E4075D135FC8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IFileDialog {
        [PreserveSig] int Show(IntPtr owner);
        void SetFileTypes(uint count, IntPtr specs);
        void SetFileTypeIndex(uint index);
        void GetFileTypeIndex(out uint index);
        void Advise(IntPtr events, out uint cookie);
        void Unadvise(uint cookie);
        void SetOptions(uint options);
        void GetOptions(out uint options);
        void SetDefaultFolder(IShellItem folder);
        void SetFolder(IShellItem folder);
        void GetFolder(out IShellItem folder);
        void GetCurrentSelection(out IShellItem item);
        void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string name);
        void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string name);
        void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string title);
        void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string label);
        void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string label);
        void GetResult(out IShellItem item);
        void AddPlace(IShellItem item, uint location);
        void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string extension);
        void Close(int hr);
        void SetClientGuid(ref Guid guid);
        void ClearClientData();
        void SetFilter(IntPtr filter);
    }
    [DllImport("shell32.dll", CharSet=CharSet.Unicode, PreserveSig=false)]
    private static extern void SHCreateItemFromParsingName(string path, IntPtr context, ref Guid iid, out IShellItem item);
    [DllImport("user32.dll")] private static extern IntPtr GetForegroundWindow();

    public static string Choose(string path, string title) {
        IntPtr owner = GetForegroundWindow();
        IFileDialog dialog = (IFileDialog)new FileOpenDialog();
        try {
            uint options; dialog.GetOptions(out options);
            dialog.SetOptions(options | 0x20u | 0x40u | 0x800u | 0x8u);
            dialog.SetTitle(title);
            dialog.SetOkButtonLabel("Select folder");
            if (!String.IsNullOrWhiteSpace(path) && System.IO.Directory.Exists(path)) {
                Guid iid = typeof(IShellItem).GUID; IShellItem initial;
                SHCreateItemFromParsingName(path, IntPtr.Zero, ref iid, out initial);
                try { dialog.SetFolder(initial); } finally { Marshal.ReleaseComObject(initial); }
            }
            int hr = dialog.Show(owner);
            if (hr == unchecked((int)0x800704C7)) return null;
            Marshal.ThrowExceptionForHR(hr);
            IShellItem result; dialog.GetResult(out result);
            try {
                IntPtr name; result.GetDisplayName(0x80058000u, out name);
                try { return Marshal.PtrToStringUni(name); } finally { Marshal.FreeCoTaskMem(name); }
            } finally { Marshal.ReleaseComObject(result); }
        } finally { Marshal.ReleaseComObject(dialog); }
    }
}
'@
    # Compile-only is used by validation; it never opens a desktop window.
    if ($env:LUDWIG_DIALOG_COMPILE_ONLY -eq '1') { exit 0 }
    $chosen = [LudwigFolderDialog]::Choose($env:LUDWIG_DIALOG_PATH, $env:LUDWIG_DIALOG_TITLE)
    $response = @{ ok = $true; cancelled = ($null -eq $chosen); path = $chosen }
} catch {
    $response = @{ ok = $false; error = $_.Exception.Message }
}
[System.IO.File]::WriteAllText($env:LUDWIG_DIALOG_RESULT, ($response | ConvertTo-Json -Compress), (New-Object System.Text.UTF8Encoding($false)))
