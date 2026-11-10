# One-time Windows setup for running Inferno's GPU experiments on a laptop with a small C: drive.
# Run in an ELEVATED PowerShell (it asks for admin itself). Reboot afterwards.
#
#   1. Move the pagefile to D: (keeps a small 2 GB one on C: for crash dumps). Under WSL2, GPU allocations count
#      against the Windows commit limit (RAM + pagefile), so a pagefile that cannot grow because C: is full makes
#      CUDA report "out of memory" with GBs of VRAM free and can crash the WSL VM.
#   2. Allow non-admin access to NVIDIA GPU performance counters (same switch as NVIDIA Control Panel >
#      Developer > Manage GPU Performance Counters > "Allow access to all users"), needed by Nsight Compute.
#      Only do this on a machine you control.
param([string]$PagefileDrive = "D:", [int]$MinMB = 16384, [int]$MaxMB = 32768)

$ErrorActionPreference = "Stop"
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Start-Process powershell.exe -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    return
}
$log = "$PSScriptRoot\windows_dev_setup.log"
Start-Transcript -Path $log -Force | Out-Null

# --- 1. pagefile -> D:
$cs = Get-CimInstance Win32_ComputerSystem
if ($cs.AutomaticManagedPagefile) {
    Set-CimInstance -InputObject $cs -Property @{ AutomaticManagedPagefile = $false }
}
$existing = Get-CimInstance Win32_PageFileSetting
foreach ($p in $existing) {
    if ($p.Name -like "C:*") { Set-CimInstance -InputObject $p -Property @{ InitialSize = 2048; MaximumSize = 2048 } }
}
$target = "$PagefileDrive\pagefile.sys"
$d = Get-CimInstance Win32_PageFileSetting | Where-Object Name -eq $target
if (-not $d) { $d = New-CimInstance -ClassName Win32_PageFileSetting -Property @{ Name = $target } }
Set-CimInstance -InputObject $d -Property @{ InitialSize = $MinMB; MaximumSize = $MaxMB }
if (-not (Get-CimInstance Win32_PageFileSetting | Where-Object Name -like "C:*")) {
    New-CimInstance -ClassName Win32_PageFileSetting -Property @{ Name = "C:\pagefile.sys"; InitialSize = 2048; MaximumSize = 2048 } | Out-Null
}
Write-Host "Pagefile settings (active after reboot):"
Get-CimInstance Win32_PageFileSetting | Format-Table Name, InitialSize, MaximumSize -AutoSize

# --- 2. GPU performance counters for all users
$key = "HKLM:\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak"
New-Item -Path $key -Force | Out-Null
New-ItemProperty -Path $key -Name RmProfilingAdminOnly -PropertyType DWord -Value 0 -Force | Out-Null
Write-Host "RmProfilingAdminOnly = $((Get-ItemProperty $key).RmProfilingAdminOnly) (0 = counters open to all users; active after reboot)"

Stop-Transcript | Out-Null
Write-Host "`nDone. Reboot Windows for both changes to take effect. Log: $log"
Read-Host "Press Enter to close"
