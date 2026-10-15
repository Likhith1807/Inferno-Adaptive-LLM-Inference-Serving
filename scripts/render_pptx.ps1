param([string]$In, [string]$OutDir, [int]$Width = 1600)
# Renders every slide of a .pptx to PNG using the installed PowerPoint (exactly what the presenter will see).
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$app = New-Object -ComObject PowerPoint.Application
try {
    $pres = $app.Presentations.Open((Resolve-Path $In).Path, $true, $false, $false)
    $h = [int]($Width * 9 / 16)
    foreach ($s in $pres.Slides) {
        $f = Join-Path (Resolve-Path $OutDir).Path ("slide_{0:D2}.png" -f $s.SlideIndex)
        $s.Export($f, "PNG", $Width, $h)
    }
    $pres.Close()
} finally { $app.Quit() }
Write-Output "rendered to $OutDir"
