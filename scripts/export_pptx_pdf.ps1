param([string]$In, [string]$OutPdf)
# Exports a .pptx to PDF with the installed PowerPoint (handy as a backup if the presentation laptop lacks PowerPoint).
$app = New-Object -ComObject PowerPoint.Application
try {
    $pres = $app.Presentations.Open((Resolve-Path $In).Path, $true, $false, $false)
    $pres.SaveAs([System.IO.Path]::GetFullPath($OutPdf), 32)   # ppSaveAsPDF
    $pres.Close()
} finally { $app.Quit() }
Write-Output "wrote $OutPdf"
