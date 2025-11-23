<#
PowerShell script to download and install Tesseract Mongolian traineddata (mon.traineddata).
Usage (run as Administrator):
  .\install_mon_traineddata.ps1
Or specify destination tessdata folder:
  .\install_mon_traineddata.ps1 -Dest "C:\Program Files\Tesseract-OCR\tessdata"
#>
param(
    [string]$Dest
)

# Determine default tessdata folder
if (-not $Dest) {
    if ($env:TESSDATA_PREFIX) {
        $Dest = Join-Path $env:TESSDATA_PREFIX "tessdata"
    }
    else {
        $Dest = "C:\Program Files\Tesseract-OCR\tessdata"
    }
}

if (-not (Test-Path $Dest)) {
    Write-Host "ERROR: tessdata folder not found at: $Dest" -ForegroundColor Red
    Write-Host "Please make sure Tesseract is installed or provide -Dest path." -ForegroundColor Yellow
    exit 1
}

$rawUrl = "https://raw.githubusercontent.com/tesseract-ocr/tessdata/main/mon.traineddata"
$temp = Join-Path $env:TEMP "mon.traineddata"

Write-Host "Downloading mon.traineddata from: $rawUrl"
try {
    Invoke-WebRequest -Uri $rawUrl -OutFile $temp -UseBasicParsing -ErrorAction Stop
}
catch {
    Write-Host "Download failed: $_" -ForegroundColor Red
    exit 2
}

Write-Host "Saving to: $Dest"
try {
    Copy-Item -Path $temp -Destination $Dest -Force -ErrorAction Stop
}
catch {
    Write-Host "Copy failed: $_" -ForegroundColor Red
    Write-Host "If you're running into permission errors, run PowerShell as Administrator and try again." -ForegroundColor Yellow
    exit 3
}

Write-Host "Installed mon.traineddata to: $Dest" -ForegroundColor Green
Write-Host "You can verify with:`n  Get-ChildItem '$Dest' | Select-Object Name`" -ForegroundColor Cyan
Remove-Item $temp -ErrorAction SilentlyContinue
