param(
    [string]$OutDir = ".\\downloads\\visdrone"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python is required. Install Python and run again."
}

Write-Host "Installing/Updating gdown..."
python -m pip install --upgrade gdown

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}

$files = @(
    @{ Name = "VisDrone2019-DET-train.zip"; Id = "1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn" },
    @{ Name = "VisDrone2019-DET-val.zip"; Id = "1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59" },
    @{ Name = "VisDrone2019-DET-test-dev.zip"; Id = "1PFdW_VFSCfZ_sTSZAGjQdifF_Xd5mf0V" }
)

foreach ($item in $files) {
    $target = Join-Path $OutDir $item.Name
    Write-Host "Downloading $($item.Name)..."
    python -m gdown --id $item.Id --output $target
}

Write-Host "Download complete: $OutDir"
Write-Host "Note: VisDrone labels are not YOLO format. Convert annotations before training YOLO."
