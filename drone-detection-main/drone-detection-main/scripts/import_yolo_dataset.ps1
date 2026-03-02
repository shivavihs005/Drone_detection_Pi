param(
    [Parameter(Mandatory = $true)]
    [string]$ZipPath,

    [string]$ProjectRoot = "."
)

$ErrorActionPreference = "Stop"

$project = Resolve-Path $ProjectRoot
$zip = Resolve-Path $ZipPath

$tempExtract = Join-Path $project "_tmp_dataset_extract"
$targetRoot = Join-Path $project "datasets\drone"

if (Test-Path $tempExtract) {
    Remove-Item -Path $tempExtract -Recurse -Force
}

New-Item -ItemType Directory -Path $tempExtract | Out-Null
Expand-Archive -Path $zip -DestinationPath $tempExtract -Force

function Get-SplitFolder {
    param(
        [string]$Root,
        [string]$Base,
        [string[]]$Candidates
    )

    foreach ($name in $Candidates) {
        $candidate = Join-Path (Join-Path $Root $Base) $name
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

function Sync-Files {
    param(
        [string]$Source,
        [string]$Target,
        [string]$Filter
    )

    if (-not (Test-Path $Target)) {
        New-Item -ItemType Directory -Path $Target -Force | Out-Null
    }

    if (Test-Path $Source) {
        Get-ChildItem -Path $Source -File -Filter $Filter | ForEach-Object {
            Copy-Item -Path $_.FullName -Destination (Join-Path $Target $_.Name) -Force
        }
    }
}

$datasetRoot = Get-ChildItem -Path $tempExtract -Directory | Select-Object -First 1
if (-not $datasetRoot) {
    throw "No extracted dataset folder found in ZIP."
}

$imagesTrain = Get-SplitFolder -Root $datasetRoot.FullName -Base "images" -Candidates @("train")
$imagesVal = Get-SplitFolder -Root $datasetRoot.FullName -Base "images" -Candidates @("val", "valid")
$imagesTest = Get-SplitFolder -Root $datasetRoot.FullName -Base "images" -Candidates @("test")

$labelsTrain = Get-SplitFolder -Root $datasetRoot.FullName -Base "labels" -Candidates @("train")
$labelsVal = Get-SplitFolder -Root $datasetRoot.FullName -Base "labels" -Candidates @("val", "valid")
$labelsTest = Get-SplitFolder -Root $datasetRoot.FullName -Base "labels" -Candidates @("test")

if (-not $imagesTrain -or -not $labelsTrain) {
    throw "Could not find required train images/labels folders in YOLO dataset ZIP."
}

Sync-Files -Source $imagesTrain -Target (Join-Path $targetRoot "images\train") -Filter "*"
if ($imagesVal) { Sync-Files -Source $imagesVal -Target (Join-Path $targetRoot "images\val") -Filter "*" }
if ($imagesTest) { Sync-Files -Source $imagesTest -Target (Join-Path $targetRoot "images\test") -Filter "*" }

Sync-Files -Source $labelsTrain -Target (Join-Path $targetRoot "labels\train") -Filter "*.txt"
if ($labelsVal) { Sync-Files -Source $labelsVal -Target (Join-Path $targetRoot "labels\val") -Filter "*.txt" }
if ($labelsTest) { Sync-Files -Source $labelsTest -Target (Join-Path $targetRoot "labels\test") -Filter "*.txt" }

Remove-Item -Path $tempExtract -Recurse -Force

Write-Host "Dataset import complete."
Write-Host "Target: $targetRoot"
