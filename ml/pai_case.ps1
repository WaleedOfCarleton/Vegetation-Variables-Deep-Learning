param(
  [Parameter(Position=0)]
  [string]$Case = "002",

  [Parameter(Position=1)]
  [string]$Env = "hemipy-gpu",

  [Parameter()]
  [switch]$Cpu,

  [Parameter()]
  [int]$PrintMax = 30
)

$ErrorActionPreference = "Stop"

# Resolve repo root from this script's location (repo_root\ml\pai_case.ps1)
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

# Normalize case input: allow "2", "002", "Case 002"
$caseStr = $Case.Trim()
if ($caseStr -match '^(?i)case\s*\d+$') {
  # already like "Case 002"
} elseif ($caseStr -match '^\d+$') {
  $caseStr = "Case {0:D3}" -f [int]$caseStr
} else {
  throw "Unrecognized case format: '$Case' (use 2, 002, or 'Case 002')"
}

# Find conda
$condaCmd = Get-Command conda -ErrorAction SilentlyContinue
if ($condaCmd) {
  $condaExe = $condaCmd.Source
} elseif (Test-Path "C:\ProgramData\anaconda3\Scripts\conda.exe") {
  $condaExe = "C:\ProgramData\anaconda3\Scripts\conda.exe"
} else {
  throw "Could not find conda on PATH or at C:\ProgramData\anaconda3\Scripts\conda.exe"
}

$argsList = @(
  "--no-plugins",
  "run",
  "-n", $Env,
  "python",
  "ml\\cnn_baseline\\predict_case_pai.py",
  "--case", $caseStr,
  "--print-max", "$PrintMax"
)

if ($Cpu) {
  $argsList += "--cpu"
}

& $condaExe @argsList
