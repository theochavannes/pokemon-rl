# Launch TensorBoard for pokemon_rl training runs
# Usage:
#   .\scripts\tensorboard.ps1              # all runs
#   .\scripts\tensorboard.ps1 run_033      # specific run

param([string]$Run = "")

$env:Path = "C:\Users\theoc\miniconda3\envs\pokemon_rl;C:\Users\theoc\miniconda3\envs\pokemon_rl\Scripts;" + $env:Path

if ($Run) {
    $logdir = "runs/$Run/logs"
    Write-Host "TensorBoard: $logdir" -ForegroundColor Cyan
} else {
    $logdir = "runs"
    Write-Host "TensorBoard: all runs" -ForegroundColor Cyan
}

Write-Host "Open http://localhost:6006 in your browser" -ForegroundColor Green
tensorboard --logdir $logdir --port 6006
