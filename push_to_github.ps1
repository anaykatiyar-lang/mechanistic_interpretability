<#
.SYNOPSIS
    PowerShell script to initialize and push gpt2-doubling-falsification to GitHub.
#>

param(
    [string]$GitHubUsername = "YOUR_GITHUB_USERNAME",
    [string]$RepoName = "gpt2-doubling-falsification"
)

$ErrorActionPreference = "Stop"

Write-Host "=== [1/5] Entering repository directory ===" -ForegroundColor Cyan
Set-Location $PSScriptRoot

Write-Host "=== [2/5] Initializing Git repository ===" -ForegroundColor Cyan
git init -b main

Write-Host "=== [3/5] Staging all files ===" -ForegroundColor Cyan
git add .

Write-Host "=== [4/5] Creating initial commit ===" -ForegroundColor Cyan
git commit -m "feat: complete research workspace and falsification toolkit for GPT-2 Small doubling anomaly"

Write-Host "=== [5/5] Remote configuration ===" -ForegroundColor Cyan
Write-Host "To link and push to your remote GitHub repository, execute:" -ForegroundColor Yellow
Write-Host "  git remote add origin https://github.com/$GitHubUsername/$RepoName.git" -ForegroundColor Green
Write-Host "  git push -u origin main" -ForegroundColor Green
Write-Host "`nOr using GitHub CLI (gh):" -ForegroundColor Yellow
Write-Host "  gh repo create $RepoName --public --source=. --remote=origin --push" -ForegroundColor Green
