#!/usr/bin/env bash
# ==============================================================================
# Step-by-Step Shell Script to Initialize & Push Repository to GitHub
# Repository: gpt2-doubling-falsification
# ==============================================================================

set -euo pipefail

# Configuration: Replace with your GitHub username or organization
GITHUB_USERNAME="YOUR_GITHUB_USERNAME"
REPO_NAME="gpt2-doubling-falsification"

echo "=== [1/5] Entering repository directory ==="
cd "$(dirname "$0")"

echo "=== [2/5] Initializing Git repository ==="
git init -b main

echo "=== [3/5] Staging files ==="
git add .

echo "=== [4/5] Creating initial publication-grade commit ==="
git commit -m "feat: complete research workspace and falsification toolkit for GPT-2 Small doubling anomaly"

echo "=== [5/5] Configuring remote and pushing to GitHub ==="
echo "Note: If you have already created the remote repository on GitHub, run:"
echo "  git remote add origin https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
echo "  git push -u origin main"
echo ""
echo "Or using the GitHub CLI (gh):"
echo "  gh repo create ${REPO_NAME} --public --source=. --remote=origin --push"
echo ""
echo "=== Done! ==="
