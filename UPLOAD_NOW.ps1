# Gesture Recognition - Upload to GitHub Script
# Username: rayhanros2030

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GESTURE RECOGNITION REPOSITORY" -ForegroundColor Yellow
Write-Host "  GitHub Upload Script" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if already have remote
$hasRemote = git remote -v 2>$null
if ($hasRemote) {
    Write-Host "Remote already configured:" -ForegroundColor Yellow
    git remote -v
    Write-Host ""
    $continue = Read-Host "Do you want to remove and re-add it? (y/n)"
    if ($continue -eq "y") {
        git remote remove origin
        Write-Host "Removed existing remote." -ForegroundColor Green
    } else {
        Write-Host "Keeping existing remote. Exiting." -ForegroundColor Yellow
        exit
    }
}

Write-Host "Setting up GitHub remote..." -ForegroundColor White

# Add remote
git remote add origin https://github.com/rayhanros2030/Gesture-Recognition.git

# Rename branch to main
git branch -M main

Write-Host ""
Write-Host "Remote configured:" -ForegroundColor Green
git remote -v

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "READY TO UPLOAD!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "NEXT STEP: Create repository on GitHub first!" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Go to: https://github.com/new" -ForegroundColor White
Write-Host "2. Repository name: Gesture-Recognition" -ForegroundColor White
Write-Host "3. DO NOT add README, .gitignore, or license" -ForegroundColor White
Write-Host "4. Click 'Create repository'" -ForegroundColor White
Write-Host ""
Write-Host "Then run: git push -u origin main" -ForegroundColor Cyan
Write-Host ""
Write-Host "OR run this script again and I'll do it for you!" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$autoPush = Read-Host "Do you want to push now? (y/n)"
if ($autoPush -eq "y") {
    Write-Host ""
    Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
    git push -u origin main
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "SUCCESS! Repository uploaded!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "View your repository at:" -ForegroundColor Cyan
    Write-Host "https://github.com/rayhanros2030/Gesture-Recognition" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "Manual upload: git push -u origin main" -ForegroundColor Yellow
    Write-Host ""
}

