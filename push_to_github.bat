@echo off
echo ========================================
echo  GitHub Push Script
echo ========================================
echo.

REM Navigate to project directory
cd /d "c:\Users\ASUS\Downloads\mansi"

echo Step 1: Initializing Git (if not already)...
git init

echo.
echo Step 2: Adding remote repository...
git remote remove origin 2>nul
git remote add origin https://github.com/mansityagi01/Predictive-Project.git

echo.
echo Step 3: Adding all files (except those in .gitignore)...
git add .

echo.
echo Step 4: Committing changes...
git commit -m "Complete ML project: Customer Spending Classification with 4 models"

echo.
echo Step 5: Setting main branch...
git branch -M main

echo.
echo Step 6: Pushing to GitHub...
git push -u origin main --force

echo.
echo ========================================
echo  Done! Check your repository at:
echo  https://github.com/mansityagi01/Predictive-Project
echo ========================================
pause
