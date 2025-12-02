# Graphviz Installation Script (PowerShell - Run as Administrator)
# This script cleans up lock files and installs Graphviz via Chocolatey

Write-Host "=== Graphviz Installation Script ===" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ERROR: This script must be run as Administrator!" -ForegroundColor Red
    Write-Host ""
    Write-Host "To run as Administrator:" -ForegroundColor Yellow
    Write-Host "  1. Right-click PowerShell" -ForegroundColor Yellow
    Write-Host "  2. Select 'Run as Administrator'" -ForegroundColor Yellow
    Write-Host "  3. Navigate to: c:\Users\17175\Desktop\the agent maker\docs\graphviz" -ForegroundColor Yellow
    Write-Host "  4. Run: .\install_graphviz_admin.ps1" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "✓ Running as Administrator" -ForegroundColor Green
Write-Host ""

# Step 1: Clean up lock files
Write-Host "Step 1: Cleaning up lock files..." -ForegroundColor Cyan
$lockFile = "C:\ProgramData\chocolatey\lib\4be59cffbfa8f5656eec25b49f9c7ba499afb7c1"
$graphvizLib = "C:\ProgramData\chocolatey\lib\Graphviz"

if (Test-Path $lockFile) {
    Remove-Item $lockFile -Force
    Write-Host "  ✓ Removed lock file: $lockFile" -ForegroundColor Green
} else {
    Write-Host "  - Lock file not found (already clean)" -ForegroundColor Gray
}

if (Test-Path $graphvizLib) {
    Remove-Item $graphvizLib -Recurse -Force
    Write-Host "  ✓ Removed partial installation: $graphvizLib" -ForegroundColor Green
} else {
    Write-Host "  - No partial installation found" -ForegroundColor Gray
}

Write-Host ""

# Step 2: Install Graphviz via Chocolatey
Write-Host "Step 2: Installing Graphviz via Chocolatey..." -ForegroundColor Cyan
Write-Host ""

try {
    choco install graphviz -y

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✓ Graphviz installed successfully!" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "✗ Installation failed with exit code: $LASTEXITCODE" -ForegroundColor Red
        Write-Host ""
        Write-Host "Alternative installation methods:" -ForegroundColor Yellow
        Write-Host "  1. Download from: https://graphviz.org/download/" -ForegroundColor Yellow
        Write-Host "  2. See: INSTALL_GRAPHVIZ.md for manual installation" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
} catch {
    Write-Host ""
    Write-Host "✗ Installation error: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 3: Refresh environment
Write-Host ""
Write-Host "Step 3: Refreshing environment variables..." -ForegroundColor Cyan
refreshenv
Write-Host "  ✓ Environment refreshed" -ForegroundColor Green

# Step 4: Verify installation
Write-Host ""
Write-Host "Step 4: Verifying installation..." -ForegroundColor Cyan

try {
    $version = & dot -V 2>&1
    Write-Host "  ✓ Graphviz version: $version" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Verification failed: dot command not found" -ForegroundColor Red
    Write-Host "  ! You may need to restart your terminal" -ForegroundColor Yellow
}

# Step 5: Render diagrams
Write-Host ""
Write-Host "Step 5: Rendering diagrams..." -ForegroundColor Cyan
$graphvizDir = "c:\Users\17175\Desktop\the agent maker\docs\graphviz"

if (Test-Path $graphvizDir) {
    Set-Location $graphvizDir
    Write-Host "  - Changed directory to: $graphvizDir" -ForegroundColor Gray

    if (Test-Path ".\render_all.bat") {
        Write-Host "  - Running render_all.bat..." -ForegroundColor Gray
        & .\render_all.bat
    } else {
        Write-Host "  ! render_all.bat not found" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ! Graphviz directory not found: $graphvizDir" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "=== Installation Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Close and reopen your terminal (to refresh PATH)" -ForegroundColor Yellow
Write-Host "  2. Verify: dot -V" -ForegroundColor Yellow
Write-Host "  3. Render diagrams: cd docs\graphviz && .\render_all.bat" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to exit"
