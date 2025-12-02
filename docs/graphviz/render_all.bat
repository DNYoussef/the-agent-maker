@echo off
REM Render all Graphviz diagrams to PNG (Windows)
REM Usage: render_all.bat

echo === Agent Forge V2 - Graphviz Diagram Renderer ===
echo.

REM Check if Graphviz is installed
where dot >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Graphviz 'dot' command not found!
    echo.
    echo Please install Graphviz:
    echo   Windows: choco install graphviz
    echo   Or download from: https://graphviz.org/download/
    echo.
    echo After installation, add Graphviz bin directory to PATH
    echo   Example: C:\Program Files\Graphviz\bin
    echo.
    exit /b 1
)

cd /d "%~dp0"
echo Rendering diagrams in: %CD%
echo.

set total=0
set success=0
set failed=0

REM Render all .dot files
for %%f in (*.dot) do (
    set /a total+=1
    set "input=%%f"
    set "output=%%~nf.png"

    echo Rendering: %%f -^> %%~nf.png

    dot -Tpng "%%f" -o "%%~nf.png" 2>nul
    if %errorlevel% equ 0 (
        set /a success+=1
        echo   [32m✓ Success[0m
    ) else (
        set /a failed+=1
        echo   [31m✗ Failed[0m
    )
    echo.
)

REM Summary
echo === Rendering Complete ===
echo Total:   %total% diagrams
echo Success: %success% diagrams
echo Failed:  %failed% diagrams
echo.

if %failed% equ 0 (
    echo [32m✓ All diagrams rendered successfully![0m
    echo.
    echo Output files:
    dir /b *.png
) else (
    echo [31m✗ Some diagrams failed to render. Check errors above.[0m
)

pause
