# Graphviz Installation Guide for Windows

**Issue**: Chocolatey installation is failing due to lock file and permissions issues.

**Status**: Manual installation required

---

## Option 1: Manual Installation (Recommended)

### Step 1: Download Graphviz

1. Visit: https://graphviz.org/download/
2. Download **Windows install packages** → **Stable 2.50.0 Windows install packages**
3. Choose: `stable_windows_10_cmake_Release_x64_graphviz-install-2.50.0-win64.exe`

### Step 2: Install

1. Run the downloaded `.exe` file
2. **IMPORTANT**: During installation, check the option **"Add Graphviz to the system PATH for all users"**
3. Default installation path: `C:\Program Files\Graphviz`
4. Complete the installation wizard

### Step 3: Verify Installation

Open a **new** Command Prompt or PowerShell and run:

```bash
dot -V
```

You should see:
```
dot - graphviz version 2.50.0 (...)
```

### Step 4: Render Diagrams

Navigate to the graphviz directory and run the render script:

```bash
cd "c:\Users\17175\Desktop\the agent maker\docs\graphviz"
.\render_all.bat
```

---

## Option 2: Chocolatey with Admin Privileges

If you prefer Chocolatey, you need to run PowerShell **as Administrator**.

### Step 1: Open PowerShell as Admin

1. Press `Win + X`
2. Select **"Windows PowerShell (Admin)"** or **"Terminal (Admin)"**

### Step 2: Remove Lock Files

```powershell
Remove-Item "C:\ProgramData\chocolatey\lib\4be59cffbfa8f5656eec25b49f9c7ba499afb7c1" -Force -ErrorAction SilentlyContinue
Remove-Item "C:\ProgramData\chocolatey\lib\Graphviz" -Recurse -Force -ErrorAction SilentlyContinue
```

### Step 3: Install Graphviz

```powershell
choco install graphviz -y
```

### Step 4: Refresh Environment

```powershell
refreshenv
```

### Step 5: Verify

```powershell
dot -V
```

---

## Option 3: Portable Installation (No Admin Required)

If you cannot get admin privileges:

### Step 1: Download Portable ZIP

1. Visit: https://graphviz.org/download/
2. Download **Portable ZIP** version
3. Extract to a local directory (e.g., `C:\Users\17175\Tools\graphviz`)

### Step 2: Add to PATH (Current User Only)

**PowerShell**:
```powershell
$graphvizPath = "C:\Users\17175\Tools\graphviz\bin"
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
[Environment]::SetEnvironmentVariable("Path", "$currentPath;$graphvizPath", "User")
```

### Step 3: Restart Terminal and Verify

```bash
dot -V
```

---

## Troubleshooting

### "dot: command not found"

**Cause**: Graphviz not in PATH

**Solution**:
1. Find installation directory (usually `C:\Program Files\Graphviz\bin`)
2. Add to PATH manually:
   - Search Windows for "Environment Variables"
   - Edit "Path" variable
   - Add `C:\Program Files\Graphviz\bin`
   - Click OK and restart terminal

### "Error: layout engine not found"

**Cause**: Graphviz plugins not configured

**Solution**:
```bash
dot -c
```

This regenerates the plugin configuration.

### Lock File Issues (Chocolatey)

**Cause**: Previous installation crashed

**Solution** (requires Admin):
```powershell
Remove-Item "C:\ProgramData\chocolatey\lib\4be59cffbfa8f5656eec25b49f9c7ba499afb7c1" -Force
Remove-Item "C:\ProgramData\chocolatey\lib\Graphviz" -Recurse -Force
choco install graphviz -y
```

---

## After Installation - Render All Diagrams

Once Graphviz is installed:

### Windows (Command Prompt):
```cmd
cd "c:\Users\17175\Desktop\the agent maker\docs\graphviz"
render_all.bat
```

### Windows (PowerShell):
```powershell
cd "c:\Users\17175\Desktop\the agent maker\docs\graphviz"
.\render_all.bat
```

### Linux/macOS (Bash):
```bash
cd "c:/Users/17175/Desktop/the agent maker/docs/graphviz"
chmod +x render_all.sh
./render_all.sh
```

---

## Expected Output

After rendering, you should have 8 PNG files:

```
docs/graphviz/
├── 8-phase-pipeline-complete.png       ✅ New
├── agent-forge-master-flow.png         ✅ Existing
├── cicd-pipeline-flow.png              ✅ New
├── infrastructure-systems.png          ✅ New
├── phase-integration-flow.png          ✅ Existing
├── quality-assurance-flow.png          ✅ New
├── testing-infrastructure-flow.png     ✅ New
└── week-1-12-timeline.png              ✅ New
```

---

## Alternative: Online Rendering

If installation continues to fail, you can render diagrams online:

1. Visit: https://dreampuf.github.io/GraphvizOnline/
2. Copy contents of `.dot` file
3. Paste into the online editor
4. Download PNG/SVG

---

## Summary

**Recommended**: Option 1 (Manual Installation)
- Most reliable
- Official installer
- Adds to PATH automatically

**Current Issue**: Chocolatey lock file prevents installation
**Root Cause**: Requires admin privileges + lock file from crashed installation
**Workaround**: Manual installation or run PowerShell as Admin

---

**Created**: 2025-10-16
**Status**: Awaiting manual installation
**Next Step**: Install Graphviz using Option 1, then run `render_all.bat`
