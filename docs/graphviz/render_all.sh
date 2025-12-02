#!/bin/bash
# Render all Graphviz diagrams to PNG
# Usage: ./render_all.sh

echo "=== Agent Forge V2 - Graphviz Diagram Renderer ==="
echo ""

# Check if Graphviz is installed
if ! command -v dot &> /dev/null; then
    echo "ERROR: Graphviz 'dot' command not found!"
    echo ""
    echo "Please install Graphviz:"
    echo "  Ubuntu/Debian: sudo apt-get install graphviz"
    echo "  macOS:         brew install graphviz"
    echo "  Windows:       choco install graphviz"
    echo ""
    exit 1
fi

# Get directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Rendering diagrams in: $SCRIPT_DIR"
echo ""

# Counter
total=0
success=0
failed=0

# Render all .dot files
for file in *.dot; do
    if [ -f "$file" ]; then
        output="${file%.dot}.png"
        echo "Rendering: $file -> $output"

        if dot -Tpng "$file" -o "$output"; then
            ((success++))
            echo "  ✓ Success"
        else
            ((failed++))
            echo "  ✗ Failed"
        fi

        ((total++))
        echo ""
    fi
done

# Summary
echo "=== Rendering Complete ==="
echo "Total:   $total diagrams"
echo "Success: $success diagrams"
echo "Failed:  $failed diagrams"
echo ""

if [ $failed -eq 0 ]; then
    echo "✓ All diagrams rendered successfully!"
    echo ""
    echo "Output files:"
    ls -1 *.png
else
    echo "✗ Some diagrams failed to render. Check errors above."
fi
