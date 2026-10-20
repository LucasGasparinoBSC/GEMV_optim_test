#!/usr/bin/env fish

# Run GEMV benchmark across multiple matrix-size cases.
# Default case format: nrows:ncols

set -l script_dir (cd (dirname (status --current-filename)); and pwd)
cd "$script_dir"

function usage
    echo "Usage: ./run_sweep.fish [nrows:ncols ...]"
    echo ""
    echo "Examples:"
    echo "  ./run_sweep.fish"
    echo "  ./run_sweep.fish 20:500 50:2000 200:20000"
end

function parse_case --argument-names spec
    set -l parts (string split ':' -- "$spec")
    if test (count $parts) -ne 2
        return 1
    end

    set -l nr $parts[1]
    set -l nc $parts[2]

    if not string match -qr '^[0-9]+$' -- "$nr"
        return 1
    end
    if not string match -qr '^[0-9]+$' -- "$nc"
        return 1
    end
    if test "$nr" -le 0 -o "$nc" -le 0
        return 1
    end

    echo "$nr $nc"
    return 0
end

set -l cases
if test (count $argv) -gt 0
    for spec in $argv
        set -l parsed (parse_case "$spec")
        if test $status -ne 0
            echo "Invalid case: '$spec'"
            usage
            exit 1
        end
        set cases $cases "$parsed"
    end
else
    # Defaults: light-to-heavier sweep.
    set cases \
        "20 500" \
        "50 2000" \
        "100 5000" \
        "200 20000"
end

# Pick an executable if already built.
set -l bin
if test -x ./gemv
    set bin ./gemv
else if test -x ./gemv_gpp
    set bin ./gemv_gpp
end

# Build if needed.
if test -z "$bin"
    echo "No benchmark binary found. Trying to build..."
    if test -f ./Makefile
        make
        and set bin ./gemv
    end

    if test -z "$bin" -o not -x "$bin"
        echo "Falling back to g++ build..."
        c++ -O3 -march=native -std=c++17 main.cpp gemv.cpp -o gemv_gpp
        or begin
            echo "Build failed."
            exit 1
        end
        set bin ./gemv_gpp
    end
end

echo "Using binary: $bin"

echo ""
echo "Running cases..."

for case in $cases
    set -l parts (string split ' ' -- "$case")
    set -l nr $parts[1]
    set -l nc $parts[2]

    echo "------------------------------------------------------------"
    echo "Case nrows=$nr ncols=$nc"
    "$bin" "$nr" "$nc"
    or begin
        echo "Run failed for nrows=$nr ncols=$nc"
        exit 1
    end
end

echo "------------------------------------------------------------"
echo "Sweep complete. CSV files were written as timing_<nrows>_<ncols>.csv"
