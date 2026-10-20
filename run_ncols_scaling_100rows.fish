#!/usr/bin/env fish

# Sweep ncols for fixed nrows=100 and collect data for scaling plots.
# Usage:
#   ./run_ncols_scaling_100rows.fish
#   ./run_ncols_scaling_100rows.fish 100 200 500 1000 2000

set -l script_dir (cd (dirname (status --current-filename)); and pwd)
cd "$script_dir"

set -l fixed_nrows 100

function usage
    echo "Usage: ./run_ncols_scaling_100rows.fish [ncols ...]"
    echo ""
    echo "Examples:"
    echo "  ./run_ncols_scaling_100rows.fish"
    echo "  ./run_ncols_scaling_100rows.fish 100 200 500 1000 2000 5000 10000"
end

# Tuning knobs (can be overridden via environment variables):
#   GEMV_WORKERS=4
#   GEMV_STABLE_CLOCKS=1
#   GEMV_DISABLE_BOOST=1
set -l workers 1
if set -q GEMV_WORKERS
    set workers $GEMV_WORKERS
end

set -l stable_clocks 1
if set -q GEMV_STABLE_CLOCKS
    set stable_clocks $GEMV_STABLE_CLOCKS
end

set -l disable_boost 0
if set -q GEMV_DISABLE_BOOST
    set disable_boost $GEMV_DISABLE_BOOST
end

set -g __gemv_saved_governor ""
set -g __gemv_boost_file ""
set -g __gemv_saved_boost ""
set -g __gemv_clock_tuned 0

function _gemv_restore_clocks
    if test "$__gemv_clock_tuned" -ne 1
        return
    end

    if test -n "$__gemv_saved_governor"
        if command -sq sudo
            sudo -n sh -c 'for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo "'"$__gemv_saved_governor"'" > "$f"; done' >/dev/null 2>&1
        end
    end

    if test -n "$__gemv_boost_file" -a -n "$__gemv_saved_boost"
        if command -sq sudo
            sudo -n sh -c 'echo "'"$__gemv_saved_boost"'" > "'"$__gemv_boost_file"'"' >/dev/null 2>&1
        end
    end

    echo "Restored CPU clock settings."
end

# Always rebuild first so we run the current source, not a stale binary.
set -l bin
if test -f ./Makefile
    echo "Building with Makefile..."
    if make
        set bin ./gemv
    end
end

if test -z "$bin"
    or not test -x "$bin"
    echo "Falling back to g++ build..."
    c++ -O3 -march=native -std=c++17 main.cpp gemv.cpp -o gemv_gpp
    or begin
        echo "Build failed."
        exit 1
    end
    set bin ./gemv_gpp
end

# Default ncols sweep (roughly logarithmic spacing from 100 to 100000).
set -l ncols_list
if test (count $argv) -gt 0
    for nc in $argv
        if not string match -qr '^[0-9]+$' -- "$nc"
            echo "Invalid ncols value: '$nc'"
            usage
            exit 1
        end
        if test "$nc" -le 0
            echo "Invalid ncols value (must be > 0): '$nc'"
            exit 1
        end
        set ncols_list $ncols_list "$nc"
    end
else
    set ncols_list \
    100 500 1000 5000 10000 \
    15000 20000 30000 40000 50000 65000 80000 100000
end

set -l summary_csv "scalability_nrows_100.csv"
echo "nrows,ncols,kernel,avg_us,speedup_vs_basic" > "$summary_csv"
set -l run_dir "run_logs_nrows_"$fixed_nrows
mkdir -p "$run_dir"

if test "$stable_clocks" -eq 1
    echo "Attempting stable clock mode (performance governor"(test "$disable_boost" -eq 1; and echo ", boost off"; or echo "")")..."

    if test -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
        set -g __gemv_saved_governor (string trim (cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor))
    end

    if test -e /sys/devices/system/cpu/cpufreq/boost
        set -g __gemv_boost_file /sys/devices/system/cpu/cpufreq/boost
    else if test -e /sys/devices/system/cpu/amd_pstate/cpb
        set -g __gemv_boost_file /sys/devices/system/cpu/amd_pstate/cpb
    else if test -e /sys/devices/system/cpu/intel_pstate/no_turbo
        set -g __gemv_boost_file /sys/devices/system/cpu/intel_pstate/no_turbo
    end

    if test -n "$__gemv_boost_file" -a -r "$__gemv_boost_file"
        set -g __gemv_saved_boost (string trim (cat "$__gemv_boost_file"))
    end

    if command -sq sudo
        sudo -n sh -c 'for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$f"; done' >/dev/null 2>&1
        if test $status -eq 0
            set -g __gemv_clock_tuned 1
        else
            echo "Warning: could not set governor to performance (sudo -n failed)."
            echo "         Run: sudo -v  (then rerun script)"
        end

        if test "$__gemv_clock_tuned" -eq 1 -a "$disable_boost" -eq 1 -a -n "$__gemv_boost_file"
            if string match -q "*no_turbo" "$__gemv_boost_file"
                # intel_pstate/no_turbo: 1 disables turbo
                sudo -n sh -c 'echo 1 > "'"$__gemv_boost_file"'"' >/dev/null 2>&1
            else
                # cpufreq/boost and amd_pstate/cpb: 0 disables boost
                sudo -n sh -c 'echo 0 > "'"$__gemv_boost_file"'"' >/dev/null 2>&1
            end
        end
    else
        echo "Warning: sudo not found, cannot enforce stable clocks."
    end
end

echo "Using binary: $bin"
echo "Fixed nrows: $fixed_nrows"
echo "ncols cases: $ncols_list"
echo "Parallel workers: $workers (dynamic scheduling)"
echo ""

for nc in $ncols_list
    echo "Queueing case nrows=$fixed_nrows ncols=$nc"
end

for nc in $ncols_list
    echo $nc
end | xargs -P $workers -I{} fish -c '
    set nc {}
    set nr '"$fixed_nrows"'
    set bin "'"$bin"'"
    set run_dir "'"$run_dir"'"
    set log_file "$run_dir/run_"$nr"_"$nc".log"
    set status_file "$run_dir/run_"$nr"_"$nc".status"

    "$bin" "$nr" "$nc" > "$log_file" 2>&1
    echo $status > "$status_file"
'

set -l failures 0
for nc in $ncols_list
    set -l status_file "$run_dir/run_"$fixed_nrows"_"$nc".status"
    set -l log_file "$run_dir/run_"$fixed_nrows"_"$nc".log"

    if not test -f "$status_file"
        echo "Run failed for nrows=$fixed_nrows ncols=$nc (missing status file)"
        set failures (math "$failures + 1")
        continue
    end

    set -l run_status (string trim (cat "$status_file"))
    if test "$run_status" -ne 0
        echo "Run failed for nrows=$fixed_nrows ncols=$nc (status=$run_status)"
        echo "See log: $log_file"
        set failures (math "$failures + 1")
    else
        echo "Completed case nrows=$fixed_nrows ncols=$nc"
    end
end

if test $failures -gt 0
    _gemv_restore_clocks
    echo "Aborting: $failures case(s) failed."
    exit 1
end

echo ""
echo "Aggregating per-case CSV files into $summary_csv ..."
for nc in $ncols_list
    set -l timing_csv "timing_"$fixed_nrows"_"$nc".csv"
    if test -f "$timing_csv"
        # Pull kernel-level and block-sweep averages into one summary table for plotting.
        # Sweep rows are expanded into distinct series names, e.g. GEMV_block_bs_BN512.
        awk -F, -v nr="$fixed_nrows" -v nc="$nc" \
            'NR>1 {
                if ($1=="kernels") {
                    printf "%s,%s,%s,%s,%s\n", nr, nc, $2, $4, $5;
                } else if ($1=="sweep") {
                    printf "%s,%s,%s_BN%s,%s,%s\n", nr, nc, $2, $3, $4, $5;
                }
            }' \
            "$timing_csv" >> "$summary_csv"
    else
        echo "Warning: expected timing file not found: $timing_csv"
    end
end

echo "------------------------------------------------------------"
echo "Sweep complete."
echo "Per-case files: timing_<nrows>_<ncols>.csv"
echo "Scalability summary: $summary_csv"

_gemv_restore_clocks

set -l plot_script "plot_avg_time_vs_ncols.py"
if test -f "$plot_script"
    echo "Generating plots from $summary_csv ..."
    python3 "$plot_script" \
        --input "$summary_csv" \
        --output "avg_time_vs_ncols_loglog.png" \
        --speedup-output "speedup_vs_ncols_semilogx.png"

    if test $status -eq 0
        echo "Generated plots: avg_time_vs_ncols_loglog.png and speedup_vs_ncols_semilogx.png"
    else
        echo "Warning: plotting step failed."
    end
else
    echo "Warning: plotting script not found: $plot_script"
end
