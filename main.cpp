#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <chrono>
#include <limits>
#include <string>
#include "gemv.hpp"

#if defined(GEMV_ENABLE_AVX512) && (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
static bool cpu_has_avx512() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq");
}
#endif

// Macro for linearized indexing of a 2D array
#define IDX(i, j, ncols) ((i) * (ncols) + (j))

int main(int argc, char* argv[]) {

    // Make sure the user provided the correct number of command-line arguments: ./gemv <nrows> <ncols>
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s <nrows> <ncols>\n", argv[0]);
        return 1;
    }

    // Parse command-line arguments for matrix dimensions
    const int nrows = std::atoi(argv[1]);
    const int ncols = std::atoi(argv[2]);

    // Memory alloc
    const int remove_count = ncols / 10;
    const int nIdx_guided = ncols - remove_count;

    float* A = (float*)calloc(nrows * ncols, sizeof(float));
    float* x = (float*)calloc(ncols, sizeof(float));
    int* listIdx = (int*)calloc(nIdx_guided, sizeof(int));
    double *y_opt = (double *)calloc(nrows, sizeof(double));
    double *y_opt_2 = (double *)calloc(nrows, sizeof(double));
    double *y_opt_3 = (double *)calloc(nrows, sizeof(double));
    double *y_opt_arm = (double *)calloc(nrows, sizeof(double));
    double *y_block = (double *)calloc(nrows, sizeof(double));
    double *y_guided_basic = (double *)calloc(nrows, sizeof(double));
    double *y_guided_opt_2 = (double *)calloc(nrows, sizeof(double));
    double *y_basic = (double *)calloc(nrows, sizeof(double));

    if (!A || !x || !listIdx || !y_opt || !y_opt_2 || !y_opt_3 || !y_opt_arm || !y_block || !y_guided_basic || !y_guided_opt_2 || !y_basic) {
        std::fprintf(stderr, "Allocation failed\n");
        free(A);
        free(x);
        free(listIdx);
        free(y_opt);
        free(y_opt_2);
        free(y_opt_3);
        free(y_opt_arm);
        free(y_block);
        free(y_guided_basic);
        free(y_guided_opt_2);
        free(y_basic);
        return 1;
    }

    std::srand((unsigned)std::time(nullptr));

    // Fill A with random values in [-10, 10].
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            const float r = (float)std::rand() / (float)RAND_MAX;
            A[IDX(i, j, ncols)] = -10.0f + 20.0f * r;
        }
    }

    // Fill x with 1..ncols.
    for (int j = 0; j < ncols; ++j) {
        x[j] = (float)(j + 1);
    }

    // Build a sequential guided list with 10% randomly removed indices.
    unsigned char* removed = (unsigned char*)calloc(ncols, sizeof(unsigned char));
    if (!removed) {
        std::fprintf(stderr, "Allocation failed\n");
        free(A);
        free(x);
        free(listIdx);
        free(y_opt);
        free(y_opt_2);
        free(y_opt_3);
        free(y_opt_arm);
        free(y_block);
        free(y_guided_basic);
        free(y_guided_opt_2);
        free(y_basic);
        return 1;
    }

    int picked = 0;
    while (picked < remove_count) {
        const int idx = std::rand() % ncols;
        if (!removed[idx]) {
            removed[idx] = 1;
            ++picked;
        }
    }

    int write_pos = 0;
    for (int j = 0; j < ncols; ++j) {
        if (!removed[j]) {
            listIdx[write_pos++] = j;
        }
    }

    if (write_pos != nIdx_guided) {
        std::fprintf(stderr, "Guided index generation failed: expected %d, got %d\n", nIdx_guided, write_pos);
        free(A);
        free(x);
        free(listIdx);
        free(y_opt);
        free(y_opt_2);
        free(y_opt_3);
        free(y_opt_arm);
        free(y_block);
        free(y_guided_basic);
        free(y_guided_opt_2);
        free(y_basic);
        return 1;
    }

    std::printf("Guided pattern: kept %d / %d indices (removed %d = %.1f%%)\n",
                nIdx_guided, ncols, remove_count, 100.0 * (double)remove_count / (double)ncols);
    std::printf("removedIdx: ");
    bool first_removed = true;
    for (int j = 0; j < ncols; ++j) {
        if (removed[j]) {
            if (!first_removed) {
                std::printf(",");
            }
            std::printf("%d", j);
            first_removed = false;
        }
    }
    if (first_removed) {
        std::printf("(none)\n");
    } else {
        std::printf("\n");
    }
    std::printf("listIdx: ");
    for (int j = 0; j < nIdx_guided; ++j) {
        std::printf("%d%s", listIdx[j], (j + 1 < nIdx_guided) ? "," : "\n");
    }
    free(removed);

    // Run benchmark:
    const int warmup_runs = 10;
    const int num_runs = 1000;

    struct BenchResult {
        double avg_s;
        double checksum;
    };

    auto bench = [&](const char* label, void (*fn)(const float*, const float*, double*, int, int), double* y) {
        // Warm-up helps avoid one-time effects (cold cache/frequency ramp-up).
        for (int run = 0; run < warmup_runs; ++run) {
            fn(A, x, y, nrows, ncols);
        }

        const auto t0 = std::chrono::steady_clock::now();
        double checksum = 0.0;
        for (int run = 0; run < num_runs; ++run) {
            fn(A, x, y, nrows, ncols);
            // Consume outputs to prevent overly aggressive dead-code elimination.
            checksum += y[run % nrows];
        }
        const auto t1 = std::chrono::steady_clock::now();

        const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
        const double avg_s = elapsed_s / num_runs;
        const double avg_us = avg_s * 1.0e6;
        std::printf("%s avg time: %.3f us/call (checksum=%.6e)\n",
                    label, avg_us, checksum);
        return BenchResult{avg_s, checksum};
    };

    auto bench_guided = [&](const char* label, void (*fn)(const int*, const float*, const float*, double*, int, int, int), double* y) {
        // Warm-up helps avoid one-time effects (cold cache/frequency ramp-up).
        for (int run = 0; run < warmup_runs; ++run) {
            fn(listIdx, A, x, y, nrows, ncols, nIdx_guided);
        }

        const auto t0 = std::chrono::steady_clock::now();
        double checksum = 0.0;
        for (int run = 0; run < num_runs; ++run) {
            fn(listIdx, A, x, y, nrows, ncols, nIdx_guided);
            // Consume outputs to prevent overly aggressive dead-code elimination.
            checksum += y[run % nrows];
        }
        const auto t1 = std::chrono::steady_clock::now();

        const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
        const double avg_s = elapsed_s / num_runs;
        const double avg_us = avg_s * 1.0e6;
        std::printf("%s avg time: %.3f us/call (checksum=%.6e)\n",
                    label, avg_us, checksum);
        return BenchResult{avg_s, checksum};
    };

    // Time both implementations.
    const BenchResult basic_res = bench("GEMV_basic", GEMV_basic, y_basic);
    const BenchResult opt_res = bench("GEMV_opt", GEMV_opt, y_opt);
    const BenchResult opt2_res = bench("GEMV_opt_2", GEMV_opt_2, y_opt_2);
    BenchResult opt3_res{0.0, 0.0};
    bool opt3_ran = false;
#if defined(GEMV_ENABLE_AVX512) && (defined(__GNUC__) || defined(__clang__))
    if (cpu_has_avx512()) {
        opt3_res = bench("GEMV_opt_3", GEMV_opt_3, y_opt_3);
        opt3_ran = true;
    } else {
        std::printf("GEMV_opt_3 skipped: CPU lacks AVX-512F/AVX-512DQ support.\n");
    }
#else
    std::printf("GEMV_opt_3 skipped: AVX-512 kernel was not enabled at compile time.\n");
#endif
    BenchResult opt_arm_res{0.0, 0.0};
    bool opt_arm_ran = false;
#if defined(__aarch64__)
    opt_arm_res = bench("GEMV_opt_arm", GEMV_opt_arm, y_opt_arm);
    opt_arm_ran = true;
#else
    std::printf("GEMV_opt_arm skipped: AArch64 NEON kernel is only enabled on ARM64.\n");
#endif
    const BenchResult block_res = bench("GEMV_block", GEMV_block, y_block);
    const BenchResult guided_basic_res = bench_guided("guided_GEMV_basic", guided_GEMV_basic, y_guided_basic);
    const BenchResult guided_opt2_res = bench_guided("guided_GEMV_opt_2", guided_GEMV_opt_2, y_guided_opt_2);

    const double avg_basic_s = basic_res.avg_s;
    const double avg_opt_s = opt_res.avg_s;
    const double avg_opt_2_s = opt2_res.avg_s;
    const double avg_opt_3_s = opt3_res.avg_s;
    const double avg_opt_arm_s = opt_arm_res.avg_s;
    const double avg_block_s = block_res.avg_s;
    const double avg_guided_basic_s = guided_basic_res.avg_s;
    const double avg_guided_opt_2_s = guided_opt2_res.avg_s;

    char csv_name[128];
    std::snprintf(csv_name, sizeof(csv_name), "timing_%d_%d.csv", nrows, ncols);
    FILE* csv = std::fopen(csv_name, "w");
    if (!csv) {
        std::fprintf(stderr, "Warning: could not open %s for writing\n", csv_name);
    } else {
        std::fprintf(csv, "section,kernel,block_n,avg_us,speedup_vs_basic,checksum,warmup_runs,runs\n");
        std::fprintf(csv, "kernels,GEMV_basic,-1,%.6f,1.000000,%.6e,%d,%d\n",
                     avg_basic_s * 1.0e6, basic_res.checksum, warmup_runs, num_runs);
        std::fprintf(csv, "kernels,GEMV_opt,-1,%.6f,%.6f,%.6e,%d,%d\n",
                     avg_opt_s * 1.0e6, avg_basic_s / avg_opt_s, opt_res.checksum, warmup_runs, num_runs);
        std::fprintf(csv, "kernels,GEMV_opt_2,-1,%.6f,%.6f,%.6e,%d,%d\n",
                     avg_opt_2_s * 1.0e6, avg_basic_s / avg_opt_2_s, opt2_res.checksum, warmup_runs, num_runs);
        if (opt3_ran) {
            std::fprintf(csv, "kernels,GEMV_opt_3,-1,%.6f,%.6f,%.6e,%d,%d\n",
                         avg_opt_3_s * 1.0e6, avg_basic_s / avg_opt_3_s, opt3_res.checksum, warmup_runs, num_runs);
        }
        if (opt_arm_ran) {
            std::fprintf(csv, "kernels,GEMV_opt_arm,-1,%.6f,%.6f,%.6e,%d,%d\n",
                         avg_opt_arm_s * 1.0e6, avg_basic_s / avg_opt_arm_s,
                         opt_arm_res.checksum, warmup_runs, num_runs);
        }
        std::fprintf(csv, "kernels,GEMV_block,-1,%.6f,%.6f,%.6e,%d,%d\n",
                     avg_block_s * 1.0e6, avg_basic_s / avg_block_s, block_res.checksum, warmup_runs, num_runs);
        std::fprintf(csv, "kernels,guided_GEMV_basic,-1,%.6f,%.6f,%.6e,%d,%d\n",
                 avg_guided_basic_s * 1.0e6, 1.000000,
                 guided_basic_res.checksum, warmup_runs, num_runs);
        std::fprintf(csv, "kernels,guided_GEMV_opt_2,-1,%.6f,%.6f,%.6e,%d,%d\n",
                 avg_guided_opt_2_s * 1.0e6, avg_guided_basic_s / avg_guided_opt_2_s,
                 guided_opt2_res.checksum, warmup_runs, num_runs);
    }

    // Compare the results
    bool correct = true;
    for (int i = 0; i < nrows; ++i) {
        if (std::fabs(y_opt[i] - y_basic[i]) > 1e-6) {
            correct = false;
            printf("Mismatch at row %d: y_opt = %.6f, y_basic = %.6f\n", i, y_opt[i], y_basic[i]);
        }
        if (std::fabs(y_opt_2[i] - y_basic[i]) > 1e-6) {
            correct = false;
            printf("Mismatch at row %d: y_opt_2 = %.6f, y_basic = %.6f\n", i, y_opt_2[i], y_basic[i]);
        }
        if (opt3_ran && std::fabs(y_opt_3[i] - y_basic[i]) > 1e-6) {
            correct = false;
            printf("Mismatch at row %d: y_opt_3 = %.6f, y_basic = %.6f\n", i, y_opt_3[i], y_basic[i]);
        }
        if (opt_arm_ran && std::fabs(y_opt_arm[i] - y_basic[i]) > 1e-6) {
            correct = false;
            printf("Mismatch at row %d: y_opt_arm = %.6f, y_basic = %.6f\n", i, y_opt_arm[i], y_basic[i]);
        }
        if (std::fabs(y_block[i] - y_basic[i]) > 1e-6) {
            correct = false;
            printf("Mismatch at row %d: y_block = %.6f, y_basic = %.6f\n", i, y_block[i], y_basic[i]);
        }
    }
    if (correct) {
        printf("Results match!\n");
    } else {
        printf("Results do not match!\n");
    }
    std::printf("Guided kernels use a reduced index set; non-guided correctness comparison is intentionally skipped.\n");

    if (avg_opt_s > 0.0) {
        std::printf("Speedup (basic/opt): %.2fx\n", avg_basic_s / avg_opt_s);
    }
    if (avg_opt_2_s > 0.0) {
        std::printf("Speedup (basic/opt_2): %.2fx\n", avg_basic_s / avg_opt_2_s);
    }
    if (avg_opt_3_s > 0.0) {
        std::printf("Speedup (basic/opt_3): %.2fx\n", avg_basic_s / avg_opt_3_s);
    }
    if (avg_opt_arm_s > 0.0) {
        std::printf("Speedup (basic/opt_arm): %.2fx\n", avg_basic_s / avg_opt_arm_s);
    }
    if (avg_block_s > 0.0) {
        std::printf("Speedup (basic/block): %.2fx\n", avg_basic_s / avg_block_s);
    }
    if (avg_guided_opt_2_s > 0.0) {
        std::printf("Speedup (guided_basic/guided_opt_2): %.2fx\n", avg_guided_basic_s / avg_guided_opt_2_s);
    }

    // Sweep block sizes for GEMV_block to find the best setting on this machine.
    const int block_sizes[] = {128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096};
    const int sweep_warmup_runs = 5;
    const int sweep_runs = num_runs;
    const double basic_per_call_s = avg_basic_s;
    double best_block_time = std::numeric_limits<double>::infinity();
    double best_block_checksum = 0.0;
    int best_block_n = block_sizes[0];

    std::printf("\nBlock-size sweep for GEMV_block_bs (warmup=%d, runs=%d)\n",
                sweep_warmup_runs, sweep_runs);

    for (int bs : block_sizes) {
        for (int run = 0; run < sweep_warmup_runs; ++run) {
            GEMV_block_bs(A, x, y_block, nrows, ncols, bs);
        }

        const auto t0 = std::chrono::steady_clock::now();
        double checksum = 0.0;
        for (int run = 0; run < sweep_runs; ++run) {
            GEMV_block_bs(A, x, y_block, nrows, ncols, bs);
            checksum += y_block[run % nrows];
        }
        const auto t1 = std::chrono::steady_clock::now();

        const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
        const double avg_s = elapsed_s / sweep_runs;
        const double avg_us = avg_s * 1.0e6;
        const double speedup_vs_basic = basic_per_call_s / avg_s;
        std::printf("  BLOCK_N=%4d: avg %.3f us/call, speedup vs basic=%.2fx (checksum=%.6e)\n",
            bs, avg_us, speedup_vs_basic, checksum);

        if (csv) {
            std::fprintf(csv, "sweep,GEMV_block_bs,%d,%.6f,%.6f,%.6e,%d,%d\n",
                         bs, avg_us, speedup_vs_basic, checksum, sweep_warmup_runs, sweep_runs);
        }

        if (avg_s < best_block_time) {
            best_block_time = avg_s;
            best_block_checksum = checksum;
            best_block_n = bs;
        }
    }

    std::printf("Best block size in sweep: BLOCK_N=%d (avg %.3f us/call over %d runs)\n",
        best_block_n, best_block_time * 1.0e6, sweep_runs);

    if (csv) {
        std::fprintf(csv, "best,GEMV_block_bs,%d,%.6f,%.6f,%.6e,%d,%d\n",
                     best_block_n, best_block_time * 1.0e6, basic_per_call_s / best_block_time,
                     best_block_checksum, sweep_warmup_runs, sweep_runs);
        std::fclose(csv);
        std::printf("Wrote timing data to %s\n", csv_name);
    }

    free(A);
    free(x);
    free(listIdx);
    free(y_opt);
    free(y_opt_2);
    free(y_opt_3);
    free(y_opt_arm);
    free(y_block);
    free(y_guided_basic);
    free(y_guided_opt_2);
    free(y_basic);
    return 0;
}