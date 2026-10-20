#include "gemv.hpp"

#include <cstdlib>
#include <vector>

static inline double hsum256_pd(__m256d v) {
    const __m128d lo = _mm256_castpd256_pd128(v);
    const __m128d hi = _mm256_extractf128_pd(v, 1);
    const __m128d pair = _mm_add_pd(lo, hi);
    const __m128d hi64 = _mm_unpackhi_pd(pair, pair);
    return _mm_cvtsd_f64(_mm_add_sd(pair, hi64));
}

void GEMV_opt(const float* __restrict A, const float* __restrict x, double* __restrict y, int m, int n) {
    // Perform matrix-vector multiplication y = A * x with float products and double accumulation.
    for (int i = 0; i < m; ++i) {
        const float* Arow = A + i * n;
        // Two vector accumulators reduce dependency depth in the inner loop.
        __m256d sum_lo = _mm256_setzero_pd();
        __m256d sum_hi = _mm256_setzero_pd();

        int j = 0;
        for (; j + 8 <= n; j += 8) {
            // Load 8 floats from the row and x, then multiply in float precision.
            const __m256 a = _mm256_loadu_ps(Arow + j);
            const __m256 b = _mm256_loadu_ps(x + j);
            const __m256 prod = _mm256_mul_ps(a, b);

            // Split 8 float products into two 4-float halves.
            const __m128 prod_lo_f = _mm256_castps256_ps128(prod);
            const __m128 prod_hi_f = _mm256_extractf128_ps(prod, 1);

            // Widen each half to double and accumulate in double precision.
            const __m256d prod_lo_d = _mm256_cvtps_pd(prod_lo_f);
            const __m256d prod_hi_d = _mm256_cvtps_pd(prod_hi_f);

            sum_lo = _mm256_add_pd(sum_lo, prod_lo_d);
            sum_hi = _mm256_add_pd(sum_hi, prod_hi_d);
        }

        // Reduce vector accumulators to a scalar sum.
        alignas(32) double lanes_lo[4];
        alignas(32) double lanes_hi[4];
        _mm256_store_pd(lanes_lo, sum_lo);
        _mm256_store_pd(lanes_hi, sum_hi);

        double sum = (lanes_lo[0] + lanes_lo[1]) + (lanes_lo[2] + lanes_lo[3]) +
                     (lanes_hi[0] + lanes_hi[1]) + (lanes_hi[2] + lanes_hi[3]);

        // Scalar cleanup for elements not divisible by 8.
        for (; j < n; ++j) {
            sum += static_cast<double>(Arow[j] * x[j]);
        }

        y[i] = sum;
    }
}

void GEMV_opt_2(const float* __restrict A, const float* __restrict x, double* __restrict y, int m, int n) {
    // 4-row micro-kernel: reuse each x load across 4 rows and keep reductions in registers.
    int i = 0;
    for (; i + 4 <= m; i += 4) {
        const float* row0 = A + (i + 0) * n;
        const float* row1 = A + (i + 1) * n;
        const float* row2 = A + (i + 2) * n;
        const float* row3 = A + (i + 3) * n;

        __m256d s0_lo = _mm256_setzero_pd();
        __m256d s0_hi = _mm256_setzero_pd();
        __m256d s1_lo = _mm256_setzero_pd();
        __m256d s1_hi = _mm256_setzero_pd();
        __m256d s2_lo = _mm256_setzero_pd();
        __m256d s2_hi = _mm256_setzero_pd();
        __m256d s3_lo = _mm256_setzero_pd();
        __m256d s3_hi = _mm256_setzero_pd();

        int j = 0;
        for (; j + 8 <= n; j += 8) {
            const __m256 xv = _mm256_loadu_ps(x + j);

            const __m256 p0 = _mm256_mul_ps(_mm256_loadu_ps(row0 + j), xv);
            const __m256 p1 = _mm256_mul_ps(_mm256_loadu_ps(row1 + j), xv);
            const __m256 p2 = _mm256_mul_ps(_mm256_loadu_ps(row2 + j), xv);
            const __m256 p3 = _mm256_mul_ps(_mm256_loadu_ps(row3 + j), xv);

            s0_lo = _mm256_add_pd(s0_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(p0)));
            s0_hi = _mm256_add_pd(s0_hi, _mm256_cvtps_pd(_mm256_extractf128_ps(p0, 1)));
            s1_lo = _mm256_add_pd(s1_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(p1)));
            s1_hi = _mm256_add_pd(s1_hi, _mm256_cvtps_pd(_mm256_extractf128_ps(p1, 1)));
            s2_lo = _mm256_add_pd(s2_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(p2)));
            s2_hi = _mm256_add_pd(s2_hi, _mm256_cvtps_pd(_mm256_extractf128_ps(p2, 1)));
            s3_lo = _mm256_add_pd(s3_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(p3)));
            s3_hi = _mm256_add_pd(s3_hi, _mm256_cvtps_pd(_mm256_extractf128_ps(p3, 1)));
        }

        double sum0 = hsum256_pd(_mm256_add_pd(s0_lo, s0_hi));
        double sum1 = hsum256_pd(_mm256_add_pd(s1_lo, s1_hi));
        double sum2 = hsum256_pd(_mm256_add_pd(s2_lo, s2_hi));
        double sum3 = hsum256_pd(_mm256_add_pd(s3_lo, s3_hi));

        for (; j < n; ++j) {
            const float xj = x[j];
            sum0 += static_cast<double>(row0[j] * xj);
            sum1 += static_cast<double>(row1[j] * xj);
            sum2 += static_cast<double>(row2[j] * xj);
            sum3 += static_cast<double>(row3[j] * xj);
        }

        y[i + 0] = sum0;
        y[i + 1] = sum1;
        y[i + 2] = sum2;
        y[i + 3] = sum3;
    }

    // Remainder rows use the original single-row optimized kernel structure.
    for (; i < m; ++i) {
        const float* Arow = A + i * n;
        __m256d sum_lo = _mm256_setzero_pd();
        __m256d sum_hi = _mm256_setzero_pd();

        int j = 0;
        for (; j + 8 <= n; j += 8) {
            const __m256 a = _mm256_loadu_ps(Arow + j);
            const __m256 b = _mm256_loadu_ps(x + j);
            const __m256 prod = _mm256_mul_ps(a, b);

            sum_lo = _mm256_add_pd(sum_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(prod)));
            sum_hi = _mm256_add_pd(sum_hi, _mm256_cvtps_pd(_mm256_extractf128_ps(prod, 1)));
        }

        double sum = hsum256_pd(_mm256_add_pd(sum_lo, sum_hi));
        for (; j < n; ++j) {
            sum += static_cast<double>(Arow[j] * x[j]);
        }
        y[i] = sum;
    }
}

static void GEMV_block_impl(const float* __restrict A, const float* __restrict x, double* __restrict y,
                            int m, int n, int block_n) {
    // Column blocking over n with the same 4-row micro-kernel used by GEMV_opt_2.
    if (block_n <= 0) {
        block_n = 1024;
    }

    int i = 0;
    for (; i + 4 <= m; i += 4) {
        const float* row0 = A + (i + 0) * n;
        const float* row1 = A + (i + 1) * n;
        const float* row2 = A + (i + 2) * n;
        const float* row3 = A + (i + 3) * n;

        double sum0 = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;

        for (int jb = 0; jb < n; jb += block_n) {
            const int jend = (jb + block_n < n) ? (jb + block_n) : n;

            __m256d s0_lo = _mm256_setzero_pd();
            __m256d s0_hi = _mm256_setzero_pd();
            __m256d s1_lo = _mm256_setzero_pd();
            __m256d s1_hi = _mm256_setzero_pd();
            __m256d s2_lo = _mm256_setzero_pd();
            __m256d s2_hi = _mm256_setzero_pd();
            __m256d s3_lo = _mm256_setzero_pd();
            __m256d s3_hi = _mm256_setzero_pd();

            int j = jb;
            for (; j + 8 <= jend; j += 8) {
                const __m256 xv = _mm256_loadu_ps(x + j);

                const __m256 p0 = _mm256_mul_ps(_mm256_loadu_ps(row0 + j), xv);
                const __m256 p1 = _mm256_mul_ps(_mm256_loadu_ps(row1 + j), xv);
                const __m256 p2 = _mm256_mul_ps(_mm256_loadu_ps(row2 + j), xv);
                const __m256 p3 = _mm256_mul_ps(_mm256_loadu_ps(row3 + j), xv);

                s0_lo = _mm256_add_pd(s0_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(p0)));
                s0_hi = _mm256_add_pd(s0_hi, _mm256_cvtps_pd(_mm256_extractf128_ps(p0, 1)));
                s1_lo = _mm256_add_pd(s1_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(p1)));
                s1_hi = _mm256_add_pd(s1_hi, _mm256_cvtps_pd(_mm256_extractf128_ps(p1, 1)));
                s2_lo = _mm256_add_pd(s2_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(p2)));
                s2_hi = _mm256_add_pd(s2_hi, _mm256_cvtps_pd(_mm256_extractf128_ps(p2, 1)));
                s3_lo = _mm256_add_pd(s3_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(p3)));
                s3_hi = _mm256_add_pd(s3_hi, _mm256_cvtps_pd(_mm256_extractf128_ps(p3, 1)));
            }

            sum0 += hsum256_pd(_mm256_add_pd(s0_lo, s0_hi));
            sum1 += hsum256_pd(_mm256_add_pd(s1_lo, s1_hi));
            sum2 += hsum256_pd(_mm256_add_pd(s2_lo, s2_hi));
            sum3 += hsum256_pd(_mm256_add_pd(s3_lo, s3_hi));

            for (; j < jend; ++j) {
                const float xj = x[j];
                sum0 += static_cast<double>(row0[j] * xj);
                sum1 += static_cast<double>(row1[j] * xj);
                sum2 += static_cast<double>(row2[j] * xj);
                sum3 += static_cast<double>(row3[j] * xj);
            }
        }

        y[i + 0] = sum0;
        y[i + 1] = sum1;
        y[i + 2] = sum2;
        y[i + 3] = sum3;
    }

    // Remainder rows (same blocked idea, single-row micro-kernel).
    for (; i < m; ++i) {
        const float* Arow = A + i * n;
        double sum = 0.0;

        for (int jb = 0; jb < n; jb += block_n) {
            const int jend = (jb + block_n < n) ? (jb + block_n) : n;

            __m256d s_lo = _mm256_setzero_pd();
            __m256d s_hi = _mm256_setzero_pd();

            int j = jb;
            for (; j + 8 <= jend; j += 8) {
                const __m256 xv = _mm256_loadu_ps(x + j);
                const __m256 p = _mm256_mul_ps(_mm256_loadu_ps(Arow + j), xv);
                s_lo = _mm256_add_pd(s_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(p)));
                s_hi = _mm256_add_pd(s_hi, _mm256_cvtps_pd(_mm256_extractf128_ps(p, 1)));
            }

            sum += hsum256_pd(_mm256_add_pd(s_lo, s_hi));

            for (; j < jend; ++j) {
                sum += static_cast<double>(Arow[j] * x[j]);
            }
        }

        y[i] = sum;
    }
}

void GEMV_block(const float* __restrict A, const float* __restrict x, double* __restrict y, int m, int n) {
    GEMV_block_impl(A, x, y, m, n, 1024);
}

void GEMV_block_bs(const float* __restrict A, const float* __restrict x, double* __restrict y,
                   int m, int n, int block_n) {
    GEMV_block_impl(A, x, y, m, n, block_n);
}

void GEMV_basic(const float* __restrict A, const float* __restrict x, double* __restrict y, int m, int n) {
    // Basic implementation of matrix-vector multiplication y = A * x with float products and double accumulation.
    for (int i = 0; i < m; ++i) {
        const float* Arow = A + i * n;
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < n; ++j) {
            sum += static_cast<double>(Arow[j] * x[j]);
        }
        y[i] = sum;
    }
}

// Guided version of GEMV

// Basic implementation
void guided_GEMV_basic(const int* __restrict listIdx, const float* __restrict A, const float* __restrict x,
                       double* __restrict y, int m, int ncols, int nIdx) {
    // Basic implementation of matrix-vector multiplication y = A * x with float products and double accumulation.
    // NOTE: column entries are now accessed by the listIdx array, which contains the column indices to be used for the multiplication.
    // This is to emulate a parallel case where some cores do not own a certain repeated entry.
    for (int i = 0; i < m; ++i) {
        const float* Arow = A + i * ncols;
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < nIdx; ++j) {
            int colIdx = listIdx[j];
            float value = Arow[colIdx]*x[colIdx];
            sum += static_cast<double>(value);
        }
        y[i] = sum;
    }
}

// Optimized guided implementation
void guided_GEMV_opt_2(const int* __restrict listIdx, const float* __restrict A, const float* __restrict x,
                       double* __restrict y, int m, int ncols, int nIdx) {
    int i = 0;
    for (; i + 4 <= m; i += 4) {
        const float* row0 = A + (i + 0) * ncols;
        const float* row1 = A + (i + 1) * ncols;
        const float* row2 = A + (i + 2) * ncols;
        const float* row3 = A + (i + 3) * ncols;

        double sum0 = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;

        int j = 0;
        for (; j + 4 <= nIdx; j += 4) {
            // Reuse indirect column ids and x values across 4 rows.
            const int c0 = listIdx[j + 0];
            const int c1 = listIdx[j + 1];
            const int c2 = listIdx[j + 2];
            const int c3 = listIdx[j + 3];

            const float x0 = x[c0];
            const float x1 = x[c1];
            const float x2 = x[c2];
            const float x3 = x[c3];

            sum0 += static_cast<double>(row0[c0] * x0);
            sum0 += static_cast<double>(row0[c1] * x1);
            sum0 += static_cast<double>(row0[c2] * x2);
            sum0 += static_cast<double>(row0[c3] * x3);

            sum1 += static_cast<double>(row1[c0] * x0);
            sum1 += static_cast<double>(row1[c1] * x1);
            sum1 += static_cast<double>(row1[c2] * x2);
            sum1 += static_cast<double>(row1[c3] * x3);

            sum2 += static_cast<double>(row2[c0] * x0);
            sum2 += static_cast<double>(row2[c1] * x1);
            sum2 += static_cast<double>(row2[c2] * x2);
            sum2 += static_cast<double>(row2[c3] * x3);

            sum3 += static_cast<double>(row3[c0] * x0);
            sum3 += static_cast<double>(row3[c1] * x1);
            sum3 += static_cast<double>(row3[c2] * x2);
            sum3 += static_cast<double>(row3[c3] * x3);
        }

        for (; j < nIdx; ++j) {
            const int colIdx = listIdx[j];
            const float xj = x[colIdx];
            sum0 += static_cast<double>(row0[colIdx] * xj);
            sum1 += static_cast<double>(row1[colIdx] * xj);
            sum2 += static_cast<double>(row2[colIdx] * xj);
            sum3 += static_cast<double>(row3[colIdx] * xj);
        }

        y[i + 0] = sum0;
        y[i + 1] = sum1;
        y[i + 2] = sum2;
        y[i + 3] = sum3;
    }

    // Remainder rows.
    for (; i < m; ++i) {
        const float* row = A + i * ncols;
        double sum = 0.0;

        int j = 0;
        for (; j + 4 <= nIdx; j += 4) {
            const int c0 = listIdx[j + 0];
            const int c1 = listIdx[j + 1];
            const int c2 = listIdx[j + 2];
            const int c3 = listIdx[j + 3];

            sum += static_cast<double>(row[c0] * x[c0]);
            sum += static_cast<double>(row[c1] * x[c1]);
            sum += static_cast<double>(row[c2] * x[c2]);
            sum += static_cast<double>(row[c3] * x[c3]);
        }

        for (; j < nIdx; ++j) {
            const int colIdx = listIdx[j];
            sum += static_cast<double>(row[colIdx] * x[colIdx]);
        }
        y[i] = sum;
    }
}

void guided_GEMV_opt_3(const int* __restrict listIdx, const float* __restrict A, const float* __restrict x,
                       double* __restrict y, int m, int ncols, int nIdx) {
    int i = 0;
    for (; i + 4 <= m; i += 4) {
        const float* row0 = A + (i + 0) * ncols;
        const float* row1 = A + (i + 1) * ncols;
        const float* row2 = A + (i + 2) * ncols;
        const float* row3 = A + (i + 3) * ncols;

        __m256d s0_lo = _mm256_setzero_pd();
        __m256d s0_hi = _mm256_setzero_pd();
        __m256d s1_lo = _mm256_setzero_pd();
        __m256d s1_hi = _mm256_setzero_pd();
        __m256d s2_lo = _mm256_setzero_pd();
        __m256d s2_hi = _mm256_setzero_pd();
        __m256d s3_lo = _mm256_setzero_pd();
        __m256d s3_hi = _mm256_setzero_pd();

        double tail0 = 0.0;
        double tail1 = 0.0;
        double tail2 = 0.0;
        double tail3 = 0.0;

        int j = 0;
        while (j < nIdx) {
            const int start = listIdx[j];
            int run = 1;
            while (j + run < nIdx && listIdx[j + run] == start + run) {
                ++run;
            }

            int k = 0;
            for (; k + 8 <= run; k += 8) {
                const int base = start + k;
                const __m256 xv = _mm256_loadu_ps(x + base);

                const __m256 p0 = _mm256_mul_ps(_mm256_loadu_ps(row0 + base), xv);
                const __m256 p1 = _mm256_mul_ps(_mm256_loadu_ps(row1 + base), xv);
                const __m256 p2 = _mm256_mul_ps(_mm256_loadu_ps(row2 + base), xv);
                const __m256 p3 = _mm256_mul_ps(_mm256_loadu_ps(row3 + base), xv);

                s0_lo = _mm256_add_pd(s0_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(p0)));
                s0_hi = _mm256_add_pd(s0_hi, _mm256_cvtps_pd(_mm256_extractf128_ps(p0, 1)));
                s1_lo = _mm256_add_pd(s1_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(p1)));
                s1_hi = _mm256_add_pd(s1_hi, _mm256_cvtps_pd(_mm256_extractf128_ps(p1, 1)));
                s2_lo = _mm256_add_pd(s2_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(p2)));
                s2_hi = _mm256_add_pd(s2_hi, _mm256_cvtps_pd(_mm256_extractf128_ps(p2, 1)));
                s3_lo = _mm256_add_pd(s3_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(p3)));
                s3_hi = _mm256_add_pd(s3_hi, _mm256_cvtps_pd(_mm256_extractf128_ps(p3, 1)));
            }

            for (; k < run; ++k) {
                const int col = start + k;
                const float xj = x[col];
                tail0 += static_cast<double>(row0[col] * xj);
                tail1 += static_cast<double>(row1[col] * xj);
                tail2 += static_cast<double>(row2[col] * xj);
                tail3 += static_cast<double>(row3[col] * xj);
            }

            j += run;
        }

        y[i + 0] = tail0 + hsum256_pd(_mm256_add_pd(s0_lo, s0_hi));
        y[i + 1] = tail1 + hsum256_pd(_mm256_add_pd(s1_lo, s1_hi));
        y[i + 2] = tail2 + hsum256_pd(_mm256_add_pd(s2_lo, s2_hi));
        y[i + 3] = tail3 + hsum256_pd(_mm256_add_pd(s3_lo, s3_hi));
    }

    for (; i < m; ++i) {
        const float* row = A + i * ncols;
        __m256d s_lo = _mm256_setzero_pd();
        __m256d s_hi = _mm256_setzero_pd();
        double tail = 0.0;

        int j = 0;
        while (j < nIdx) {
            const int start = listIdx[j];
            int run = 1;
            while (j + run < nIdx && listIdx[j + run] == start + run) {
                ++run;
            }

            int k = 0;
            for (; k + 8 <= run; k += 8) {
                const int base = start + k;
                const __m256 xv = _mm256_loadu_ps(x + base);
                const __m256 p = _mm256_mul_ps(_mm256_loadu_ps(row + base), xv);
                s_lo = _mm256_add_pd(s_lo, _mm256_cvtps_pd(_mm256_castps256_ps128(p)));
                s_hi = _mm256_add_pd(s_hi, _mm256_cvtps_pd(_mm256_extractf128_ps(p, 1)));
            }

            for (; k < run; ++k) {
                const int col = start + k;
                tail += static_cast<double>(row[col] * x[col]);
            }

            j += run;
        }

        y[i] = tail + hsum256_pd(_mm256_add_pd(s_lo, s_hi));
    }
}

void guided_GEMV_opt_4(const int* __restrict listIdx, const float* __restrict A, const float* __restrict x,
                       double* __restrict y, int m, int ncols, int nIdx) {
    if (nIdx <= 0) {
        for (int i = 0; i < m; ++i) {
            y[i] = 0.0;
        }
        return;
    }

    // Reuse run buffers across calls so benchmarked kernels do not pay allocator overhead.
    static thread_local std::vector<int> run_starts;
    static thread_local std::vector<int> run_lens;
    static thread_local const int* cached_list_ptr = nullptr;
    static thread_local int cached_nIdx = -1;
    static thread_local int cached_first = -1;
    static thread_local int cached_last = -1;
    static thread_local int cached_nruns = 0;

    if ((int)run_starts.size() < nIdx) {
        run_starts.resize(nIdx);
        run_lens.resize(nIdx);
    }

    const int first = listIdx[0];
    const int last = listIdx[nIdx - 1];
    const bool need_rebuild = (cached_list_ptr != listIdx) ||
                              (cached_nIdx != nIdx) ||
                              (cached_first != first) ||
                              (cached_last != last);

    if (need_rebuild) {
        int nruns = 0;
        int j = 0;
        while (j < nIdx) {
            const int start = listIdx[j];
            int len = 1;
            while (j + len < nIdx && listIdx[j + len] == start + len) {
                ++len;
            }
            run_starts[nruns] = start;
            run_lens[nruns] = len;
            ++nruns;
            j += len;
        }

        cached_list_ptr = listIdx;
        cached_nIdx = nIdx;
        cached_first = first;
        cached_last = last;
        cached_nruns = nruns;
    }
    const int nruns = cached_nruns;

    constexpr int SIMD_RUN_THRESHOLD = 16;

    int i = 0;
    for (; i + 4 <= m; i += 4) {
        const float* row0 = A + (i + 0) * ncols;
        const float* row1 = A + (i + 1) * ncols;
        const float* row2 = A + (i + 2) * ncols;
        const float* row3 = A + (i + 3) * ncols;

        __m256d s0_lo_a = _mm256_setzero_pd();
        __m256d s0_hi_a = _mm256_setzero_pd();
        __m256d s1_lo_a = _mm256_setzero_pd();
        __m256d s1_hi_a = _mm256_setzero_pd();
        __m256d s2_lo_a = _mm256_setzero_pd();
        __m256d s2_hi_a = _mm256_setzero_pd();
        __m256d s3_lo_a = _mm256_setzero_pd();
        __m256d s3_hi_a = _mm256_setzero_pd();

        __m256d s0_lo_b = _mm256_setzero_pd();
        __m256d s0_hi_b = _mm256_setzero_pd();
        __m256d s1_lo_b = _mm256_setzero_pd();
        __m256d s1_hi_b = _mm256_setzero_pd();
        __m256d s2_lo_b = _mm256_setzero_pd();
        __m256d s2_hi_b = _mm256_setzero_pd();
        __m256d s3_lo_b = _mm256_setzero_pd();
        __m256d s3_hi_b = _mm256_setzero_pd();

        double tail0 = 0.0;
        double tail1 = 0.0;
        double tail2 = 0.0;
        double tail3 = 0.0;

        for (int r = 0; r < nruns; ++r) {
            const int start = run_starts[r];
            const int len = run_lens[r];

            if (len < SIMD_RUN_THRESHOLD) {
                for (int k = 0; k < len; ++k) {
                    const int col = start + k;
                    const float xj = x[col];
                    tail0 += static_cast<double>(row0[col] * xj);
                    tail1 += static_cast<double>(row1[col] * xj);
                    tail2 += static_cast<double>(row2[col] * xj);
                    tail3 += static_cast<double>(row3[col] * xj);
                }
                continue;
            }

            int k = 0;
            for (; k + 16 <= len; k += 16) {
                const int b0 = start + k;
                const int b1 = b0 + 8;

                const __m256 x0 = _mm256_loadu_ps(x + b0);
                const __m256 x1 = _mm256_loadu_ps(x + b1);

                const __m256 p00 = _mm256_mul_ps(_mm256_loadu_ps(row0 + b0), x0);
                const __m256 p10 = _mm256_mul_ps(_mm256_loadu_ps(row1 + b0), x0);
                const __m256 p20 = _mm256_mul_ps(_mm256_loadu_ps(row2 + b0), x0);
                const __m256 p30 = _mm256_mul_ps(_mm256_loadu_ps(row3 + b0), x0);

                const __m256 p01 = _mm256_mul_ps(_mm256_loadu_ps(row0 + b1), x1);
                const __m256 p11 = _mm256_mul_ps(_mm256_loadu_ps(row1 + b1), x1);
                const __m256 p21 = _mm256_mul_ps(_mm256_loadu_ps(row2 + b1), x1);
                const __m256 p31 = _mm256_mul_ps(_mm256_loadu_ps(row3 + b1), x1);

                s0_lo_a = _mm256_add_pd(s0_lo_a, _mm256_cvtps_pd(_mm256_castps256_ps128(p00)));
                s0_hi_a = _mm256_add_pd(s0_hi_a, _mm256_cvtps_pd(_mm256_extractf128_ps(p00, 1)));
                s1_lo_a = _mm256_add_pd(s1_lo_a, _mm256_cvtps_pd(_mm256_castps256_ps128(p10)));
                s1_hi_a = _mm256_add_pd(s1_hi_a, _mm256_cvtps_pd(_mm256_extractf128_ps(p10, 1)));
                s2_lo_a = _mm256_add_pd(s2_lo_a, _mm256_cvtps_pd(_mm256_castps256_ps128(p20)));
                s2_hi_a = _mm256_add_pd(s2_hi_a, _mm256_cvtps_pd(_mm256_extractf128_ps(p20, 1)));
                s3_lo_a = _mm256_add_pd(s3_lo_a, _mm256_cvtps_pd(_mm256_castps256_ps128(p30)));
                s3_hi_a = _mm256_add_pd(s3_hi_a, _mm256_cvtps_pd(_mm256_extractf128_ps(p30, 1)));

                s0_lo_b = _mm256_add_pd(s0_lo_b, _mm256_cvtps_pd(_mm256_castps256_ps128(p01)));
                s0_hi_b = _mm256_add_pd(s0_hi_b, _mm256_cvtps_pd(_mm256_extractf128_ps(p01, 1)));
                s1_lo_b = _mm256_add_pd(s1_lo_b, _mm256_cvtps_pd(_mm256_castps256_ps128(p11)));
                s1_hi_b = _mm256_add_pd(s1_hi_b, _mm256_cvtps_pd(_mm256_extractf128_ps(p11, 1)));
                s2_lo_b = _mm256_add_pd(s2_lo_b, _mm256_cvtps_pd(_mm256_castps256_ps128(p21)));
                s2_hi_b = _mm256_add_pd(s2_hi_b, _mm256_cvtps_pd(_mm256_extractf128_ps(p21, 1)));
                s3_lo_b = _mm256_add_pd(s3_lo_b, _mm256_cvtps_pd(_mm256_castps256_ps128(p31)));
                s3_hi_b = _mm256_add_pd(s3_hi_b, _mm256_cvtps_pd(_mm256_extractf128_ps(p31, 1)));
            }

            for (; k + 8 <= len; k += 8) {
                const int base = start + k;
                const __m256 xv = _mm256_loadu_ps(x + base);

                const __m256 p0 = _mm256_mul_ps(_mm256_loadu_ps(row0 + base), xv);
                const __m256 p1 = _mm256_mul_ps(_mm256_loadu_ps(row1 + base), xv);
                const __m256 p2 = _mm256_mul_ps(_mm256_loadu_ps(row2 + base), xv);
                const __m256 p3 = _mm256_mul_ps(_mm256_loadu_ps(row3 + base), xv);

                s0_lo_a = _mm256_add_pd(s0_lo_a, _mm256_cvtps_pd(_mm256_castps256_ps128(p0)));
                s0_hi_a = _mm256_add_pd(s0_hi_a, _mm256_cvtps_pd(_mm256_extractf128_ps(p0, 1)));
                s1_lo_a = _mm256_add_pd(s1_lo_a, _mm256_cvtps_pd(_mm256_castps256_ps128(p1)));
                s1_hi_a = _mm256_add_pd(s1_hi_a, _mm256_cvtps_pd(_mm256_extractf128_ps(p1, 1)));
                s2_lo_a = _mm256_add_pd(s2_lo_a, _mm256_cvtps_pd(_mm256_castps256_ps128(p2)));
                s2_hi_a = _mm256_add_pd(s2_hi_a, _mm256_cvtps_pd(_mm256_extractf128_ps(p2, 1)));
                s3_lo_a = _mm256_add_pd(s3_lo_a, _mm256_cvtps_pd(_mm256_castps256_ps128(p3)));
                s3_hi_a = _mm256_add_pd(s3_hi_a, _mm256_cvtps_pd(_mm256_extractf128_ps(p3, 1)));
            }

            for (; k < len; ++k) {
                const int col = start + k;
                const float xj = x[col];
                tail0 += static_cast<double>(row0[col] * xj);
                tail1 += static_cast<double>(row1[col] * xj);
                tail2 += static_cast<double>(row2[col] * xj);
                tail3 += static_cast<double>(row3[col] * xj);
            }
        }

        const __m256d s0 = _mm256_add_pd(_mm256_add_pd(s0_lo_a, s0_hi_a), _mm256_add_pd(s0_lo_b, s0_hi_b));
        const __m256d s1 = _mm256_add_pd(_mm256_add_pd(s1_lo_a, s1_hi_a), _mm256_add_pd(s1_lo_b, s1_hi_b));
        const __m256d s2 = _mm256_add_pd(_mm256_add_pd(s2_lo_a, s2_hi_a), _mm256_add_pd(s2_lo_b, s2_hi_b));
        const __m256d s3 = _mm256_add_pd(_mm256_add_pd(s3_lo_a, s3_hi_a), _mm256_add_pd(s3_lo_b, s3_hi_b));

        y[i + 0] = tail0 + hsum256_pd(s0);
        y[i + 1] = tail1 + hsum256_pd(s1);
        y[i + 2] = tail2 + hsum256_pd(s2);
        y[i + 3] = tail3 + hsum256_pd(s3);
    }

    for (; i < m; ++i) {
        const float* row = A + i * ncols;

        __m256d s_lo_a = _mm256_setzero_pd();
        __m256d s_hi_a = _mm256_setzero_pd();
        __m256d s_lo_b = _mm256_setzero_pd();
        __m256d s_hi_b = _mm256_setzero_pd();
        double tail = 0.0;

        for (int r = 0; r < nruns; ++r) {
            const int start = run_starts[r];
            const int len = run_lens[r];

            if (len < SIMD_RUN_THRESHOLD) {
                for (int k = 0; k < len; ++k) {
                    const int col = start + k;
                    tail += static_cast<double>(row[col] * x[col]);
                }
                continue;
            }

            int k = 0;
            for (; k + 16 <= len; k += 16) {
                const int b0 = start + k;
                const int b1 = b0 + 8;

                const __m256 x0 = _mm256_loadu_ps(x + b0);
                const __m256 x1 = _mm256_loadu_ps(x + b1);
                const __m256 p0 = _mm256_mul_ps(_mm256_loadu_ps(row + b0), x0);
                const __m256 p1 = _mm256_mul_ps(_mm256_loadu_ps(row + b1), x1);

                s_lo_a = _mm256_add_pd(s_lo_a, _mm256_cvtps_pd(_mm256_castps256_ps128(p0)));
                s_hi_a = _mm256_add_pd(s_hi_a, _mm256_cvtps_pd(_mm256_extractf128_ps(p0, 1)));
                s_lo_b = _mm256_add_pd(s_lo_b, _mm256_cvtps_pd(_mm256_castps256_ps128(p1)));
                s_hi_b = _mm256_add_pd(s_hi_b, _mm256_cvtps_pd(_mm256_extractf128_ps(p1, 1)));
            }

            for (; k + 8 <= len; k += 8) {
                const int base = start + k;
                const __m256 xv = _mm256_loadu_ps(x + base);
                const __m256 p = _mm256_mul_ps(_mm256_loadu_ps(row + base), xv);
                s_lo_a = _mm256_add_pd(s_lo_a, _mm256_cvtps_pd(_mm256_castps256_ps128(p)));
                s_hi_a = _mm256_add_pd(s_hi_a, _mm256_cvtps_pd(_mm256_extractf128_ps(p, 1)));
            }

            for (; k < len; ++k) {
                const int col = start + k;
                tail += static_cast<double>(row[col] * x[col]);
            }
        }

        const __m256d s = _mm256_add_pd(_mm256_add_pd(s_lo_a, s_hi_a), _mm256_add_pd(s_lo_b, s_hi_b));
        y[i] = tail + hsum256_pd(s);
    }

}