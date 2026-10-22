#ifndef GEMV_HPP
#define GEMV_HPP

void GEMV_opt(const float* __restrict A, const float* __restrict x, double* __restrict y, int m, int n);
void GEMV_opt_2(const float* __restrict A, const float* __restrict x, double* __restrict y, int m, int n);
void GEMV_opt_3(const float* __restrict A, const float* __restrict x, double* __restrict y, int m, int n);
void GEMV_opt_arm(const float* __restrict A, const float* __restrict x, double* __restrict y, int m, int n);
void GEMV_opt_arm_sve_vla(const float* __restrict A, const float* __restrict x, double* __restrict y, int m, int n);
void GEMV_opt_arm_sve_128(const float* __restrict A, const float* __restrict x, double* __restrict y, int m, int n);
bool GEMV_arm_sve_available();
void GEMV_block(const float* __restrict A, const float* __restrict x, double* __restrict y, int m, int n);
void GEMV_block_bs(const float* __restrict A, const float* __restrict x, double* __restrict y, int m, int n, int block_n);
void GEMV_basic(const float* __restrict A, const float* __restrict x, double* __restrict y, int m, int n);

void guided_GEMV_basic(const int* __restrict listIdx, const float* __restrict A, const float* __restrict x,
					  double* __restrict y, int m, int ncols, int nIdx);
void guided_GEMV_opt_2(const int* __restrict listIdx, const float* __restrict A, const float* __restrict x,
					  double* __restrict y, int m, int ncols, int nIdx);
void guided_GEMV_opt_3(const int* __restrict listIdx, const float* __restrict A, const float* __restrict x,
					  double* __restrict y, int m, int ncols, int nIdx);
void guided_GEMV_opt_4(const int* __restrict listIdx, const float* __restrict A, const float* __restrict x,
					  double* __restrict y, int m, int ncols, int nIdx);

#endif //! GEMV_HPP