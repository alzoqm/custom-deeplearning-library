#ifndef _MATOPER_H_
#define _MATOPER_H_

#include "tensor.h"

using namespace std;

template <typename T>
__global__ void Matrix_Add(T *d_a, T *d_b, T *d_c, unsigned int n, unsigned int m, int max_b);

template <typename T>
__global__ void Matrix_Sub_a(T *d_a, T *d_b, T *d_c, unsigned int n, unsigned int m, int max_a);

template <typename T>
__global__ void Matrix_Sub_b(T *d_a, T *d_b, T *d_c, unsigned int n, unsigned int m, int max_b);

template <typename T>
__global__ void Matrix_Div_a(T *d_a, T *d_b, T *d_c, unsigned int n, unsigned int m, int max_a);

template <typename T>
__global__ void Matrix_Div_b(T *d_a, T *d_b, T *d_c, unsigned int n, unsigned int m, int max_b);

template <typename T>
__global__ void MatrixMul(T *d_a, T *d_b, T *d_c, unsigned int n, unsigned int m, int max_b); // normal multiplication

template <typename T>
__global__ void Matrix_Mul(T *a,T *b, T *c, unsigned int bs, unsigned int n, unsigned int m, unsigned int p);

template <typename T>
void cpu_Matrix_Add(T &a, T &b, T &c);

template <typename T>
void cpu_Matrix_Sub_a(T &a, T &b, T &c);

template <typename T>
void cpu_Matrix_Sub_b(T &a, T &b, T &c);

template <typename T>
void cpu_Matrix_Div_a(T &a, T &b, T &c);

template <typename T>
void cpu_Matrix_Div_b(T &a, T &b, T &c);

template <typename T>
void cpu_MatrixMul(T &a, T &b, T &c);

template <typename T>
void cpu_Matrix_Mul(T *a, T *b, T *c, unsigned short bs, unsigned short n, unsigned short m, unsigned short p);

template <typename T>
void matoper_check_shape1(Tensor<T> &a, Tensor<T> &b, bool &check_a_b_dim ,char &min_dim, char &max_dim, char &diff);

template <typename T>
void matoper_check_shape2(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c, bool &check_a_b_dim ,char &min_dim, char &max_dim, char &diff);

template <typename T>
Tensor<T> matadd(Tensor<T> &a, Tensor<T> &b);

template <typename T>
void matadd(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c);

template <typename T>
Tensor<T> matsub(Tensor<T> &a, Tensor<T> &b);

template <typename T>
void matsub(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c);

template <typename T>
Tensor<T> matdiv(Tensor<T> &a, Tensor<T> &b);

template <typename T>
void matdiv(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c);

template <typename T>
Tensor<T> matmul(Tensor<T> &a, Tensor<T> &b);

template <typename T>
void matmul(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c);

template <typename T>
Tensor<T> mat_mul(Tensor<T> &a, Tensor<T> &b);

template <typename T>
void mat_mul(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c);

#endif
