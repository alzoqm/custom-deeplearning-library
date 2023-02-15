#ifndef _MATOPER_H_
#define _MATOPER_H_

#include "tensor.h"

using namespace std;

template <typename T>
__global__ void Matrix_Add(T *d_a, T *d_b, T *d_c, unsigned int n, unsigned int m);

template <typename T>
__global__ void Matrix_Sub(T *d_a, T *d_b, T *d_c, unsigned int n, unsigned int m);

template <typename T>
__global__ void Matrix_Div(T *d_a, T *d_b, T *d_c, unsigned int n, unsigned int m);

template <typename T>
__global__ void MatrixMul(T *d_a, T *d_b, T *d_c, unsigned int n, unsigned int m);

template <typename T>
__global__ void Matrix_Mul(T *a,T *b, T *c, unsigned int bs, unsigned int n, unsigned int m, unsigned int p);

template <typename T>
void cpu_Matrix_Add(T &a, T &b, T &c);

template <typename T>
void cpu_Matrix_Sub(T &a, T &b, T &c);

template <typename T>
void cpu_Matrix_Div(T &a, T &b, T &c);

template <typename T>
void cpu_MatrixMul(T &a, T &b, T &c);

template <typename T>
void cpu_Matrix_Mul(T *a, T *b, T *c, unsigned short bs, unsigned short n, unsigned short m, unsigned short p);

template <typename T>
Tensor<T> matadd(Tensor<T> &a, Tensor<T> &b);

template <typename T>
Tensor<T> matsub(Tensor<T> &a, Tensor<T> &b);

template <typename T>
Tensor<T> matdiv(Tensor<T> &a, Tensor<T> &b);

template <typename T>
Tensor<T> matmul(Tensor<T> &a, Tensor<T> &b);

template <typename T>
Tensor<T> mat_mul(Tensor<T> &a, Tensor<T> &b);

template <typename T>
void matadd(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c);

template <typename T>
void matsub(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c);

template <typename T>
void matdiv(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c);

template <typename T>
void matmul(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c);

template <typename T>
void mat_mul(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c);

#endif
