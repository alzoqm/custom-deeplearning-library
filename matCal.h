#ifndef _MATCAL_H_
#define _MATCAL_H_

#include "tensor.h"

using namespace std;

template <typename T>
__global__ void gpu_max(T *a, T *b, int upper_sum_shape, int lower_sum_shape, unsigned short t_shape)

template <typename T>
void cpu_max(T *a, T *b, int upper_sum_shape, int lower_sum_shape, unsigned short t_shape);

template <typename T>
void max(Tensor<T> &input, Tensor<T> &output, int dim=300, bool keepdim=false);

template <typename T>
__global__ void gpu_min(T *a, T *b, int upper_sum_shape, int lower_sum_shape, unsigned short t_shape);

template <typename T>
void cpu_min(T *a, T *b, int upper_sum_shape, int lower_sum_shape, unsigned short t_shape);

template <typename T>
void min(Tensor<T> &input, Tensor<T> &output, int dim=300, bool keepdim=false);

template <typename T>
__global__ void gpu_sum(T *a, T *b, int upper_sum_shape, int lower_sum_shape, unsigned short t_shape);

template <typename T>
void cpu_sum(T *a, T *b, int upper_sum_shape, int lower_sum_shape, unsigned short t_shape);

template <typename T>
void sum(Tensor<T> &input, Tensor<T> &output, int dim=300, bool keepdim=false);

template <typename T>
__global__ void gpu_mean(T *a, T *b, int upper_sum_shape, int lower_sum_shape, unsigned short t_shape);

template <typename T>
void cpu_mean(T *a, T *b, int upper_sum_shape, int lower_sum_shape, unsigned short t_shape);

template <typename T>
void mean(Tensor<T> &input, Tensor<T> &output, int dim=300, bool keepdim=false);

#endif