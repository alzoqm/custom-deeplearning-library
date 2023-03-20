#ifndef _MATCAL_H_
#define _MATCAL_H_

#include "tensor.h"

using namespace std;

template <typename T>
__global__ void gpu_max(T *a, T *b, int upper_sum_shape, int lower_sum_shape, uint16_t t_shape)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < upper_sum_shape && j < lower_sum_shape)
    {
        T max_value = a[j+i*lower_sum_shape*t_shape];
        for(int k=1; k<t_shape; k++)
        {
            if(a[k*lower_sum_shape + j + i*lower_sum_shape*t_shape] > max_value)
            {
                max_value = a[k*lower_sum_shape + j + i*lower_sum_shape*t_shape];
            }
        }
        b[i * lower_sum_shape + j] = max_value;
    }
}

template <typename T>
void cpu_max(T *a, T *b, int upper_sum_shape, int lower_sum_shape, uint16_t t_shape)
{
    for(int i=0; i<upper_sum_shape; i++)
    {
        
        for(int j=0; j<lower_sum_shape; j++)
        {
            
            T max_value = a[j+i*lower_sum_shape*t_shape]; // initialize max_value with the first element in the current slice
            
            for(int k = 1; k<t_shape; k++)
            {
                if(a[k*lower_sum_shape + j + i*lower_sum_shape*t_shape] > max_value)
                {
                    max_value = a[k*lower_sum_shape + j + i*lower_sum_shape*t_shape];
                }
            }
            b[i * lower_sum_shape + j] = max_value; // store the max value for the current slice in the output array
        }
    }
}

template <typename T>
void max(Tensor<T> &input, Tensor<T> &output, int dim=300, bool keepdim=false)
{
    if(dim < 0)
    {
        dim = input.dim + dim;
    }
    if(input.is_cuda != output.is_cuda)
    {
        throw std::runtime_error("input and output is not same device\n");
    }
    if(input.dim != output.dim)
    {
        throw std::runtime_error("input and output is not same dim\n");
    }
    if(output.tensor_shape[dim] != 1) // Output target dim must be 1
    {
        throw std::runtime_error("out.tensor_shape[dim] must be 1\n");
    }
    for(int i=0; i<input.dim; i++)
    {
        if(i==dim)
        {
            continue;
        }
        if(input.tensor_shape[i] != output.tensor_shape[i])
        {
            throw std::runtime_error("input and output is not same shape\n");
        }
    }

    int upper_sum_shape = 1;
    int lower_sum_shape = 1;
    //printf("%d\n", dim);
    for(int i=0; i<dim; i++)
    {
        upper_sum_shape *= input.tensor_shape[i];
    }
    for(int i=input.dim-1; i>dim; i--)
    {
        lower_sum_shape *= input.tensor_shape[i];
    }

    if(input.is_cuda==true)
    {
        dim3 block(16, 16);
        dim3 grid((upper_sum_shape + block.x - 1) / block.x, (lower_sum_shape + block.y - 1) / block.y);
        gpu_max<T><<<grid, block>>>(input.value, output.value, upper_sum_shape, lower_sum_shape, input.tensor_shape[dim]);
    }
    else
    {
        cpu_max(input.value, output.value, upper_sum_shape, lower_sum_shape, input.tensor_shape[dim]);
    }

    if(keepdim==false)
    {
        output.squeeze(dim);
    }
}

template <typename T>
__global__ void gpu_min(T *a, T *b, int upper_sum_shape, int lower_sum_shape, uint16_t t_shape)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < upper_sum_shape && j < lower_sum_shape)
    {
        T min_value = a[j+i*lower_sum_shape*t_shape];
        for(int k=1; k<t_shape; k++)
        {
            if(a[k*lower_sum_shape + j + i*lower_sum_shape*t_shape] < min_value)
            {
                min_value = a[k*lower_sum_shape + j + i*lower_sum_shape*t_shape];
            }
        }
        b[i * lower_sum_shape + j] = min_value;
    }
}

template <typename T>
void cpu_min(T *a, T *b, int upper_sum_shape, int lower_sum_shape, uint16_t t_shape)
{
    for(int i=0; i<upper_sum_shape; i++)
    {
        
        for(int j=0; j<lower_sum_shape; j++)
        {
            
            T min_value = a[j+i*lower_sum_shape*t_shape]; // initialize max_value with the first element in the current slice
            
            for(int k = 1; k<t_shape; k++)
            {
                if(a[k*lower_sum_shape + j + i*lower_sum_shape*t_shape] < min_value)
                {
                    min_value = a[k*lower_sum_shape + j + i*lower_sum_shape*t_shape];
                }
            }
            b[i * lower_sum_shape + j] = min_value; // store the max value for the current slice in the output array
        }
    }
}

template <typename T>
void min(Tensor<T> &input, Tensor<T> &output, int dim=300, bool keepdim=false)
{
    if(dim < 0)
    {
        dim = input.dim + dim;
    }
    if(input.is_cuda != output.is_cuda)
    {
        throw std::runtime_error("input and output is not same device\n");
    }
    if(input.dim != output.dim)
    {
        throw std::runtime_error("input and output is not same dim\n");
    }
    if(output.tensor_shape[dim] != 1) // Output target dim must be 1
    {
        throw std::runtime_error("out.tensor_shape[dim] must be 1\n");
    }
    for(int i=0; i<input.dim; i++)
    {
        if(i==dim)
        {
            continue;
        }
        if(input.tensor_shape[i] != output.tensor_shape[i])
        {
            throw std::runtime_error("input and output is not same shape\n");
        }
    }

    int upper_sum_shape = 1;
    int lower_sum_shape = 1;
    //printf("%d\n", dim);
    for(int i=0; i<dim; i++)
    {
        upper_sum_shape *= input.tensor_shape[i];
    }
    for(int i=input.dim-1; i>dim; i--)
    {
        lower_sum_shape *= input.tensor_shape[i];
    }

    if(input.is_cuda==true)
    {
        dim3 block(16, 16);
        dim3 grid((upper_sum_shape + block.x - 1) / block.x, (lower_sum_shape + block.y - 1) / block.y);
        gpu_min<T><<<grid, block>>>(input.value, output.value, upper_sum_shape, lower_sum_shape, input.tensor_shape[dim]);
    }
    else
    {
        cpu_min(input.value, output.value, upper_sum_shape, lower_sum_shape, input.tensor_shape[dim]);
    }

    if(keepdim==false)
    {
        output.squeeze(dim);
    }
}


template <typename T>
__global__ void gpu_sum(T *a, T *b, int upper_sum_shape, int lower_sum_shape, uint16_t t_shape)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < upper_sum_shape && j < lower_sum_shape)
    {
        T sum_value = a[j+i*lower_sum_shape*t_shape];
        for(int k=1; k<t_shape; k++)
        {
            sum_value += a[k*lower_sum_shape + j + i*lower_sum_shape*t_shape];
        }
        b[i * lower_sum_shape + j] = sum_value;
    }
}

template <typename T>
void cpu_sum(T *a, T *b, int upper_sum_shape, int lower_sum_shape, uint16_t t_shape)
{
    for(int i=0; i<upper_sum_shape; i++)
    {
        
        for(int j=0; j<lower_sum_shape; j++)
        {
            
            T sum_value = a[j+i*lower_sum_shape*t_shape]; // initialize max_value with the first element in the current slice
            for(int k=1; k<t_shape; k++)
            {
                sum_value += a[k*lower_sum_shape + j + i*lower_sum_shape*t_shape];
            }
            b[i * lower_sum_shape + j] = sum_value; // store the max value for the current slice in the output array
        }
    }
}

template <typename T>
void sum(Tensor<T> &input, Tensor<T> &output, int dim=300, bool keepdim=false)
{
    if(dim < 0)
    {
        dim = input.dim + dim;
    }
    if(input.is_cuda != output.is_cuda)
    {
        throw std::runtime_error("input and output is not same device\n");
    }
    if(input.dim != output.dim)
    {
        throw std::runtime_error("input and output is not same dim\n");
    }
    if(output.tensor_shape[dim] != 1) // Output target dim must be 1
    {
        throw std::runtime_error("out.tensor_shape[dim] must be 1\n");
    }
    for(int i=0; i<input.dim; i++)
    {
        if(i==dim)
        {
            continue;
        }
        if(input.tensor_shape[i] != output.tensor_shape[i])
        {
            throw std::runtime_error("input and output is not same shape\n");
        }
    }

    int upper_sum_shape = 1;
    int lower_sum_shape = 1;
    //printf("%d\n", dim);
    for(int i=0; i<dim; i++)
    {
        upper_sum_shape *= input.tensor_shape[i];
    }
    for(int i=input.dim-1; i>dim; i--)
    {
        lower_sum_shape *= input.tensor_shape[i];
    }

    if(input.is_cuda==true)
    {
        dim3 block(16, 16);
        dim3 grid((upper_sum_shape + block.x - 1) / block.x, (lower_sum_shape + block.y - 1) / block.y);
        gpu_sum<T><<<grid, block>>>(input.value, output.value, upper_sum_shape, lower_sum_shape, input.tensor_shape[dim]);
    }
    else
    {
        cpu_sum(input.value, output.value, upper_sum_shape, lower_sum_shape, input.tensor_shape[dim]);
    }

    if(keepdim==false)
    {
        output.squeeze(dim);
    }
}

template <typename T>
Tensor<T> *sum(Tensor<T> &input, int dim=300, bool keepdim=false)
{
    if(dim < 0)
    {
        dim = input.dim + dim;
    }
    vector<uint16_t> output_shape(input.dim);
    for(int i=0; i<input.dim; i++)
    {
        if(i==dim)
        {
            output_shape[i] = 1;
        }
        else
        {
            output_shape[i] = input.tensor_shape[i];
        }
    }
    Tensor<T> *output = new Tensor<T>(output_shape, input.is_cuda);
    int upper_sum_shape = 1;
    int lower_sum_shape = 1;
    for(int i=0; i<dim; i++)
    {
        upper_sum_shape *= input.tensor_shape[i];
    }
    for(int i=input.dim-1; i>dim; i--)
    {
        lower_sum_shape *= input.tensor_shape[i];
    }

    if(input.is_cuda==true)
    {   
        dim3 block(16, 16);
        dim3 grid((upper_sum_shape + block.x - 1) / block.x, (lower_sum_shape + block.y - 1) / block.y);
        gpu_sum<T><<<grid, block>>>(input.value, output->value, upper_sum_shape, lower_sum_shape, input.tensor_shape[dim]);
    }
    else
    {
        cpu_sum(input.value, output->value, upper_sum_shape, lower_sum_shape, input.tensor_shape[dim]);
    }
    if(keepdim==false)
    {
        output->squeeze(dim);
    }
    return output;
}

template <typename T>
__global__ void gpu_mean(T *a, T *b, int upper_sum_shape, int lower_sum_shape, uint16_t t_shape)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < upper_sum_shape && j < lower_sum_shape)
    {
        T sum_value = a[j+i*lower_sum_shape*t_shape];
        for(int k=1; k<t_shape; k++)
        {
            sum_value += a[k*lower_sum_shape + j + i*lower_sum_shape*t_shape];
        }
        b[i * lower_sum_shape + j] = sum_value / t_shape;
    }
}

template <typename T>
void cpu_mean(T *a, T *b, int upper_sum_shape, int lower_sum_shape, uint16_t t_shape)
{
    for(int i=0; i<upper_sum_shape; i++)
    {
        
        for(int j=0; j<lower_sum_shape; j++)
        {
            
            T sum_value = a[j+i*lower_sum_shape*t_shape]; // initialize max_value with the first element in the current slice
            for(int k=1; k<t_shape; k++)
            {
                sum_value += a[k*lower_sum_shape + j + i*lower_sum_shape*t_shape];
            }
            b[i * lower_sum_shape + j] = sum_value / t_shape; // store the max value for the current slice in the output array
        }
    }
}

template <typename T>
void mean(Tensor<T> &input, Tensor<T> &output, int dim=300, bool keepdim=false)
{
    if(dim < 0)
    {
        dim = input.dim + dim;
    }
    if(input.is_cuda != output.is_cuda)
    {
        throw std::runtime_error("input and output is not same device\n");
    }
    if(input.dim != output.dim)
    {
        throw std::runtime_error("input and output is not same dim\n");
    }
    if(output.tensor_shape[dim] != 1) // Output target dim must be 1
    {
        throw std::runtime_error("out.tensor_shape[dim] must be 1\n");
    }
    for(int i=0; i<input.dim; i++)
    {
        if(i==dim)
        {
            continue;
        }
        if(input.tensor_shape[i] != output.tensor_shape[i])
        {
            throw std::runtime_error("input and output is not same shape\n");
        }
    }

    int upper_sum_shape = 1;
    int lower_sum_shape = 1;
    //printf("%d\n", dim);
    for(int i=0; i<dim; i++)
    {
        upper_sum_shape *= input.tensor_shape[i];
    }
    for(int i=input.dim-1; i>dim; i--)
    {
        lower_sum_shape *= input.tensor_shape[i];
    }

    if(input.is_cuda==true)
    {
        dim3 block(16, 16);
        dim3 grid((upper_sum_shape + block.x - 1) / block.x, (lower_sum_shape + block.y - 1) / block.y);
        gpu_mean<T><<<grid, block>>>(input.value, output.value, upper_sum_shape, lower_sum_shape, input.tensor_shape[dim]);
    }
    else
    {
        cpu_mean(input.value, output.value, upper_sum_shape, lower_sum_shape, input.tensor_shape[dim]);
    }

    if(keepdim==false)
    {
        output.squeeze(dim);
    }
}

#endif