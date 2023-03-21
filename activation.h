#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include "tensor.h"
#include "matOper.h"
#include "matCal.h"

using namespace std;

template <typename T>
__global__ void gpu_relu_forward(T *value, bool *mask, uint32_t sum_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < sum_size && value[i] <= 0)
    {
        value[i] = 0;
        mask[i] = true;
    }
    else if(i<sum_size)
    {
        mask[i] = false;
    }
}

template <typename T>
__global__ void gpu_relu_backward(T *value, bool *mask, uint32_t sum_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < sum_size && mask[i] == true)
    {
        value[i] = 0;
    }
}

template <typename T>
class ReLU
{
private:
    Tensor<bool> *mask = nullptr;
    void cpu_relu_forward(T *value, bool *mask, uint32_t sum_size)
    {
        for(int i=0; i<sum_size; i++)
        {
            if(value[i] <= 0)
            {
                value[i] = 0;
                mask[i] = true;
            }
            else
            {
                mask[i] = false;
            }
        }
    }

    void cpu_relu_backward(T *value, bool *mask, uint32_t sum_size)
    {
        for(int i=0; i<sum_size; i++)
        {
            if(mask[i] == true)
            {
                value[i] = 0;
            }
        }
    }

public:
    void forward(Tensor<T> *X)
    {
        if(this->mask==nullptr)
        {
            this->mask = new Tensor<bool>(X->tensor_shape, X->is_cuda);
        }
        if(X->is_cuda)
        {
            dim3 block(1024);
            dim3 grid((X->sum_size + block.x - 1) / block.x);
            gpu_relu_forward<<<grid, block>>>(X->value, this->mask->value, X->sum_size);
        }
        else
        {
            cpu_relu_forward(X->value, this->mask->value, X->sum_size);
        }
    }

    void backward(Tensor<T> *dout)
    {
        if(this->mask->is_cuda)
        {
            dim3 block(1024);
            dim3 grid((dout->sum_size + block.x - 1) / block.x);
            gpu_relu_backward<<<grid, block>>>(dout->value, this->mask->value, dout->sum_size);
        }
        else
        {
            cpu_relu_backward(dout->value, this->mask->value, dout->sum_size);
        }
    }
};

#endif