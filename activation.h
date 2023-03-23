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
__global__ void gpu_sigmoid_forward(T *value, uint32_t sum_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < sum_size)
    {
        value[i] = 1 / (1 + exp(-value[i]));
    }
}

template <typename T>
__global__ void gpu_sigmoid_backward(T *dout, T *save_X, uint32_t sum_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < sum_size)
    {
        dout[i] = dout[i] * save_X[i] * (1 - save_X[i]);
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
    ~ReLU()
    {
        if(this->mask != nullptr)
        {
            delete mask;
            this->mask=nullptr;
        }
    }
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

template <typename T>
class Sigmoid
{
private:
    Tensor<T> *save_X = nullptr;
    void cpu_sigmoid_forward(T *value, uint32_t sum_size)
    {
        for(int i=0; i<sum_size; i++)
        {
            value[i] = 1 / (1 + exp(-value[i]));
        }
    }

    void cpu_sigmoid_backward(T *dout, T *save_X, uint32_t sum_size)
    {
        for(int i=0; i<sum_size; i++)
        {
            dout[i] = dout[i] * save_X[i] * (1 - save_X[i]);
        }
    }
public:
    ~Sigmoid()
    {
        if(this->save_X!=nullptr)
        {
            delete this->save_X;
            this->save_X = nullptr;
        }
    }

    void forward(Tensor<T> *X)
    {
        if(X->is_cuda)
        {
            dim3 block(1024);
            dim3 grid((X->sum_size + block.x - 1) / block.x);
            gpu_sigmoid_forward<<<grid, block>>>(X->value, X->sum_size);  
        }
        else
        {
            cpu_sigmoid_forward(X->value, X->sum_size);
        }
        if(this->save_X==nullptr)
        {
            vector<uint16_t> save_X_shape(X->dim-1);
            for(int i=1; i<X->dim; i++)
            {
                save_X_shape[i-1] = X->tensor_shape[i];
            }
            this->save_X = new Tensor<T>(save_X_shape, X->is_cuda);
        }
        this->save_X->unsqueeze(0);
        mean(*X, *save_X, 0, false);
    }

    void backward(Tensor<T> *dout)
    {
        if(dout->is_cuda)
        {
            dim3 block(1024);
            dim3 grid((dout->sum_size + block.x - 1) / block.x);
            gpu_sigmoid_backward<<<grid, block>>>(dout->value, this->save_X->value, save_X->sum_size);
        }
        else
        {
            cpu_sigmoid_backward(dout->value, this->save_X->value, save_X->sum_size);
        }
    }
};

#endif