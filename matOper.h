#ifndef _MATOPER_H_
#define _MATOPER_H_

#include "tensor.h"
#include <iostream>
#include <vector>

using namespace std;

template <typename T>
__global__ void Matrix_Add(T *d_a, T *d_b, T *d_c, uint32_t n, uint32_t m, int max_b) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < m) 
  {
    d_c[i * m + j] = d_a[i * m + j] + d_b[(i * m)%max_b + j];
  }
}

template <typename T>
__global__ void Matrix_Sub_a(T *d_a, T *d_b, T *d_c, uint32_t n, uint32_t m, int max_a) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < m) 
  {
    d_c[i * m + j] = d_a[(i * m)%max_a + j] - d_b[i * m + j];
  }
}

template <typename T>
__global__ void Matrix_Sub_b(T *d_a, T *d_b, T *d_c, uint32_t n, uint32_t m, int max_b) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < m) 
  {
    d_c[i * m + j] = d_a[i * m + j] - d_b[(i * m)%max_b + j];
  }
}

template <typename T>
__global__ void Matrix_Div_a(T *d_a, T *d_b, T *d_c, uint32_t n, uint32_t m, int max_a) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < m) 
  {
    d_c[i * m + j] = d_a[(i * m)%max_a + j] / d_b[i * m + j];
  }
}

template <typename T>
__global__ void Matrix_Div_b(T *d_a, T *d_b, T *d_c, uint32_t n, uint32_t m, int max_b) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < m) 
  {
    d_c[i * m + j] = d_a[i * m + j] / d_b[(i * m)%max_b + j];
  }
}

template <typename T>
__global__ void MatrixMul(T *d_a, T *d_b, T *d_c, uint32_t n, uint32_t m, int max_b) // normal multiplication
{ 
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < m) 
  {
    d_c[i * m + j] = d_a[i * m + j] * d_b[(i * m)%max_b + j];
  }
}

template <typename T>
__global__ void Matrix_Mul_a(T *a,T *b, T *c, uint32_t bs, uint32_t n, uint32_t m, uint32_t p, int max_a)
{ 
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.z * blockDim.z + threadIdx.z;
    
    if(l < bs && i < n && j < p ) 
    {
        T sum = 0.;
        for(int k = 0; k < m; k++) 
        {
            //sum += a[l*n*m + i * m + k] * b[l*m*p + k * p + j];
            sum += a[(l%max_a)*n*m + i * m + k] * b[l*m*p + k * p + j];
        }
        c[l*n*p + i * p + j] = sum;
    }
}

template <typename T>
__global__ void Matrix_Mul_b(T *a,T *b, T *c, uint32_t bs, uint32_t n, uint32_t m, uint32_t p, int max_b)
{ 
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.z * blockDim.z + threadIdx.z;
    
    if(l < bs && i < n && j < p ) 
    {
        T sum = 0.;
        for(int k = 0; k < m; k++) 
        {
            //sum += a[l*n*m + i * m + k] * b[l*m*p + k * p + j];
            sum += a[l*n*m + i * m + k] * b[(l%max_b)*m*p + k * p + j];
        }
        c[l*n*p + i * p + j] = sum;
    }
}

template <typename T>
__global__ void gpu_trans(T *a, T *b, uint32_t bs, uint32_t n, uint32_t m)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.z * blockDim.z + threadIdx.z;
    if(l < bs && i < n && j < m)
    {
        b[l*n*m + j*n + i] = a[l*n*m + i*m + j];
    }
}

template <typename T>
void cpu_Matrix_Add(T &a, T &b, T &c)
{
    int max_b = b.n*b.m;
    for(int i=0; i<c.n; i++)
    {
        for(int j=0; j<c.m; j++)
        {
            c.value[i*c.m + j] = a.value[i*c.m + j] + b.value[(i*c.m + j)%max_b];
        }
    }
}

template <typename T>
void cpu_Matrix_Sub_a(T &a, T &b, T &c)
{
    int max_a = a.n*a.m;
    for(int i=0; i<c.n; i++)
    {
        for(int j=0; j<c.m; j++)
        {
            c.value[i*c.m + j] = a.value[(i*c.m)%max_a + j] - b.value[i*c.m + j];
        }
    }
}

template <typename T>
void cpu_Matrix_Sub_b(T &a, T &b, T &c)
{
    int max_b = b.n*b.m;
    for(int i=0; i<c.n; i++)
    {
        for(int j=0; j<c.m; j++)
        {
            c.value[i*c.m + j] = a.value[i*c.m + j] - b.value[(i*c.m)%max_b + j];
        }
    }
}

template <typename T>
void cpu_Matrix_Div_a(T &a, T &b, T &c)
{
    int max_a = a.n*a.m;
    for(int i=0; i<c.n; i++)
    {
        for(int j=0; j<c.m; j++)
        {
            c.value[i*c.m + j] = a.value[(i*c.m)%max_a + j] / b.value[i*c.m + j];
        }
    }
}

template <typename T>
void cpu_Matrix_Div_b(T &a, T &b, T &c)
{
    int max_b = b.n*b.m;
    for(int i=0; i<c.n; i++)
    {
        for(int j=0; j<c.m; j++)
        {
            c.value[i*c.m + j] = a.value[i*c.m + j] / b.value[(i*c.m)%max_b + j];
        }
    }
}

template <typename T>
void cpu_MatrixMul(T &a, T &b, T &c)
{
    int max_b = b.n*b.m;
    for(int i=0; i<c.n; i++)
    {
        for(int j=0; j<c.m; j++)
        {
            c.value[i*c.m + j] = a.value[i*c.m + j] * b.value[(i*c.m)%max_b + j];
        }
    }
}

template <typename T>
void cpu_Matrix_Mul_a(T *a, T *b, T *c, uint16_t bs, uint16_t n, uint16_t m, uint16_t p, int max_a)
{
    for(int l=0; l<bs; l++)
    {
        for (int i = 0; i < n; i++) 
        {
            for (int j = 0; j < p; j++) 
            {
                T sum = 0.;
                for (int k = 0; k < m; k++) 
                {
                    //sum += a[l*n*m + i * m + k] * b[l*m*p + k * p + j];
                    sum += a[(l%max_a)*n*m + i * m + k] * b[l*m*p + k * p + j];
                }
                c[l*n*p + i * p + j] = sum;
            }
        }
    }
}

template <typename T>
void cpu_Matrix_Mul_b(T *a, T *b, T *c, uint16_t bs, uint16_t n, uint16_t m, uint16_t p, int max_b)
{
    for(int l=0; l<bs; l++)
    {
        //printf("%d\n\n", l%max_b*m*p );
        for (int i = 0; i < n; i++) 
        {
            for (int j = 0; j < p; j++) 
            {
                T sum = 0.;
                for (int k = 0; k < m; k++) 
                {
                    //sum += a[l*n*m + i * m + k] * b[l*m*p + k * p + j];
                    sum += a[l*n*m + i * m + k] * b[(l%max_b)*m*p + k * p + j];
                }
                c[l*n*p + i * p + j] = sum;
            }
        }
    }
}

template <typename T>
void cpu_trans(T *a, T *b, uint16_t bs, uint16_t n, uint16_t m)
{
    for(int l=0; l<bs; l++)
    {
        for(int i=0; i<n; i++)
        {
            for(int j=0; j<m; j++)
            {
                b[l*n*m + j*n + i] = a[l*n*m + i*m + j];
            }
        }
    }
}

template <typename T>
void check_shape1(Tensor<T> &a, Tensor<T> &b, bool &check_a_b_dim ,char &min_dim, char &max_dim, char &diff)
{
    if(a.dim < b.dim)
    {
        min_dim = a.dim;
        max_dim = b.dim;
        diff = max_dim-min_dim;
        check_a_b_dim = true; // a < b
    }
    else
    {
        min_dim = b.dim;
        max_dim = a.dim;
        diff = max_dim-min_dim;
    }
    if(check_a_b_dim==true)
    {
        for (int i = max_dim-1; i >= diff; i--) 
        {
            if (a.tensor_shape[i-diff] != b.tensor_shape[i]) 
            {
                throw std::invalid_argument("tensor1 and tensor2 shapes is different\n");
            }
        }
    }
    else
    {
        for (int i = max_dim-1; i >= diff; i--)
        {
            if (a.tensor_shape[i] != b.tensor_shape[i-diff]) 
            {
                throw std::invalid_argument("tensor1 and tensor2 shapes is different\n");
            }
        }
    }

    if (a.is_cuda != b.is_cuda) 
    {
        if (a.is_cuda) 
        {
            throw std::runtime_error("tensor1 is on CUDA but tensor2 is on CPU\n");
        }
        else 
        {
            throw std::runtime_error("tensor1 is on CPU but tensor2 is on CUDA\n");
        }
    }  
}

template <typename T>
void check_shape2(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c, bool &check_a_b_dim ,char &min_dim, char &max_dim, char &diff)
{
    if(a.dim < b.dim)
    {
        min_dim = a.dim;
        max_dim = b.dim;
        diff = max_dim-min_dim;
        check_a_b_dim = true; // a < b
    }
    else
    {
        min_dim = b.dim;
        max_dim = a.dim;
        diff = max_dim-min_dim;
    }
    if(check_a_b_dim==true)
    {
        if (b.dim != c.dim)
        {
            throw std::invalid_argument("input/output dimensions is different\n");
        }
        for (int i = max_dim-1; i >= diff; i--) 
        {
            if (a.tensor_shape[i-diff] != b.tensor_shape[i]) 
            {
                throw std::invalid_argument("tensor1 and tensor2 shapes is different\n");
            }
        }
        for (int i = 0; i < max_dim; i++) 
        {
            if (b.tensor_shape[i] != c.tensor_shape[i]) 
            {
                throw std::invalid_argument("input and output shapes is different\n");
            }
        }
    }
    else
    {
        if (a.dim != c.dim)
        {
            throw std::invalid_argument("input/output dimensions is different\n");
        }
        for (int i = max_dim-1; i >= diff; i--)
        {
            if (a.tensor_shape[i] != b.tensor_shape[i-diff]) 
            {
                throw std::invalid_argument("tensor1 and tensor2 shapes is different\n");
            }
        }
        for (int i = 0; i < max_dim; i++) 
        {
            if (a.tensor_shape[i] != c.tensor_shape[i]) 
            {
                throw std::invalid_argument("input and output shapes is different\n");
            }
        }
    }

    if (a.is_cuda != b.is_cuda) 
    {
        if (a.is_cuda) 
        {
            throw std::runtime_error("tensor1 is on CUDA but tensor2 is on CPU\n");
        }
        else 
        {
            throw std::runtime_error("tensor1 is on CPU but tensor2 is on CUDA\n");
        }
    }

    if (a.is_cuda != c.is_cuda) 
    {
        if (a.is_cuda) 
        {
            throw std::runtime_error("input is on CUDA but output is on CPU\n");
        }
        else 
        {
            throw std::runtime_error("input is on CPU but output is on CUDA\n");
        }
    }
}

template <typename T>
Tensor<T> matadd(Tensor<T> &a, Tensor<T> &b)
{
    bool check_a_b_dim = false;
    char min_dim=-1;
    char max_dim=-1;
    char diff=0;
    check_shape1(a, b, check_a_b_dim, min_dim, max_dim, diff);
    if(check_a_b_dim==true)
    {
        vector<std::uint16_t> c_shape(max_dim);
        copy(b.tensor_shape, b.tensor_shape + b.dim, c_shape.begin());
        Tensor<T> c({ c_shape.begin(), c_shape.end() });
        if(a.is_cuda==true)
        {
            c.cuda();
            dim3 block(16, 16);
            dim3 grid((c.n + block.x - 1) / block.x, (c.m + block.y - 1) / block.y);
            int max_a = a.n*a.m;
            Matrix_Add<T><<<grid, block>>>(b.value, a.value, c.value, c.n, c.m, max_a);
            return c;
        }
        else
        {
            cpu_Matrix_Add(b, a, c);
            return c;
        }
    }
    else
    {
        vector<std::uint16_t> c_shape(max_dim);
        copy(a.tensor_shape, a.tensor_shape + a.dim, c_shape.begin());
        Tensor<T> c({ c_shape.begin(), c_shape.end() });
        if(a.is_cuda==true)
        {
            c.cuda();
            dim3 block(16, 16);
            dim3 grid((c.n + block.x - 1) / block.x, (c.m + block.y - 1) / block.y);
            int max_b = b.n*b.m;
            Matrix_Add<T><<<grid, block>>>(a.value, b.value, c.value, c.n, c.m, max_b);
            return c;
        }
        else
        {
            cpu_Matrix_Add(a, b, c);
            return c;
        }
    }
}

template <typename T>
void matadd(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c)
{
    bool check_a_b_dim = false;
    char min_dim=-1;
    char max_dim=-1;
    char diff=0;
    check_shape2(a, b, c, check_a_b_dim, min_dim, max_dim, diff);

    if(a.is_cuda==true)
    {
        dim3 block(16, 16);
        dim3 grid((c.n + block.x - 1) / block.x, (c.m + block.y - 1) / block.y);
        if(check_a_b_dim==true)
        {
            int max_a = a.n*a.m;
            Matrix_Add<T><<<grid, block>>>(b.value, a.value, c.value, c.n, c.m, max_a);
        }
        else
        {
            int max_b = b.n*b.m;
            Matrix_Add<T><<<grid, block>>>(a.value, b.value, c.value, c.n, c.m, max_b);
        }
        
    }
    else
    {
        if(check_a_b_dim==true)
        {
            cpu_Matrix_Add(b, a, c);
        }
        else
        {
            cpu_Matrix_Add(a, b, c);
        }
    }
}

template <typename T>
Tensor<T> matsub(Tensor<T> &a, Tensor<T> &b)
{
    bool check_a_b_dim = false;
    char min_dim=-1;
    char max_dim=-1;
    char diff=0;
    check_shape1(a, b, check_a_b_dim, min_dim, max_dim, diff);
    if(check_a_b_dim==true)
    {
        vector<std::uint16_t> c_shape(max_dim);
        copy(b.tensor_shape, b.tensor_shape + b.dim, c_shape.begin());
        Tensor<T> c({ c_shape.begin(), c_shape.end() });
        if(a.is_cuda==true)
        {
            c.cuda();
            dim3 block(16, 16);
            dim3 grid((c.n + block.x - 1) / block.x, (c.m + block.y - 1) / block.y);
            int max_a = a.n*a.m;
            Matrix_Sub_a<T><<<grid, block>>>(a.value, b.value, c.value, c.n, c.m, max_a);
            return c;
        }
        else
        {
            cpu_Matrix_sub_a(a, b, c);
            return c;
        }
    }
    else
    {
        vector<std::uint16_t> c_shape(max_dim);
        copy(a.tensor_shape, a.tensor_shape + a.dim, c_shape.begin());
        Tensor<T> c({ c_shape.begin(), c_shape.end() });
        if(a.is_cuda==true)
        {
            c.cuda();
            dim3 block(16, 16);
            dim3 grid((c.n + block.x - 1) / block.x, (c.m + block.y - 1) / block.y);
            int max_b = b.n*b.m;
            Matrix_Sub_b<T><<<grid, block>>>(a.value, b.value, c.value, c.n, c.m, max_b);
            return c;
        }
        else
        {
            cpu_Matrix_sub_b(a, b, c);
            return c;
        }
    }
}

template <typename T>
void matsub(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c)
{
    bool check_a_b_dim = false;
    char min_dim=-1;
    char max_dim=-1;
    char diff=0;
    check_shape2(a, b, c, check_a_b_dim, min_dim, max_dim, diff);
    if(a.is_cuda==true)
    {
        dim3 block(16, 16);
        dim3 grid((c.n + block.x - 1) / block.x, (c.m + block.y - 1) / block.y);
        if(check_a_b_dim==true)
        {
            int max_a = a.n*a.m;
            Matrix_Sub_a<T><<<grid, block>>>(a.value, b.value, c.value, c.n, c.m, max_a);
        }
        else
        {
            int max_b = b.n*b.m;
            Matrix_Sub_b<T><<<grid, block>>>(a.value, b.value, c.value, c.n, c.m, max_b);
        }
    }
    else
    {
        if(check_a_b_dim==true)
        {
            cpu_Matrix_Sub_a(a, b, c);
        }
        else
        {
            cpu_Matrix_Sub_b(a, b, c);
        }
        
    }
}

template <typename T>
Tensor<T> matdiv(Tensor<T> &a, Tensor<T> &b)
{
    bool check_a_b_dim = false;
    char min_dim=-1;
    char max_dim=-1;
    char diff=0;
    check_shape1(a, b, check_a_b_dim, min_dim, max_dim, diff);
    if(check_a_b_dim==true)
    {
        vector<std::uint16_t> c_shape(max_dim);
        copy(b.tensor_shape, b.tensor_shape + b.dim, c_shape.begin());
        Tensor<T> c({ c_shape.begin(), c_shape.end() });
        if(a.is_cuda==true)
        {
            c.cuda();
            dim3 block(16, 16);
            dim3 grid((c.n + block.x - 1) / block.x, (c.m + block.y - 1) / block.y);
            int max_a = a.n*a.m;
            Matrix_Div_a<T><<<grid, block>>>(a.value, b.value, c.value, c.n, c.m, max_a);
            return c;
        }
        else
        {
            cpu_Matrix_div_a(a, b, c);
            return c;
        }
    }
    else
    {
        vector<std::uint16_t> c_shape(max_dim);
        copy(a.tensor_shape, a.tensor_shape + a.dim, c_shape.begin());
        Tensor<T> c({ c_shape.begin(), c_shape.end() });
        if(a.is_cuda==true)
        {
            c.cuda();
            dim3 block(16, 16);
            dim3 grid((c.n + block.x - 1) / block.x, (c.m + block.y - 1) / block.y);
            int max_b = b.n*b.m;
            Matrix_Div_b<T><<<grid, block>>>(a.value, b.value, c.value, c.n, c.m, max_b);
            return c;
        }
        else
        {
            cpu_Matrix_div_b(a, b, c);
            return c;
        }
    }
}

template <typename T>
void matdiv(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c)
{
    bool check_a_b_dim = false;
    char min_dim=-1;
    char max_dim=-1;
    char diff=0;
    check_shape2(a, b, c, check_a_b_dim, min_dim, max_dim, diff);
    if(a.is_cuda==true)
    {
        dim3 block(16, 16);
        dim3 grid((c.n + block.x - 1) / block.x, (c.m + block.y - 1) / block.y);
        if(check_a_b_dim==true)
        {
            int max_a = a.n*a.m;
            Matrix_Div_a<T><<<grid, block>>>(a.value, b.value, c.value, c.n, c.m, max_a);
        }
        else
        {
            int max_b = b.n*b.m;
            Matrix_Div_b<T><<<grid, block>>>(a.value, b.value, c.value, c.n, c.m, max_b);
        }

    }
    else
    {
        if(check_a_b_dim==true)
        {
            cpu_Matrix_Div_a(a, b, c);
        }
        else
        {
            cpu_Matrix_Div_b(a, b, c);
        }
        
    }
}

template <typename T>
Tensor<T> matmul(Tensor<T> &a, Tensor<T> &b)
{
    bool check_a_b_dim = false;
    char min_dim=-1;
    char max_dim=-1;
    char diff=0;
    check_shape1(a, b, check_a_b_dim, min_dim, max_dim, diff);
    if(check_a_b_dim==true)
    {
        vector<std::uint16_t> c_shape(max_dim);
        copy(b.tensor_shape, b.tensor_shape + b.dim, c_shape.begin());
        Tensor<T> c({ c_shape.begin(), c_shape.end() });
        if(a.is_cuda==true)
        {
            c.cuda();
            dim3 block(16, 16);
            dim3 grid((c.n + block.x - 1) / block.x, (c.m + block.y - 1) / block.y);
            int max_a = a.n*a.m;
            MatrixMul<T><<<grid, block>>>(b.value, a.value, c.value, c.n, c.m, max_a);
            return c;
        }
        else
        {
            cpu_MatrixMul(b, a, c);
            return c;
        }
    }
    else
    {
        vector<std::uint16_t> c_shape(max_dim);
        copy(a.tensor_shape, a.tensor_shape + a.dim, c_shape.begin());
        Tensor<T> c({ c_shape.begin(), c_shape.end() });
        if(a.is_cuda==true)
        {
            c.cuda();
            dim3 block(16, 16);
            dim3 grid((c.n + block.x - 1) / block.x, (c.m + block.y - 1) / block.y);
            int max_b = b.n*b.m;
            MatrixMul<T><<<grid, block>>>(a.value, b.value, c.value, c.n, c.m, max_b);
            return c;
        }
        else
        {
            cpu_MatrixMul(a, b, c);
            return c;
        }
    }
}

template <typename T>
void matmul(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c)
{
    bool check_a_b_dim = false;
    char min_dim=-1;
    char max_dim=-1;
    char diff=0;
    check_shape2(a, b, c, check_a_b_dim, min_dim, max_dim, diff);
    if(a.is_cuda==true)
    {
        dim3 block(16, 16);
        dim3 grid((c.n + block.x - 1) / block.x, (c.m + block.y - 1) / block.y);
        if(check_a_b_dim==true)
        {
            int max_a = a.n*a.m;
            MatrixMul<T><<<grid, block>>>(b.value, a.value, c.value, c.n, c.m, max_a);
        }
        else
        {
            int max_b = b.n*b.m;
            MatrixMul<T><<<grid, block>>>(a.value, b.value, c.value, c.n, c.m, max_b);
        }
    }
    else
    {
        if(check_a_b_dim==true)
        {
            cpu_MatrixMul(b, a, c);
        }
        else
        {
            cpu_MatrixMul(a, b, c);
        }
    }
}

template <typename T>
Tensor<T> mat_mul(Tensor<T> &a, Tensor<T> &b)
{
    bool check_a_b_dim = false;
    char min_dim=-1;
    char max_dim=-1;
    char diff=0;
    if(a.dim < b.dim)
    {
        min_dim = a.dim;
        max_dim = b.dim;
        diff = max_dim-min_dim;
        check_a_b_dim = true; // a < b
    }
    else
    {
        min_dim = b.dim;
        max_dim = a.dim;
        diff = max_dim-min_dim;
    }
    if(a.m != b.tensor_shape[b.dim-2])
    {
        throw std::runtime_error("tensor1 col size and tensor2 row size must be same\n");
    }
    if(check_a_b_dim==true)
    {
        if(max_dim != 2)
        {
            for (int i = max_dim-3; i >= diff; i--) 
            {
                if (a.tensor_shape[i-diff] != b.tensor_shape[i]) 
                {
                    throw std::invalid_argument("tensor1 and tensor2 shapes is different\n");
                }
            }
        }
    }
    else
    {
        if(max_dim != 2)
        {
            for (int i = 0; i <min_dim-2; i++)
            {
                if (a.tensor_shape[i+diff] != b.tensor_shape[i]) 
                {
                    
                    throw std::invalid_argument("tensor1 and tensor2 shapes is different\n");
                }
            }
        }
    }

    if (a.is_cuda != b.is_cuda) 
    {
        if (a.is_cuda) 
        {
            throw std::runtime_error("tensor1 is on CUDA but tensor2 is on CPU\n");
        }
        else 
        {
            throw std::runtime_error("tensor1 is on CPU but tensor2 is on CUDA\n");
        }
    }
    char c_dim = max_dim;
    vector<std::uint16_t> c_shape(max_dim);
    int a_n = a.tensor_shape[a.dim-2];
    int max_a = a.n / a_n;
    int a_m = a.m;
    int max_b = b.n / b.tensor_shape[b.dim-2];
    int b_m = b.m;
    if(check_a_b_dim==true)
    {
        for(int i=0; i<max_dim-2; i++)
        {
            c_shape[i] = b.tensor_shape[i];
        }
        c_shape[c_dim-2] = a.tensor_shape[a.dim-2];
        c_shape[c_dim-1] = b.m;
        Tensor<T> c({ c_shape.begin(), c_shape.end() });

        if(a.is_cuda==true)
        {
            c.cuda();
            dim3 block(4, 16, 16);
            dim3 grid((max_b + block.x - 1) / block.x, (a_n + block.y - 1) / block.y, (b_m + block.z - 1) / block.z);
            Matrix_Mul_a<T><<<grid, block>>>(a.value, b.value, c.value, max_b, a_n, a_m, b_m, max_a);
            return c;
        }
        else
        {
            cpu_Matrix_Mul_a<T>(a.value, b.value, c.value, max_b, a_n, a_m, b_m, max_a);
            return c;
        }
    }
    else
    {
        for(int i=0; i<max_dim-2; i++)
        {
            c_shape[i] = a.tensor_shape[i];
        }
        c_shape[c_dim-2] = a.tensor_shape[a.dim-2];
        c_shape[c_dim-1] = b.m;
        Tensor<T> c({ c_shape.begin(), c_shape.end() });

        if(a.is_cuda==true)
        {
            c.cuda();
            dim3 block(4, 16, 16);
            dim3 grid((max_a + block.x - 1) / block.x, (a_n + block.y - 1) / block.y, (b_m + block.z - 1) / block.z);
            Matrix_Mul_b<T><<<grid, block>>>(a.value, b.value, c.value, max_a, a_n, a_m, b_m, max_b);
            return c;
        }
        else
        {
            cpu_Matrix_Mul_b<T>(a.value, b.value, c.value, max_a, a_n, a_m, b_m, max_b);
            return c;
        }
    }

}

template <typename T>
void mat_mul(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c)
{
    bool check_a_b_dim = false;
    char min_dim=-1;
    char max_dim=-1;
    char diff=0;
    if(a.dim < b.dim)
    {
        min_dim = a.dim;
        max_dim = b.dim;
        diff = max_dim-min_dim;
        check_a_b_dim = true; // a < b
    }
    else
    {
        min_dim = b.dim;
        max_dim = a.dim;
        diff = max_dim-min_dim;
    }
    if(a.m != b.tensor_shape[b.dim-2])
    {
        throw std::runtime_error("tensor1 col size and tensor2 row size must be same\n");
    }
    if(check_a_b_dim==true)
    {
        if(max_dim != 2)
        {
            for (int i = max_dim-3; i >= diff; i--) 
            {
                if (a.tensor_shape[i-diff] != b.tensor_shape[i]) 
                {
                    throw std::invalid_argument("tensor1 and tensor2 shapes is different\n");
                }
            }
        }
        for(int i=0; i<max_dim-2; i++)
        {
            if(b.tensor_shape[i] != c.tensor_shape[i])
            {
                throw std::invalid_argument("input and output shapes is different\n");
            }
        }
    }
    else
    {
        if(max_dim != 2)
        {
            for (int i = 0; i <min_dim-2; i++)
            {
                if (a.tensor_shape[i+diff] != b.tensor_shape[i]) 
                {
                    
                    throw std::invalid_argument("tensor1 and tensor2 shapes is different\n");
                }
            }
            for(int i=0; i<max_dim-2; i++)
            {
                if(a.tensor_shape[i] != c.tensor_shape[i])
                {
                    throw std::invalid_argument("input and output shapes is different\n");
                }
            }
        }
    }
    if(c.m != b.m || c.tensor_shape[c.dim-2] != a.tensor_shape[a.dim-2])
    {
         throw std::invalid_argument("input and output shapes is different1\n");
    }

    if (a.is_cuda != b.is_cuda || a.is_cuda != c.is_cuda) 
    {
        if (a.is_cuda) 
        {
            throw std::runtime_error("tensor1 is on CUDA but tensor2/output is on CPU\n");
        }
        else 
        {
            throw std::runtime_error("tensor1 is on CPU but tensor2/output is on CUDA\n");
        }
    }

    int a_n = a.tensor_shape[a.dim-2];
    int max_a = a.n / a_n;
    int a_m = a.m;
    int max_b = b.n / b.tensor_shape[b.dim-2];
    int b_m = b.m;
    if(check_a_b_dim==true)
    {
        if(a.is_cuda==true)
        {
            c.cuda();
            dim3 block(4, 16, 16);
            dim3 grid((max_b + block.x - 1) / block.x, (a_n + block.y - 1) / block.y, (b_m + block.z - 1) / block.z);
            Matrix_Mul_a<T><<<grid, block>>>(a.value, b.value, c.value, max_b, a_n, a_m, b_m, max_a);
        }
        else
        {
            cpu_Matrix_Mul_a<T>(a.value, b.value, c.value, max_b, a_n, a_m, b_m, max_a);
        }
    }
    else
    {
        if(a.is_cuda==true)
        {
            c.cuda();
            dim3 block(4, 16, 16);
            dim3 grid((max_a + block.x - 1) / block.x, (a_n + block.y - 1) / block.y, (b_m + block.z - 1) / block.z);
            Matrix_Mul_b<T><<<grid, block>>>(a.value, b.value, c.value, max_a, a_n, a_m, b_m, max_b);
        }
        else
        {
            cpu_Matrix_Mul_b<T>(a.value, b.value, c.value, max_a, a_n, a_m, b_m, max_b);
        }
    }
}

// template <typename T>
// Tensor<T> trans(Tensor<T> &a)
// {
//     if(a.dim==1)
//     {
//         Tensor<T> out(a.tensor_shape);
//         return out;
//     }
//     vector<uint16_t> out_shape(a.dim);
//     for(int i=0; i<a.dim-2; i++)
//     {
//         out_shape[i] = a.tensor_shape[i];
//     }
//     out_shape[a.dim-2] = a.tensor_shape[a.dim-1];
//     out_shape[a.dim-1] = a.tensor_shape[a.dim-2];
//     Tensor<T> out(out_shape);
//     int n = a.tensor_shape[a.dim-2];
//     int bs = a.n / n;
//     int m = a.m;
//     if(a.is_cuda==true)
//     {
//         out.cuda();
//         dim3 block(4, 16, 16);
//         dim3 grid((bs + block.x - 1) / block.x, (n + block.y - 1) / block.y, (m + block.z - 1) / block.z);
//         gpu_trans<<<grid, block>>>(a.value, out.value, bs, n, m);
//         return out;
//     }
//     else
//     {
//         cpu_trans(a.value, out.value, bs, n, m);
//         out.print();
//         return out;
//     }
// }

template <typename T>
Tensor<T> *trans(Tensor<T> &a)
{
    if(a.dim==1)
    {
        Tensor<T> *out = new Tensor<T>(a.tensor_shape);
        return out;
    }
    vector<uint16_t> out_shape(a.dim);
    for(int i=0; i<a.dim-2; i++)
    {
        out_shape[i] = a.tensor_shape[i];
    }
    out_shape[a.dim-2] = a.tensor_shape[a.dim-1];
    out_shape[a.dim-1] = a.tensor_shape[a.dim-2];
    Tensor<T> *out = new Tensor<T>(out_shape, a.is_cuda);
    int n = a.tensor_shape[a.dim-2];
    int bs = a.n / n;
    int m = a.m;
    if(a.is_cuda==true)
    {
        //out->cuda();
        dim3 block(4, 16, 16);
        dim3 grid((bs + block.x - 1) / block.x, (n + block.y - 1) / block.y, (m + block.z - 1) / block.z);
        gpu_trans<<<grid, block>>>(a.value, out->value, bs, n, m);
        return out;
    }
    else
    {
        cpu_trans(a.value, out->value, bs, n, m);
        return out;
    }
}

template <typename T>
void trans(Tensor<T> &a, Tensor<T> &out)
{
    if(a.dim != out.dim)
    {
        throw std::invalid_argument("input and output must be same dim\n");
    }
    if(a.is_cuda != out.is_cuda)
    {
        throw std::runtime_error("input and output device is not same\n");
    }
    if(a.dim==1)
    {
        return;
    }
    for(int i=0; i<a.dim-2; i++)
    {
        if(a.tensor_shape[i] != out.tensor_shape[i])
        {
            throw std::invalid_argument("input and output shape is not match\n");
        }
    }
    if(out.tensor_shape[out.dim-2] != a.tensor_shape[a.dim-1] || out.tensor_shape[out.dim-1] != a.tensor_shape[a.dim-2])
    {
        throw std::invalid_argument("input and output shape is not match\n");
    }
    int n = a.tensor_shape[a.dim-2];
    int bs = a.n / n;
    int m = a.m;
    if(a.is_cuda==true)
    {
        out.cuda();
        dim3 block(4, 16, 16);
        dim3 grid((bs + block.x - 1) / block.x, (n + block.y - 1) / block.y, (m + block.z - 1) / block.z);
        gpu_trans<<<grid, block>>>(a.value, out.value, bs, n, m);
    }
    else
    {
        cpu_trans(a.value, out.value, bs, n, m);
    }
}

#endif