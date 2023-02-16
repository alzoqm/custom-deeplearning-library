#include "matOper.h"

using namespace std;

template <typename T>
__global__ void Matrix_Add(T *d_a, T *d_b, T *d_c, unsigned int n, unsigned int m) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < m) 
  {
    d_c[i * m + j] = d_a[i * m + j] + d_b[i * m + j];
  }
}

template <typename T>
__global__ void Matrix_Sub(T *d_a, T *d_b, T *d_c, unsigned int n, unsigned int m) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < m) 
  {
    d_c[i * m + j] = d_a[i * m + j] - d_b[i * m + j];
  }
}

template <typename T>
__global__ void Matrix_Div(T *d_a, T *d_b, T *d_c, unsigned int n, unsigned int m) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < m) 
  {
    d_c[i * m + j] = d_a[i * m + j] / d_b[i * m + j];
  }
}

template <typename T>
__global__ void MatrixMul(T *d_a, T *d_b, T *d_c, unsigned int n, unsigned int m) // normal multiplication
{ 
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < m) 
  {
    d_c[i * m + j] = d_a[i * m + j] * d_b[i * m + j];
  }
}

template <typename T>
__global__ void Matrix_Mul(T *a,T *b, T *c, unsigned int bs, unsigned int n, unsigned int m, unsigned int p)
{ 
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.z * blockDim.z + threadIdx.z;
    
    if(l < bs && i < n && j < p ) 
    {
        T sum = 0.;
        for(int k = 0; k < m; k++) 
        {
            //sum += a[batch_size*n*m + row * m + i] * b[batch_size*m*p + i * p + col];
            sum += a[l*n*m + i * m + k] * b[l*m*p + k * p + j];
        }
        c[l*n*p + i * p + j] = sum;
    }
}

template <typename T>
void cpu_Matrix_Add(T &a, T &b, T &c)
{
    for(int i=0; i<c.n; i++)
    {
        for(int j=0; j<c.m; j++)
        {
            c.value[i*c.m + j] = a.value[i*c.m + j] + b.value[i*c.m + j];
        }
    }
}

template <typename T>
void cpu_Matrix_Sub(T &a, T &b, T &c)
{
    for(int i=0; i<c.n; i++)
    {
        for(int j=0; j<c.m; j++)
        {
            c.value[i*c.m + j] = a.value[i*c.m + j] - b.value[i*c.m + j];
        }
    }
}

template <typename T>
void cpu_Matrix_Div(T &a, T &b, T &c)
{
    for(int i=0; i<c.n; i++)
    {
        for(int j=0; j<c.m; j++)
        {
            c.value[i*c.m + j] = a.value[i*c.m + j] / b.value[i*c.m + j];
        }
    }
}

template <typename T>
void cpu_MatrixMul(T &a, T &b, T &c)
{
    for(int i=0; i<c.n; i++)
    {
        for(int j=0; j<c.m; j++)
        {
            c.value[i*c.m + j] = a.value[i*c.m + j] * b.value[i*c.m + j];
        }
    }
}

template <typename T>
void cpu_Matrix_Mul(T *a, T *b, T *c, unsigned short bs, unsigned short n, unsigned short m, unsigned short p)
{
    for(int l=0; l<bs; l++)
    {
        for (int i = 0; i < n; ++i) 
        {
            for (int j = 0; j < p; ++j) 
            {
                T sum = 0.;
                for (int k = 0; k < m; ++k) 
                {
                    sum += a[l*n*m + i * m + k] * b[l*m*p + k * p + j];
                }
                c[l*n*p + i * p + j] = sum;
            }
        }
    }
}

template <typename T>
Tensor<T> matadd(Tensor<T> &a, Tensor<T> &b)
{
    if(a.dim != b.dim)
    {
        throw "tensor1 and tensor2 dim is not same!\n";
    }
    for (int i = 0; i < a.dim; i++) 
    {
        if (a.tensor_shape[i] != b.tensor_shape[i]) 
        {
            throw invalid_argument("tensor1 and tensor2 shape is not same!");
        }
    }
    if(a.is_cuda != b.is_cuda)
    {
        if(a.is_cuda==true)
        {
            throw "tensor1 is cuda but tensor2 is cpu.\n";
        }
        else
        {
            throw "tensor1 is cpu but tensor2 is cuda.\n";
        }
    }
    unsigned short c_shape[a.dim];
    memcpy(c_shape, a.tensor_shape, sizeof(short)*a.dim);
    Tensor<T> c(c_shape, a.dim);
    if(a.is_cuda==true)
    {
        c.cuda();
        dim3 block(16, 16);
        dim3 grid((c.n + block.x - 1) / block.x, (c.m + block.y - 1) / block.y);
        Matrix_Add<T><<<grid, block>>>(a.value, b.value, c.value, c.n, c.m);
        return c;
    }
    else
    {
        cpu_Matrix_Add(a, b, c);
        return c;
    }
}

template <typename T>
void matadd(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c)
{
    if(a.dim != b.dim)
    {
        throw "tensor1 and tensor2 dim is not same!\n";
    }
    if(a.dim != c.dim)
    {
        throw "input and output dim is not same!\n";
    }
    for (int i = 0; i < a.dim; i++) 
    {
        if (a.tensor_shape[i] != b.tensor_shape[i]) 
        {
            throw invalid_argument("tensor1 and tensor2 shape is not same!");
        }
        if(a.tensor_shape[i] != c.tensor_shape[i])
        {
            throw invalid_argument("input shape and output shape is not same!");
        }
    }
    if(a.is_cuda != b.is_cuda)
    {
        if(a.is_cuda==true)
        {
            throw "tensor1 is cuda but tensor2 is cpu.\n";
        }
        else
        {
            throw "tensor1 is cpu but tensor2 is cuda.\n";
        }
    }
    if(a.is_cuda != c.is_cuda)
    {
        if(a.is_cuda==true)
        {
            throw "input is cuda but output is cpu.\n";
        }
        else
        {
            throw "input is cpu but output is cuda.\n";
        }
    }

    if(a.is_cuda==true)
    {
        dim3 block(16, 16);
        dim3 grid((c.n + block.x - 1) / block.x, (c.m + block.y - 1) / block.y);
        Matrix_Add<T><<<grid, block>>>(a.value, b.value, c.value, c.n, c.m);
    }
    else
    {
        cpu_Matrix_Add(a, b, c);
    }
}

template <typename T>
Tensor<T> matsub(Tensor<T> &a, Tensor<T> &b)
{
    if(a.dim != b.dim)
    {
        throw "tensor1 and tensor2 dim is not same!\n";
    }
    for (int i = 0; i < a.dim; i++) 
    {
        if (a.tensor_shape[i] != b.tensor_shape[i]) 
        {
            throw invalid_argument("tensor1 and tensor2 shape is not same!");
        }
    }
    if(a.is_cuda != b.is_cuda)
    {
        if(a.is_cuda==true)
        {
            throw "tensor1 is cuda but tensor2 is cpu.\n";
        }
        else
        {
            throw "tensor1 is cpu but tensor2 is cuda.\n";
        }
    }
    unsigned short c_shape[a.dim];
    memcpy(c_shape, a.tensor_shape, sizeof(short)*a.dim);
    Tensor<T> c(c_shape, a.dim);
    if(a.is_cuda==true)
    {
        c.cuda(); 
        dim3 block(16, 16);
        dim3 grid((c.n + block.x - 1) / block.x, (c.m + block.y - 1) / block.y);
        Matrix_Sub<T><<<grid, block>>>(a.value, b.value, c.value, c.n, c.m);
        return c;
    }
    else
    {
        cpu_Matrix_Sub(a, b, c);
        return c;
    }
}

template <typename T>
void matsub(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c)
{
    if(a.dim != b.dim)
    {
        throw "tensor1 and tensor2 dim is not same!\n";
    }
    if(a.dim != c.dim)
    {
        throw "input and output dim is not same!\n";
    }
    for (int i = 0; i < a.dim; i++) 
    {
        if (a.tensor_shape[i] != b.tensor_shape[i]) 
        {
            throw invalid_argument("tensor1 and tensor2 shape is not same!");
        }
        if(a.tensor_shape[i] != c.tensor_shape[i])
        {
            throw invalid_argument("input shape and output shape is not same!");
        }
    }
    if(a.is_cuda != b.is_cuda)
    {
        if(a.is_cuda==true)
        {
            throw "tensor1 is cuda but tensor2 is cpu.\n";
        }
        else
        {
            throw "tensor1 is cpu but tensor2 is cuda.\n";
        }
    }
    if(a.is_cuda != c.is_cuda)
    {
        if(a.is_cuda==true)
        {
            throw "input is cuda but output is cpu.\n";
        }
        else
        {
            throw "input is cpu but output is cuda.\n";
        }
    }
    if(a.is_cuda==true)
    {
        dim3 block(16, 16);
        dim3 grid((c.n + block.x - 1) / block.x, (c.m + block.y - 1) / block.y);
        Matrix_Sub<T><<<grid, block>>>(a.value, b.value, c.value, c.n, c.m);
    }
    else
    {
        cpu_Matrix_Sub(a, b, c);
    }
}

template <typename T>
Tensor<T> matdiv(Tensor<T> &a, Tensor<T> &b)
{
    if(a.dim != b.dim)
    {
        throw "tensor1 and tensor2 dim is not same!\n";
    }
    for (int i = 0; i < a.dim; i++) 
    {
        if (a.tensor_shape[i] != b.tensor_shape[i]) 
        {
            throw invalid_argument("tensor1 and tensor2 shape is not same!");
        }
    }
    if(a.is_cuda != b.is_cuda)
    {
        if(a.is_cuda==true)
        {
            throw "tensor1 is cuda but tensor2 is cpu.\n";
        }
        else
        {
            throw "tensor1 is cpu but tensor2 is cuda.\n";
        }
    }
    unsigned short c_shape[a.dim];
    memcpy(c_shape, a.tensor_shape, sizeof(short)*a.dim);
    Tensor<T> c(c_shape, a.dim);
    if(a.is_cuda==true)
    {
        c.cuda();
        dim3 block(16, 16);
        dim3 grid((c.n + block.x - 1) / block.x, (c.m + block.y - 1) / block.y);
        Matrix_Div<T><<<grid, block>>>(a.value, b.value, c.value, c.n, c.m);
        return c;
    }
    else
    {
        cpu_Matrix_Div(a, b, c);
        return c;
    }
}

template <typename T>
void matdiv(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c)
{
    if(a.dim != b.dim)
    {
        throw "tensor1 and tensor2 dim is not same!\n";
    }
    if(a.dim != c.dim)
    {
        throw "input and output dim is not same!\n";
    }
    for (int i = 0; i < a.dim; i++) 
    {
        if (a.tensor_shape[i] != b.tensor_shape[i]) 
        {
            throw invalid_argument("tensor1 and tensor2 shape is not same!");
        }
        if(a.tensor_shape[i] != c.tensor_shape[i])
        {
            throw invalid_argument("input shape and output shape is not same!");
        }
    }
    if(a.is_cuda != b.is_cuda)
    {
        if(a.is_cuda==true)
        {
            throw "tensor1 is cuda but tensor2 is cpu.\n";
        }
        else
        {
            throw "tensor1 is cpu but tensor2 is cuda.\n";
        }
    }
    if(a.is_cuda != c.is_cuda)
    {
        if(a.is_cuda==true)
        {
            throw "input is cuda but output is cpu.\n";
        }
        else
        {
            throw "input is cpu but output is cuda.\n";
        }
    }
    if(a.is_cuda==true)
    {
        dim3 block(16, 16);
        dim3 grid((c.n + block.x - 1) / block.x, (c.m + block.y - 1) / block.y);
        Matrix_Div<T><<<grid, block>>>(a.value, b.value, c.value, c.n, c.m);
    }
    else
    {
        cpu_Matrix_Div(a, b, c);
    }
}

template <typename T>
Tensor<T> matmul(Tensor<T> &a, Tensor<T> &b)
{
    if(a.dim != b.dim)
    {
        throw "tensor1 and tensor2 dim is not same!\n";
    }
    for (int i = 0; i < a.dim; i++) 
    {
        if (a.tensor_shape[i] != b.tensor_shape[i]) 
        {
            throw invalid_argument("tensor1 and tensor2 shape is not same!");
        }
    }
    if(a.is_cuda != b.is_cuda)
    {
        if(a.is_cuda==true)
        {
            throw "tensor1 is cuda but tensor2 is cpu.\n";
        }
        else
        {
            throw "tensor1 is cpu but tensor2 is cuda.\n";
        }
    }
    unsigned short c_shape[a.dim];
    memcpy(c_shape, a.tensor_shape, sizeof(short)*a.dim);
    Tensor<T> c(c_shape, a.dim);
    if(a.is_cuda==true)
    {
        c.cuda();
        dim3 block(16, 16);
        dim3 grid((c.n + block.x - 1) / block.x, (c.m + block.y - 1) / block.y);
        MatrixMul<T><<<grid, block>>>(a.value, b.value, c.value, c.n, c.m);
        return c;
    }
    else
    {
        cpu_MatrixMul(a, b, c);
        return c;
    }
}

template <typename T>
void matmul(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c)
{
    if(a.dim != b.dim)
    {
        throw "tensor1 and tensor2 dim is not same!\n";
    }
    if(a.dim != c.dim)
    {
        throw "input and output dim is not same!\n";
    }
    for (int i = 0; i < a.dim; i++) 
    {
        if (a.tensor_shape[i] != b.tensor_shape[i]) 
        {
            throw invalid_argument("tensor1 and tensor2 shape is not same!");
        }
        if(a.tensor_shape[i] != c.tensor_shape[i])
        {
            throw invalid_argument("input shape and output shape is not same!");
        }
    }
    if(a.is_cuda != b.is_cuda)
    {
        if(a.is_cuda==true)
        {
            throw "tensor1 is cuda but tensor2 is cpu.\n";
        }
        else
        {
            throw "tensor1 is cpu but tensor2 is cuda.\n";
        }
    }
    if(a.is_cuda != c.is_cuda)
    {
        if(a.is_cuda==true)
        {
            throw "input is cuda but output is cpu.\n";
        }
        else
        {
            throw "input is cpu but output is cuda.\n";
        }
    }
    if(a.is_cuda==true)
    {
        dim3 block(16, 16);
        dim3 grid((c.n + block.x - 1) / block.x, (c.m + block.y - 1) / block.y);
        MatrixMul<T><<<grid, block>>>(a.value, b.value, c.value, c.n, c.m);
    }
    else
    {
        cpu_MatrixMul(a, b, c);
    }
}

template <typename T>
Tensor<T> mat_mul(Tensor<T> &a, Tensor<T> &b)
{
    if(a.dim != b.dim)
    {
        throw "tensor1 and tensor2 dim is not same!\n";
    }
    unsigned int b_n = b.tensor_shape[b.dim-2];
    if(a.m != b_n)
    {
        throw "tensor1 col and tensor2 row is not same!\n";
    }
    for(int i=0; i<a.dim-2; i++)
    {
        if(a.tensor_shape[i] != b.tensor_shape[i])
        {
            throw "tensor1 shape and tensor2 shape is not same!\n";
        }
    }
    if(a.is_cuda != b.is_cuda)
    {
        if(a.is_cuda==true)
        {
            throw "tensor1 is cuda but tensor2 is cpu.\n";
        }
        else
        {
            throw "tensor1 is cpu but tensor2 is cuda.\n";
        }
    }
    char c_dim = a.dim;
    unsigned short c_shape[c_dim];
    for(int i=0; i<a.dim-2; i++)
    {
        c_shape[i] = a.tensor_shape[i];
    }
    c_shape[c_dim-2] = a.tensor_shape[c_dim-2];
    c_shape[c_dim-1] = b.tensor_shape[c_dim-1];
    Tensor<T> c(c_shape, c_dim);
    int a_n = a.tensor_shape[a.dim-2];
    int a_b = a.n / a_n;
    int a_m = a.m;
    int b_m = b.m;
    if(a.is_cuda==true)
    {
        c.cuda();
        // dim3 block(16, 16);
        // dim3 grid((c.n + block.x - 1) / block.x, (c.m + block.y - 1) / block.y);
        dim3 block(4, 16, 16);
        dim3 grid((a_b + block.x - 1) / block.x, (a_n + block.y - 1) / block.y, (b_m + block.z - 1) / block.z);
        Matrix_Mul<T><<<grid, block>>>(a.value, b.value, c.value, a_b, a_n, a_m, b_m);
        return c;
    }
    else
    {
        cpu_Matrix_Mul<T>(a.value, b.value, c.value, a_b, a_n, a_m, b_m);
        return c;
    }
}

template <typename T>
void mat_mul(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c)
{
    if(a.dim != b.dim)
    {
        throw "tensor1 and tensor2 dim is not same!\n";
    }
    unsigned int b_n = b.tensor_shape[b.dim-2];
    if(a.m != b_n)
    {
        throw "tensor1 col and tensor2 row is not same!\n";
    }
    for(int i=0; i<a.dim-2; i++)
    {
        if(a.tensor_shape[i] != b.tensor_shape[i])
        {
            throw "tensor1 shape and tensor2 shape is not same!\n";
        }
    }
    if(a.is_cuda != b.is_cuda || a.is_cuda != c.is_cuda)
    {
        if(a.is_cuda==true)
        {
            throw "tensor1 is cuda but tensor2 is cpu.\n";
        }
        else
        {
            throw "tensor1 is cpu but tensor2 is cuda.\n";
        }
    }
    int a_n = a.tensor_shape[a.dim-2];
    int a_b = a.n / a_n;
    int a_m = a.m;
    int b_m = b.m;
    if(a.is_cuda==true)
    {
        c.cuda();
        dim3 block(4, 16, 16);
        dim3 grid((a_b + block.x - 1) / block.x, (a_n + block.y - 1) / block.y, (b_m + block.z - 1) / block.z);
        Matrix_Mul<T><<<grid, block>>>(a.value, b.value, c.value, a_b, a_n, a_m, b_m);
    }
    else
    {
        cpu_Matrix_Mul<T>(a.value, b.value, c.value, a_b, a_n, a_m, b_m);
    }
}