#include "matOper.h"
#include "tensor.h"

using namespace std;

template <typename T>
__global__ void Matrix_Add(T *d_a, T *d_b, T *d_c, unsigned int n, unsigned int m, int max_b) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < m) 
  {
    d_c[i * m + j] = d_a[i * m + j] + d_b[(i * m)%max_b + j];
  }
}

template <typename T>
__global__ void Matrix_Sub_a(T *d_a, T *d_b, T *d_c, unsigned int n, unsigned int m, int max_a) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < m) 
  {
    d_c[i * m + j] = d_a[(i * m)%max_a + j] - d_b[i * m + j];
  }
}

template <typename T>
__global__ void Matrix_Sub_b(T *d_a, T *d_b, T *d_c, unsigned int n, unsigned int m, int max_b) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < m) 
  {
    d_c[i * m + j] = d_a[i * m + j] - d_b[(i * m)%max_b + j];
  }
}

template <typename T>
__global__ void Matrix_Div_a(T *d_a, T *d_b, T *d_c, unsigned int n, unsigned int m, int max_a) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < m) 
  {
    d_c[i * m + j] = d_a[(i * m)%max_a + j] / d_b[i * m + j];
  }
}

template <typename T>
__global__ void Matrix_Div_b(T *d_a, T *d_b, T *d_c, unsigned int n, unsigned int m, int max_b) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < m) 
  {
    d_c[i * m + j] = d_a[i * m + j] / d_b[(i * m)%max_b + j];
  }
}

template <typename T>
__global__ void MatrixMul(T *d_a, T *d_b, T *d_c, unsigned int n, unsigned int m, int max_b) // normal multiplication
{ 
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < m) 
  {
    d_c[i * m + j] = d_a[i * m + j] * d_b[(i * m)%max_b + j];
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
void matoper_check_shape1(Tensor<T> &a, Tensor<T> &b, bool &check_a_b_dim ,char &min_dim, char &max_dim, char &diff)
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
void matoper_check_shape2(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c, bool &check_a_b_dim ,char &min_dim, char &max_dim, char &diff)
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
    matoper_check_shape1(a, b, check_a_b_dim, min_dim, max_dim, diff);
    if(check_a_b_dim==true)
    {
        unsigned short c_shape[b.dim];
        memcpy(c_shape, b.tensor_shape, sizeof(short)*b.dim);
        Tensor<T> c(c_shape, b.dim);
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
        unsigned short c_shape[a.dim];
        memcpy(c_shape, a.tensor_shape, sizeof(short)*a.dim);
        Tensor<T> c(c_shape, a.dim);
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
    matoper_check_shape2(a, b, c, check_a_b_dim, min_dim, max_dim, diff);

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
    matoper_check_shape1(a, b, check_a_b_dim, min_dim, max_dim, diff);
    if(check_a_b_dim==true)
    {
        unsigned short c_shape[b.dim];
        memcpy(c_shape, b.tensor_shape, sizeof(short)*b.dim);
        Tensor<T> c(c_shape, b.dim);
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
        unsigned short c_shape[a.dim];
        memcpy(c_shape, a.tensor_shape, sizeof(short)*a.dim);
        Tensor<T> c(c_shape, a.dim);
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
    matoper_check_shape2(a, b, c, check_a_b_dim, min_dim, max_dim, diff);
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
    matoper_check_shape1(a, b, check_a_b_dim, min_dim, max_dim, diff);
    if(check_a_b_dim==true)
    {
        unsigned short c_shape[b.dim];
        memcpy(c_shape, b.tensor_shape, sizeof(short)*b.dim);
        Tensor<T> c(c_shape, b.dim);
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
        unsigned short c_shape[a.dim];
        memcpy(c_shape, a.tensor_shape, sizeof(short)*a.dim);
        Tensor<T> c(c_shape, a.dim);
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
    matoper_check_shape2(a, b, c, check_a_b_dim, min_dim, max_dim, diff);
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
    matoper_check_shape1(a, b, check_a_b_dim, min_dim, max_dim, diff);
    if(check_a_b_dim==true)
    {
        unsigned short c_shape[b.dim];
        memcpy(c_shape, b.tensor_shape, sizeof(short)*b.dim);
        Tensor<T> c(c_shape, b.dim);
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
        unsigned short c_shape[a.dim];
        memcpy(c_shape, a.tensor_shape, sizeof(short)*a.dim);
        Tensor<T> c(c_shape, a.dim);
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
    matoper_check_shape2(a, b, c, check_a_b_dim, min_dim, max_dim, diff);
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
    if (a.dim != b.dim) 
    {
        throw std::invalid_argument("tensor1 and tensor2 dimensions do not match");
    }

    if (b.dim >= 2) 
    {
        unsigned int b_n = b.tensor_shape[b.dim-2];
        if (a.m != b_n) 
        {
            throw std::invalid_argument("tensor1 column and tensor2 row do not match");
        }
    }

    for (int i = 0; i < a.dim-2; i++) 
    {
        if (a.tensor_shape[i] != b.tensor_shape[i]) 
        {
            throw std::invalid_argument("tensor1 and tensor2 shapes do not match");
        }
    }

    if (a.is_cuda != b.is_cuda) 
    {
        if (a.is_cuda) 
        {
            throw std::runtime_error("tensor1 is on CUDA device but tensor2 is on CPU");
        }
        else 
        {
            throw std::runtime_error("tensor1 is on CPU but tensor2 is on CUDA device");
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
    if (a.dim != b.dim) 
    {
        throw std::invalid_argument("tensor1 and tensor2 dimensions do not match");
    }

    unsigned int b_n = b.tensor_shape[b.dim-2];
    if (a.m != b_n) 
    {
        throw std::invalid_argument("tensor1 columns and tensor2 rows do not match");
    }

    for (int i = 0; i < a.dim-2; i++) 
    {
        if (a.tensor_shape[i] != b.tensor_shape[i]) 
        {
            throw std::invalid_argument("tensor1 and tensor2 shapes do not match");
        }
    }

    if (a.is_cuda != b.is_cuda || a.is_cuda != c.is_cuda) 
    {
        if (a.is_cuda) 
        {
            throw std::runtime_error("tensor1 is on CUDA device but tensor2/output is on CPU");
        }
        else 
        {
            throw std::runtime_error("tensor1 is on CPU but tensor2/output is on CUDA device");
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
