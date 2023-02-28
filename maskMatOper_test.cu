#include "tensor.h"
#include "matOper.h"

using namespace std;

template <typename T>
__global__ void mask_Matrix_Mul_a(T *a,T *b, T *c, unsigned int bs, unsigned int n, unsigned int m, unsigned int p, int max_a)
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
__global__ void mask_Matrix_Mul_b(T *a,T *b, T *c, unsigned int bs, unsigned int n, unsigned int m, unsigned int p, int max_b)
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
void cpu_mask_Matrix_Mul_a(T *a, T *b, T *c, unsigned short bs, unsigned short n, unsigned short m, unsigned short p, int max_a)
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
                    //sum += a[l*n*m + i * m + k] * b[l*m*p + k * p + j];
                    sum += a[(l%max_a)*n*m + i * m + k] * b[l*m*p + k * p + j];
                }
                c[l*n*p + i * p + j] = sum;
            }
        }
    }
}

template <typename T>
void cpu_mask_Matrix_Mul_b(T *a, T *b, T *c, unsigned short bs, unsigned short n, unsigned short m, unsigned short p, int max_b)
{
    for(int l=0; l<bs; l++)
    {
        //printf("%d\n\n", l%max_b*m*p );
        for (int i = 0; i < n; ++i) 
        {
            for (int j = 0; j < p; ++j) 
            {
                T sum = 0.;
                for (int k = 0; k < m; ++k) 
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
void mask_mat_mul(Tensor<T> &a, Tensor<T> &b, Tensor<T> &c)
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