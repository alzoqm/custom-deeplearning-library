#include "tensor.h"

using namespace std;


template <typename T>
Tensor<T>::Tensor(unsigned short *shape, char dim) // Constructor for the Tensor class
{
    this->dim = dim; // Assign the number of dimensions

    // Allocate memory for the 'tensor_shape' array and store the shape information
    this->tensor_shape = new unsigned short[dim];
    memcpy(this->tensor_shape, shape, sizeof(ushort)*dim);

    // Calculate the total size of the tensor by multiplying the shape information
    this->sum_size = 1;
    for(int i=0; i<dim; i++)
    {
        this->sum_size = this->sum_size*shape[i];
    }
    
    // Calculate the number of rows and columns in the tensor
    this->m = this->tensor_shape[this->dim-1];
    this->n = this->sum_size / this->m;
    
    // Allocate memory for the 'value' array and fill it with random values
    this->value = new T[this->sum_size];
    for(int i=0; i<this->sum_size; ++i)
    {
        value[i] = (((T)rand())/RAND_MAX) + 0.5; // 0~1 range
    }

    // Set the 'is_cuda' member to false (indicating that the tensor is not stored on a GPU)
    this->is_cuda = false;
}

template <typename T>
Tensor<T>::Tensor(T *value, unsigned short *shape, char dim) // Constructor allocation Value to Tensor
{
    this->dim = dim; // Assign the number of dimensions

    // Allocate memory for the 'tensor_shape' array and store the shape information
    this->tensor_shape = new unsigned short[dim];
    memcpy(this->tensor_shape, shape, sizeof(ushort)*dim);
    
    // Calculate the total size of the tensor by multiplying the shape information
    this->sum_size = 1;
    for(int i=0; i<dim; i++)
    {
        this->sum_size = this->sum_size*shape[i];
    }
    
    // Calculate the number of rows and columns in the tensor
    this->m = this->tensor_shape[this->dim-1];
    this->n = this->sum_size / this->m;
    
    // Allocate memory for the 'value' array and copy the values from the input 'value' argument
    this->value = new T[this->sum_size];
    memcpy(this->value, value, this->sum_size*sizeof(T));

    // Set the 'is_cuda' member to false (indicating that the tensor is not stored on a GPU)
    this->is_cuda = false;
}

template <typename T>
Tensor<T>::Tensor(T value, unsigned short *shape, char dim) // Constructor allocation Value to Tensor
{
    this->dim = dim; // Assign the number of dimensions
    // Allocate memory for the 'tensor_shape' array and store the shape information
    this->tensor_shape = new unsigned short[dim];
    memcpy(this->tensor_shape, shape, sizeof(ushort)*dim);
    
    // Calculate the total size of the tensor by multiplying the shape information
    this->sum_size = 1;
    for(int i=0; i<dim; i++)
    {
        this->sum_size = this->sum_size*shape[i];
    }
    
    // Calculate the number of rows and columns in the tensor
    this->m = this->tensor_shape[this->dim-1];
    this->n = this->sum_size / this->m;
    
    // Allocate memory for the 'value' and copy the values from the input 'value' argument
    this->value = new T[this->sum_size];
    for(int i=0; i<this->sum_size; i++)
    {
        this->value[i] = value;
    }
    //memcpy(this->value, value, this->sum_size*sizeof(T));

    // Set the 'is_cuda' member to false (indicating that the tensor is not stored on a GPU)
    this->is_cuda = false;
}

template <typename T>
Tensor<T>::~Tensor() // Destructor for the Tensor class
{   
    if(this->is_cuda==true) // Check if the tensor is stored on a GPU or in CPU memory
    {
        cudaFree(this->value);
    }
    else // else in cpu
    {
        delete this->value;
    }
}

template <typename T>
unsigned short *Tensor<T>::shape() // Returns the shape of the Tensor2D as an array of unsigned short integers
{
    // Allocate memory for a temporary shape array
    unsigned short *temp_shape;
    temp_shape = new unsigned short[this->dim];

    // Copy the shape from 'tensor_shape' to 'temp_shape'
    memcpy(temp_shape, this->tensor_shape, this->dim);

    // Return the temporary shape array
    return temp_shape;
}

template <typename T>
T *Tensor<T>::return_value() // Returns the value of the Tensor2D as an array of template
{
    if(this->is_cuda==true) // Check if the tensor is stored on a GPU or in CPU memory
    {
        this->cpu(); // If stored on a GPU, transfer it to CPU memory
        T *value = new T[this->sum_size]; // Allocate memory for a temporary value array
        memcpy(value, this->value, this->sum_size*sizeof(T)); // Copy the value from 'this->value' to 'value'
        this->cuda(); // Transfer the tensor back to GPU memory
        return value; // Return the temporary value array
    }
    else
    {
        T *value = new T[this->sum_size]; // If stored in CPU memory, allocate memory for a temporary value array
        memcpy(value, this->value, this->sum_size*sizeof(T)); // Copy the value from 'this->value' to 'value'
        return value; // Return the temporary value array
    }
}

template <typename T>
void Tensor<T>::print()
{   
    if(this->is_cuda==true)
    {
        this->cpu();
        printf("(value: \n");
        unsigned int *check_size = new unsigned int[this->dim+1];
        for (unsigned int i = 0; i <= this->dim; i++) 
        {
            check_size[i] = 1;
        }
        for(int i=this->dim-1; i>=0; i--)
        {
            check_size[i] = this->tensor_shape[i] * check_size[i+1];
        }
        if(this->n <= 6)
        {
            for(int i=0; i<this->n; i++)
            {
                if(this->m <= 6)
                {
                    for(int j=0; j<this->m; j++)
                    {
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k]==0)
                            {
                                printf("[");
                            }
                        }
                        cout<<value[i*this->m+j];
                        if(j != this->m-1)
                        {
                            printf(", ");
                        }
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k] == check_size[k]-1)
                            {
                                printf("]");
                            }
                        }
                    }
                    printf("\n");
                }
                else
                {
                    for(int j=0; j<3; j++)
                    {
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k]==0)
                            {
                                printf("[");
                            }
                        }
                        cout<<value[i*this->m+j];
                        printf(", ");
                    }
                    printf("..... ");
                    for(int j=this->m-3; j<this->m; j++)
                    {
                        cout<<value[i*this->m+j];
                        if(j != this->m-1)
                        {
                            printf(", ");
                        }
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k] == check_size[k]-1)
                            {
                                printf("]");
                            }
                        }
                    }
                    printf("\n");
                }
            }
        }
        else
        {
            for(int i=0; i<3; i++)
            {
                if(this->m <= 6)
                {
                    for(int j=0; j<this->m; j++)
                    {
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k]==0)
                            {
                                printf("[");
                            }
                        }
                        cout<<value[i*this->m+j];
                        if(j != this->m-1)
                        {
                            printf(", ");
                        }
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k] == check_size[k]-1)
                            {
                                printf("]");
                            }
                        }
                    }
                    printf("\n");
                }
                else
                {
                    for(int j=0; j<3; j++)
                    {
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k]==0)
                            {
                                printf("[");
                            }
                        }
                        cout<<value[i*this->m+j];
                        printf(", ");
                    }
                    printf("..... ");
                    for(int j=this->m-3; j<this->m; j++)
                    {
                        cout<<value[i*this->m+j];
                        if(j != this->m-1)
                        {
                            printf(", ");
                        }
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k] == check_size[k]-1)
                            {
                                printf("]");
                            }
                        }
                    }
                    printf("\n");
                }
            }
            printf(". . .\n");
            printf(". . .\n");
            for(int i=this->n-3; i<this->n; i++)
            {
                if(this->m <= 6)
                {
                    for(int j=0; j<this->m; j++)
                    {
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k]==0)
                            {
                                printf("[");
                            }
                        }
                        cout<<value[i*this->m+j];
                        if(j != this->m-1)
                        {
                            printf(", ");
                        }
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k] == check_size[k]-1)
                            {
                                printf("]");
                            }
                        }
                    }
                    printf("\n");
                }
                else
                {
                    for(int j=0; j<3; j++)
                    {
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k]==0)
                            {
                                printf("[");
                            }
                        }
                        cout<<value[i*this->m+j];
                        printf(", ");
                    }
                    printf("....., ");
                    for(int j=this->m-3; j<this->m; j++)
                    {
                        cout<<value[i*this->m+j];
                        if(j != this->m-1)
                        {
                            printf(", ");
                        }
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k] == check_size[k]-1)
                            {
                                printf("]");
                            }
                        }
                    }
                    printf("\n");
                }
            }
        }
        printf("shape: (");
        for(int i=0; i<this->dim; i++)
        {
            printf("%d", this->tensor_shape[i]);
            if(i!=this->dim-1)
            {
                printf(", ");
            }
        }
        printf(")\n");
        this->cuda();
        cout<<"is_cuda: "<<this->is_cuda<<")\n";
    }
    else
    {
        printf("(value: \n");
        unsigned int *check_size = new unsigned int[this->dim+1];
        for (unsigned int i = 0; i <= this->dim; i++) 
        {
            check_size[i] = 1;
        }
        for(int i=this->dim-1; i>=0; i--)
        {
            check_size[i] = this->tensor_shape[i] * check_size[i+1];
        }
        if(this->n <= 6)
        {
            for(int i=0; i<this->n; i++)
            {
                if(this->m <= 6)
                {
                    for(int j=0; j<this->m; j++)
                    {
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k]==0)
                            {
                                printf("[");
                            }
                        }
                        cout<<value[i*this->m+j];
                        if(j != this->m-1)
                        {
                            printf(", ");
                        }
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k] == check_size[k]-1)
                            {
                                printf("]");
                            }
                        }
                    }
                    printf("\n");
                }
                else
                {
                    for(int j=0; j<3; j++)
                    {
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k]==0)
                            {
                                printf("[");
                            }
                        }
                        cout<<value[i*this->m+j];
                        printf(", ");
                    }
                    printf("....., ");
                    for(int j=this->m-3; j<this->m; j++)
                    {
                        cout<<value[i*this->m+j];
                        if(j != this->m-1)
                        {
                            printf(", ");
                        }
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k] == check_size[k]-1)
                            {
                                printf("]");
                            }
                        }
                    }
                    printf("\n");
                }
            }
        }
        else
        {
            for(int i=0; i<3; i++)
            {
                if(this->m <= 6)
                {
                    for(int j=0; j<this->m; j++)
                    {
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k]==0)
                            {
                                printf("[");
                            }
                        }
                        cout<<value[i*this->m+j];
                        if(j != this->m-1)
                        {
                            printf(", ");
                        }
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k] == check_size[k]-1)
                            {
                                printf("]");
                            }
                        }
                    }
                    printf("\n");
                }
                else
                {
                    for(int j=0; j<3; j++)
                    {
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k]==0)
                            {
                                printf("[");
                            }
                        }
                        cout<<value[i*this->m+j];
                        printf(", ");
                    }
                    printf("....., ");
                    for(int j=this->m-3; j<this->m; j++)
                    {
                        cout<<value[i*this->m+j];
                        if(j != this->m-1)
                        {
                            printf(", ");
                        }
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k] == check_size[k]-1)
                            {
                                printf("]");
                            }
                        }
                    }
                    printf("\n");
                }
            }
            printf(". . .\n");
            printf(". . .\n");
            for(int i=this->n-3; i<this->n; i++)
            {
                if(this->m <= 6)
                {
                    for(int j=0; j<this->m; j++)
                    {
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k]==0)
                            {
                                printf("[");
                            }
                        }
                        cout<<value[i*this->m+j];
                        if(j != this->m-1)
                        {
                            printf(", ");
                        }
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k] == check_size[k]-1)
                            {
                                printf("]");
                            }
                        }
                    }
                    printf("\n");
                }
                else
                {
                    for(int j=0; j<3; j++)
                    {
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k]==0)
                            {
                                printf("[");
                            }
                        }
                        cout<<value[i*this->m+j];
                        printf(", ");
                    }
                    printf("....., ");
                    for(int j=this->m-3; j<this->m; j++)
                    {
                        cout<<value[i*this->m+j];
                        if(j != this->m-1)
                        {
                            printf(", ");
                        }
                        for(int k=0; k<this->dim; k++)
                        {
                            if((i*this->m+j)%check_size[k] == check_size[k]-1)
                            {
                                printf("]");
                            }
                        }
                    }
                    printf("\n");
                }
            }
        }
        printf("shape: (");
        for(int i=0; i<this->dim; i++)
        {
            printf("%d", this->tensor_shape[i]);
            if(i!=this->dim-1)
            {
                printf(", ");
            }
        }
        printf(")\n");
        cout<<"is_cuda: "<<this->is_cuda<<")\n";
    }
}

template <typename T>
void Tensor<T>::cuda() // Copy the tensor data from the CPU memory to the GPU memory.
    {
        if(this->is_cuda==true)
        {
            return;
        }
        // Allocate temporary memory on the GPU to store the tensor data
        cudaMalloc((void**)&this->temp_value, sizeof(T)*this->sum_size); 
        // Copy the tensor data from the CPU memory to the temporary GPU memory
        cudaMemcpy(this->temp_value, this->value, sizeof(T)*this->sum_size, cudaMemcpyHostToDevice);
        delete this->value; // Delete the original tensor data in the CPU memory

        // Allocate memory on the GPU to store the tensor data
        cudaMalloc((void**)&this->value, sizeof(T)*this->sum_size);
        // Copy the tensor data from the temporary GPU memory to the GPU memory
        cudaMemcpy(this->value, this->temp_value, sizeof(T)*this->sum_size, cudaMemcpyDeviceToDevice);
        // Free the temporary GPU memory
        cudaFree(this->temp_value);
        // Update the flag to indicate that the tensor data is now stored on the GPU
        this->is_cuda = true;
    }

template <typename T>
void Tensor<T>::cpu() // allocate the tensor data from the GPU memory to the CPU memory.
{
    if(this->is_cuda==false)
    {
        return;
    }
    // Allocate temporary memory on the CPU to store the tensor data
    this->temp_value = new T[this->sum_size];
    // Copy the tensor data from the GPU memory to the temporary CPU memory
    cudaMemcpy(this->temp_value, this->value, sizeof(T)*this->sum_size, cudaMemcpyDeviceToHost);
    cudaFree(this->value); // Free the GPU memory

    // Allocate memory on the CPU to store the tensor data
    this->value = new T[this->sum_size];
    // Copy the tensor data from the temporary CPU memory to the CPU memory
    memcpy(this->value, this->temp_value, this->sum_size*sizeof(T));
    // Delete the temporary CPU memory
    delete this->temp_value;
    // Update the flag to indicate that the tensor data is now stored on the CPU
    this->is_cuda = false;
}

template <typename T>
void Tensor<T>::squeeze(int dim=300) // Later, change default value custom NoneType
{
    if(dim==300)
    {
        if(this->sum_size==1)
        {
            delete this->tensor_shape;
            this->dim = 1;
            this->tensor_shape = new unsigned short[this->dim];
            this->tensor_shape[0] = 1;
            return;
        }

        unsigned short *temp_tensor_shape;
        int one_dim_cnt=0;
        for(int i=0; i<this->dim; i++)
        {
            if(this->tensor_shape[i] == 1)
            {
                one_dim_cnt += 1;
            }
        }
        if(one_dim_cnt==0)
        {
            return;
        }

        temp_tensor_shape = new unsigned short[this->dim-one_dim_cnt];
        int cnt=0;
        for(int i=0; i<this->dim; i++)
        {
            if(this->tensor_shape[i] != 1)
            {
                temp_tensor_shape[cnt] = this->tensor_shape[i];
                cnt += 1;
            }
        }

        this->dim = this->dim-one_dim_cnt;
        delete this->tensor_shape;
        this->tensor_shape = new unsigned short(this->dim);

        for(int i=0; i<this->dim; i++)
        {
            this->tensor_shape[i] = temp_tensor_shape[i];
        }
        delete temp_tensor_shape;
        return;
    }
    else
    {
        if(dim < 0)
        {
            dim = this->dim+dim;
        }
        if(this->tensor_shape[dim] != 1)
        {
            throw std::invalid_argument("tensor.tensor_shape[argument] is not 1");
        }
        unsigned short *temp_tensor_shape = new unsigned short[this->dim-1];
        int cnt = 0;
        for(int i=0; i<this->dim; i++)
        {
            if(i==dim)
            {
                continue;
            }
            temp_tensor_shape[cnt] = this->tensor_shape[i];
            cnt += 1;
        }
        delete this->tensor_shape;
        this->dim = this->dim-1;
        this->tensor_shape = new unsigned short[this->dim];
        for(int i=0; i<this->dim; i++)
        {
            this->tensor_shape[i] = temp_tensor_shape[i];
        }
        delete temp_tensor_shape;
        return;
    }
}

template <typename T>
void Tensor<T>::unsqueeze(int dim)
{
    if(dim < 0)
    {
        dim = this->dim + 1 + dim;
    }
    if(dim > this->dim+1)
    {
        throw std::invalid_argument("argument > this->dim+1");
    }
    unsigned short *temp_tensor_shape = new unsigned short[this->dim+1];

    int cnt = 0;
    for(int i=0; i<this->dim+1; i++)
    {
        if(dim==i)
        {
            temp_tensor_shape[i] = 1;
        }
        else
        {
            temp_tensor_shape[i] = this->tensor_shape[cnt];
            cnt+=1;
        }
    }
    delete this->tensor_shape;
    this->dim += 1;
    this->tensor_shape = new unsigned short[this->dim];
    memcpy(this->tensor_shape, temp_tensor_shape, this->dim * sizeof(unsigned short));
    delete temp_tensor_shape;
    return;
}

template <typename T>
void Tensor<T>::reshape(short *reshape_array, int dim)
{
    int temp_reshape_sum=1;
    unsigned int reshape_sum = 1;
    char m1_check = 0; // -1 check
    char m1_index = -1;
    for(int i=0; i<dim; i++)
    {
        if(reshape_array[i] == -1)
        {
            m1_check += 1;
            m1_index = i;
        }
        temp_reshape_sum *= reshape_array[i];
    }
    if(m1_check >= 2)
    {
        throw std::runtime_error("The value '-1' can only be used once.\n");
    }
    if(temp_reshape_sum < 0) // using -1
    {
        reshape_sum = (-temp_reshape_sum);
        printf("value: %d\n", reshape_sum);
        if(this->sum_size % reshape_sum!=0)
        {
            throw std::runtime_error("The total size of the original tensor and the size of the newly defined shape must be the same.1\n");
        }
        short m1_value = this->sum_size / reshape_sum;
        reshape_array[m1_index] = m1_value;
        reshape_sum *= reshape_array[m1_index];
    }
    else
    {
        reshape_sum = temp_reshape_sum;
    }
    
    if(this->sum_size != reshape_sum)
    {
        printf("%d %d\n", this->sum_size, reshape_sum);
        throw std::runtime_error("The total size of the original tensor and the size of the newly defined shape must be the same.\n");
    }
    delete this->tensor_shape;
    this->dim = dim;
    this->tensor_shape = new unsigned short[this->dim];
    memcpy(this->tensor_shape, reshape_array, sizeof(ushort)*dim);
}