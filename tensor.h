#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <cuda_runtime.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

using namespace std;

template <typename T>
class Tensor{
public:
    T *value; // Pointer to the data stored in the tensor
    T *temp_value; // Pointer to a temporary storage for the data in the tensor
    unsigned short *tensor_shape; // Array of unsigned short values that represent the shape of the tensor
    unsigned int n; // Number of rows in the tensor
    unsigned int m; // Number of columns in the tensor
    char dim; // Character that represents the number of dimensions of the tensor
    unsigned int sum_size; // Integer that represents the sum of the size of the tensor
    bool *mask; // bitfiled로 교체 필요
    bool is_cuda; // Boolean flag that indicates whether the tensor is stored on a GPU or not
public:
    Tensor(unsigned short *shape, char dim) // Constructor for the Tensor class
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

    Tensor(T *value, unsigned short *shape, char dim) // Constructor allocation Value to Tensor
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

    Tensor(T value, unsigned short *shape, char dim) // Constructor allocation Value to Tensor
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

    ~Tensor() // Destructor for the Tensor class
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

    unsigned short *shape() // Returns the shape of the Tensor2D as an array of unsigned short integers
    {
        // Allocate memory for a temporary shape array
        unsigned short *temp_shape;
        temp_shape = new unsigned short[this->dim];

        // Copy the shape from 'tensor_shape' to 'temp_shape'
        memcpy(temp_shape, this->tensor_shape, this->dim);

        // Return the temporary shape array
        return temp_shape;
    }

    T *return_value() // Returns the value of the Tensor2D as an array of template
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

    void print()
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

    void cuda() // Copy the tensor data from the CPU memory to the GPU memory.
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

    void cpu() // Copys the tensor data from the GPU memory to the CPU memory.
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
};

#endif
