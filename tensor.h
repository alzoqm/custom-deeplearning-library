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
    Tensor(unsigned short *shape, char dim); // Constructor for the Tensor class
    Tensor(T *value, unsigned short *shape, char dim); // Constructor allocation Value to Tensor
    Tensor(T value, unsigned short *shape, char dim); // Constructor allocation Value to Tensor
    ~Tensor(); // Destructor for the Tensor class
    unsigned short *shape(); // Returns the shape of the Tensor2D as an array of unsigned short integers
    T *return_value(); // Returns the value of the Tensor2D as an array of template
    void print();
    void cuda(); // Copy the tensor data from the CPU memory to the GPU memory.
    void cpu(); // allocate the tensor data from the GPU memory to the CPU memory.
    void squeeze(int dim);
    void unsqueeze(int dim);
};

#endif
