#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;

typedef struct BitField{
    uint8_t a: 1;
    uint8_t b: 1;
    uint8_t c: 1;
    uint8_t d: 1;
    uint8_t e: 1;
    uint8_t f: 1;
    uint8_t g: 1;
    uint8_t h: 1;
}bit_field;

template <typename T>
class Tensor{
public:
    T *value; // Pointer to the data stored in the tensor
    T *temp_value; // Pointer to a temporary storage for the data in the tensor
    vector<uint16_t> tensor_shape; // Array of uint16_t values that represent the shape of the tensor
    uint32_t n; // Number of rows in the tensor
    uint32_t m; // Number of columns in the tensor
    int8_t dim; // int8_tacter that represents the number of dimensions of the tensor
    uint32_t sum_size; // Integer that represents the sum of the size of the tensor
    bool *mask; // Later, implement the relevant function
    bool is_cuda; // Boolean flag that indicates whether the tensor is stored on a GPU or not
    bool requeird_grad;
    bool is_leaf;
public:
    Tensor() {};
    Tensor(initializer_list<uint16_t> shape); // Constructor for the Tensor class
    Tensor(initializer_list<T> value, initializer_list<uint16_t> shape); // Constructor allocation Value to Tensor
    Tensor(T value, initializer_list<uint16_t> shape); // Constructor allocation Value to Tensor
    ~Tensor(); // Destructor for the Tensor class
    vector<uint16_t> shape(); // Returns the shape of the Tensor2D as an array of unsigned short integers
    T *return_value(); // Returns the value of the Tensor2D as an array of template
    void print();
    void cuda(); // Copy the tensor data from the CPU memory to the GPU memory.
    void cpu(); // allocate the tensor data from the GPU memory to the CPU memory.
    void squeeze(int dim);
    void unsqueeze(int dim);
    void reshape(initializer_list<int16_t> reshape_array);
};

#endif
