#include "tensor.cuh"
#include "matOper.cuh"

__host__ int main()
{
    srand(time(NULL));
    unsigned short size1[] = {2, 128, 256};
    unsigned short size2[] = {2, 256, 128};
    unsigned short size3[] = {2, 128, 128};

    Tensor<double> tensor1(size1, sizeof(size1) / sizeof(unsigned short));
    Tensor<double> tensor2(size2, sizeof(size2) / sizeof(unsigned short));
    Tensor<double> tensor3(size3, sizeof(size3) / sizeof(unsigned short));
    Tensor<double> tensor3_cpu(size3, sizeof(size3) / sizeof(unsigned short));

    clock_t cpu_start = clock();
    mat_mul(tensor1, tensor2, tensor3_cpu);
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    size_t before_free, before_total;
    cudaMemGetInfo(&before_free, &before_total);
    printf("Total GPU memory: %lu MB\nbefore Free GPU memory: %lu MB\n", before_total/1000000, before_free/1000000);
    
    clock_t compile_start = clock();
    tensor1.cuda();
    tensor2.cuda();
    tensor3.cuda();
    clock_t compile_end = clock();
    double compile_time = (double)(compile_end - compile_start) / CLOCKS_PER_SEC;

    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("after Free GPU memory: %lu MB\n", free/1000000);
    printf("diff before after memory: %lu MB\n\n", before_free/1000000- free/1000000);

    clock_t gpu_start = clock();
    mat_mul(tensor1, tensor2, tensor3);
    clock_t gpu_end = clock();
    double gpu_time = (double)(gpu_end - gpu_start) / CLOCKS_PER_SEC;

    printf("cpu_time: %f\n", compile_time);
    printf("GPU compile time: %f\n", compile_time);
    printf("GPU time: %f\n", gpu_time);
    printf("GPU sum time: %f\n", compile_time+gpu_time);
    printf("\n");
    tensor3_cpu.print();
    tensor3.print();

    return 0;
}
