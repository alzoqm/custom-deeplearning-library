#include "tensor.h"
#include "matOper.h"
#include "matCal.h"
#include "Linear.h"
#include "init.h"

__host__ int main()
{
    // srand(time(NULL));
    // Tensor<float> tensor1({128, 1024, 1024});
    // Tensor<float> tensor2(1, {1024, 1024});
    // Tensor<float> tensor3({128, 1024, 1024});
    // //Tensor<float> tensor3_cpu({128, 1024, 1024});
    // Tensor<float> tensor4(3, {1024});
    // // clock_t cpu_start = clock();
    // // mat_mul(tensor1, tensor2, tensor3_cpu);
    // // clock_t cpu_end = clock();
    // // double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // size_t before_free, before_total;
    // cudaMemGetInfo(&before_free, &before_total);
    // printf("Total GPU memory: %lu MB\nbefore Free GPU memory: %lu MB\n", before_total/1000000, before_free/1000000);
    
    // clock_t compile_start = clock();
    // tensor1.cuda();
    // tensor2.cuda();
    // tensor3.cuda();
    // tensor4.cuda();
    // clock_t compile_end = clock();
    // double compile_time = (double)(compile_end - compile_start) / CLOCKS_PER_SEC;

    // size_t free, total;
    // cudaMemGetInfo(&free, &total);
    // printf("after Free GPU memory: %lu MB\n", free/1000000);
    // printf("diff before after memory: %lu MB\n\n", before_free/1000000- free/1000000);

    // clock_t gpu_start = clock();
    // mat_mul(tensor1, tensor2, tensor3);
    // matadd(tensor3, tensor4, tensor3);
    // clock_t gpu_end = clock();
    // double gpu_time = (double)(gpu_end - gpu_start) / CLOCKS_PER_SEC;

    // //printf("cpu_time: %f\n", cpu_time);
    // printf("GPU compile time: %f\n", compile_time);
    // printf("GPU time: %f\n", gpu_time);
    // printf("GPU sum time: %f\n", compile_time+gpu_time);
    // printf("\n");
    // //tensor3_cpu.print();
    // tensor3.print();

    size_t before_free, before_total;
    cudaMemGetInfo(&before_free, &before_total);
    printf("Total GPU memory: %lu MB\nbefore Free GPU memory: %lu MB\n", before_total/1000000, before_free/1000000);
    clock_t create_start = clock();
    Tensor<float> *input = new Tensor<float>(2, {128, 1024, 1024});
    Linear<float> linear_1(1024, 1024, true);
    Tensor<float> *label = new Tensor<float>(2, {1024, 1024});
    clock_t create_end = clock();
    double create_time = (double)(create_end - create_start) / CLOCKS_PER_SEC;

    clock_t compile_start = clock();
    label->cuda();
    linear_1.cuda();
    input->cuda();
    clock_t compile_end = clock();
    double compile_time = (double)(compile_end - compile_start) / CLOCKS_PER_SEC;
    clock_t run_start = clock();
    Tensor<float>* out_ptr=nullptr;
    Tensor<float>* dx = nullptr;
    for(int i=0; i<1; i++)
    {
        if(out_ptr!=nullptr)
        {
            delete out_ptr;
        }
        out_ptr = linear_1.forward(input);
        if(dx!=nullptr)
        {
            delete dx;
        }
        dx = linear_1.backward(label);
    }
    clock_t run_end = clock();
    double run_time = (double)(run_end - run_start) / CLOCKS_PER_SEC;
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("\nafter Free GPU memory: %lu MB\n", free/1000000);
    printf("diff before after memory: %lu MB\n\n", before_free/1000000- free/1000000);

    printf("create time: %f\n", create_time);
    printf("compile time: %f\n", compile_time);
    printf("run time: %f\n", run_time);
    printf("sum time: %f\n", run_time + compile_time + create_time);
    //out_ptr->print();
    dx->print();
    return 0;
}