#include "tensor.h"
#include "matOper.h"
#include "matCal.h"
#include "activation.h"
#include "Linear.h"
#include "init.h"

__host__ int main()
{
    size_t before_free, before_total;
    cudaMemGetInfo(&before_free, &before_total);
    printf("Total GPU memory: %lu MB\nbefore Free GPU memory: %lu MB\n", before_total/1000000, before_free/1000000);
    clock_t create_start = clock();
    Tensor<float> *input = new Tensor<float>(1, {4, 512, 256}, true);
    Linear<float> linear_1(256, 128, true, true);
    Tensor<float>* output= new Tensor<float>({4, 512, 128}, true);

    ReLU<float> relu;

    Tensor<float> *dout = new Tensor<float>(1, {512, 128}, true);
    Tensor<float>* dx = new Tensor<float>({512, 256}, true);
    clock_t create_end = clock();
    double create_time = (double)(create_end - create_start) / CLOCKS_PER_SEC;

    clock_t run_start = clock();

    for(int i=0; i<1; i++)
    {

        linear_1.forward(input, output);
        relu.forward(output);
        relu.backward(dout);
        linear_1.backward(dout, dx);
    }
    clock_t run_end = clock();
    double run_time = (double)(run_end - run_start) / CLOCKS_PER_SEC;
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("\nafter Free GPU memory: %lu MB\n", free/1000000);
    printf("diff before after memory: %lu MB\n\n", before_free/1000000- free/1000000);

    printf("create time: %f\n", create_time);
    printf("run time: %f\n", run_time);
    printf("sum time: %f\n", run_time + create_time);
    // output->print();
    // dx->print();

    return 0;
}