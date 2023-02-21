#include "tensor.h"
#include "matOper.h"
#include "matCal.h"

using namespace std;

template <typename T>
class Linear
{
private:
    Tensor<T> *W=nullptr;
    Tensor<T> *b=nullptr;
public:
    Tensor<T> *save_X=nullptr;
public:
    Linear(unsigned short *weight_shape, bool add_bias=true)
    {
        this->W = new Tensor<T>(1, weight_shape, 2);
        if(add_bias==true)
        {
            unsigned short *bias_shape = new unsigned short[1];
            bias_shape[0] = weight_shape[1];
            this->b = new Tensor<T>(2, bias_shape, 1);
        }
    }

    void weight_print()
    {
        printf("Weight: \n");
        this->W->print();
    }

    void bias_print()
    {
        if(this->b!=nullptr)
        {
            printf("Bias: \n");
            this->b->print();
        }
    }

    void print()
    {
        this->weight_print();
        printf("\n");
        this->bias_print();
    }
    
    void cuda()
    {
        this->W->cuda();
        if(this->b!=nullptr)
        {
            this->b->cuda();
        }
        if(this->out!=nullptr)
        {
            this->out->cuda();
        }
    }

    void cpu()
    {
        this->W->cpu();
        if(this->b!=nullptr)
        {
            this->b->cpu();
        }
        if(this->out!=nullptr)
        {
            this->out->cpu();
        }
    }
};