#include"tensor.h"
#include"activation.h"
#include"matCal.h"
#include"matOper.h"

using namespace std;

template <typename T>
class Conv2D
{
private:
    uint16_t in_chans;
    uint16_t out_chans;
    uint16_t kernel_size;
    vector<uint16_t> strides;
    vector<uint16_t> padding;
    Tensor<T> *W=nullptr;
    Tensor<T> *b=nullptr;
    Tensor<T> *save_X=nullptr;
    Tensor<T> *dW=nullptr;
    Tensor<T> *db=nullptr;
    bool have_bias=false;
public:
    Conv2D(uint16_t in_chans, uint16_t out_chans, uint16_t kernel_size=1, initializer_list<uint16_t> strides={1, 1},
    initializer_list<uint16_t> padding={0, 0}, bool add_bias=false, bool is_cuda=false)
    {
        this->in_chans = in_chans;
        this->out_chans = out_chans;
        this->kernel_size = kernel_size;
        this->strides = vector<uint16_t>(strides);
        this->padding = vector<uint16_t>(padding);
        this->have_bias = add_bias;
        this->W = new Tensor<T>(1, {in_chans, out_chans, kernel_size, kernel_size}, is_cuda);
        if(add_bias)
        {
            this->b = new Tensor<T>(1, {out_chans}, is_cuda);
        }
    }

    ~Conv2D()
    {
        if(this->W!=nullptr)
        {
            delete W;
            this->W=nullptr;
        }
        if(this->b!=nullptr)
        {
            delete b;
            this->b=nullptr;
        }
        if(this->dW!=nullptr)
        {
            delete dW;
            this->dW=nullptr;
        }
        if(this->db!=nullptr)
        {
            delete db;
            this->db=nullptr;
        }
        if(this->save_X!=nullptr)
        {
            delete save_X;
            this->save_X=nullptr;
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
        return;
    }

    void print()
    {
        this->weight_print();
        printf("\n");
        if(this->have_bias != false)
        {
            this->bias_print();
        }
    }
    
    void cuda()
    {
        this->W->cuda();
        if(this->b!=nullptr)
        {
            this->b->cuda();
        }
    }

    void cpu()
    {
        this->W->cpu();
        if(this->b!=nullptr)
        {
            this->b->cpu();
        }
    }

    void forward(Tensor<T> *X, Tensor<T> *out)
    {
        if(this->save_X == nullptr)
        {
            this->save_X = new Tensor<T>({1, X->tensor_shape[X->dim-2], X->tensor_shape[X->dim-1]}, X->is_cuda);
        }
        else
        {
            this->save_X->unsqueeze(0);
        }
        sum(*X, *save_X, 0, false);
        uint16_t out_h = out->tensor_shape[1];
        uint16_t out_w = out->tensor_shape[2];
        // conv cal_code
    }

    Tensor<T> *forward(Tensor<T> *X)
    {
        vector<uint16_t> output_shape;
        output_shape.push_back(X->tensor_shape[0]);
        uint16_t out_h = ((X->tensor_shape[1] + 2*this->padding[0] - this->kernel_size) / this->strides[0]) + 1;
        uint16_t out_w = ((X->tensor_shape[2] + 2*this->padding[1] - this->kernel_size) / this->strides[1]) + 1;
        output_shape.push_back(out_h);
        output_shape.push_back(out_w);
        output_shape.push_back(X->tensor_shape[X->dim-1]);
        Tensor<T> *out = new Tensor<T>(output_shape, X->is_cuda);
        if(this->save_X == nullptr)
        {
            this->save_X = new Tensor<T>({1, X->tensor_shape[X->dim-2], X->tensor_shape[X->dim-1]}, X->is_cuda);
        }
        else
        {
            this->save_X->unsqueeze(0);
        }

        return out;
    }
};