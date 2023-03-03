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
    Tensor<T> *dW=nullptr;
    Tensor<T> *db=nullptr;
    bool have_bias=false;
public:
    Tensor<T> *save_X=nullptr;
public:
    Linear(uint16_t in_dim, uint16_t out_dim, bool add_bias = true)
    {
        this->W = new Tensor<T>(1, {in_dim, out_dim}); // allocate memory using new operator
        if(add_bias==true)
        {
            this->have_bias = true;
            this->b = new Tensor<T>(1, {out_dim}); // allocate memory using new operator
        }
        else
        {
            this->have_bias = false;
        }
    }

    ~Linear()
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

    Tensor<T> forward(Tensor<T> X)
    {
        vector<uint16_t> output_shape;
        for(int i=0; i<X.dim-1; i++)
        {
            output_shape.push_back(X.tensor_shape[i]);
        }
        output_shape.push_back(this->W->m);
        Tensor<T> out(output_shape);
        if(X.is_cuda == true)
        {
            out.cuda();
        }

        this->save_X = new Tensor<T>(X);
        
        mat_mul(X, *W, out);
        if(this->b!=nullptr)
        {
            matadd(out, *b, out);
        }
        out.is_leaf=false;
        return out;
    }
};