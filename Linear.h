#ifndef _LINEAR_H_
#define _LINEAR_H_

#include "tensor.h"
#include "matOper.h"
#include "matCal.h"
#include "activation.h"

using namespace std;

template <typename T>
class Linear
{
private:
    Tensor<T> *W=nullptr;
    Tensor<T> *b=nullptr;
    Tensor<T> *dW=nullptr;
    Tensor<T> *db=nullptr;
    Tensor<T> *W_T=nullptr;
    Tensor<T> *save_X_T=nullptr;
    bool have_bias=false;
    Tensor<T> *save_X=nullptr;

public:
    Linear(uint16_t in_dim, uint16_t out_dim, bool add_bias = true, bool is_cuda=false)
    {
        this->W = new Tensor<T>(1, {in_dim, out_dim}, is_cuda); // allocate memory using new operator
        this->have_bias = add_bias;
        if(add_bias==true)
        {
            this->b = new Tensor<T>(1, {out_dim}, is_cuda); // allocate memory using new operator
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
        if(this->W_T!=nullptr)
        {
            delete W_T;
            this->W_T=nullptr;
        }
        if(this->save_X_T!=nullptr)
        {
            delete save_X_T;
            this->save_X_T=nullptr;
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

    Tensor<T>* forward(Tensor<T> *X) 
    {
        vector<uint16_t> output_shape;
        for (int i = 0; i < X->dim - 1; i++) 
        {
            output_shape.push_back(X->tensor_shape[i]);
        }
        output_shape.push_back(this->W->m);
        Tensor<T>* out = new Tensor<T>(output_shape, X->is_cuda);

        if(this->save_X==nullptr)
        {
            this->save_X = new Tensor<T>({1, X->tensor_shape[X->dim-2], X->tensor_shape[X->dim-1]}, X->is_cuda);
        }
        else
        {
            this->save_X->unsqueeze(0);
        }
        
        sum(*X, *save_X, 0, false); 
        
        mat_mul(*X, *W, *out);
        if (this->b != nullptr) 
        {
            matadd(*out, *b, *out);
        }
        //out->is_leaf = false;
        return out;
    }

    void forward(Tensor<T> *X, Tensor<T> *out) 
    {
        if(this->save_X==nullptr)
        {
            this->save_X = new Tensor<T>({1, X->tensor_shape[X->dim-2], X->tensor_shape[X->dim-1]}, X->is_cuda);
        }
        else
        {
            this->save_X->unsqueeze(0);
        }
        
        sum(*X, *save_X, 0, false); // 다른 차원 간의 연산 지원하기
        
        mat_mul(*X, *W, *out);
        
        if (this->b != nullptr) 
        {
            matadd(*out, *b, *out);
        }
    }

    Tensor<T> *backward(Tensor<T> *dout)
    {
        if(this->dW==nullptr)
        {
            vector<uint16_t> dW_shape(this->save_X->dim);
            for(int i=0; i<this->save_X->dim-2; i++)
            {
                dW_shape[i] = this->save_X->tensor_shape[i];
            }
            dW_shape[this->save_X->dim-2] = this->W->tensor_shape[0];
            dW_shape[this->save_X->dim-1] = dout->tensor_shape[dout->dim-1];
            this->dW = new Tensor<T>(dW_shape, this->save_X->is_cuda);
        }
        if(this->db==nullptr && this->have_bias==true)
        {
            vector<uint16_t> db_shape(2);
            db_shape[1] = this->b->tensor_shape[0];
            db_shape[0] = 1;
            this->db = new Tensor<T>(db_shape, this->save_X->is_cuda);
        }

        if(this->save_X_T==nullptr)
        {
            this->save_X_T = new Tensor<T>({this->save_X->tensor_shape[1], this->save_X->tensor_shape[0]}, this->save_X->is_cuda);
        }
        
        trans(*save_X, *save_X_T);
        mat_mul(*save_X_T, *dout, *dW);
        sum(*dout, *db, 0, true);


        // learning rate를 통한 학습 코드 넣기
        
        
        vector<uint16_t> dX_shape(2);
        dX_shape[0] = dout->tensor_shape[0];
        dX_shape[1] = this->W->tensor_shape[0];
        if(this->W_T==nullptr)
        {
            this->W_T = new Tensor<T>({W->tensor_shape[1], W->tensor_shape[0]}, W->is_cuda);
        }
        trans(*W, *W_T);
        Tensor<T> *dX = new Tensor<T>(dX_shape, dout->is_cuda);
        mat_mul(*dout, *W_T, *dX);

        return dX;
    }

    void backward(Tensor<T> *dout, Tensor<T> *dX)
    {
        if(this->dW==nullptr)
        {
            vector<uint16_t> dW_shape(this->save_X->dim);

            for(int i=0; i<this->save_X->dim-2; i++)
            {
                dW_shape[i] = this->save_X->tensor_shape[i];
            }
            dW_shape[this->save_X->dim-2] = this->W->tensor_shape[0];
            dW_shape[this->save_X->dim-1] = dout->tensor_shape[dout->dim-1];
            this->dW = new Tensor<T>(dW_shape, this->save_X->is_cuda);
        }
        if(this->db==nullptr && this->have_bias==true)
        {
            vector<uint16_t> db_shape(2);
            db_shape[1] = this->b->tensor_shape[0];
            db_shape[0] = 1;
            this->db = new Tensor<T>(db_shape, this->save_X->is_cuda);
        }

        if(this->save_X_T==nullptr)
        {
            this->save_X_T = new Tensor<T>({this->save_X->tensor_shape[1], this->save_X->tensor_shape[0]}, this->save_X->is_cuda);
        }
        trans(*save_X, *save_X_T);
        mat_mul(*save_X_T, *dout, *dW);

        //learning rate를 통한 학습 코드 넣기

        if(this->W_T==nullptr)
        {
            this->W_T = new Tensor<T>({W->tensor_shape[1], W->tensor_shape[0]}, W->is_cuda);
        }
        trans(*W, *W_T);
        mat_mul(*dout, *W_T, *dX);
    }
};


#endif