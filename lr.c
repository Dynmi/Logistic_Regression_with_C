#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "lr.h"


float mul(float *x, float *w)
{
    float y = 0;
    for(int i=0; i<N_FEATS; i++)
    {
        y += x[i]*w[i];
    } 
    return y;
}


void cal_gradients(float *w, float *x, float *y, float *gradients, int BATCH_SIZE)
{
    float y_mul[BATCH_SIZE];
    float y_pred[BATCH_SIZE];

    for(int p=0; p<BATCH_SIZE; p++)
    {
        y_mul[p] = mul(x+p*N_FEATS,w);
        y_pred[p] = SIGMOID_FORWARD(y_mul[p]); 
    }

    for(int i=0; i<N_FEATS; i++)
    {
        for(int p=0; p<BATCH_SIZE; p++)
        {
            gradients[i]+= (y[p]-y_mul[p]) * x[p*N_FEATS+i];
        }
        gradients[i]/=BATCH_SIZE;
    }
}


//
// W          array of weights
// grads      gradients corresponding to each w in W
// a          learning rate
//
void update_params(float *W, float *grads, float a)
{

    for(int i=0; i<N_FEATS; i++)
    {
        W[i] = W[i]-a*grads[i];
    }
}


//
// input 
//      x  feature [1,n]
//      w  weights [1,n]
// output
//      y  pred    [1]
//
float inference(float *x, float *w)
{
    float y_mul = mul(x, w);
    return SIGMOID_FORWARD(y_mul);
}

