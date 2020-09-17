#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define N_FEATS 30

#define sigmoid(x) 1/(1+exp(-x))
#define CrossEntropy(y_true, y_pred) y_true*log(y_pred)+(1-y_true)*(log(1-y_pred)) 


float predict(float *x, float *w)
{
    float y = 0;
    for(int i=0; i<N_FEATS; i++)
    {
        y += x[i]*w[i];
    } 
    return sigmoid(y);
}


void cal_gradients(float *w, float *x, float *y, int batch_size, float *gradients)
{
    float y_pred[batch_size];
    for(int p=0; p<batch_size; p++)
    {
        y_pred[p] = predict(x+p*N_FEATS,w); 
    }

    for(int i=0; i<N_FEATS; i++)
    {
        for(int p=0; p<batch_size; p++)
        {
            gradients[i]+= *(x+p*N_FEATS+i) * (y_pred[p]-*(y+p));
        }
        gradients[i]/=batch_size;
    }
}


void update_params(float *W, float *grads, float a)
{
    /**
     * W          array of weights
     * gradients  gradients corresponding to each w in W
     * a          learning rate
     **/
    for(int i=0; i<N_FEATS; i++)
    {
        W[i] = W[i]-a*grads[i];
    }
}