#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define N_FEATS 30

#define SIGMOID_FORWARD(x)   1/(1+exp(-x))
#define CrossEntropy(y_,y)   -(y*log(y_)+(1-y)*log(1-y_))

void  cal_gradients(float *w, float *x, float *y, float *gradients, int BATCH_SIZE);
void  update_params(float *W, float *grads, float a);
float inference(float *x, float *w);

