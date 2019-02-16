#include "DevFunc.h"
#include <stdlib.h>

__global__ void kernBinary(int n, float* in_vec, float* rand_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n)
    {
			if(in_vec[i] > rand_vec[i])
			{
				in_vec[i] = 1.0f;
			}
			else
			{
				in_vec[i] = 0.0f;
			}
		}
}
__global__ void kernWeightMultiP( int n, float p, float* in_vec )
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if( i < n ){
		in_vec[i] = in_vec[i] * p;
	}
}
__global__ void kernDropout( int n, float p, float* in, float* rand_vec )
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if( i < n ){
		if(rand_vec[i]<p){
			in[i] = 0;
		}
	}
}

__global__ void kernSigmoid(int n, float* in_vec, float* out_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n){
#ifdef RELU
	//RELU
		if(in_vec[i]>0)
			out_vec[i]=in_vec[i];
		else
			out_vec[i]=0.0f;
#else
	//sigmoid
			out_vec[i] = 1.0f/(1.0f + expf(- in_vec[i]));
#endif
	}
}

__global__ void kernDsigmoid(int n, float* in_vec, float *invec2, float* out_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i<n)
    {
#ifdef RELU
	//RELU
		if(in_vec[i]>0)
			out_vec[i]=in_vec2[i];
		else
			out_vec[i]=0.0f;
#else
	//sigmoid
		const float y = in_vec[i];
		out_vec[i] = (1.0f - y) * y * invec2[i];
#endif
    }
}

__global__ void  kernSoftmax(int rows, int cols, float* in_vec, float* out_vec)
{
    int row = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (row < rows)
    {
			int i;
			const int index = row * cols;
			const float* invec = &in_vec[index];
		  float* outvec = &out_vec[index];
			const float* inptr;
			float* outptr;
		
			// First find the max of each vector
			float max;
			
			inptr = invec;
			max = *inptr++;
			for (i=cols-1; i!=0; i--)
			{
			    float val;
		
			    val = *inptr++;
			    if (val>max)
				max = val;
			}
			// Now put exp(in-max) in out
			inptr = invec;
			outptr = outvec;
			float sumexp = 0;
			for (i=cols; i!=0; i--)
			{
			    float f, e;
			    
			    f = *inptr++;
			    e = expf(f - max);
			    *outptr++ = e;
			    sumexp += e;
			}
			// Now scale the output
			float scale = 1.0f/sumexp;
			outptr = outvec;
			for (i=cols; i!=0; i--)
			{
			    *outptr = (*outptr) * scale;
			    outptr++;
			}
    }
}

__global__ void  kernLinearOutCopy(int rows, int cols, float* in_vec, float* out_vec)
{
    int row = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (row < rows)
    {
		 int j;
	 	 for(j =0; j< cols;j++)
		 	out_vec[cols *row +j] = in_vec[cols *row +j];
		 	
    }
}

__global__ void kernMultiCopy(int vec_len, float* vec, float* mat)
{
	int nums = (vec_len + blockDim.x -1)/blockDim.x;
	int baseMat = blockIdx.x * vec_len + threadIdx.x;
	int baseVec = threadIdx.x;

	int posVec;
	int posMat;
#pragma unroll
	for(int i = 0;i< nums; ++i){
		posVec = baseVec + i * blockDim.x;
		posMat = baseMat + i * blockDim.x;
		if(posVec < vec_len){
			mat[posMat] =  vec[posVec];
		}
	}
#if 0
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (col < vec_len)
    { 
			int j;
			float val = vec[col];
			float* top = &mat[col];
			for (j=mat_height; j!=0; j--)
			{
			    *top = val;
			    top += vec_len;
			}
    }
#endif
}

__global__ void kernSumcol(int rows, int cols, float* in, float* res)
{
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (col < cols)
    {
			int j;
			const float* fromp = &in[col];
			float* top = &res[col];
			
			(*top) = (*fromp);
			fromp +=cols;
			for (j=rows-1; j!=0; j--)
			{
			    (*top) += (*fromp);
			    fromp+=cols;
			}
    }
}
__global__ void kernabsolutevalus(int n, float* in_vec, float* out_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i<n)
		out_vec[i] = fabs(in_vec[i]);
}
__global__ void kernlog(int n, float* in_vec, float* out_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i<n)
			out_vec[i] = logf(in_vec[i]);
}
__global__ void kernexp(int n, float* in_vec, float* out_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i<n)
			out_vec[i] = expf(in_vec[i]);
}

__global__ void kernindex(int n1,int n2, float* mat, float* vec,float* mat2)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i<n1)
	{
		for(int j=0;j<n2;j++)
		{
			mat2[i*n2+j] = pow(mat[i*n2+j],vec[j]);
		}
	}
}
__global__ void kernindex2(int n, float* mat, float alpha,float* mat2)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i<n)
	{
			mat2[i] = pow(mat[i],alpha);
	}
}

__global__ void kernDivideeachelement(int n, float* in_vec1, float* in_vec2,float* out_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n)
			out_vec[i] = in_vec1[i]/in_vec2[i];
}
__global__ void kernmultieachelement(int n, float* in_vec1, float* in_vec2,float* out_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n)
			out_vec[i] = in_vec1[i]*in_vec2[i];
}
__global__ void kernminuseachelement(int n, float* in_vec1, float* in_vec2,float* out_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n)
			out_vec[i] = in_vec1[i]-in_vec2[i];
}
__global__ void kernAccSumcol(int rows, int cols, float* in, float* res, float alpha, float beta)
{
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (col < cols)
    {
			int j;
			const float* fromp = &in[col];
			float* top = &res[col];
			
			(*top) = (*top) *alpha + beta *(*fromp);
			fromp +=cols;
			for (j=rows-1; j!=0; j--)
			{
			    (*top) += beta *(*fromp);
			    fromp+=cols;
			}
    }
}

__global__ void kernAccSumrow(int rows, int cols, float* in, float* res, float alpha, float beta)
{
    int row = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (row < rows)
    {
			int j;
			const float* fromp = &in[row];
			float* top = &res[row];
			
			(*top) = (*top) *alpha + beta *(*fromp);
			fromp +=rows;
			for (j= cols -1; j!=0; j--)
			{
			    (*top) += beta *(*fromp);
			    fromp += rows;
			}
    }
}

__global__ void kernVecMulNum(int n, float* in_vec, float number, float* out_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i<n)
			out_vec[i] = in_vec[i] * number;
}

__global__ void kernmatrixsum(int n, float* in_vec,  float* out_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i<n)
			in_vec[i] = in_vec[i]+out_vec[i];
}
__global__ void kernmatrixsum2(int n, float* in_vec,  float* out_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i<n)
			in_vec[i] = 0.9f*in_vec[i]+out_vec[i];
}

__global__ void kernVecMulNumpt(int n, float* in_vec, float* number, float* out_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i<n)
			out_vec[i] = in_vec[i] * (*number);
}

__global__ void kernMatAddVec(int rows, int cols, float* vec, float* mat, float beta)
{
	int row = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(row < rows){
		int j=0;
		int base = row * cols;
		for(j=0; j<cols; j++){
			mat[ base + j ] += beta * vec[j];
		}
	}
}

__global__ void kernNormRow(int rows, int cols, float* vec, float* vec_norm)
{
	int row = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(row < rows){
		int j=0;
		int base = row * cols;
		for(j=0; j<cols; j++){
			vec_norm[ row ] +=  vec[ base + j ] * vec[ base + j ];
		}
		vec_norm[ row ] = sqrt(vec_norm[ row ]);
	}
}

__global__ void kernDiagMatrixInv(int n, float* covar, float* covar_inv)
{
	int row = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(row < n){
		covar_inv[ row + row * n ] = 1.0f/covar[ row + row * n ];
	}
}

__global__ void kernVecMul(int n, float* in_vec1, float* in_vec2, float* out_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i<n)
			out_vec[i] = in_vec1[i] * in_vec2[i];
}
__global__ void kernVecDiv(int n, float *in_vec1, float *in_vec2, float *out_vec)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i<n)
			out_vec[i] = in_vec1[i] / in_vec2[i];
}

__global__ void kernSubClean( int rows , int cols, const float *in_vec1, const float *in_clean, float *res_vec)
{
	 int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	 if(i < rows)
	 {
	 	 int j;
	 	 for(j =0; j< cols;j++){
			 res_vec[cols *i + j] = (2.0f/rows)*(in_vec1[cols *i +j]-in_clean[cols *i +j]);
		 }
	 }
}
__global__ void kernSubClean2( int rows , int cols,float beta, const float *in_vec1, const float *in_clean, float *res_vec)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i < rows)
	{
		int j;
		for(j =0; j< cols;j++)
		{
			if(in_vec1[i*cols+j]>in_clean[cols *i +j])
			{
				res_vec[cols *i + j]=beta*pow((in_vec1[i*cols+j]-in_clean[cols *i +j]),(beta-1));
			}
			else if(in_vec1[i*cols+j]==in_clean[cols *i +j])
			{
				res_vec[cols *i + j]=0;
			}
			else
			{
				res_vec[cols *i + j]=-beta*pow((in_clean[cols *i +j]-in_vec1[i*cols+j]),(beta-1));
			}
		}
	}
}
__global__ void kernerror( int rows , int cols, const float *in_vec1, const float *in_clean, float *res_vec)
{
	 int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	 if(i < rows)
	 {
	 	 int j;
	 	 for(j =0; j< cols;j++){
			 res_vec[cols *i + j] = in_vec1[cols *i +j]-in_clean[cols *i +j];
		 }
	 }
}
__global__ void kernmatrixdia(int n1,int n2, float* in_vec,  float* out_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (i<n1)
	{
		int j;
		for(j=0;j<n2;j++)
		{
			if(i==j)
			out_vec[i*n2+j] = in_vec[i*n2+j];
			else
			out_vec[i*n2+j] =0;
		}
	}
}

__global__ void kernAccSum(int n, float* in, float* res, float beta)
{
	int tid = threadIdx.x;
	int nums = (n+ blockDim.x -1)/blockDim.x;
	int step = blockDim.x;
	int base = blockIdx.x * n + tid;
	int pos;
	int thres = blockIdx.x * n + n ;

#pragma unroll
	for(int i =0; i <nums; ++i){
		pos = base + step * i;
		if(pos <thres){
			res[pos] =  in[pos] + beta * res[pos];
		}
	}
}

__global__ void kernDivide(int n, float* in_vec, float* out_vec,float beta)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n)
			out_vec[i] = in_vec[i]/beta;
}
__global__ void kernconsDivide(int n, float* in_vec, float* out_vec,float beta)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n)
			out_vec[i] = beta/in_vec[i];
}
__global__ void kernfunc(int n, float* in_vec, float* out_vec)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n)
	{
		out_vec[i]=0.0f;
		for(int k=1;k>100*in_vec[i];k++)
			out_vec[i] += 1.0f/k-1.0f/(k+in_vec[i]);
		out_vec[i]=out_vec[i]*in_vec[i]*in_vec[i]-0.5772156649f*in_vec[i]*in_vec[i];
	}
}
__global__ void kernfunc2(int n1,int n2, float* in_vec,float* vec, float* out_vec,float alpha)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n1)
	{
		for(int j=0;j<n2;j++)
		{
			if(in_vec[i*n2+j]>0)
			{
				out_vec[i*n2+j]=pow(in_vec[i*n2+j],(alpha-1.0f))*alpha/pow(vec[j],alpha);
			}
			else if(in_vec[i*n2+j]==0)
			{
				out_vec[i*n2+j]=0;
			}
			else
			{
				out_vec[i*n2+j]=-pow((-in_vec[i*n2+j]),(alpha-1.0f))*alpha/pow(vec[j],alpha);
			}
		}
	}
}
__global__ void kernUpdatedelta(int cols, float* delta, float* weights, float* gradient, int n, float momentum, float lr, float weightcost)
{
	int tid = threadIdx.x;
	int nums = (blockDim.x -1 + cols)/blockDim.x;
	int base = blockIdx.x * cols + tid;
	int step = blockDim.x;
	int pos;
#pragma unroll
	for(int i=0; i< nums; i++ )
	{
		pos = base + i*step;
		if(tid + i*step <cols){
            delta[pos] = momentum * delta[pos] - lr*(gradient[pos] / n + weightcost * weights[pos]);
			//delta[pos] = momentum * delta[pos] - (1-momentum)*lr*(gradient[pos] / n + weightcost * weights[pos]); //dropout 
		}

	}
}
#if 0
__global__ void kernUpdatedelta(int size, float* delta, float* weights, float* gradient, int n, float momentum, float lr, float weightcost)

    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size)
			//delta[i] = momentum * delta[i] - lr * (gradient[i] / n + weightcost * weights[i]);
			delta[i] = momentum * delta[i] - (1-momentum)* lr * (gradient[i] / n + weightcost * weights[i]);//dropout
}
#endif