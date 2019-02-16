#include <stdio.h>
#include <stdlib.h>
#include "/usr/local/cuda-9.0/include/cublas_v2.h"

static const int CUDA_MAXBLOCKS = 65535;
static const int NTHREADS = 256;
static const int BASICSIZE = 32;

__global__ void kernSigmoid(int n, float* in_vec, float* out_vec);
__global__ void kernBinary(int n, float* in_vec, float* rand_vec);
__global__ void kernMultiCopy(int vec_len, float* vec, float* mat);
__global__ void kernSumcol(int rows, int cols, float* in, float* res);
__global__ void kernAccSumcol(int rows, int cols, float* in, float* res, float alpha, float beta);
__global__ void kernAccSumrow(int rows, int cols, float* in, float* res, float alpha, float beta);
__global__ void kernSoftmax(int rows, int cols, float *in_vec, float* out_vec); //kernLinearOutCopy
__global__ void kernLinearOutCopy(int rows, int cols, float *in_vec, float* out_vec);
__global__ void kernDsigmoid(int n, float* in_vec, float *invec2, float* out_vec);
__global__ void kernSubClean2( int rows , int cols,float beta, const float *in_vec1, const float *in_clean, float *res_vec);
__global__ void kernerror( int rows , int cols, const float *in_vec1, const float *in_clean, float *res_vec);

__global__ void kernVecMulNum(int n, float* in_vec, float number, float* out_vec);
__global__ void kernVecMulNumpt(int n, float* in_vec, float* number, float* out_vec);
__global__ void kernMatAddVec(int rows, int cols, float* vec, float* mat, float beta);
__global__ void kernNormRow(int rows, int cols, float* vec, float* vec_norm);
__global__ void kernDiagMatrixInv(int n, float* covar, float* covar_inv);
__global__ void kernmatrixsum(int n, float* in_vec,  float* out_vec);
__global__ void kernmatrixsum2(int n, float* in_vec,  float* out_vec);
__global__ void kernmatrixdia(int n1,int n2, float* in_vec,  float* out_vec);
__global__ void kernindex2(int n, float* mat, float alpha,float* mat2);
__global__ void kernVecMul(int n, float *in_vec1, float *in_vec2, float *res_vec);

__global__ void kernVecDiv(int n, float *in_vec1, float *in_vec2, float *res_vec);
__global__ void kernSubClean( int rows , int cols, const float *in_vec1, const float *in_clean, float *res_vec);
__global__ void kernAccSum(int n, float* in, float* res, float beta);
__global__ void kernDivide(int n, float* in_vec, float* out_vec,float beta);
__global__ void kernUpdatedelta(int size, float* delta, float* weights, float* gradient, int n, float momentum, float lr, float weightcost);
__global__ void kernWeightMultiP( int n, float p, float* in_vec );
__global__ void kernDropout( int n, float p, float* in, float* rand_vec );
__global__ void kernabsolutevalus(int n, float* in_vec, float* out_vec);
__global__ void kernlog(int n, float* in_vec, float* out_vec);
__global__ void kernindex(int n1,int n2, float* mat, float* vec,float* mat2);
__global__ void kernDivideeachelement(int n, float* in_vec1, float* in_vec2,float* out_vec);
__global__ void kernmultieachelement(int n, float* in_vec1, float* in_vec2,float* out_vec);
__global__ void kernconsDivide(int n, float* in_vec, float* out_vec,float beta);
__global__ void kernfunc(int n, float* in_vec, float* out_vec);
__global__ void kernminuseachelement(int n, float* in_vec1, float* in_vec2,float* out_vec);
__global__ void kernfunc2(int n1,int n2, float* in_vec,float* vec, float* out_vec,float alpha);
__global__ void kernexp(int n, float* in_vec, float* out_vec);
inline void SgemmTN(cublasHandle_t handle,int m, int k,
			int n, const float* A, const float* B, float* C, 
			const float alpha, const float beta)
{		
    cublasStatus_t e =cublasSgemm(handle,CUBLAS_OP_T, CUBLAS_OP_N,
		m, n, k, &beta, (float*)A, k, (float*) B, k, &alpha, C, m);
		if(e != CUBLAS_STATUS_SUCCESS)
		{
			printf("%d,%d,%d...........SgemmTN wrong\n",m,k,n);
		}
		if(e == CUBLAS_STATUS_EXECUTION_FAILED)
		{
			printf("...........1\n");
		}
}

inline void SgemmNN(cublasHandle_t handle,int m, int k,
			int n, const float* A,const float* B, float* C,
			const float alpha, const float beta)
{
    cublasStatus_t e =cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,
		m, n, k, &beta, (float*)A, m, (float*) B, k, &alpha, C, m);
		if(e != CUBLAS_STATUS_SUCCESS)
		{
			printf("...........SgemmNN wrong\n");
		}
}

inline void SgemmNT(cublasHandle_t handle,int m, int k,
			int n, const float* A,
			const float* B, float* C, const float alpha, const float beta)
{
    cublasStatus_t e =cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_T,
		m, n, k, &beta, (float*)A, m, (float*) B, n, &alpha, C, m);
		if(e != CUBLAS_STATUS_SUCCESS)
		{
			printf("...........SgemmNT wrong\n");
		}
}

inline void DevWeightMultiP(cudaStream_t stream, int n, float p, float* in_vec)
{
	int nblocks = (n + NTHREADS-1)/NTHREADS;
	if (nblocks > CUDA_MAXBLOCKS)
			printf("DevWeightMultiP: nblocks too large\n");
    kernWeightMultiP<<<nblocks,NTHREADS,0,stream>>>(n, p, in_vec);
}

inline void DevDropout(cudaStream_t stream, int n, float p, float* in_vec, float* rand_vec)
{
	int nblocks = (n + NTHREADS-1)/NTHREADS;
	if (nblocks > CUDA_MAXBLOCKS)
			printf("DevDropout: nblocks too large\n");
    kernDropout<<<nblocks,NTHREADS,0,stream>>>(n, p, in_vec, rand_vec);
}

inline void DevSigmoid(cudaStream_t stream, int n, float* in_vec, float* out_vec)
{
    int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("DevSigmoid: nblocks too large\n");
    kernSigmoid<<<nblocks,NTHREADS,0,stream>>>(n, in_vec, out_vec);
}

inline void DevDsigmoid(cudaStream_t stream, int n, float* in_vec, float* in_vec2, float* out_vec)
{
    int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks> CUDA_MAXBLOCKS)
				printf("DevDsigmoid: nblocks too large\n");
    kernDsigmoid<<<nblocks,NTHREADS,0,stream>>>(n, in_vec, in_vec2, out_vec);
}

inline void DevSoftmax(cudaStream_t stream, int rows, int cols, float* in_vecs, float* out_vecs)
{
    int nblocks = (rows + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("DevSoftmax: nblocks too large\n");
    kernSoftmax<<<nblocks, NTHREADS,0,stream>>>(rows, cols, in_vecs, out_vecs);
}

inline void DevLinearOutCopy(cudaStream_t stream, int rows, int cols, float* in_vecs, float* out_vecs)
{
    int nblocks = (rows + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("DevLinearOutCopy: nblocks too large\n");
    kernLinearOutCopy<<<nblocks, NTHREADS,0,stream>>>(rows, cols, in_vecs, out_vecs);
}

inline void DevMultiCopy(cudaStream_t stream,int mat_height, int vec_len,
		  float* vec, float* mat)
{
	kernMultiCopy<<<mat_height, NTHREADS>>>(vec_len, vec, mat);
}

inline void DevSumcol(cudaStream_t stream,int rows, int cols, float* in, float* res)
{
    int nblocks = (cols + NTHREADS-1)/NTHREADS;
    if (nblocks>CUDA_MAXBLOCKS)
			printf("DevSumcol: nblocks too large\n");
    kernSumcol<<<nblocks, NTHREADS,0,stream>>>(rows, cols, in, res);
}
inline void Devabsolutevalus(cudaStream_t stream, int n, float *in_vec, float *res_vec)
{
    int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("Devabsolutevalus: nblocks too large\n");
    kernabsolutevalus<<<nblocks,NTHREADS,0,stream>>>(n, in_vec, res_vec);
}
inline void Devlog(cudaStream_t stream, int n, float *in_vec, float *res_vec)
{
    int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("Devlog: nblocks too large\n");
    kernlog<<<nblocks,NTHREADS,0,stream>>>(n, in_vec, res_vec);
}
inline void Devexp(cudaStream_t stream, int n, float *in_vec, float *res_vec)
{
    int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("Devexp: nblocks too large\n");
    kernexp<<<nblocks,NTHREADS,0,stream>>>(n, in_vec, res_vec);
}
inline void Devindex(cudaStream_t stream, int n1,int n2, float *mat, float *vec,float* mat2)
{
    int nblocks = (n1 + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("Devindex: nblocks too large\n");
    kernindex<<<nblocks,NTHREADS,0,stream>>>(n1, n2, mat,vec,mat2);
}
inline void Devindex2(cudaStream_t stream, int n, float *mat, float alpha,float* mat2)
{
    int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("Devindex2: nblocks too large\n");
    kernindex2<<<nblocks,NTHREADS,0,stream>>>(n, mat,alpha,mat2);
}

inline void DevAccSumcol(cudaStream_t stream,int rows, int cols, float* in, float* res, float alpha, float beta)
{
    int nblocks = (cols + NTHREADS-1)/NTHREADS;
    if (nblocks>CUDA_MAXBLOCKS)
			printf("DevSumcol: nblocks too large\n");
    kernAccSumcol<<<nblocks, NTHREADS,0,stream>>>(rows, cols, in, res, alpha, beta);
}
inline void Devmatrixdia(cudaStream_t stream,int n1,int n2, float* in_vec,  float* out_vec)
{
	int nblocks = (n1 + NTHREADS-1)/NTHREADS;
    if (nblocks>CUDA_MAXBLOCKS)
			printf("Devmatirxdia: nblocks too large\n");
    kernmatrixdia<<<nblocks, NTHREADS,0,stream>>>(n1, n2, in_vec,out_vec);
}

inline void DevAccSumrow(cudaStream_t stream,int rows, int cols, float* in, float* res, float alpha, float beta)
{
    int nblocks = (rows + NTHREADS-1)/NTHREADS;
    if (nblocks>CUDA_MAXBLOCKS)
			printf("DevSumrow: nblocks too large\n");
    kernAccSumrow<<<nblocks, NTHREADS,0,stream>>>(rows, cols, in, res, alpha, beta);
}
inline void Devmatrixsum(cudaStream_t stream,int n, float* in_vec,  float* out_vec)
{
    int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks>CUDA_MAXBLOCKS)
			printf("Devmatrixsum: nblocks too large\n");
    kernmatrixsum<<<nblocks, NTHREADS,0,stream>>>(n, in_vec, out_vec);
}

inline void Devmatrixsum2(cudaStream_t stream,int n, float* in_vec,  float* out_vec)
{
    int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks>CUDA_MAXBLOCKS)
			printf("Devmatrixsum2: nblocks too large\n");
    kernmatrixsum2<<<nblocks, NTHREADS,0,stream>>>(n, in_vec, out_vec);
}


inline void DevAccSum(cudaStream_t stream, int n, int m, float* in, float* res, float beta)
{
	kernAccSum<<<m, NTHREADS,0,stream>>>(n, in, res,  beta);
}

inline void DevVecMul(cudaStream_t stream, int n, float *in_vec1, float *in_vec2, float *res_vec)
{
    int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("DevVecMul: nblocks too large\n");
    kernVecMul<<<nblocks,NTHREADS,0,stream>>>(n, in_vec1, in_vec2, res_vec);
}
inline void DevVecDiv(cudaStream_t stream, int n, float *in_vec1, float *in_vec2, float *res_vec)
{
    int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("DevVecMul: nblocks too large\n");
    kernVecDiv<<<nblocks,NTHREADS,0,stream>>>(n, in_vec1, in_vec2, res_vec);
}
inline void Devfunc(cudaStream_t stream, int n, float* in_vec, float* out_vec)
{
    int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("Devfunc: nblocks too large\n");
    kernfunc<<<nblocks,NTHREADS,0,stream>>>(n, in_vec, out_vec);
}
inline void Devfunc2(cudaStream_t stream, int n1,int n2, float* in_vec,float* vec, float* out_vec,float alpha)
{
    int nblocks = (n1 + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("Devfunc2: nblocks too large\n");
    kernfunc2<<<nblocks,NTHREADS,0,stream>>>(n1, n2,in_vec,vec, out_vec,alpha);
}
inline void DevVecMulNum(cudaStream_t stream, int n, float *in_vec, float num, float *res_vec)
{
    int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("DevVecMulNum: nblocks too large\n");
    kernVecMulNum<<<nblocks,NTHREADS,0,stream>>>(n, in_vec, num, res_vec);
}
inline void DevVecMulNumpt(cudaStream_t stream, int n, float *in_vec, float *num, float *res_vec)
{
    int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("DevVecMulNumpt: nblocks too large\n");
    kernVecMulNumpt<<<nblocks,NTHREADS,0,stream>>>(n, in_vec, num, res_vec);
}

inline void DevMatAddVec(cudaStream_t stream, int rows, int cols, float* vec, float* mat, float beta)
{
	int nblocks = (rows + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("DevVecMulNum: nblocks too large\n");
    kernMatAddVec<<<nblocks,NTHREADS,0,stream>>>(rows, cols, vec, mat, beta);
}

inline void DevNormRow(cudaStream_t stream, int rows, int cols, float* vec, float* vec_norm)
{
	int nblocks = (rows + NTHREADS-1)/NTHREADS;
	if (nblocks > CUDA_MAXBLOCKS)
			printf("DevVecMulNum: nblocks too large\n");
    kernNormRow<<<nblocks,NTHREADS,0,stream>>>(rows, cols, vec, vec_norm);
}
inline void DevDiagMatrixInv(cudaStream_t stream, int n, float* covar, float* covar_inv)
{
	int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("DevVecMulNum: nblocks too large\n");
    kernDiagMatrixInv<<<nblocks,NTHREADS,0,stream>>>(n, covar, covar_inv);
}

inline void DevSubClean(cudaStream_t stream, int rows , int cols, const float *in_vec1, const float *in_clean, float *res_vec)
{
	 int nblocks = (rows + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("DevSubClean: nblocks too large\n");
	kernSubClean<<<nblocks,NTHREADS,0,stream>>>( rows, cols, in_vec1, in_clean, res_vec);
}
inline void DevSubClean2(cudaStream_t stream, int rows , int cols, float beta,const float *in_vec1, const float *in_clean, float *res_vec)
{
	 int nblocks = (rows + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("DevSubClean2: nblocks too large\n");
	kernSubClean2<<<nblocks,NTHREADS,0,stream>>>( rows, cols,beta, in_vec1, in_clean, res_vec);
}

inline void Deverror(cudaStream_t stream, int rows , int cols, const float *in_vec1, const float *in_clean, float *res_vec)
{
	 int nblocks = (rows + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("Deverror: nblocks too large\n");
	kernerror<<<nblocks,NTHREADS,0,stream>>>( rows, cols, in_vec1, in_clean, res_vec);
}
inline void DevDivide(cudaStream_t stream, int n, float* in_vec, float* out_vec,float beta)
{
	 int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("DevDevide: nblocks too large\n");
    kernDivide<<<nblocks,NTHREADS,0,stream>>>( n, in_vec, out_vec, beta);
}
inline void DevconsDivide(cudaStream_t stream, int n, float* in_vec, float* out_vec,float beta)
{
	 int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("DevconsDevide: nblocks too large\n");
    kernconsDivide<<<nblocks,NTHREADS,0,stream>>>( n, in_vec, out_vec, beta);
}
inline void DevDivideeachelement(cudaStream_t stream, int n, float* in_vec1, float* in_vec2,float* out_vec)
{
	 int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("DevDevideeachelement: nblocks too large\n");
    kernDivideeachelement<<<nblocks,NTHREADS,0,stream>>>( n, in_vec1,in_vec2, out_vec);
}
inline void Devmultieachelement(cudaStream_t stream, int n, float* in_vec1, float* in_vec2,float* out_vec)
{
	 int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("Devmultieachelement: nblocks too large\n");
    kernmultieachelement<<<nblocks,NTHREADS,0,stream>>>( n, in_vec1,in_vec2, out_vec);
}
inline void Devminuseachelement(cudaStream_t stream, int n, float* in_vec1, float* in_vec2,float* out_vec)
{
	 int nblocks = (n + NTHREADS-1)/NTHREADS;
    if (nblocks > CUDA_MAXBLOCKS)
			printf("Devminuseachelement: nblocks too large\n");
    kernminuseachelement<<<nblocks,NTHREADS,0,stream>>>( n, in_vec1,in_vec2, out_vec);
}

inline void updatedelta(cudaStream_t stream, int rows, int cols, float* delta, float* weights, float* gradient, int n, float momentum, float lr, float weightcost)
{
	kernUpdatedelta<<<rows,NTHREADS,0,stream>>>( cols, delta, weights, gradient, n, momentum, lr, weightcost);
}