#ifndef __BP_GPU_H_
#define __BP_GPU_H_
#include "/usr/local/cuda-9.0/include/cuda_runtime.h"
#include "/usr/local/cuda-9.0/include/cublas_v2.h"
#include "/usr/local/cuda-9.0/include/curand.h"
#define MAXLAYER	10
#define MAXCACHEFRAME 200000
#define timeElapsed(setence)\
{\
	struct timeval timerStart, timerStop;\
	gettimeofday(&timerStart, NULL);\
	setence;\
	gettimeofday(&timerStop, NULL);\
	int timeElapsed = (timerStop.tv_sec - timerStart.tv_sec) *1000 + (timerStop.tv_usec - timerStart.tv_usec)/1000;\
	printf("%s time elapsed %d ms\n",#setence, timeElapsed);\
}
struct BP_WorkSpace
{
	float *in; 								/// input data
	float *out;								/// Output data
	float *targ;
	float *vec1;			
	float *vec2 ;	
	float *scalefactor;	    
	float *errorabsolute;
	float *errorabsolute2;
	float *realerror;
	float *newobj;
	float *weights[MAXLAYER];   	
	float *bias[MAXLAYER];      	
	float *layer_x[MAXLAYER];  	  /// Input to layer
	float *layer_y[MAXLAYER]; 		/// Output from layer
	float *layer_dedy[MAXLAYER];  /// de/dy
	float *layer_dydx[MAXLAYER];  /// dy/dx
	float *layer_dedx[MAXLAYER];  /// de/dx
	float *layer_ydedx[MAXLAYER];
	float *layer_sumdedx[MAXLAYER]; 
	float *delta_bias[MAXLAYER]; // Output bias update
	float *delta_weights[MAXLAYER]; // Output bias update
	float *DevRandVector; 
	int *DevSeed;
	
};

class BP_GPU
{
public:
	BP_GPU(int random_seed, int a_GPU_selected, int a_numlayers, int *a_layersizes, int a_bunchsize, float a_lrate, float a_momentum, float  a_weightcost,
		float **weights, float **bias,float shapefactor,int MLflag, int dropoutflat, float visible_omit, float hid_omit);
	~BP_GPU();
public:
	void train(int n_frames, float* in, const float *targ);
	void train_bunch_single(int n_frames, float* in, const float *targ);
	float CrossValid(int n_frames, const float* in, const float *targ);
	float CrossValiddB(int n_frames, const float* in, const float *targ);
	float CrossValid2(int n_frames, const float* in, const float *targ);
	void cv_bunch_single(int n_frames, const float* in, float *out);
	void returnWeights(float **weights, float **bias);    /// copy weights and biases from gpu memory to cpu memory 
	float Gamma( float x );
	int numlayers;
	int layersizes[MAXLAYER];
	int bunchsize;
	float lrate;
	float shapefactor;
	float momentum;
	float weightcost;
	int dropoutflag;
	int MLflag;
	float visible_omit;
	float hid_omit;
private:
	void devnew_vf(const char* varname, int n, float **devptr);
	void devnew_vi(const char* varname, int n, int **devptr);
	void devfree_vf(const char* varname,  float* devptr);
	void devfree_vi(const char* varname,  int* devptr);
	void todev_vf_vf(const char* varname, int n, const float* from, float* devto, cudaStream_t stream);
	void fromdev_vf_vf(const char* varname, int n, const float* devfrom, float* to, cudaStream_t stream);

	BP_WorkSpace dev;  //viaribles for devices
	int GPU_total;							//devices used num,
	int GPU_selected;				//devices selected

	cublasHandle_t handles;
	cublasHandle_t handles2;
	cudaStream_t streams;
	cudaStream_t streams2;
	curandGenerator_t gen;

};
#endif
