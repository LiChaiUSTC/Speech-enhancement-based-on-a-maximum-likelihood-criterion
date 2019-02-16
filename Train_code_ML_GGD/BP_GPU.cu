#include <stdio.h>
#include <stdlib.h>
#include<math.h>
#include <sys/time.h>
#include "BP_GPU.h"
#include "DevFunc.h"
#define THREUSEMULTIGPU 256

BP_GPU::BP_GPU(int random_seed, int a_GPU_selected, int a_numlayers, int *a_layersizes, int a_bunchsize, float a_lrate, float a_momentum,  
	float a_weightcost,float **weights, float **bias,float a_shapefactor, int a_MLflag,int a_dropoutflag,float a_visible_omit, float a_hid_omit)
	:GPU_selected(a_GPU_selected),numlayers(a_numlayers),bunchsize(a_bunchsize),momentum(a_momentum),lrate(a_lrate),weightcost(a_weightcost),shapefactor(a_shapefactor),MLflag(a_MLflag), dropoutflag(a_dropoutflag), visible_omit(a_visible_omit),hid_omit(a_hid_omit)
{
	int maxlayersize = 0;
	//// set GPU num
	cudaGetDeviceCount(&GPU_total);
	printf("Total GPU Device : %d\n",GPU_total);
	if(GPU_selected > GPU_total || GPU_selected < 0)
	{
		printf("GPU Num %d Not In Range %d-%d\n",GPU_selected,0,GPU_total-1);
		exit(0);
	}
	cudaSetDevice(GPU_selected);
	printf("Use GPU Device : %d\n",GPU_selected);
	
	cudaError_t er;
	curandStatus_t eg;
		er = cudaSetDevice(GPU_selected);
		if (er!=cudaSuccess)
			printf("cudaSetDevice(%d) failed\n",GPU_selected);
	
		er =cudaStreamCreate(&(streams));
		if (er!=cudaSuccess)
			printf("cudaStreamCreate 1 failed\n");
		er =cudaStreamCreate(&(streams2));
		if (er!=cudaSuccess)
			printf("cudaStreamCreate 2 failed\n");

		cublasStatus_t eb = cublasCreate(&handles);
		if (eb!=CUBLAS_STATUS_SUCCESS)
			printf("cublasCreate(%d) failed\n",1);
		eb = cublasCreate(&handles2);
		if (eb!=CUBLAS_STATUS_SUCCESS)
			printf("cublasCreate(%d) failed\n",2);

		eb = cublasSetStream(handles,streams);
		if (eb!=CUBLAS_STATUS_SUCCESS)
			printf("cublasSetStream(handles[%d],streams[%d]) failed\n",1,1);
		eb = cublasSetStream(handles2,streams2);
		if (eb!=CUBLAS_STATUS_SUCCESS)
			printf("cublasSetStream(handles[%d],streams[%d]) failed\n",2,2);

		eg = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
		if (eg!=CURAND_STATUS_SUCCESS)
			printf("curandCreateGenerator failed\n");
		eg = curandSetStream(gen, streams);
		if (eg!=CURAND_STATUS_SUCCESS)
			printf("curandSetStream failed\n");

		srand(random_seed);
		curandSetPseudoRandomGeneratorSeed(gen, rand());

	//// Alloc device Memory
	int i;
	for(i =0; i < numlayers;i++)
	{
		layersizes[i] = a_layersizes[i];
		if( maxlayersize < layersizes[i])
		{
			maxlayersize = layersizes[i];
		}
	}
    devnew_vf("in", MAXCACHEFRAME *layersizes[0], &(dev.in));
    devnew_vf("out", bunchsize *layersizes[numlayers -1], &(dev.out));
	devnew_vf("targ", MAXCACHEFRAME*layersizes[numlayers -1], &(dev.targ));

    devnew_vf("realerror", bunchsize *layersizes[numlayers -1], &(dev.realerror));
    devnew_vf("errorabsolute", bunchsize *layersizes[numlayers -1], &(dev.errorabsolute));
    devnew_vf("errorabsolute2", bunchsize *layersizes[numlayers -1], &(dev.errorabsolute2));
    devnew_vf("newobj", bunchsize *layersizes[numlayers -1], &(dev.newobj));

	devnew_vf("vec1", layersizes[numlayers -1], &(dev.vec1));	
	devnew_vf("vec2", layersizes[numlayers -1], &(dev.vec2));
	devnew_vf("scalefactor", layersizes[numlayers -1], &(dev.scalefactor));
	
	devnew_vf("DevRandVector", maxlayersize * bunchsize, &(dev.DevRandVector));
	devnew_vi("DevSeed", BASICSIZE, &(dev.DevSeed));
    for (i = 1; i< numlayers; i++)
    {
		devnew_vf("bias", layersizes[i], &(dev.bias[i]));
		devnew_vf("weights", layersizes[i] *layersizes[i-1], &(dev.weights[i]));
		devnew_vf("delta_bias", layersizes[i], &(dev.delta_bias[i]));
		devnew_vf("delta_weights", layersizes[i] *layersizes[i-1], &(dev.delta_weights[i]));
		devnew_vf("layer_y", bunchsize *layersizes[i], &(dev.layer_y[i]));
		devnew_vf("layer_x", bunchsize *layersizes[i], &(dev.layer_x[i]));
		devnew_vf("layer_dedy", bunchsize *layersizes[i], &(dev.layer_dedy[i]));
		devnew_vf("layer_dydx", bunchsize *layersizes[i], &(dev.layer_dydx[i]));
		devnew_vf("layer_dedx", bunchsize *layersizes[i], &(dev.layer_dedx[i]));
		devnew_vf("layer_ydedx", layersizes[i] *layersizes[i-1], &(dev.layer_ydedx[i]));
		devnew_vf("layer_sumdedx", layersizes[i], &(dev.layer_sumdedx[i]));
    }

	cudaSetDevice(GPU_selected);
		  
		for(i = 1; i< numlayers; i++)
		{
			todev_vf_vf("weights", layersizes[i-1] *layersizes[i], weights[i], dev.weights[i], streams);
			todev_vf_vf("bias", layersizes[i], bias[i], dev.bias[i], streams);
		}
	
	printf("Created net with %d layers, bunchsize %d.\n", numlayers, bunchsize);
}

BP_GPU::~BP_GPU()
{
	cudaSetDevice(GPU_selected);
	
	devfree_vf("in", dev.in);
	devfree_vf("out", dev.out);
	devfree_vf("targ", dev.targ);
	devfree_vf("DevRandVector", dev.DevRandVector);
	devfree_vi("DevSeed", dev.DevSeed);

	devfree_vf("realerror", dev.realerror);
	devfree_vf("errorabsolute", dev.errorabsolute);
	devfree_vf("errorabsolute2", dev.errorabsolute2);

	devfree_vf("vec1", dev.vec1);
	devfree_vf("vec2", dev.vec2);
	devfree_vf("newobj", dev.newobj);
	devfree_vf("scalefactor", dev.scalefactor);

	int i;
	for (i = 1; i< numlayers; i++)
	{
		devfree_vf("weights", dev.weights[i]);
		devfree_vf("bias", dev.bias[i]);
		devfree_vf("delta_weights", dev.delta_weights[i]);
		devfree_vf("delta_bias", dev.delta_bias[i]);
		devfree_vf("layer_x", dev.layer_x[i]);
		devfree_vf("layer_y", dev.layer_y[i]);
		devfree_vf("layer_dedx", dev.layer_dedx[i]);
		devfree_vf("layer_dydx", dev.layer_dydx[i]);
		devfree_vf("layer_dedy", dev.layer_dedy[i]);
		devfree_vf("layer_ydedx", dev.layer_ydedx[i]);
		devfree_vf("layer_sumdedx", dev.layer_sumdedx[i]);
	}
	cublasDestroy(handles);
	cudaStreamDestroy(streams);
	curandDestroyGenerator(gen);
}

void BP_GPU::train(int n_frames, float* in, const float *targ)
{
	int frames_this_bunch; // Number of frames to handle this bunch
	int n_input = layersizes[0];
	int out_dims= layersizes[numlayers-1];
	
	float *realin;
	float *realtarg;
	
	cudaSetDevice(GPU_selected);

	todev_vf_vf("in",n_frames * n_input, in, dev.in, streams);
	todev_vf_vf("targ", n_frames * out_dims, targ, dev.targ,streams);

	realin = dev.in;
	realtarg = dev.targ;    

	int i;
	for (i=0; i< n_frames; i+= bunchsize)
	{
		frames_this_bunch = (bunchsize > n_frames - i)?(n_frames - i):bunchsize;
		if(frames_this_bunch == bunchsize)
		{
			train_bunch_single(frames_this_bunch, realin, realtarg);
		}
		else
		{
			printf("this bunch has only %d samples and is ignored.\n",frames_this_bunch);
		}
		
		realin += n_input * frames_this_bunch;
		realtarg += out_dims * frames_this_bunch;
	}
}

float BP_GPU::CrossValid(int n_frames, const float* in, const float *targ)
{
	//only use one GPU
	float squared_err=0.0f;
	int out_dims= layersizes[numlayers-1];
	float *out = new float [bunchsize*out_dims];
	int i,j,d;
	int frames_this_bunch;	// Number of frames to handle this bunch
	int n_input = layersizes[0];
	float *realin;
	// First copy data to GPU
	cudaSetDevice(GPU_selected);
	todev_vf_vf("in", n_frames* n_input, in, dev.in, streams);
	realin = dev.in;

	for (i=0; i< n_frames; i+= bunchsize)
	{
		
		frames_this_bunch = (bunchsize > n_frames - i)?(n_frames - i):bunchsize;
		cv_bunch_single(frames_this_bunch, realin, out);
		for(j =0; j< frames_this_bunch;j++)
		{
		  for(d=0;d<out_dims;d++)
		  {
		    squared_err = squared_err + (out[j*out_dims+d]-targ[j*out_dims+d])*(out[j*out_dims+d]-targ[j*out_dims+d]);
			}
		}
		realin += n_input * frames_this_bunch;
		targ += out_dims * frames_this_bunch;
	}
	delete []out;
	return squared_err;
}
float BP_GPU::CrossValiddB(int n_frames, const float* in, const float *targ)
{
	//only use one GPU
	float squared_err=0.0f;
	int out_dims= layersizes[numlayers-1];
	float *out = new float [bunchsize*out_dims];
	int i,j,d;
	int frames_this_bunch;	// Number of frames to handle this bunch
	int n_input = layersizes[0];
	float *realin;
	// First copy data to GPU
	cudaSetDevice(GPU_selected);
	todev_vf_vf("in", n_frames* n_input, in, dev.in, streams);
	realin = dev.in;

	for (i=0; i< n_frames; i+= bunchsize)
	{
		
		frames_this_bunch = (bunchsize > n_frames - i)?(n_frames - i):bunchsize;
		cv_bunch_single(frames_this_bunch, realin, out);
		for(j =0; j< frames_this_bunch;j++)
		{
		  for(d=0;d<out_dims;d++)
		  {
		    squared_err = squared_err + abs(out[j*out_dims+d]-targ[j*out_dims+d]);
			}
		}
		realin += n_input * frames_this_bunch;
		targ += out_dims * frames_this_bunch;
	}
	squared_err=squared_err/out_dims;
	delete []out;
	return squared_err;
}
float BP_GPU::CrossValid2(int n_frames, const float* in, const float *targ)
{
	int out_dims= layersizes[numlayers-1];
	float *err = new float[n_frames * out_dims];
	float *out = new float [bunchsize*out_dims];
	int i,j,d;
	int frames_this_bunch;	// Number of frames to handle this bunch
	int n_input = layersizes[0];
	float *realin;     
	cudaSetDevice(GPU_selected);
	todev_vf_vf("in", n_frames* n_input, in, dev.in, streams);
	int h=0;
	realin = dev.in;
	for (i=0; i< n_frames; i+= bunchsize)
	{
		frames_this_bunch = (bunchsize > n_frames - i)?(n_frames - i):bunchsize;
		cv_bunch_single(frames_this_bunch, realin, out);
		for(j =0; j< frames_this_bunch;j++)
		{
		  for(d=0;d<out_dims;d++)
		  {
		   err[(j+h)*out_dims+d] =targ[j*out_dims+d]-out[j*out_dims+d];
		  }
		}
		realin += n_input * frames_this_bunch;
		targ += out_dims * frames_this_bunch;
		h=h+frames_this_bunch;
	}
	float density1;
	float density2=0;
	float density3=0;
	float density;
	float *scalefac=new float[out_dims];
	fromdev_vf_vf("scalefac", out_dims , dev.scalefactor, scalefac, streams);
	density1=n_frames*out_dims*log(shapefactor/(2*Gamma(1.0/shapefactor)));
	for(int u=0;u<out_dims;u++)
	{
		density2+=log(scalefac[u]);
	}
	density2=density2*n_frames;
	for(int uu=0;uu<n_frames;uu++)
	{
		for(int uuu=0;uuu<out_dims;uuu++)
		{
			 density3+=pow((abs(err[uu*out_dims+uuu])/scalefac[uuu]),shapefactor);
		}
	}
	density=density1-density2-density3;
	delete []out;
	delete []err;
	delete []scalefac;
	return density;
}

void BP_GPU::train_bunch_single(int n_frames, float *in, const float* targ)
{
	const float one  = 1.0f;
	const float zero = 0.0f;
	int cur_layer;			// The index of the current layer.
	int prev_layer;			// The index of the previous layer.
	int cur_layer_units;	// The number of units in the current layer.
	int prev_layer_units;	// The number of units in the previous layer.
	int cur_layer_size;		// The size of the current layer.
	int prev_layer_size;

	float* cur_layer_x;
	float* cur_layer_y;				// Output from the current layer
	const float* prev_layer_y;	// Output from the previous non-linearity.
	float* cur_layer_dydx;	// dydx for the current layer.
	float* cur_layer_dedy;	// dedy for the current layer.
	float* prev_layer_dedy;	// dedy for the previous layer.
	float* cur_layer_dedx;	// dedx for the current layer.
	float* cur_layer_ydedx;
	float* cur_layer_sumdedx;
	float* cur_layer_bias;	// Biases for the current layer.
	float* cur_layer_delta_bias; // Delta biases for the current layer.
	float* cur_layer_delta_weights;
	float* cur_weights;		// Weights inputing to the current layer.
	float cur_lrate =  lrate;
	//// Forward
	for (cur_layer=1; cur_layer< numlayers; cur_layer++)
	{
		prev_layer = cur_layer - 1;
		cur_layer_units = layersizes[cur_layer];
		prev_layer_units = layersizes[prev_layer];
		cur_layer_size = cur_layer_units * n_frames;
		prev_layer_size = prev_layer_units * n_frames;
		cur_layer_x = dev.layer_x[cur_layer];
		cur_layer_y = dev.layer_y[cur_layer];
		if (cur_layer==1){
			if(dropoutflag == 1){
				curandGenerateUniform(gen, dev.DevRandVector, prev_layer_size);
				DevDropout(streams, prev_layer_size, visible_omit, in, dev.DevRandVector);
			}
			prev_layer_y = in;
		}
		else{
			if(dropoutflag == 1){
				curandGenerateUniform(gen, dev.DevRandVector, prev_layer_size);
				DevDropout(streams, prev_layer_size, hid_omit, dev.layer_y[prev_layer], dev.DevRandVector);
			}
			prev_layer_y = dev.layer_y[prev_layer];
		}
		cur_layer_bias = dev.bias[cur_layer];
		cur_weights = dev.weights[cur_layer];

		DevMultiCopy(streams,n_frames, cur_layer_units, cur_layer_bias, cur_layer_x);
		SgemmNN(handles,cur_layer_units, prev_layer_units, n_frames, cur_weights, prev_layer_y, cur_layer_x, one, one); 

		if (cur_layer != numlayers - 1){
			DevSigmoid(streams,cur_layer_size, cur_layer_x, cur_layer_y);
		}
		else{  
	    cudaMemcpy(dev.out,cur_layer_x,n_frames*cur_layer_units*sizeof(float),cudaMemcpyDeviceToDevice);
		}
	}
	// Backward
	for (cur_layer = numlayers -1; cur_layer >0; cur_layer--)
	{
		prev_layer = cur_layer - 1;
		cur_layer_units = layersizes[cur_layer];
		prev_layer_units = layersizes[prev_layer];
		cur_layer_size = cur_layer_units * n_frames;
		cur_layer_y = dev.layer_y[cur_layer];
		if (cur_layer==1)
			prev_layer_y = in;
		else
			prev_layer_y = dev.layer_y[prev_layer];
		cur_layer_dydx = dev.layer_dydx[cur_layer];
		cur_layer_dedy = dev.layer_dedy[cur_layer];
		prev_layer_dedy = dev.layer_dedy[prev_layer];
		cur_layer_dedx = dev.layer_dedx[cur_layer];
		cur_layer_ydedx = dev.layer_ydedx[cur_layer];
		cur_layer_sumdedx = dev.layer_sumdedx[cur_layer];
		cur_layer_bias = dev.bias[cur_layer];
		cur_layer_delta_bias = dev.delta_bias[cur_layer];
		cur_layer_delta_weights = dev.delta_weights[cur_layer];
		cur_weights = dev.weights[cur_layer];
		
		float *vec1 = dev.vec1;
		float *vec2 = dev.vec2;
		float *scalefactor=dev.scalefactor;   
		float *realerror=dev.realerror;
		float *errorabsolute=dev.errorabsolute;
		float *errorabsolute2=dev.errorabsolute2;
                float *newobj=dev.newobj;
		if (cur_layer != numlayers - 1)
		{
			DevDsigmoid(streams, cur_layer_size, cur_layer_y, cur_layer_dedy, cur_layer_dedx);
		}
		else
		{
			//"shapefactor" refers to the shape factor β in GGD. "scalefactor" refers to the scale factor α in GGD. 
			//When MLflag!=1, the classic β-norm function is selected as the objective function, where β=1 corresponds to the LAD criterion, β=2 corresponds to the MMSE criterion.
			DevSubClean2(streams, n_frames, cur_layer_units,shapefactor, dev.out, targ, cur_layer_dedx);
			DevVecMulNum(streams, cur_layer_units * n_frames, cur_layer_dedx, 1.0f/n_frames, cur_layer_dedx);
			// When MLflag==1, the proposed log-likelihood function based on the GGD error model is selected as the objective function.
			if(MLflag == 1) //The scale factors and DNN parameters are optimized alternatively.
			{	//Estimate the scale factor in each dimension according to the ML criterion.
				Deverror(streams, n_frames, cur_layer_units, dev.out, targ, realerror);
				Devabsolutevalus(streams,cur_layer_units * n_frames,realerror,errorabsolute);
				Devindex2(streams,n_frames*cur_layer_units, errorabsolute, shapefactor,errorabsolute2);
				DevSumcol(streams, n_frames, cur_layer_units, errorabsolute2, vec1);
				DevDivide(streams, cur_layer_units, vec1, vec1, n_frames);
				DevVecMulNum(streams, cur_layer_units, vec1, shapefactor, vec2);
				float ppp=1.0f/shapefactor;
				Devindex2(streams,cur_layer_units, vec2,ppp, scalefactor);
				//Calculate the gradient of the proposed objective function with respect to the output
				Devfunc2(streams, n_frames,cur_layer_units, realerror,scalefactor, newobj,shapefactor);
				DevVecMulNum(streams, cur_layer_units * n_frames, newobj, 1.0f/n_frames, cur_layer_dedx);
			}
		
		}
		cudaStreamSynchronize(streams);
		if (cur_layer != 1)
		{
			SgemmTN(handles, prev_layer_units, cur_layer_units, n_frames, cur_weights, cur_layer_dedx, prev_layer_dedy, zero, one);
		}
		SgemmNT(handles2, cur_layer_units, n_frames, prev_layer_units, cur_layer_dedx, prev_layer_y, cur_layer_ydedx ,zero, one);
		updatedelta(streams2, cur_layer_units, prev_layer_units, cur_layer_delta_weights, cur_weights, cur_layer_ydedx, n_frames, momentum, cur_lrate, weightcost);
		DevAccSumrow(streams2, cur_layer_units, n_frames, cur_layer_dedx, cur_layer_sumdedx, zero, one);
		updatedelta(streams2, 1, cur_layer_units, cur_layer_delta_bias, cur_layer_bias, cur_layer_sumdedx, n_frames, momentum, cur_lrate, zero);
		DevAccSum(streams2,	cur_layer_units, prev_layer_units, cur_layer_delta_weights,cur_weights, 1.0);
		DevAccSum(streams2,	cur_layer_units, 1, cur_layer_delta_bias,cur_layer_bias, 1.0);
	}
	cudaStreamSynchronize(streams);
}

void BP_GPU::cv_bunch_single(int n_frames, const float *in, float* out)
{
	const float one  = 1.0f;
	int cur_layer;			// The index of the current layer.
	int prev_layer;			// The index of the previous layer.
	int cur_layer_units;	// The number of units in the current layer.
	int prev_layer_units;	// The number of units in the previous layer.
	int cur_layer_size;		// The size of the current layer.
	int out_dims= layersizes[numlayers-1];
  
	float* cur_layer_x;
	float* cur_layer_y;				// Output from the current layer
	const float* prev_layer_y;	// Output from the previous non-linearity.
	float* cur_layer_bias;	// Biases for the current layer.
	float* cur_weights;		// Weights inputing to the current layer.

	float *devout;
	devnew_vf("devout", n_frames*out_dims, &devout);
	//dropout parameters
	int weight_size;
	float vis_keep;
	float hid_keep;
	vis_keep = 1.0f-visible_omit;
	hid_keep = 1.0f-hid_omit;
	//// Forward
	for (cur_layer=1; cur_layer< numlayers; cur_layer++)
	{
		prev_layer = cur_layer - 1;
		cur_layer_units = layersizes[cur_layer];
		prev_layer_units = layersizes[prev_layer];
		cur_layer_size = cur_layer_units * n_frames;
		cur_layer_x = dev.layer_x[cur_layer];
		cur_layer_y = dev.layer_y[cur_layer];

		weight_size = prev_layer_units*cur_layer_units;

		if (cur_layer==1)
			prev_layer_y = in;
		else
			prev_layer_y = dev.layer_y[prev_layer];
		cur_layer_bias = dev.bias[cur_layer];

		if(dropoutflag==1){
			if(cur_layer==1)
				DevWeightMultiP(streams, weight_size, vis_keep, dev.weights[cur_layer]);
			else
				DevWeightMultiP(streams, weight_size, hid_keep, dev.weights[cur_layer]);
		}

		cur_weights = dev.weights[cur_layer];

		DevMultiCopy(streams,n_frames, cur_layer_units, cur_layer_bias, cur_layer_x);
		SgemmNN(handles,cur_layer_units, prev_layer_units, n_frames, cur_weights, prev_layer_y, cur_layer_x, one, one); 

		if(dropoutflag==1){
			if(cur_layer==1)
				DevWeightMultiP(streams, weight_size, 1.0f/vis_keep, dev.weights[cur_layer]);
			else
				DevWeightMultiP(streams, weight_size, 1.0f/hid_keep, dev.weights[cur_layer]);
		}

		if (cur_layer != numlayers - 1){
			DevSigmoid(streams,cur_layer_size, cur_layer_x, cur_layer_y);
		}
		else{  
		cudaMemcpy(devout,cur_layer_x,n_frames*cur_layer_units*sizeof(float),cudaMemcpyDeviceToDevice);
		}
	}
    fromdev_vf_vf("devout",n_frames*out_dims,devout,out, streams);
	devfree_vf("devout",devout);
}

void BP_GPU::returnWeights(float **weights, float **bias)
{
	int i;
	////copy weights && biases to devices
	cudaSetDevice(GPU_selected);
  // cudaSetDevice(1);
	for(i = 1; i< numlayers; i++)
	{
		fromdev_vf_vf("weights", layersizes[i-1] *layersizes[i], dev.weights[i], weights[i], streams);
		fromdev_vf_vf("bias", layersizes[i], dev.bias[i], bias[i], streams);
	}
}

///// following are alloc and free functions
void BP_GPU::devnew_vf(const char* varname, int n, float **devptr)
{
	cudaError_t cudaStat =  cudaMalloc((void **) devptr, n* sizeof(float));
	if(cudaStat !=cudaSuccess ) 
	{
		printf("%s device momory alloc error\n", varname);
		exit(0);
	}
	float *zero;
	cudaMallocHost((void**)&zero,n*sizeof(float));

	for(int i=0;i< n;i++)
		zero[i] = 0.0f;
	cublasSetVector(n,sizeof(float),zero,1,(*devptr),1);
	cudaFreeHost(zero);
}

void BP_GPU::devnew_vi(const char* varname, int n, int **devptr)
{
	cudaError_t cudaStat = cudaMalloc((void **) devptr, n* sizeof(int));
	if(cudaStat !=cudaSuccess ) 
	{
		printf( "%s device momory alloc error\n", varname);
		exit(0);
	}
	int *zero;
	cudaMallocHost((void**)&zero,n*sizeof(int));

	for(int i=0;i< n;i++)
		zero[i] = 0;
	cublasSetVector(n,sizeof(int),zero,1,(*devptr),1);
	cudaFreeHost(zero);
}

void BP_GPU::devfree_vf(const char* varname, float* devptr)
{
	cudaFree((void *) devptr);
}

void BP_GPU::devfree_vi(const char* varname, int* devptr)
{
	cudaFree((void *) devptr);
}

void BP_GPU::todev_vf_vf(const char* varname, int n, const float* from, float* devto, cudaStream_t stream)
{
	cublasStatus_t  e = cublasSetVectorAsync(n, sizeof(float), from, 1, devto, 1, stream);
	if (e != CUBLAS_STATUS_SUCCESS)
	{
		printf("cuda blas todev_vf_vf error variable %s\n",varname);
		exit(0);
	}
}

void BP_GPU::fromdev_vf_vf(const char* varname, int n, const float* devfrom, float* to, cudaStream_t stream)
{
	cublasStatus_t e = cublasGetVectorAsync(n, sizeof(float), devfrom, 1, to, 1, stream);
	if (e != CUBLAS_STATUS_SUCCESS)
	{
		printf("cuda blas fromdev_vf_vf error variable %s\n",varname);
		exit(0);
	}
}


float BP_GPU::Gamma( float x )
{//x>0
	if( x > 2 && x<= 3 )
	{
		const float c0 =  0.0000677106;
		const float c1 = -0.0003442342;
		const float c2 =  0.0015397681;
		const float c3 = -0.0024467480;
		const float c4 =  0.0109736958;
		const float c5 = -0.0002109075;
		const float c6 =  0.0742379071;
		const float c7 =  0.0815782188;
		const float c8 =  0.4118402518;
		const float c9 =  0.4227843370;
		const float c10 = 1.0000000000;
		float temp = 0;
		temp = temp + c0*pow( x-2.0, 10.0) + c1*pow( x-2.0, 9.0);
		temp = temp + c2*pow( x-2.0, 8.0) + c3*pow( x-2.0 , 7.0);
		temp = temp + c4*pow( x-2.0, 6.0) + c5*pow( x-2.0, 5.0 );
		temp = temp + c6*pow( x-2.0, 4.0 ) + c7*pow( x-2.0, 3.0 );
		temp = temp + c8*pow( x-2.0, 2.0 ) + c9*( x-2.0) + c10;
		return temp;
	}
	else if( x>0 && x<=1 )
	{
		return Gamma( x+2 )/(x*(x+1) );
	}
	else if( x > 1 && x<=2 )
	{
		return Gamma( x+1 )/x;
	}
	else if( x > 3 )
	{
		int i = 1;
		float temp = 1;
		while( ((x-i)>2 && (x-i) <= 3 ) == false )
		{
			temp = (x-i) * temp;
			i++;
		}
		temp = temp*(x-i);
		return temp*Gamma( x-i);
	}
	else
	{
		return 0;
	}
}

