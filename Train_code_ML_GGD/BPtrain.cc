#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <time.h>
#include <unistd.h>

#include "BP_GPU.h"
#include "Interface.h"
#include <sys/time.h>

void test();
void test2(int argc,char*argv[]);

void *threadFetch(void *s)
{
	Interface * InterObj = (Interface *)s;
	float *tmpdata, *tmptarg;
	int i = 0;
	int *chunk_index = InterObj->chunk_index;
	struct WorkPara *para;
	para = InterObj->para;

	InterObj->cur_chunk_samples = InterObj->Readchunk(chunk_index[0]);
	tmpdata = para->indata[1];
	tmptarg = para->targ[1];

	para->indata[1] = para->indata[0];
	para->targ[1]	= para->targ[0];

	para->indata[0] = tmpdata;
	para->targ[0]	= tmptarg;

	setSignal(InterObj->syncChunk, FULL);

	int tmp;
	for(i = 1; i< InterObj->total_chunks; ++i)
	{
		tmp = InterObj->Readchunk(chunk_index[i]);
		waitSignal(InterObj->syncChunk, EMPTY);

		InterObj->cur_chunk_samples = tmp;

		tmpdata = para->indata[1];
		tmptarg = para->targ[1];

		para->indata[1] = para->indata[0];
		para->targ[1]	= para->targ[0];

		para->indata[0] = tmpdata;
		para->targ[0]	= tmptarg;
		setSignal(InterObj->syncChunk, FULL);
	}
}
int main(int argc, char *argv[])
{
	
	struct WorkPara *paras;
	int *chunk_index;
	int cur_chunk_samples;
	int i,j;
	float squared_err=0.0;
	float dB_squared_err=0.0;
	float likelihood=0.0;
	double timenow;
	timenow = time(NULL);
	
#ifdef RELU
	printf("\033\[41;30m--------activation functin is relu--------\033[0m\n");
#else
	printf("\033\[41;30m--------activation functin is sigmoid--------\033[0m\n");
#endif

	Interface *InterObj = new Interface;
	InterObj->Initial(argc, argv);
	paras = InterObj->para;
	BP_GPU *TrainObj = new BP_GPU(paras->init_randem_seed, paras->gpu_used, InterObj->numlayers, paras->layersizes, paras->bunchsize,  paras->lrate, paras->momentum,
				paras->weightcost, paras->weights, paras->bias,paras->shapefactor,paras->MLflag, paras->dropoutflag, paras->visible_omit, paras->hid_omit);
	InterObj->get_pfile_info();
	/////////train	
	InterObj->get_chunk_info(paras->train_sent_range);
	InterObj->chunk_index = new int [InterObj->total_chunks];
	for(i=0;i< InterObj->total_chunks;i++){
		InterObj->chunk_index[i] =i;
	}
	
	InterObj->GetRandIndex(InterObj->chunk_index,InterObj->total_chunks);

	chunk_index = InterObj->chunk_index;

	pthread_t fetch;
	pthread_create(&fetch, NULL, threadFetch, (void *) InterObj);

	for(i=0;i< InterObj->total_chunks; i++){
		waitSignal(InterObj->syncChunk, FULL);
		fprintf(InterObj->fp_log,"Starting chunk %d of %d containing %d samples.\n", i+1 ,InterObj->total_chunks, InterObj->cur_chunk_samples);
		fflush(InterObj->fp_log);
		TrainObj->train(InterObj->cur_chunk_samples, paras->indata[1] ,paras->targ[1]);
		setSignal(InterObj->syncChunk, EMPTY);
	}
	
	pthread_join(fetch, NULL);
	
	timenow = time(NULL) - timenow;
	fprintf(InterObj->fp_log, "Total cost time: %.1f s.\n", timenow);

	printf("begin to write weights\n");
	TrainObj->returnWeights(paras->weights,paras->bias);
	InterObj->Writeweights();
    printf("finish to write weights\n\n");
	
	printf("begin to CV\n");
	////CV
	fprintf(InterObj->fp_log,"Starting CV.\n");
	InterObj->get_chunk_info_cv(paras->cv_sent_range);
	chunk_index = new int [InterObj->cv_total_chunks];
	for(i=0;i< InterObj->cv_total_chunks;i++){
		chunk_index[i] =i;
	}

	for(i=0;i< InterObj->cv_total_chunks; i++){
		cur_chunk_samples = InterObj->Readchunk_cv(chunk_index[i]);
		printf("cur_chunk_samples=%d\n",cur_chunk_samples);
		squared_err += TrainObj->CrossValid(cur_chunk_samples, paras->indata[0] ,paras->targ[0]);
		dB_squared_err += TrainObj->CrossValiddB(cur_chunk_samples, paras->indata[0] ,paras->targ[0]);
		if (paras->MLflag==1)
		{
			likelihood += TrainObj->CrossValid2(cur_chunk_samples, paras->indata[0] ,paras->targ[0]);
		}
	}
	float cvacc = ((float) squared_err/InterObj->cv_total_samples);
	fprintf(InterObj->fp_log,"CV over. squared error: %f\n", cvacc);
	float cvacc1 = ((float) dB_squared_err/InterObj->cv_total_samples);
	fprintf(InterObj->fp_log,"CV over. square root squared error: %f\n", cvacc1);
	if (paras->MLflag==1)
	{
		float cvacc2 = ((float) likelihood/InterObj->cv_total_samples);
		fprintf(InterObj->fp_log,"CV2 over. CV log likelihood: %f\n", cvacc2);
	}
	fflush(InterObj->fp_log);
	delete [] chunk_index;
	printf("all finish!\n");
	delete TrainObj;
	delete InterObj;	
	return 0;
}
