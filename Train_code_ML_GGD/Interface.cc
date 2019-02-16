#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <assert.h>
#include <time.h>


#define NOTDEBUG 1
#include "Interface.h"
#include "BP_GPU.h"

#define PFILE_HEADER_SIZE  (32768)
int waitSignal(SyncInfo & syncInfo, bool flag)
{
	Lock &lock = syncInfo.lock;
	Signal &signal = syncInfo.signal;

	pthread_mutex_lock(&lock.mutex);

	if(lock.flag == flag){
		pthread_mutex_unlock(&lock.mutex);
		return 0;
	}
	if(flag == FULL){
		pthread_cond_wait(&signal.full, &lock.mutex);
	}
	else{
		pthread_cond_wait(&signal.empty, &lock.mutex);
	}

	pthread_mutex_unlock(&lock.mutex);
	return 0;
}
int setSignal(SyncInfo & syncInfo, bool flag)
{
	Lock &lock = syncInfo.lock;
	Signal &signal = syncInfo.signal;

	pthread_mutex_lock(&lock.mutex);

	lock.flag =flag;

	if(flag == FULL){
		pthread_cond_signal(&signal.full);
	}
	else{
		pthread_cond_signal(&signal.empty);
	}

	pthread_mutex_unlock(&lock.mutex);
	return 0;
}
int initLock(Lock &lock, bool flag)
{
	lock.flag =flag;
	int ret = pthread_mutex_init(&lock.mutex, NULL);
	if(ret) {
		printf("ERROR: failed to call pthread_mutex_init()\n");
		return -1;
	}
	return 0;
}
int initSignal(Signal &signal)
{
	int ret = 0;
	ret = pthread_cond_init(&signal.empty, NULL);
	if( ret ){
		printf("ERROR: failed to call pthread_cond_init()\n");
		return -1;
	}

	ret = pthread_cond_init(&signal.full, NULL);
	if( ret ){
		printf("ERROR: failed to call pthread_cond_init()\n");
		return -1;
	}
	return 0;
}

void swap32(int *val)
{
	unsigned int uval;
	unsigned int res;
	int b0, b1, b2, b3;

	uval = (unsigned int) (*val);
	b0 = uval >> 24;
	b1 = (uval >> 8) & 0x0000ff00;
	b2 = (uval << 8) & 0x00ff0000;
	b3 = uval << 24;

	res = b0|b1|b2|b3;

	*val = (int) res;
	return;
}

Interface::Interface()
{
	para = new struct WorkPara;
	chunk_frame_st = new int[MAXCHUNK];
	cv_chunk_frame_st = new int[MAXCHUNK];
	initSignal(syncChunk.signal);
	initLock(syncChunk.lock, EMPTY);
}

Interface::~Interface()
{
	int i;
	fclose(fp_data);
	fclose(fp_targ);
	fclose(fp_log);
	fclose(fp_out);

	delete []mean;
	delete []dVar;

	for(i =1; i< numlayers; i++)
	{
		delete [](para->weights[i]);
		delete [](para->bias[i]);
	}
	delete para;
	delete []chunk_frame_st;
	delete []cv_chunk_frame_st;
	delete []framesBeforeSent;
}

////Read input parameters to varibles and assert that files input are all accessed 
////then initialize the weights and biases , feature mean and variance ,data randem index
////Lastly,allocate memory for data and targets
void Interface::Initial(int argc, char **argv)
{
	int i;
	char *p;
	char *argname;
	char *argvalue;
	char buff[MAXLINE];
	para->init_randem_weight_min = -0.1;
	para->init_randem_weight_max = 0.1;
	para->init_randem_bias_min = -0.1;
	para->init_randem_bias_max = 0.1;
	for(i =0; i< MAXLINE; i++)
	{
		data_rand_index[i] = i;
	}

	////Read Args
	for(i =1; i < argc; i++)
	{
		p = strstr(argv[i],"=");
		if(p ==NULL)
		{
			fprintf(fp_log,"Arg: %s  Format Error\n",argv[i]);
			exit(0);
		}
		argname = argv[i];
		argvalue = p+1;
		*p = '\0';
		if(0 == strcmp(argname, "fea_file"))
		{
			strcpy(para->fea_FN, argvalue); 
			continue;
		}
		if(0 == strcmp(argname, "norm_file"))
		{
			strcpy(para->fea_normFN, argvalue); 
			continue;
		}
		if(0 == strcmp(argname, "targ_file"))
		{
			strcpy(para->targ_FN, argvalue); 
			continue;
		}
		if(0 == strcmp(argname, "outwts_file"))
		{
			strcpy(para->out_weightFN, argvalue); 
			continue;
		}
		if(0 == strcmp(argname, "log_file"))
		{
			strcpy(para->log_FN, argvalue); 
			continue;
		}
		if(0 == strcmp(argname, "initwts_file"))
		{
			strcpy(para->init_weightFN, argvalue); 
			continue;
		}
		if(0 == strcmp(argname, "train_sent_range"))
		{
			strcpy(para->train_sent_range, argvalue); 
			continue;
		}
		if(0 == strcmp(argname, "cv_sent_range"))
		{
			strcpy(para->cv_sent_range, argvalue); 
			continue;
		}

		if(0 == strcmp(argname, "fea_dim"))
		{
			para->fea_dim = atoi(argvalue);
			continue;
		}
		if(0 == strcmp(argname, "fea_context"))
		{
			para->fea_context = atoi(argvalue);
			continue;
		}
		if(0 == strcmp(argname, "targ_offset"))
		{
			para->targ_offset = atoi(argvalue);
			continue;
		}
		if(0 == strcmp(argname, "dropoutflag"))
		{
			para->dropoutflag = atoi(argvalue);
			continue;
		}
		if(0 == strcmp(argname, "MLflag"))
		{
			para->MLflag = atoi(argvalue);
			continue;
		}
		if(0 == strcmp(argname, "traincache"))
		{
			para->traincache = atoi(argvalue);
			continue;
		}
		if(0 == strcmp(argname, "bunchsize"))
		{
			para->bunchsize = atoi(argvalue);
			continue;
		}
		if(0 == strcmp(argname, "gpu_used"))
		{
			para->gpu_used = atoi(argvalue);
			continue;
		}
		if(0 == strcmp(argname, "init_randem_seed"))
		{
			para->init_randem_seed = atoi(argvalue);
			continue;
		}

		if(0 == strcmp(argname, "momentum"))
		{
			para->momentum = atof(argvalue);
			continue;
		}
		if(0 == strcmp(argname, "shapefactor"))
		{
			para->shapefactor = atof(argvalue);
			continue;
		}
		if(0 == strcmp(argname, "weightcost"))
		{
			para->weightcost = atof(argvalue);
			continue;
		}
		if(0 == strcmp(argname, "lrate"))
		{
			para->lrate = atof(argvalue);
			continue;
		}
		if(0 == strcmp(argname, "visible_omit"))
		{
			para->visible_omit = atof(argvalue);
			continue;
		}
		if(0 == strcmp(argname, "hid_omit"))
		{
			para->hid_omit = atof(argvalue);
			continue;
		}
		if(0 == strcmp(argname, "init_randem_weight_min"))
		{
			para->init_randem_weight_min = atof(argvalue);
			continue;
		}
		if(0 == strcmp(argname, "init_randem_weight_max"))
		{
			para->init_randem_weight_max = atof(argvalue);
			continue;
		}
		if(0 == strcmp(argname, "init_randem_bias_max"))
		{
			para->init_randem_bias_max = atof(argvalue);
			continue;
		}
		if(0 == strcmp(argname, "init_randem_bias_min"))
		{
			para->init_randem_bias_min = atof(argvalue);
			continue;
		}
		if(0 == strcmp(argname, "layersizes"))
		{
			char *pp = argvalue;
			int count =0;
			int j;
			p = strstr(argvalue, ",");
			while(p != NULL)
			{
				*p = '\0';
				para->layersizes[count++] = atoi(pp);
				pp = p +1;
				p = strstr(pp, ",");
			}
			para->layersizes[count++] = atoi(pp);
			numlayers = count;
			continue;
		}
	}
	////Check Files
	if(NULL ==(fp_log = fopen(para->log_FN, "wt")))
	{
		printf("can not open output log file: %s\n", para->log_FN);
		exit(0);
	}
	if(NULL ==(fp_data = fopen(para->fea_FN, "rb")))
	{
		fprintf(fp_log,"can not open feature file: %s\n", para->fea_FN);
		exit(0);
	}
	if(NULL ==(fp_targ = fopen(para->targ_FN, "rb")))
	{
		fprintf(fp_log,"can not open target file: %s\n", para->targ_FN);
		exit(0);
	}
	if(NULL ==(fp_out = fopen(para->out_weightFN, "wb")))
	{
		fprintf(fp_log,"can not open output weights file: %s\n", para->out_weightFN);
		exit(0);
	}

	fprintf(fp_log,"parameters input:\n");
	fprintf(fp_log,"fea_file:             %s\n", para->fea_FN);
	fprintf(fp_log,"norm_file:            %s\n", para->fea_normFN);
	fprintf(fp_log,"targ_file:            %s\n", para->targ_FN);
	fprintf(fp_log,"outwts_file:          %s\n", para->out_weightFN);
	fprintf(fp_log,"log_file:		          %s\n", para->log_FN);
	fprintf(fp_log,"initwts_file:         %s\n", para->init_weightFN);
	fprintf(fp_log,"train_sent_range:     %s\n", para->train_sent_range);
	fprintf(fp_log,"cv_sent_range:        %s\n", para->cv_sent_range);
	fprintf(fp_log,"fea_dim:		          %d\n", para->fea_dim);
	fprintf(fp_log,"fea_context:		      %d\n", para->fea_context);
	fprintf(fp_log,"bunchsize:		        %d\n", para->bunchsize);
	fprintf(fp_log,"gpu_used:		          %d\n", para->gpu_used);
	fprintf(fp_log,"train_cache:		      %d\n", para->traincache);
	fprintf(fp_log,"init_randem_seed:		  %d\n", para->init_randem_seed);
	fprintf(fp_log,"targ_offset:		      %d\n", para->targ_offset);
	fprintf(fp_log,"dropoutflag:		      %d\n", para->dropoutflag);
	fprintf(fp_log,"MLflag:		      %d\n", para->MLflag);

	fprintf(fp_log,"init_randem_weight_max:		  %f\n", para->init_randem_weight_max);
	fprintf(fp_log,"init_randem_weight_min:		  %f\n", para->init_randem_weight_min);
	fprintf(fp_log,"init_randem_bias_max:		    %f\n", para->init_randem_bias_max);
	fprintf(fp_log,"init_randem_bias_min:		    %f\n", para->init_randem_bias_min);
	fprintf(fp_log,"momentum:		                %f\n", para->momentum);
	fprintf(fp_log,"shapefactor:		                %f\n", para->shapefactor);
	fprintf(fp_log,"weightcost:		              %f\n", para->weightcost);
	fprintf(fp_log,"learnrate:		              %f\n", para->lrate);
	fprintf(fp_log,"visible_omit:		      %d\n", para->visible_omit);
	fprintf(fp_log,"hid_omit:		      %d\n", para->hid_omit);
	fprintf(fp_log,"layersizes:		              ");
	for(int j =0; j < numlayers; j++)
		fprintf(fp_log,"%d,", para->layersizes[j]);
	fprintf(fp_log,"\n");
	fprintf(fp_log,"Please check...\n");

	////Load Norm file
	if(NULL ==(fp_norm = fopen(para->fea_normFN, "rt")))
	{
		fprintf(fp_log,"can not open normalization file: %s\n", para->fea_normFN);
		exit(0);
	}
	else
	{
		fprintf(fp_log,"Loading Norm file...\n");
		mean = new float[para->fea_dim];
		dVar = new float[para->fea_dim];

		fgets(buff,MAXLINE,fp_norm);
		for(int j=0;j< para->fea_dim;j++)
		{
			fgets(buff,MAXLINE,fp_norm);
			mean[j] = atof(buff);
		}
		fgets(buff,MAXLINE,fp_norm);
		for(int j=0;j< para->fea_dim;j++)
		{
			fgets(buff,MAXLINE,fp_norm);
			dVar[j] = atof(buff);
		}
		fclose(fp_norm);
		fprintf(fp_log,"Norm file loaded.\n");
	}

	//// Init weights
	for(i =1; i< numlayers; i++)
	{
		int size	= para->layersizes[i] *para->layersizes[i-1];
		para->weights[i] = new float [size];
		para->bias[i] = new float [para->layersizes[i]];

		memset(para->weights[i],0,size *sizeof(float));
		memset(para->bias[i],0,para->layersizes[i] *sizeof(float));
	}
	srand48(para->init_randem_seed); ////Only once for weights and data index

	if(0 == strcmp("", para->init_weightFN ))
	{
#ifdef RELU
		fprintf(fp_log,"Getting Randemed initial weights...\n");
		for(i =1; i< numlayers; i++)
		{
			int size	= para->layersizes[i] *para->layersizes[i-1];
			GetRandWeight(para->weights[i], para->init_randem_weight_min, para->init_randem_weight_max, size);
			GetRandWeight(para->bias[i], para->init_randem_bias_min, para->init_randem_bias_max, para->layersizes[i]);
		}
		fprintf(fp_log,"Randemed initial weights getted.\n");
#else
		fprintf(fp_log,"fatal_error, please set initial weights file\n");
#endif
	}
	else
	{
		if(NULL ==(fp_init_weight = fopen(para->init_weightFN, "rb")))
		{
			fprintf(fp_log,"can not open initial weights file: %s\n", para->init_weightFN);
			exit(0);
		}
		else
		{
			fprintf(fp_log,"Loading Init weight file...\n");

			int stat[10];
			char head[256];

			for(i =1; i< numlayers; i++)
			{
				fread(stat,sizeof(int),5,fp_init_weight);
				fread(head,sizeof(char),stat[4],fp_init_weight);

				if(stat[1] != para->layersizes[i] || stat[2] != para->layersizes[i -1])
				{
					fprintf(fp_log,"%d,%d,%d,%d\n",stat[1],stat[2],para->layersizes[i],para->layersizes[i -1]);
					fprintf(fp_log,"init weights node nums do not match\n");
					exit(0);
				}
				fread(para->weights[i],sizeof(float),para->layersizes[i -1] *para->layersizes[i],fp_init_weight);

				fread(stat,sizeof(int),5,fp_init_weight);
				fread(head,sizeof(char),stat[4],fp_init_weight);

				if(stat[2] != para->layersizes[i] || stat[1] != 1)
				{
					fprintf(fp_log,"init bias node nums do not match\n");
					exit(0);
				}
				fread(para->bias[i],sizeof(float),para->layersizes[i],fp_init_weight);
			}
			fclose(fp_init_weight);
			fprintf(fp_log,"Init weight file loaded.\n");
		}
	}

	//// Alloc data and target memory
	if(para->fea_dim * para->fea_context != para->layersizes[0])
	{
		fprintf(fp_log,"feadim times context must be equal to layersizes[0]\n");
		exit(0);
	}
	for(i =0; i<2; i++){
		para->indata[i] 	= new float[para->layersizes[0] * para->traincache];
		//para->targ 	= new int[para->traincache];
		para->targ[i] 	= new float[para->layersizes[numlayers-1] * para->traincache];
	}
	fflush(fp_log);
}

void Interface::Writeweights()
{
	int i,j,k,m;
	fprintf(fp_log,"Saving weights to file...\n");

	int stat[10];
	char head[256];
	float *tmpweights;
	for(i =1; i< numlayers; i++)
	{
		sprintf(head,"weights%d%d",i,i+1);
		stat[0] = 10;
		stat[1] = para->layersizes[i];
		stat[2] = para->layersizes[i -1];
		stat[3] = 0;
		stat[4] = strlen(head)+1;

		fwrite(stat,sizeof(int),5,fp_out);
		fwrite(head,sizeof(char),stat[4],fp_out);
		fwrite(para->weights[i],sizeof(float),stat[2] *stat[1],fp_out);
		sprintf(head,"bias%d",i+1);
		stat[0] = 10;
		stat[1] = 1;
		stat[2] = para->layersizes[i];
		stat[3] = 0;
		stat[4] = strlen(head)+1;

		fwrite(stat,sizeof(int),5,fp_out);
		fwrite(head,sizeof(char),stat[4],fp_out);
		fwrite(para->bias[i],sizeof(float),stat[2],fp_out);
	}
	fprintf(fp_log,"Saving over.\n");
}

///Get frames number,sentence number and frame number per sent in training data and assert them in data and target pfile is consistent
void Interface::get_pfile_info()
{
	char *header = new char[PFILE_HEADER_SIZE]; // Store header here
	long int offset;
	int sizePerFrame;
	int *tmpframepersent;
	unsigned int tmpsentnum;
	unsigned int tmpframenum;

  fprintf(fp_log,"begin to read in_pfile\n");
	/////Read data pfile
	fseek(fp_data,0,0);
	if (fread(header, PFILE_HEADER_SIZE, 1, fp_data) != 1)
	{
		fprintf(fp_log, "Failed to read data pfile header.\n");
		exit(0);
	}
	get_uint(header, "-num_sentences", &total_sents); ///get sent num
	get_uint(header, "-num_frames", &total_frames);   ///get frame num

	framesBeforeSent = new int[total_sents];
	sizePerFrame = sizeof(float)*(2+ para->fea_dim);
	offset = total_frames * (long int)sizePerFrame + PFILE_HEADER_SIZE;
	read_tail(fp_data, offset, total_sents, framesBeforeSent);  /// get frame per sent

#if NOTDEBUG
//Read target pfile (also feature file, same with the input format) 
  fprintf(fp_log,"begin to read target_pfile\n");
	fseek(fp_targ,0,0);
	if (fread(header, PFILE_HEADER_SIZE, 1, fp_targ) != 1)
	{
		fprintf(fp_log, "Failed to read target pfile header.\n");
		exit(0);
	}
	get_uint(header, "-num_sentences", &tmpsentnum); ///get sent num
	get_uint(header, "-num_frames", &tmpframenum);   ///get frame num

	tmpframepersent = new int[tmpsentnum];
	sizePerFrame = sizeof(float)*(2+ para->layersizes[numlayers-1]);
	offset = tmpframenum * (long int)sizePerFrame + PFILE_HEADER_SIZE;
	read_tail(fp_targ, offset, tmpsentnum, tmpframepersent);  /// get frame per sent

	///assert consistency
	fprintf(fp_log,"tmpsentnum=%d,tmpframenum=%d,total_frames=%d\n",tmpsentnum,tmpframenum,total_frames);
	if(tmpsentnum != total_sents || tmpframenum != total_frames)
	{
		fprintf(fp_log, "frames or sentence num in target pfile and data pfile is not consistent.\n");
		exit(0);
	}
	else
		{
			fprintf(fp_log, "frames or sentence num in target pfile and data pfile is consistent.\n");
			}
	for(int i=0; i<total_sents;i++ )
	{
		if(tmpframepersent[i] != framesBeforeSent[i] )
		{
			fprintf(fp_log, "tails in target pfile and data pfile is not consistent---%d.\n",i);
			exit(0);
		}
	}
	delete []tmpframepersent;
#endif

	delete []header;
	fprintf(fp_log, "Get pfile info over: Training data has %u frames, %u sentences.\n", total_frames, total_sents);
}

/// get chunk nums and get frame start id for every chunk
void Interface::get_chunk_info(char *range)
{
	char *p, *st, *en;
	int sentid;
	int count_chunk = 1;
	int cur_frames_num = 0;
	int cur_frames_lost =0;
	int cur_frame_id = 0;
	int cur_chunk_frames =0;
	int frames_inc;
	int next_st;

	if(NULL ==(p =strstr(range,"-")))
	{
		fprintf(fp_log,"sent range: %s format error.\n",range);
		exit(0);
	}
	else{
		en = p+1;
		*p = '\0';
		st = range;
		sent_st = atoi(st);
		sent_en = atoi(en);
		if(sent_en < sent_st || sent_st < 0 || sent_en >= total_sents){
			fprintf(fp_log,"sent range: %d to %d number error.\n",sent_st,sent_en);
			exit(0);
		}
	}

	if(sent_st ==0){
		cur_frame_id = 0;
	}
	else{
		cur_frame_id = framesBeforeSent[sent_st -1];
	}
	chunk_frame_st[0] = cur_frame_id;

	for(sentid =sent_st; sentid <= sent_en; sentid++)
	{
		frames_inc = framesBeforeSent[sentid] - cur_frame_id;
		cur_frame_id = framesBeforeSent[sentid];
		if(frames_inc >= para->fea_context){
			cur_frames_lost = para->fea_context -1;
		}
		else{
			cur_frames_lost = frames_inc;
		}
		cur_frames_num = frames_inc - cur_frames_lost;
		cur_chunk_frames += cur_frames_num;
		while(cur_chunk_frames >= para->traincache ){
			next_st = cur_frame_id -(cur_chunk_frames - para->traincache);
			if(next_st < total_frames){
				chunk_frame_st[count_chunk] = next_st;
				count_chunk++;
				cur_chunk_frames = (cur_frame_id - next_st > para->fea_context -1)?(cur_frame_id - next_st - para->fea_context +1):0;
			}
		}
	}
	total_chunks = count_chunk;
	total_samples = (total_chunks -1) *para->traincache + cur_chunk_frames;

	fprintf(fp_log, "Get chunk info over: Training sentences have %d chunks, %d samples.\n", total_chunks, total_samples);
}

/// get chunk nums and get frame start id for every chunk
void Interface::get_chunk_info_cv(char *range)
{
	char *p, *st, *en;
	int sentid;
	int count_chunk = 1;
	int cur_frames_num = 0;
	int cur_frames_lost =0;
	int cur_frame_id = 0;
	int cur_chunk_frames =0;
	int frames_inc;
	int next_st;

	if(NULL ==(p =strstr(range,"-")))
	{
		fprintf(fp_log,"cv sent range: %s format error.\n",range);
		exit(0);
	}
	else{
		en = p+1;
		*p = '\0';
		st = range;
		cv_sent_st = atoi(st);
		cv_sent_en = atoi(en);
		if(cv_sent_en < cv_sent_st || cv_sent_st < 0 || cv_sent_en >= total_sents){
			fprintf(fp_log,"cv sent range: %d to %d number error.\n",cv_sent_st, cv_sent_en);
			exit(0);
		}
	}

	if(cv_sent_st ==0){
		cur_frame_id = 0;
	}
	else{
		cur_frame_id = framesBeforeSent[cv_sent_st -1];
	}
	cv_chunk_frame_st[0] = cur_frame_id;

	for(sentid = cv_sent_st; sentid <= cv_sent_en; sentid++)
	{
		frames_inc = framesBeforeSent[sentid] - cur_frame_id;
		cur_frame_id = framesBeforeSent[sentid];
		if(frames_inc >= para->fea_context){
			cur_frames_lost = para->fea_context -1;
		}
		else{
			cur_frames_lost = frames_inc;
		}
		cur_frames_num = frames_inc - cur_frames_lost;
		cur_chunk_frames += cur_frames_num;
		while(cur_chunk_frames >= para->traincache ){
			next_st = cur_frame_id -(cur_chunk_frames - para->traincache);
			if(next_st < total_frames){
				cv_chunk_frame_st[count_chunk] = next_st;
				count_chunk++;
				cur_chunk_frames = (cur_frame_id - next_st > para->fea_context -1)?(cur_frame_id - next_st - para->fea_context +1):0;
			}
		}
	}

	cv_total_chunks = count_chunk;
	cv_total_samples = (cv_total_chunks -1) *para->traincache + cur_chunk_frames;

	fprintf(fp_log, "Get cv chunk info over: CV sentences have %d chunks, %d samples.\n", cv_total_chunks, cv_total_samples);
}

///// read one chunck frames by index
int Interface::Readchunk(int chunk_index)  
{
	long int offset;
	int frames_need_read;
	int frames_processed;
	int samples_in_chunk;
	int size_perframe;
	int cur_sent;
	int cur_frame_of_sent;
	int cur_frame_id;
	int cur_sample;
	int *sample_index;
	float *dataori;
	float *targori;
	int i,j,k,t;
	///Read data pfile
	size_perframe = (para->fea_dim +2) *sizeof(float);	
	offset = PFILE_HEADER_SIZE + chunk_frame_st[chunk_index]* (long int)size_perframe;
	if(chunk_index == total_chunks -1){
		frames_need_read = framesBeforeSent[sent_en] - chunk_frame_st[chunk_index];
		samples_in_chunk = total_samples - para->traincache *chunk_index;
	}
	else{
		samples_in_chunk = para->traincache;
		frames_need_read = chunk_frame_st[chunk_index +1] - chunk_frame_st[chunk_index];
	}
	if(0 != fseek(fp_data, offset,0)){
		fprintf(fp_log,"data pfile cannot fseek to chunk %d.\n",chunk_index);
		exit(0);
	}

	sample_index = new int [samples_in_chunk];
	for(i=0; i< samples_in_chunk; i++){
		sample_index[i] = i;
	}
	GetRandIndex(sample_index, samples_in_chunk);

	dataori = new float[frames_need_read *(para->fea_dim +2)];
	fread((char *)dataori,size_perframe,frames_need_read,fp_data);
	swap32( (int*)&dataori[0]);
	cur_sent = *((int*) &dataori[0]);
	for(i =0;i < frames_need_read; i++){
		for(j =0; j< para->fea_dim;j++){
			swap32((int *) (&dataori[2+j +i*(para->fea_dim +2)]));
			dataori[2+j +i*(para->fea_dim +2)] -= mean[j];
			dataori[2+j +i*(para->fea_dim +2)] *= dVar[j];
		}
	}
	frames_processed = 0;
	cur_frame_id = chunk_frame_st[chunk_index];
	cur_sample = 0;

	while(frames_processed != frames_need_read){
		if(framesBeforeSent[cur_sent] > frames_need_read + chunk_frame_st[chunk_index]){
			cur_frame_of_sent = frames_need_read - frames_processed;
		}
		else{
			cur_frame_of_sent = framesBeforeSent[cur_sent] - cur_frame_id;
		}
		for(j =0; j<= cur_frame_of_sent - para->fea_context;j++){
			for(i =0;i< para->fea_context;i++){
				for(k=0;k< para->fea_dim;k++){
					para->indata[0][sample_index[cur_sample]* para->layersizes[0] +k +i *para->fea_dim] = dataori[(frames_processed +j +i) *(2+para->fea_dim) +k+2];
				}
			}
			cur_sample++;
		}

		cur_frame_id = framesBeforeSent[cur_sent];
		cur_sent++;
		frames_processed += cur_frame_of_sent;
	}

	///Read targ pfile
	size_perframe = (para->layersizes[numlayers-1] +2) *sizeof(float);
	offset = PFILE_HEADER_SIZE + chunk_frame_st[chunk_index]* (long int) size_perframe;

	if(0 != fseek(fp_targ, offset,0)){
		fprintf(fp_log,"targ pfile cannot fseek to chunk %d.\n",chunk_index);
		exit(0);
	}
	targori = new float[frames_need_read *(para->layersizes[numlayers-1] +2)];
	fread((char *)targori,size_perframe,frames_need_read,fp_targ);
	swap32( (int*)&targori[0]);
	cur_sent = *((int*) &targori[0]);
	for(i =0;i < frames_need_read; i++){
		for(j = 0; j< para->layersizes[numlayers-1];j++){
			swap32((int *) (&targori[2+j +i*(para->layersizes[numlayers-1] +2)]));
			targori[2+j +i*(para->layersizes[numlayers-1] +2)] -= mean[j%para->fea_dim];
			targori[2+j +i*(para->layersizes[numlayers-1] +2)] *= dVar[j%para->fea_dim];
		}
	}
	frames_processed = 0;
	cur_frame_id = chunk_frame_st[chunk_index];
	cur_sample = 0;

	while(frames_processed != frames_need_read){
		if(framesBeforeSent[cur_sent] > frames_need_read + chunk_frame_st[chunk_index]){
			cur_frame_of_sent = frames_need_read - frames_processed;
		}
		else{
			cur_frame_of_sent = framesBeforeSent[cur_sent] - cur_frame_id;
		}
		for(j =0; j<= cur_frame_of_sent - para->fea_context;j++){
			for(k=0;k< para->layersizes[numlayers-1];k++){
			para->targ[0][sample_index[cur_sample]* para->layersizes[numlayers-1] +k ] = targori[(frames_processed +j+ para->targ_offset) *(2+para->layersizes[numlayers-1]) +k+2];	  
			  }
			cur_sample++;
		}

		cur_frame_id = framesBeforeSent[cur_sent];
		cur_sent++;
		frames_processed += cur_frame_of_sent;
	}

	delete []sample_index;
	delete []dataori;
	delete []targori;
	return samples_in_chunk;
}

///// read one chunck frames by index
int Interface::Readchunk_cv(int chunk_index)  
{
	long int offset;
	int frames_need_read;
	int frames_processed;
	int samples_in_chunk;
	int size_perframe;
	int cur_sent;
	int cur_frame_of_sent;
	int cur_frame_id;
	int cur_sample;
	int *sample_index;
	float *dataori;
	float *targori;
	int i,j,k,t;

	///Read data pfile
	size_perframe = (para->fea_dim +2) *sizeof(float);
	offset = PFILE_HEADER_SIZE + cv_chunk_frame_st[chunk_index]* (long int)size_perframe;

	if(chunk_index == cv_total_chunks -1){
		frames_need_read = framesBeforeSent[cv_sent_en] - cv_chunk_frame_st[chunk_index];
		samples_in_chunk = cv_total_samples - para->traincache *chunk_index;
	}
	else{
		samples_in_chunk = para->traincache;
		frames_need_read = cv_chunk_frame_st[chunk_index +1] - cv_chunk_frame_st[chunk_index];
	}

	if(0 != fseek(fp_data, offset,0)){
		fprintf(fp_log,"data pfile cannot fseek to chunk %d.\n",chunk_index);
		exit(0);
	}

	sample_index = new int [samples_in_chunk];
	for(i=0; i< samples_in_chunk; i++){
		sample_index[i] = i;
	}

	dataori = new float[frames_need_read *(para->fea_dim +2)];
	fread((char *)dataori,size_perframe,frames_need_read,fp_data);
	swap32( (int*)&dataori[0]);
	cur_sent = *((int*) &dataori[0]);
	for(i =0;i < frames_need_read; i++){
		for(j =0; j< para->fea_dim;j++){
			swap32((int *) (&dataori[2+j +i*(para->fea_dim +2)]));
			dataori[2+j +i*(para->fea_dim +2)] -= mean[j];
			dataori[2+j +i*(para->fea_dim +2)] *= dVar[j];
		}
	}	

	frames_processed = 0;
	cur_frame_id = cv_chunk_frame_st[chunk_index];
	cur_sample = 0;

	while(frames_processed != frames_need_read){
		if(framesBeforeSent[cur_sent] > frames_need_read + cv_chunk_frame_st[chunk_index]){
			cur_frame_of_sent = frames_need_read - frames_processed;
		}
		else{
			cur_frame_of_sent = framesBeforeSent[cur_sent] - cur_frame_id;
		}

		for(j =0; j<= cur_frame_of_sent - para->fea_context;j++){
			for(i =0;i< para->fea_context;i++){
				for(k=0;k< para->fea_dim;k++){
					para->indata[0][sample_index[cur_sample]* para->layersizes[0] +k +i *para->fea_dim] = dataori[(frames_processed +j +i) *(2+para->fea_dim) +k+2];
				}
			}

			cur_sample++;
		}

		cur_frame_id = framesBeforeSent[cur_sent];
		cur_sent++;
		frames_processed += cur_frame_of_sent;
	}

	///Read targ pfile
	size_perframe = (para->layersizes[numlayers-1] +2) *sizeof(float);
	offset = PFILE_HEADER_SIZE + cv_chunk_frame_st[chunk_index]* (long int) size_perframe;

	if(0 != fseek(fp_targ, offset,0)){
		fprintf(fp_log,"targ pfile cannot fseek to chunk %d.\n",chunk_index);
		exit(0);
	}
	targori = new float[frames_need_read *(para->layersizes[numlayers-1] +2)];
	fread((char *)targori,size_perframe,frames_need_read,fp_targ);
	swap32( (int*)&targori[0]);
	cur_sent = *((int*) &targori[0]);
	for(i =0;i < frames_need_read; i++){
		for(j = 0; j< para->layersizes[numlayers-1];j++){
			swap32((int *) (&targori[2+j +i*(para->layersizes[numlayers-1] +2)]));
			targori[2+j +i*(para->layersizes[numlayers-1] +2)] -= mean[j%para->fea_dim];
			targori[2+j +i*(para->layersizes[numlayers-1] +2)] *= dVar[j%para->fea_dim];
		}
	}
	frames_processed = 0;
	cur_frame_id = cv_chunk_frame_st[chunk_index];
	cur_sample = 0;

	while(frames_processed != frames_need_read){
		if(framesBeforeSent[cur_sent] > frames_need_read + cv_chunk_frame_st[chunk_index]){
			cur_frame_of_sent = frames_need_read - frames_processed;
		}
		else{
			cur_frame_of_sent = framesBeforeSent[cur_sent] - cur_frame_id;
		}
		for(j =0; j<= cur_frame_of_sent - para->fea_context;j++){
			for(k=0;k< para->layersizes[numlayers-1];k++){
			para->targ[0][sample_index[cur_sample]* para->layersizes[numlayers-1] +k ] = targori[(frames_processed +j+ para->targ_offset) *(2+para->layersizes[numlayers-1]) +k+2];	  
			  }
			cur_sample++;
		}

		cur_frame_id = framesBeforeSent[cur_sent];
		cur_sent++;
		frames_processed += cur_frame_of_sent;
	}

	delete []sample_index;
	delete []dataori;
	delete []targori;
	return samples_in_chunk;
}

void Interface::GetRandWeight(float *vec, float min, float max, int len)  ///// get randem vector with uniform distribution
{
	for(int i =0;i< len;i++)
	{
		vec[i] = drand48()*(max -min) +min;
	}
}

void Interface::GetRandIndex(int *vec, int len)  
{
	int i;
	int idx;
	int tmp;
	for(i =0 ;i< len -1;i++){
		idx = lrand48() % (len-i);
		tmp = vec[idx];
		vec[idx] = vec[len -1 -i];
		vec[len -1 -i] = tmp;
	}
}

void Interface::get_uint(const char* hdr, const char* argname, unsigned int* val)
{
	const char* p;		// Pointer to argument
	int count = 0;		// Number of characters scanned

	// Find argument in header
	p = strstr(hdr, argname);
	if (p==NULL){
		fprintf(fp_log, "pfile header format is Not correct.\n");
		exit(0);
	}
	// Go past argument name
	p += strlen(argname);
	// Get value from stfing
	sscanf(p, " %u%n", val, &count);

	// We expect to pass one space, so need >1 characters for success.
	if (count <= 1){
		fprintf(fp_log, "%s num in pfile header is Not correct.\n",argname);
		exit(0);
	}
}

void Interface::read_tail(FILE *fp, long int file_offset, unsigned int sentnum, int *out)
{
	long int offset = file_offset;
	offset += 4;
	fseek(fp,offset,0);
	if(sentnum !=(fread((char*)out, sizeof(int), sentnum,fp))){
		fprintf(fp_log, "pfile tail is Not correct.\n");
		exit(0);
	}

	for(int i= 0; i< sentnum; i++){
		swap32((int*) (&out[i]));
	}
}
