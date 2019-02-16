use strict;
my $ROOT_DIR = "..";
my $i;
my $j;
my $line;
my $curacc;
my $preacc;
my $numlayers=5;

	my $lrate=0.1;
	my $layersizes = "1799";
	for(my $i=0;$i<$numlayers -2;$i++)
	{
		$layersizes	  .= ",2048";
	}	
	$layersizes	  .= ",257";
	my $node=2048;
	my $exe 						= "./BPtrain_Sigmoid";
	my $gpu_used				= 0;
	my $bunchsize				= 128;
	my $momentum				= 0.9;
	#MLflag=1 corresponds to the objective function, ML-DNN-GGD with heteroscedasticity assumption; 
	#MLflag=0 corresponds to the objective funciton, classic beta-norm, namely, ML-DNN-GGD with homoscedasticity assumption.
	#When MLflag=0 and shapefactor=2, the objective funciton corresponds to the conventional MMSE, namely, L2-norm.
	my $MLflag                  =1;
	my $shapefactor             =1;  #the value of beta, namely, the shape factor in GGD
	my $weightcost			= 0.00001;
	my $fea_dim					= 257;
	my $fea_context			= 7;
	my $traincache			= 102400;  ############ how many samples per chunk #102400
	my $init_randem_seed= 27870775;   ############ every epoch must change
	my $targ_offset			= 3;#(7-1)/2
	
	my $CF_DIR					= "$ROOT_DIR/tools_pfile";
	my $norm_file				= "$CF_DIR/train_noisy.norm";
	my $fea_file				= "$CF_DIR/train_noisy.pfile";
	my $targ_file				= "$CF_DIR/train_clean.pfile";
	
	my $train_sent_range		= "0-7";	#utterances for training from the training set
	my $cv_sent_range				= "8-9"; #utterances for cross valiation from the training set
	
	my $MLP_DIR					= "./MLGGD1";
	system("mkdir $MLP_DIR");

	my $outwts_file			= "$MLP_DIR/mlp.1.wts";
	my $log_file				= "$MLP_DIR/mlp.1.log";
	my $initwts_file		="./pretraining_weights/Rand_1799_3hid2048_257_beta2.wts";
	print "iter 1 lrate is $lrate\n"; 
	if( !-e $outwts_file){
	system("$exe" .
		" gpu_used=$gpu_used".
		" numlayers=$numlayers".
		" layersizes=$layersizes".
		" bunchsize=$bunchsize".
		" MLflag=$MLflag".
		" shapefactor=$shapefactor".
		" momentum=$momentum".
		" weightcost=$weightcost".
		" lrate=$lrate".
		" fea_dim=$fea_dim".
		" fea_context=$fea_context".
		" traincache=$traincache".
		" init_randem_seed=$init_randem_seed".
		" targ_offset=$targ_offset".
		" initwts_file=$initwts_file".
		" norm_file=$norm_file".
		" fea_file=$fea_file".
		" targ_file=$targ_file".
		" outwts_file=$outwts_file".
		" log_file=$log_file".
		" train_sent_range=$train_sent_range".
		" cv_sent_range=$cv_sent_range".
		" dropoutflag=0".
		" visible_omit=0.1".
		" hid_omit=0.1"
		);
	}  
  $preacc=$curacc;
	my $destep=0;
	for($i= 2;$i <= 10;$i++){

		$j = $i -1;  #???
		$initwts_file		= "$MLP_DIR/mlp.$j.wts";
		$outwts_file		= "$MLP_DIR/mlp.$i.wts";
		$log_file				= "$MLP_DIR/mlp.$i.log";
		$init_randem_seed  += 345;
		print "iter $i lrate is $lrate\n"; 
		if( !-e $outwts_file){
		system("$exe" .
		" gpu_used=$gpu_used".
		" numlayers=$numlayers".
		" layersizes=$layersizes".
		" bunchsize=$bunchsize".
		" MLflag=$MLflag".
		" shapefactor=$shapefactor".
		" momentum=$momentum".
		" weightcost=$weightcost".
		" lrate=$lrate".
		" fea_dim=$fea_dim".
		" fea_context=$fea_context".
		" traincache=$traincache".
		" init_randem_seed=$init_randem_seed".
		" targ_offset=$targ_offset".
		" initwts_file=$initwts_file".
		" norm_file=$norm_file".
		" fea_file=$fea_file".
		" targ_file=$targ_file".
		" outwts_file=$outwts_file".
		" log_file=$log_file".
		" train_sent_range=$train_sent_range".
		" cv_sent_range=$cv_sent_range".
		" dropoutflag=0".
		" visible_omit=0.1".
		" hid_omit=0.1"
		);
		}
	}
	for($i= 11;$i <= 50;$i++){
		$j = $i -1;   #???
		$initwts_file		= "$MLP_DIR/mlp.$j.wts";
		$outwts_file		= "$MLP_DIR/mlp.$i.wts";
		$log_file				= "$MLP_DIR/mlp.$i.log";
		$lrate *= 0.9;
		$init_randem_seed  += 345;
		print "iter $i lrate is $lrate\n"; 
		if( !-e $outwts_file){
		system("$exe" .
		" gpu_used=$gpu_used".
		" numlayers=$numlayers".
		" layersizes=$layersizes".
		" bunchsize=$bunchsize".
		" MLflag=$MLflag".
		" shapefactor=$shapefactor".
		" momentum=$momentum".
		" weightcost=$weightcost".
		" lrate=$lrate".
		" fea_dim=$fea_dim".
		" fea_context=$fea_context".
		" traincache=$traincache".
		" init_randem_seed=$init_randem_seed".
		" targ_offset=$targ_offset".
		" initwts_file=$initwts_file".
		" norm_file=$norm_file".
		" fea_file=$fea_file".
		" targ_file=$targ_file".
		" outwts_file=$outwts_file".
		" log_file=$log_file".
		" train_sent_range=$train_sent_range".
		" cv_sent_range=$cv_sent_range".
		" dropoutflag=0".
		" visible_omit=0.1".
		" hid_omit=0.1"
		);
		}
	}
