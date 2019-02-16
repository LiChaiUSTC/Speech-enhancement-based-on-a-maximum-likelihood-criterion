#!/usr/local/bin/perl -w

#attentions: please confirm that all the input files from windows system are converted from DOS to UNIX
#attentions: please add path: export PATH=$PATH:/home/zhoupan/tools/QN/basic/bin

my $ROOT_DIR  = ".";
my $TOOL_DIR 	= "$ROOT_DIR/tools/QN/atlas1/bin";
my $CF_DIR		= "$ROOT_DIR";

my $len_scp   = "$CF_DIR/frame_numbers.len";

my $fea_scp   = "$CF_DIR/train_noisy.scp";

my $fea_tr    = "$CF_DIR/train_noisy.pfile";


my $part_num  = 10;
my $split_num = 1;  
my $i;
my @pid;
my $pfile_list;


print "ok\n";
system("split -l $part_num -d -a 1 $fea_scp $fea_scp");
system("split -l $part_num -d -a 1 $len_scp $len_scp");

foreach $i (0..$split_num-1)
{
	defined($pid[$i] = fork) or die "can't fork: $!";
	unless ($pid[$i])
	{
		system("$TOOL_DIR/feacat -period 16.0 -ipformat htk -deslenfile $len_scp$i -lists  -o $fea_tr$i $fea_scp$i");###这地方16ms要非常注意
		exit(0);
	}
}

$pfile_list = "";
foreach $i (0..$split_num-1)
{
	waitpid($pid[$i], 0);
	$pfile_list = $pfile_list . "$fea_tr$i ";
}

system("$TOOL_DIR/pfile_concat -o $fea_tr $pfile_list");

