use warnings;

$noisy_pfile="./train_noisy";
system("./tools/QN/atlas1/bin/qnnorm norm_ftrfile=$noisy_pfile.pfile output_normfile=$noisy_pfile.norm");