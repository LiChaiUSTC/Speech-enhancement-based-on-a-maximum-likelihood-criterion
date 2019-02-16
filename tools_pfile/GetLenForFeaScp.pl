use strict;
use File::Path;
use File::Basename;

@ARGV == 4 || die "Usage: pl scp_in scp_len nDim nSplit\n";
my ($scp_in, $scp_len, $nDim, $nSplit) = @ARGV;
my $config = "configHCopyFBK.be.txt";

::SplitFile($scp_in, $nSplit);

my @pid;
foreach my $i (1..$nSplit)
{
	defined($pid[$i]=fork) || die "Can't fork: $!";
	unless($pid[$i])
	{
		GetLenForWavScp("$scp_in.$i", "$scp_len.$i", $nDim);
		# ::LPS($scp_in, $i);
		#::FBK($scp_in, $i);
		#::Wav2RAW($scp_in, $i);
		exit(0);
	}
}

foreach my $i (1..$nSplit)
{
	waitpid($pid[$i],0);
}

foreach (1..$nSplit)
{
	unlink("$scp_in$_");
}

::MergeFile($scp_len, $nSplit);

foreach (1..$nSplit)
{
	unlink("$scp_in.$_");
	unlink("$scp_len.$_");
}

sub ::GetLenForWavScp
{
	my ($scp_in, $scp_len, $nDim) = @_;
	open(IN, $scp_in) || die $!;
	open(OUT, ">", $scp_len) || die $!;
	my $size;
	while(<IN>)
	{
		chomp;
		$size = -s $_;
		$size = ($size - 12) / 4 / $nDim;

		print OUT $size."\n";

		if($size <= 0)
		{
			print "fea len wrong: $_\n";
		}
		elsif($size <= 30)
		{
			print "fea too small(300ms): $_\n";
		}
        elsif($size >= 3000)
		{
			print "fea too large(30s): $_\n";
		}
	}
}

sub ::FBK
{
	my ($file, $i) = @_;
	MakePathForScpNEW("$file$i");
	my $cmd = "HCopy.exe -C $config -S $file$i";
	system($cmd);
}

sub ::SplitFile
{
	my ($file, $n, $suffix, $separator) = @_;
	my ($i, $j, $k, $file_iii);
	my $line;
	my $numPerSplit;
	my $numLeft;
	
	my @lines;
	my @numPerSplit;
	
	$suffix = "" if(!defined($suffix));
	$separator = "." if(!defined($separator));
	
	open(IN, "$file") || die "Error: Can't read file: $file, $!";
	@lines = <IN>;
	close(IN);
	$i = @lines;
	
	$numPerSplit = int($i/$n);
	$numLeft = $i%$n;
	die "Error: Number of lines per split is 0, Total: $i, nSplit: $n\n" if($numPerSplit == 0);
	
	$numPerSplit[0] = $i;##原文件总行数
	foreach(1..$n)
	{
		$numPerSplit[$_] = $numPerSplit;
	}
	foreach(1..$numLeft)
	{
		$numPerSplit[$_]++;
	}
	print "Number of lines for each split(SplitFile): @numPerSplit\n";
	
	foreach $k(1..$n)
	{
		$file_iii = $file.$separator.$k.$suffix;
		open(OUT,">$file_iii") || die "Error: Can't write file: $file_iii, $!";
		foreach $j(1..$numPerSplit[$k])
		{
			print OUT shift(@lines);
		}
		close(OUT);
	}
}

sub ::MergeFile
{
	my ($file, $n, $suffix, $separator, $skip) = @_;
	my ($iii, @array, $file_iii);
	
	$suffix    = "" if(!defined($suffix));
	$separator = "." if(!defined($separator));
	$skip      = 0 if(!defined($skip));
	
	open(OUTPUT, ">$file") || die "Error: Can't write file: $file, $!";
	
	foreach $iii(1..$n)
	{
		my $skip_iii = $skip;
		$file_iii = $file.$separator.$iii.$suffix;
		print $file_iii."\n";
		open(INPUT,"$file_iii") || die "Error: Can't read file: $file_iii, $!";
		@array = <INPUT>;
		close(INPUT);

		shift(@array) while($skip_iii--);
		print OUTPUT (@array);

		unlink $file_iii;
	}
	close(OUTPUT);
}

sub ::Wav2RAW
{
	my($file, $i) = @_;
	open(IN, "$file$i") || die "Can't open file: $file$i\n";
	while(<IN>)
	{
		chomp;
		my($first, $second) = split(" ", $_);
		my $pcm = $second;
		$pcm =~ s/\.wav/\.pcm/;
		my $cmd = "Bin\\WAV2RAW $second $pcm";
		system($cmd);
	}
	close(IN);
}

sub ::LPS
{
	my ($file, $i) = @_;
	MakePathForScpNEW("$file$i");
	open(IN, "$file$i") || die "Can't open file: $file$i\n";
	while(<IN>)
	{
		chomp;
		my($first, $second) = split(" ", $_);
		#firstmy $pcm = $second;
		#$pcm =~ s/\.wav/\.pcm/;
		my $cmd = "bin/Wav2LogSpec_be_25ms -F RAW -fs 16 $first $second";
		system($cmd);
	}
	close(IN);
}

sub ::LPS_readwav
{
	my ($file, $i) = @_;
	MakePathForScpNEW("$file$i");
	open(IN, "$file$i") || die "Can't open file: $file$i\n";
	while(<IN>)
	{
		chomp;
		my($first, $second) = split(" ", $_);
		my $pcm = $first;
		$pcm =~ s/\.wav/\.pcm/;
		system("bin/WAV2RAW $first $pcm");
		my $cmd = "bin/Wav2LogSpec_be -F RAW -fs 16 $pcm $second";
		system($cmd);
	}
	close(IN);
}

sub ::MakePathForScp
{
	use File::Basename;
	
	my ($scp) = @_;
	open(IN, $scp) || die "Can't open file: $scp\n";
	while(<IN>)
	{
		chomp;
		my($strFilename, $strPathname, $suffix) = fileparse($_);
		::MakePathIfNotExist($strPathname);
	}
	close(IN);
}

sub ::MakePathForScpNEW
{
	use File::Basename;
	
	my ($scp) = @_;
	open(IN, $scp) || die "Can't open file: $scp\n";
	while(<IN>)
	{
		chomp;
		my($first, $second) = split(" ", $_);
		my($strFilename, $strPathname, $suffix) = fileparse($second);
		::MakePathIfNotExist($strPathname);
	}
	close(IN);
}

sub ::MakePathIfNotExist
{
	use File::Path;
	my ($strPathname) = @_;
	
	if(!-e $strPathname)
	{
		mkpath($strPathname, 1, 0755) || die "Error: Can't make path: $strPathname, $!";
	}
}

sub ::MakeDirIfNotExist
{
	my ($strPathname) = @_;
	
	if(!-e $strPathname)
	{
		mkdir($strPathname, 0755) || die "Error: Can't make directory: $strPathname, $!";
	}
}