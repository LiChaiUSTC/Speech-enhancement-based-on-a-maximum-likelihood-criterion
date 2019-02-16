#!/usr/bin/perl
use strict;
use warnings;

my $numlayers   = 5;
my $beta        = 2;#sigmoid beta=2; ReLUS beta=0.5
my $flag        = 1;
my $root_dir    = ".";
my $fname       = "Rand_1799_3hid2048_257_beta2";
my $out_wts_dir = "$root_dir";
my $out_pfilename = "$out_wts_dir/$fname.wts";
my $cmd = "$root_dir/Gen_rand_net $numlayers 1799 2048 2048 2048 257 $out_wts_dir $out_pfilename $flag $beta";
system($cmd);

