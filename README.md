
# DNN-based speech enhancement optimized by a maximum likelihood criterion rather than the conventional MMSE criterion
This repository contains the code and supplementary result for the paper "Using Generalized Gaussian Distributions to Improve Regression Error Modeling for Deep-Learning-Based Speech Enhancement" (submitted to IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING)

## Step1: Prepare the input and output files. 
### Step1.1: Extract log-power spectrum (LPS) features
#### cd Feature_prepare
1.cd SourceCode\_Wav2LogSpec_be

Execute `make` to generate the executable file "Wav2LPS\_be"

Note: Set the sampling frequency, frame length and frame shift in the source code "Wav2LogSpec\_be.c" according to your own needs. In our paper, we set the sampling frequency, frame length and frame shift to 16kHz, 32ms and 16ms respectively.

2.Extract LPS features with command below:

`matlab -nodesktop -nosplash -r LPS_extract`

### Step1.2: Packaged into a Pfile file
#### cd tools_pfile
1.Calculate the number of frames with command below:

`perl GetLenForFeaScp.pl train_noisy.scp frame_numbers.len 257 1`

2.Use [quicknet toolset](http://www1.icsi.berkeley.edu/Speech/qn.html) to prepare Pfile as the input and the output files with command below:

`perl pfile_noisy.pl`

`perl pfile_clean.pl`

Pfile is the big file of all training features.

3.Calculate the mean and standard deviation with the command below for normalization by Z-score:

`perl get_norm.pl`

Note that the reciprocal of standard deviation rather than standard deviation is obtained after executing this command.

## Step2: Training
#### Installation
1.cuda

2.g++
#### cd Train\_code\_ML\_GGD
This code is for feed-forward DNN training, where the optimization criterion can be 
conventional MMSE or our proposed objective function with GGD error model derived according to ML criterion.

1.Execute `make` to generate the executable file "BPtrain_Sigmoid" 
    
2.You can train the feed-forward DNNs in the paper by calling "BPtrain_Sigmoid" with command below:

`perl finetune.pl`

Note: 

- The paramters "MLflag" and "shapefactor" in "finetune.pl" control the choice of the objective function.   
- When MLflag≠1, the classic β-norm function is selected as the objective function, where β=1 corresponds to the L1-norm, namely the least absolute deviation (LAD) and β=2 corresponds to the L2-norm, namely the MMSE.  
- When MLflag=1, the GGD error model based log-likelihood function is selected as the objective function, where "shapefactor" refers to the shape factor β in GGD. 

#### Implementation details
In this paper, we propose a new objective function. The codes for our proposed ML-GGD-DNN can be obtained by making minor modifications based on the code for MMSE-DNN. More specifically, we only need to modify the gradient of the objective function with respect to the output in the backpropagation part of the codes for MMSE-DNN.

The following codes are from the lines 408 to 423 of "BP_GPU.cu", which is to calculate the gradient of the objective funtion with respect to the output. The called functions are defined in "DevFunc.cu".

    DevSubClean2(streams, n_frames, cur_layer_units,shapefactor, dev.out, targ, cur_layer_dedx); 
    DevVecMulNum(streams, cur_layer_units * n_frames, cur_layer_dedx, 1.0f/n_frames, cur_layer_dedx);
    if(MLflag == 1) 
    {
    Deverror(streams, n_frames, cur_layer_units, dev.out, targ, realerror);
    Devabsolutevalus(streams,cur_layer_units * n_frames,realerror,errorabsolute);
    Devindex2(streams,n_frames*cur_layer_units, errorabsolute, shapefactor,errorabsolute2);
    DevSumcol(streams, n_frames, cur_layer_units, errorabsolute2, vec1);
    DevDivide(streams, cur_layer_units, vec1, vec1, n_frames);
    DevVecMulNum(streams, cur_layer_units, vec1, shapefactor, vec2);
    float ppp=1.0f/shapefactor;
    Devindex2(streams,cur_layer_units, vec2,ppp, scalefactor);
    Devfunc2(streams, n_frames,cur_layer_units, realerror,scalefactor, newobj,shapefactor);
	DevVecMulNum(streams, cur_layer_units * n_frames, newobj, 1.0f/n_frames, cur_layer_dedx);
    }

- When MLflag≠1, the β-norm function is selected as the objective function as follows:  
  ![公式](https://latex.codecogs.com/gif.latex?E(\boldsymbol{W})=\sum_{n=1}^{N}\sum_{d=1}^{D}|x_{n,d}-\hat{x}_{n,d}(y_{n-\tau}^{n&plus;\tau},\boldsymbol{W})|^{\beta},)

    where β=2 corresponds to the MMSE criterion and β=1 corresponds to the LAD criterion.  
    Then the backpropagation procedure with a SGD method is used to update DNN parameters **W** in the minibatch mode of M sample frames (In this paper, M=128).
    
    The function "DevSubClean2" achieves the calculation of the gradient of ![公式](https://latex.codecogs.com/gif.latex?E(\boldsymbol{W})) with respect to the output ![公式](https://latex.codecogs.com/gif.latex?\hat{x}_{n,d}(y_{n-\tau}^{n&plus;\tau},\boldsymbol{W})) as follows:

    ![公式](https://latex.codecogs.com/gif.latex?\frac{\partial&space;E(\boldsymbol{W})}{\partial&space;\hat{x}_{m,d}(y_{m-\tau}^{m&plus;\tau},\boldsymbol{W})}=\beta\mathop{\rm&space;sgn}\left(\hat{x}_{m,d}(y_{m-\tau}^{m&plus;\tau},\boldsymbol{W})-x_{m,d}\right)|x_{m,d}-\hat{x}_{m,d}(y_{m-\tau}^{m&plus;\tau},\boldsymbol{W})|^{\beta-1}.)

    

- When MLflag=1, the GGD error model based log-likelihood function is selected as the objective function as follows:
![公式](https://latex.codecogs.com/gif.latex?\ln&space;p(\boldsymbol{X}|\boldsymbol{Y},\boldsymbol{W},&space;\boldsymbol{\alpha})=ND\ln\frac{\beta}{2\Gamma(\frac{1}{\beta})}-N\sum_{d=1}^{D}&space;\ln&space;\alpha_d&space;-&space;\sum_{n=1}^{N}&space;\sum_{d=1}^{D}&space;\frac{|x_{n,d}-\hat{x}_{n,d}(\boldsymbol{y}_{n-\tau}^{n&plus;\tau},&space;\boldsymbol{W})|^{\beta}}{\alpha_d^{\beta}}.)

    We adopt maximum likelihood criterion to optimize both the DNN parameters **W** and GGD parameters **α**. In this paper, two algorithms for optimisation are proposed. Here, we only provide one optimization algorithm which is adopted in all the experiments for ML-GGD-DNN in our paper, namely the alternating two-step optimization algorithm.

    Maximizing the log-likelihood ![公式](https://latex.codecogs.com/gif.latex?\ln&space;p(\boldsymbol{X}|\boldsymbol{Y},\boldsymbol{W},&space;\boldsymbol{\alpha})) is equivalent to minimizing the following error function:
![公式](https://latex.codecogs.com/gif.latex?E(\boldsymbol{W},\boldsymbol{\alpha})=N\sum_{d=1}^{D}&space;\ln&space;\alpha_d&space;&plus;&space;\sum_{n=1}^{N}&space;\sum_{d=1}^{D}&space;\frac{|x_{n,d}-\hat{x}_{n,d}(\boldsymbol{y}_{n-\tau}^{n&plus;\tau},&space;\boldsymbol{W})|^{\beta}}{\alpha_d^{\beta}})

    Then **W** and **α** are alternatively optimized in each minibatch (M=128).  
    First, a closed solution of **α** referred to by "scalefactor" in the codes is derived by fixing **W** and minimizing ![公式](https://latex.codecogs.com/gif.latex?E(\boldsymbol{W},\boldsymbol{\alpha})) in the minibatch mode of M sample frames as follows:

    ![公式](https://latex.codecogs.com/gif.latex?\alpha_d=\left(\frac{\beta}{M}\sum_{m=1}^{M}|x_{m,d}-\hat{x}_{m,d}(\boldsymbol{y}_{m-\tau}^{m&plus;\tau},&space;\boldsymbol{W})|^{\beta}\right)^{\frac{1}{\beta}})
    
    Second, **W** is optimized by the backpropagation procedure with the SGD method by fixing **α**.
    The function "Devfun2" achieves the calculation of the gradient of ![公式](https://latex.codecogs.com/gif.latex?E(\boldsymbol{W},\boldsymbol{\alpha})) with respect to the output ![公式](https://latex.codecogs.com/gif.latex?\hat{x}_{n,d}(y_{n-\tau}^{n&plus;\tau},\boldsymbol{W})) as follows:

    ![公式](https://latex.codecogs.com/gif.latex?\frac{\partial&space;E(\boldsymbol{W},\boldsymbol{\alpha})}{\partial&space;\hat{x}_{m,d}(y_{m-\tau}^{m&plus;\tau},\boldsymbol{W})}=\frac{\beta}{\alpha_d^{\beta}}\mathop{\rm&space;sgn}\left(\hat{x}_{m,d}(y_{m-\tau}^{m&plus;\tau},\boldsymbol{W})-x_{m,d}\right)|x_{m,d}-\hat{x}_{m,d}(y_{m-\tau}^{m&plus;\tau},\boldsymbol{W})|^{\beta-1}.)


## Step3: Testing
#### cd Test_code
Select one well-trained model and change the suffix 'wts' to 'mat'. Then execute the following command:    

`matlab -nodesktop -nosplash -r decode`

## Demos:
#### cd Enh_demos
<p><img src="https://i.imgur.com/h6UB0qY.png" width="210" /> <img src="https://i.imgur.com/utBjeXl.png" width="210"/> <img src="https://i.imgur.com/lNuwlIQ.png" width="210" /> <img src="https://i.imgur.com/W23l6sf.png" width="210" /></p>
  <table width="1000" height="33" border="1">
    <tr>
      <td width="210"><div align="center">(a) <a href="Enh_demos/DestroyerOperations_SNR5_CLEAN_TEST_DR7_FDHC0_SI929.WAV">Clean </a></div></td>
      <td width="210"><div align="center">(b) <a href="Enh_demos/DestroyerOperations_SNR5_NOISY_TEST_DR7_FDHC0_SI929.wav">Noisy</a></div></td>
      <td width="210"><div align="center">(c) <a href="Enh_demos/DestroyerOperations_SNR5_MMSE_TEST_DR7_FDHC0_SI929.wav">MMSE</a></div></td>
      <td width="210"><div align="center">(d) <a href="Enh_demos/DestroyerOperations_SNR5_ML_TEST_DR7_FDHC0_SI929.wav">ML</a></div></td>
    </tr>
  </table>
<p><img src="https://i.imgur.com/lewAJ7P.png" width="210" /> <img src="https://i.imgur.com/KOM8RjY.png" width="210"/> <img src="https://i.imgur.com/wLIa5bL.png" width="210" /> <img src="https://i.imgur.com/cRy8qHS.png" width="210" /></p>
  <table width="1000" height="33" border="1">
    <tr>
      <td width="210"><div align="center">(a) <a href="Enh_demos/Factory1_SNR5_CLEAN_TEST_DR2_FPAS0_SI2204.WAV">Clean </a></div></td>
      <td width="210"><div align="center">(b) <a href="Enh_demos/Factory1_SNR5_NOISY_TEST_DR2_FPAS0_SI2204.wav">Noisy</a></div></td>
      <td width="210"><div align="center">(c) <a href="Enh_demos/Factory1_SNR5_MMSE_TEST_DR2_FPAS0_SI2204.wav">MMSE</a></div></td>
      <td width="210"><div align="center">(d) <a href="Enh_demos/Factory1_SNR5_ML_TEST_DR2_FPAS0_SI2204.wav">ML</a></div></td>
    </tr>
  </table>
<p><img src="https://i.imgur.com/qXuzGwz.png" width="210" /> <img src="https://i.imgur.com/04tXVKC.png" width="210"/> <img src="https://i.imgur.com/sel3lYn.png" width="210" /> <img src="https://i.imgur.com/g7g4VK2.png" width="210" /></p>
  <table width="1000" height="33" border="1">
    <tr>
      <td width="210"><div align="center">(a) <a href="Enh_demos/MachineGun_SNR5_CLEAN_TEST_DR2_FPAS0_SX404.WAV">Clean </a></div></td>
      <td width="210"><div align="center">(b) <a href="Enh_demos/MachineGun_SNR5_NOISY_TEST_DR2_FPAS0_SX404.wav">Noisy</a></div></td>
      <td width="210"><div align="center">(c) <a href="Enh_demos/MachineGun_SNR5_MMSE_TEST_DR2_FPAS0_SX404.wav">MMSE</a></div></td>
      <td width="210"><div align="center">(d) <a href="Enh_demos/MachineGun_SNR5_ML_TEST_DR2_FPAS0_SX404.wav">ML</a></div></td>
    </tr>
  </table>

