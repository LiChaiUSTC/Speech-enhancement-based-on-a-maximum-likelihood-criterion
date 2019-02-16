clc
clear all

expansion_frames=7;
output_dim=257;
norm=load('train_noisy.norm');
global_mean=norm(1:output_dim);   
global_var=norm((output_dim+1):length(norm)); %the reciprocal of standard deviation

load('mlp.mat'); % the well-trained model
weights12=weights12';
w1=[weights12;bias2];
weights23=weights23';
w2=[weights23;bias3];
weights34=weights34';
w3=[weights34;bias4];
weights45=weights45';
w4=[weights45;bias5];
noisy_lsp_list='data.scp'; %test set
flsp=fopen(noisy_lsp_list);
tline=fgetl(flsp);
line_num=0;
system('mkdir ./temp');
while(tline~=-1)
    line_num=line_num+1; 
    cmd=sprintf('sox %s ./temp/temp.raw',tline); 
    system(cmd);
    cmd=sprintf('./SourceCode_Wav2LogSpec_be/Wav2LPS_be -F RAW -fs 16 %s %s', './temp/temp.raw', './temp/noisy_lps'); %get noisy-test feature
    system(cmd); 
    wav_se_tline=strrep(tline,'.wav','_enhanced.wav');%the path of enhanced speech
    [htkdata,nframes,sampPeriod,sampSize,paramKind]=readHTK_new('./temp/noisy_lps','be');
    htkdata=htkdata';
    for i=1:1:size(htkdata,2)
        htkdata(:,i) = (htkdata(:,i)-global_mean(i))*global_var(i);  
    end;
    frame_expand(htkdata,expansion_frames); %%%write to ¡®input_lsp.txt¡¯
    sources=load('input_lsp.txt');    
    N=size(sources,1);
    data = [sources ones(N,1)];
    %%%% without dropout + Sigmoid
    [size1,size2]=size(w1);
    new_w1=[w1(1:1:size1-1,:);w1(size1,:)];
    w1probs = 1./(1 + exp(-data*new_w1));
    w1probs = [w1probs  ones(N,1)];
    
    [size1,size2]=size(w2);
    new_w2=[w2(1:1:size1-1,:);w2(size1,:)];
    w2probs = 1./(1 + exp(-w1probs*new_w2)); 
    w2probs = [w2probs ones(N,1)];
    
    [size1,size2]=size(w3);
    new_w3=[w3(1:1:size1-1,:);w3(size1,:)];  
    w3probs = 1./(1 + exp(-w2probs*new_w3));
    w3probs = [w3probs ones(N,1)];
    
    [size1,size2]=size(w4);
    new_w4=[w4(1:1:size1-1,:);w4(size1,:)];
    dataout = w3probs*new_w4;  
    
    for i=1:1:output_dim
        dataout(:,i) = dataout(:,i)/global_var(i)+global_mean(i);
    end;
    writeHTK_new('./temp/out.htk', dataout,nframes, 160000, output_dim*4, 9, 'be');
    cmd=sprintf('./SourceCode_LogSpec2Wav_be/LPS2Wav_be %s %s ./temp/out.htk info.txt %s -F RAW -fs 16','./temp/temp.raw','./temp/temp.raw','./temp/temp_se.raw');
    system(cmd);
    cmd=sprintf('sox -t raw -e signed-integer -r 16000 -c 1 -b 16 %s %s', './temp/temp_se.raw', wav_se_tline);
    system(cmd);
    tline=fgetl(flsp);  %% goto next line
end
fclose(flsp);
 


