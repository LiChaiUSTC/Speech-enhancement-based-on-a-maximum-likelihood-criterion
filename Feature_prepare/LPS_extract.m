clc
clear all

noisy_lsp_list='data.scp'; % the list of data address
flsp=fopen(noisy_lsp_list);
tline=fgetl(flsp);
line_num=0;
while(tline~=-1)
    line_num=line_num+1; 
    tline2=strrep(tline,'.wav','.lps');
    cmd=sprintf('sox %s ./temp.raw',tline); 
    system(cmd);
    cmd=sprintf('./SourceCode_Wav2LogSpec_be/Wav2LPS_be -F RAW -fs 16 %s %s', './temp.raw', tline2); %get LPS feature
    system(cmd); 
    tline=fgetl(flsp);  %% goto next line
end
fclose(flsp);
 


