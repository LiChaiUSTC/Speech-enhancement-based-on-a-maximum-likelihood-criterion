function frame_expand(feature,context_num)
[inFrames,inDims]=size(feature);
data1=[];
for t=1:1:inFrames
    data2=[];
for c=(context_num-1)/2:-1:1 %%%%t�������չ
    if(t-c<=0) 
    datatmp=feature(1,:);
    else
    datatmp=feature(t-c,:);
    end
    data2=[data2 datatmp];
end

datatmp=feature(t,:);
data2=[data2 datatmp]; %%%%t������֡

 for c=1:(context_num-1)/2 %%%%t���ұ���չ
    if(t+c>=inFrames) 
    datatmp=feature(inFrames,:);
    else
    datatmp=feature(t+c,:);
    end
    data2=[data2 datatmp];
end

data1=[data1;data2];
end
save input_lsp.txt data1 -ascii;
end