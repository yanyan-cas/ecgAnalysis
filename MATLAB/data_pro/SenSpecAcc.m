function [ SNS,SPC] = SenSpecAcc(class, pre_class )
%SENSPECACC Summary of this function goes here
%   Detailed explanation goes here
SNS=[];
SPC=[];
for i=1:5 
    TP=min(class(i), pre_class(i));
    TN=abs(pre_class(i)-class(i));
    if (class(i)>pre_class(i))
        FP=class(i)-pre_class(i);
        specity=FP/(FP+TN);
    else
        specity=0;
    end
    sensity=TP/(TP+TN);
    SNS=[SNS;sensity];
    SPC=[SPC;specity];
end
end

