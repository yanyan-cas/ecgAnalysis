function [ class ] = assessment( pred )
%ASSESSMENT Summary of this function goes here
%   Detailed explanation goes here
length = size(pred, 2);
N=0;
S=0;
V=0;
F=0;
Q=0;
for i=1:length
    if (pred(1,i)==1 || pred(1,i)==2 || pred(1,i)==3 || pred(1,i)==11 || pred(1,i)==34)
        N = N+1;
    end
    if (pred(1,i)==4 || pred(1,i)==7 || pred(1,i)==8 || pred(1,i)==9 )
        S = S+1;
    end
    if (pred(1,i)==5 || pred(1,i)==10 || pred(1,i)==31 )
        V = V+1;
    end
    if (pred(1,i)==6)
        F = F+1;
    end
    if (pred(1,i)==12 || pred(1,i)==13 || pred(1,i)==38 )
        Q = Q+1;
    end
end
class =[N;S; V; F; Q ];

end

