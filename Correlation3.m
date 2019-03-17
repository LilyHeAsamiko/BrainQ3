clear
zeffiro_interface
load('zef.mat');
ind = find(evalin('base','zef.sigma(zef.brain_ind,2)')==1);
ind2 = find(evalin('base','zef.sigma(zef.brain_ind,2)')==2);
ind3 = find(evalin('base','zef.sigma(zef.brain_ind,2)')==3);

Ind = unique(zef.source_interpolation_ind{1}(ind,:));
Ind2 = unique(zef.source_interpolation_ind{1}(ind2,:));
Ind3 = unique(zef.source_interpolation_ind{1}(ind3,:));

for i = 1:length(zef.reconstruction)
    X(:,i) = zef.reconstruction{i};
end
l = length(X);
XX = sqrt(X(1:l/3,:).^2+X(l/3+1:2*l/3,:).^2+X(2*l/3+1:l,:).^2);
A = mean(XX(Ind,:),2);
B = mean(XX(Ind2,:),2);

%B = gpuArray(XX(Ind,:));
for i = 1: length(A)
    for j = 1: length(B)
        C(i,j)= (A(i)-mean(A))*(B(j)-mean(B))'/sqrt(sum((A(i)-mean(A)).^2)*sum((B(j)-mean(B)).^2));
    end
end

C2 = imresize(C,0.1);
figure,
imagesc(C)
imagesc(C2)

reconstruction = zeros(length(XX),1);
reconstruction(Ind)= abs(mean(C,2));
reconstruction(Ind2)= abs(mean(C,1));
clear('zef.reconstruction')
%for i = 1:length(zef.reconstruction)
%    zef.reconstruction{i}=repmat(reconstruction,3,1);
%end
zef.number_of_frames = 1; 
re = repmat(reconstruction',3,1);
zef.reconstruction = re(:);