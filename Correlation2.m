clear
zeffiro_interface
load('zef.mat');
ind = find(evalin('base','zef.sigma(zef.brain_ind,2)')==1);
ind2 = find(evalin('base','zef.sigma(zef.brain_ind,2)')==2);
ind3 = find(evalin('base','zef.sigma(zef.brain_ind,2)')==3);

Ind = unique(zef.source_interpolation_ind{1}(ind,:));
Ind2 = unique(zef.source_interpolation_ind{1}(ind2,:));
Ind3 = unique(zef.source_interpolation_ind{1}(ind3,:));

Ind0 =[Ind;Ind2;Ind3];

for i = 1:length(zef.reconstruction)
    X(:,i) = zef.reconstruction{i};
end
l = length(X);
XX = sqrt(X(1:l/3,:).^2+X(l/3+1:2*l/3,:).^2+X(2*l/3+1:l,:).^2);
B =mean(XX(Ind0,:),2);

for i = 1: length(B)
    for j = 1: length(B)
        C(i,j)= (B(i)-mean(B))*(B(j)-mean(B))'/sqrt(sum((B(i)-mean(B)).^2)*sum((B(j)-mean(B)).^2));
    end
end

C2 = imresize(C,0.1);
figure,
imagesc(C)
imagesc(C2)

reconstruction = zeros(length(XX),1);
reconstruction(Ind0)= [diag(C)];
clear('zef.reconstruction')
%for i = 1:length(zef.reconstruction)
%    zef.reconstruction{i}=repmat(reconstruction,3,1);
%end
zef.number_of_frames = 1;
re = repmat(reconstruction',3,1);
zef.reconstruction = re(:);