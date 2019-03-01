GLMF = sum((zef.measurements-repmat(mean(zef.measurements,1),size(zef.measurements,1),1)).^2,1)/size(zef.measurements,1);
%[m,i]=max(GLMF);
figure,
plot(GLMF);

dt = zef.inv_time_1*3600:(zef.inv_time_1*3600+zef.number_of_frames);
spectrogram(zef.measurements(1,dt));
figure,
pspectrum(zef.measurements(1,dt));
SY = fft(zef.measurements(:,dt),zef.number_of_frames,2);

reconstructions = zeros(size(zef.L,2), zef.number_of_frames);
for i = 1:zef.number_of_frames
%    reconstructions(:,i) = zef.reconstruction{1,:}(:);
    reconstructions(:,i) = zef.reconstruction(:);
end
tol = sum(sum(SY-zef.L*reconstructions));

reconstructions = zeros(size(zef.L,2), zef.number_of_frames);
SX = reconstructions;
A = zef.L;

%consider graph regularization, W is the weight of edgeson the graph using
%heat kernel function. It works as finding near neibering points.
W = zeros(size(SX,1),size(SX,2));
sigma = 0.5

%supervised regularized NMF 
%inv_n_iter = 3000;
%mu controls the graph regularization
mu = 0.1
for s = 1: inv_n_iter
    s
    %lambda controls the stepsize
    lambda0 = 0.05;
    tao = 25;
    lambda = lambda0*exp(-s/tao);
    SX = SX.*(A'*SY + mu*W.*SX)./(A'*SY.*SX +lambda*SX.^(-0.5)+mu*W.*SX);
    for i = 1:size(W,1)
        for j = 1:size(W,2)
            W(i, j) = exp(-sqrt((SX(i,1)-SX(1,j)).^2)/sigma);
        end
    end
    err = sum(sum(SY-A*SX))
    if err < 0.08*tol
        break;
    end
end

    
    
    
    



