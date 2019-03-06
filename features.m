function Features = features(reconstructions, zef, tt, n, iter)
S = size(reconstructions, 1);
N = tt*3600;
amp = zeros(S/3, n);
xx = zeros(S/3, n);
yy = zeros(S/3, n);
zz = zeros(S/3, n);
dx = zeros(S/3, 1);
dy = zeros(S/3, 1);
dz = zeros(S/3, 1);
for i = tt-n: tt-2
     xx(:,i) = reconstructions(1:S/3, i+1)-reconstructions(1:S/3, i);
     yy(:,i) = reconstructions(S/3+1:2*S/3, i+1)-reconstructions(S/3+1:2*S/3, i);
     zz(:,i) = reconstructions(2*S/3+1:S, i+1)-reconstructions(2*S/3+1:S, i);
     d_amp(:, i) = sqrt(xx(:,i+1).^2 + yy(:,i+1).^2 + zz(:,i+1).^2)-sqrt(xx(:,i).^2 + yy(:,i).^2 + zz(:,i).^2);
end

%SXmotion1 = gpuArray(sqrt(reconstructions(1:S/3, tt).^2 +reconstructions(S/3+1:2*S/3, tt).^2 + reconstructions(S/3+1:2*S/3,tt).^2));
%SXmotion2 = gpuArray(mean(amp,2));
SX1 = gpuArray(reconstructions(:,i+1))+mean(gpuArray([xx;yy;zz]),2);
SX3 = gpuArray(sqrt(xx(:,i+1).^2 + yy(:,i+1).^2 + zz(:,i+1).^2))+mean(gpuArray(d_amp),2);
SX2 = gpuArray(reconstructions(:,tt));
SY = gpuArray(zef.measurements(:,tt));
%SYmotion1 = gpuArray(fft(zef.measurements(:,N)));
%SYmotion2 = gpuArray(fft(zef.measurements(:,N-1)));
%SXmotion = gpuArray([SXmotion1;SXmotion2]);
%SYmotion = gpuArray([SYmotion1;SYmotion2]);
Ax = gpuArray(zef.L(:,1:S/3));
Ay = gpuArray(zef.L(:,S/3+1:2*S/3));
Az = gpuArray(zef.L(:,2*S/3+1:S));
A1 = gpuArray(zef.L);
A2 =A1;
A3 = sqrt(Ax.^2+Ay.^2+Az.^2);


W1 = gpuArray(zeros(size(SX1,1),size(SX1,2)));
W2 = gpuArray(zeros(size(SX2,1),size(SX2,2)));
W3 = gpuArray(zeros(size(SX3,1),size(SX3,2)));
sigma = 0.5;

%supervised regularized NMF 
%inv_n_iter = 3000;
%mu controls the graph regularization
mu = 0.1;
% fun = @(A, B) A*B;
% H1 = bscfun(fun, SX1, SX1');
% HX1 = bscfun(fun, A1, H1);
% HY1 = bscfun(fun, SY1,  SX1');

for s = 1: iter
    s
    %lambda controls the stepsize
    lambda0 = 0.05;
    tao = 25;
    lambda = lambda0*exp(-s/tao);
%    A1 = bsfunc(@rdivide, bsxfun(@times, A1, HY1), bsxfun(A1, HX1));
    for i = 1: size(SX1, 1)
        for j = 1:size(SX1, 1)
            H1(i,j) = sum(bsxfun(@times, SX1(i,:), SX1(j,:)));
            H2(i,j) = sum(bsxfun(@times, SX2(i,:), SX2(j,:)));
        end
    end
    for i = 1: size(SX3, 1)
        for j = 1:size(SX3, 1)
            H3(i,j) = sum(bsxfun(@times, SX3(i,:), SX3(j,:)));
        end
    end
    for i = 1: size(A1, 1)
        for j = 1:size(H1, 2)
            X1(i,j) = sum(bsxfun(@times, A1(i,:), H1(:,j)));
            X2(i,j) = sum(bsxfun(@times, A2(i,:), H2(:,j)));
        end
    end
    for i = 1: size(A3, 1)
        for j = 1:size(H3, 2)
            X3(i,j) = sum(bsxfun(@times, A3(i,:), H3(:,j)));
        end
    end
    for i = 1: size(SY, 1)
        for j = 1:size(SX1, 2)
            Y1(i,j) = sum(bsxfun(@times, SY(i,:), SX1(:,j)));
            Y2(i,j) = sum(bsxfun(@times, SY(i,:), SX2(:,j)));
        end
    end
    for i = 1: size(SY, 1)
        for j = 1:size(SX3, 2)
            Y3(i,j) = sum(bsxfun(@times, SY(i,:), SX3(:,j)));
        end
    end 
    A1 = bsxfun(@rdivide, bsxfun(@times, A1, Y1), X1);
    A2 = bsxfun(@rdivide, bsxfun(@times, A2, Y2), X2);
    A3 = bsxfun(@rdivide, bsxfun(@times, A3, Y3), X3);    
%    A1 = A1.*(SY*SX1')./(A1*gpuArray(SX1*SX1'));
%    A2 = A2.*(SY*SX2')./(A2*gpuArray(SX2*SX2'));
%    A3 = A3.*(SY*SX3')./(A3*gpuArray(SX3*SX3'));
    SX1 = SX1.*(A1'*SY1 + mu*W1.*SX1)./(A1'*SY1.*SX1 +lambda*SX1.^(-0.5)+mu*W1.*SX1);
    SX2 = SX2.*(A2'*SY2 + mu*W2.*SX2)./(A2'*SY2.*SX2 +lambda*SX2.^(-0.5)+mu*W2.*SX2);
    SX3 = SX3.*(A3'*SY3 + mu*W3.*SX3)./(A3'*SY3.*SX3 +lambda*SX3.^(-0.5)+mu*W3.*SX3);
    for i = 1:size(W1,1)
        for j = 1:size(W1,2)
            W1(i, j) = exp(-sqrt((SX1(i,1)-SX1(1,j)).^2)/sigma);
            W2(i, j) = exp(-sqrt((SX2(i,1)-SX2(1,j)).^2)/sigma);
        end
    end
    for i = 1:size(W3,1)
        for j = 1:size(W3,2)
            W3(i, j) = exp(-sqrt((SX3(i,1)-SX3(1,j)).^2)/sigma);
        end
    end        
    err = sum(sum(SY1-A1*SX1))+sum(sum(SY2-A2*SX2))+sum(sum(SY3-A3*SX3));
    if err < 1e-5
        break;
    end
end

SX = (SX1+SX2)/2;
x = zef.source_positions(:, 1);
y = zef.source_positions(:, 2);
z= zef.source_positions(:, 3);
%dx(:) = zef.source_directions(:, 1);
%dy(:) = zef.source_directions(:, 2);
%dz(:) = zef.source_directions(:, 3);
Features = [x, y, z, SX3, SX(1:S/3,:), SX(S/3+1:2*S/3,:), SX(2*S/3+1:S, :)];
end
