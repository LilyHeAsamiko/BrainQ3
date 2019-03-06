clear;
clc;
load('zef.mat');
delete(gcp('nocreate'));
gpuDevice(1);
if gpuDeviceCount
    parpool(gpuDeviceCount);
else
    parpool();
end
GLMF = sum((zef.measurements-repmat(mean(zef.measurements,1),size(zef.measurements,1),1)).^2,1)/size(zef.measurements,1);
%[m,i]=max(GLMF);
figure,
plot(GLMF);

t = zef.inv_time_1;
dt = t*3600:(t*3600+zef.number_of_frames);
spectrogram(zef.measurements(1,dt));
figure,
pspectrum(zef.measurements(1,dt));
SY = fft(zef.measurements(:,dt),zef.number_of_frames,2);

reconstructions = zeros(size(zef.L,2), zef.number_of_frames);
for i = 1:zef.number_of_frames
    reconstructions(:,i) = zef.reconstruction{1,:}(:);
%    reconstructions(:,i) = zef.reconstruction(:);
end
tol = sum(sum(SY-zef.L*reconstructions));

%reconstructions = zeros(size(zef.L,2), zef.number_of_frames);
n = 10;
F = gpuArray(zeros(zef.number_of_frames-n-1, n+6, size(reconstructions,1)/3));
for tt = n+1:zef.number_of_frames
    F(t-n,:,:) = features(reconstructions, zef, tt, n, 300);
end

SY1 = [1 2 3];
SX1 = [1 2 3;1, 2, 3];
A1 = [1;2];
H1 = sum(bsxfun(@times, SX1, SX1),2);
HX1 = sum(bsxfun(@times, A1, H1),2);
HY1 = sum(bsxfun(@times, SY1, SX1),2);
A1 = bsxfun(@rdivide, bsxfun(@times, A1, HY1), bsxfun(A1, HX1));




