clear;
clc
load('zef.mat');
f = 20000;
n1 = f*0.49; 
nn = f*0.5;% 0.5s when the device is on
n2 = f*0.51;
n3 = f*2.017;
nn2 = f*2.0267% 2s when the device is off for 1s
n4 = f*2.037;

%% sensors
figure,
ind = find(zef.sensors(:,1)>-20 & zef.sensors(:,1) < 50 & zef.sensors(:,2)>-40 & zef.sensors(:,2) < 40)
plot(zef.sensors(:,1),zef.sensors(:,2),'bo',zef.sensors(ind,1),zef.sensors(ind,2),'ro');
figure,
plot3(zef.sensors(:,1), zef.sensors(:,2), zef.sensors(:,3),'o');

%% time series of 6 sensors near thalamus close to 0.5s
figure,
m = zef.measurements(ind,n1:n2);
m1 = mean(m,1);
GLMF = sqrt(sum((m-repmat(mean(m,1),size(m,1),1)).^2,1))/size(m,1);
%[m,i]=max(GLMF);
ylim = [-150 150];
subplot(411)
plot(n1:n2,m,'-', n1:n2,m1,'r.-',n1:n2,GLMF,'k.-');
hold on
h_line = line([nn nn],[0 ylim(2)]);
set(h_line,'linewidth',5,'color','k','linestyle','--');
title("time series of 6 sensors near thalamus close to 0.5s")
%% time series of 6 sensors far away from thalamus close to 0.5s
mm = zef.measurements(59:64,n1:n2);
mm1 = mean(mm,1);
GLMF2 = sqrt(sum((mm-repmat(mean(mm,1),size(mm,1),1)).^2,1))/size(mm,1);
%[m,i]=max(GLMF);
subplot(412)
ylim = [-40 40];
plot(n1:n2,mm,'-', n1:n2,mm1,'r.-',n1:n2,GLMF2,'k.-');
hold on
h_line = line([nn nn],[0 ylim(2)]);
set(h_line,'linewidth',5,'color','k','linestyle','--');
title("time series of 6 sensors far away from thalamus close to 0.5s")
%% time series of 6 sensors near thalamus close to 2s
subplot(413)
ylim = [-50 50];
m2 = zef.measurements(ind,n3:n4);
m3 = mean(m2,1);
GLMF3 = sqrt(sum((m2-repmat(mean(m2,1),size(m2,1),1)).^2,1))/size(m2,1);
%[m,i]=max(GLMF);
plot(n3:n4,m2,'b-', n3:n4,m3,'r.-',n3:n4,GLMF3,'k.-');
hold on
h_line = line([nn2 nn2],[0 ylim(2)]);
set(h_line,'linewidth',5,'color','k','linestyle','--');
title("time series of 6 sensors near thalamus close to 2s")
%% time series of 6 sensors far away from thalamus close to 2s
mm2 = zef.measurements(59:64,n3:n4);
mm3 = mean(mm2,1);
GLMF4 = sqrt(sum((mm2-repmat(mean(mm2,1),size(mm2,1),1)).^2,1))/size(mm2,1);
%[m,i]=max(GLMF);
subplot(414)
ylim = [-100 100];
plot(n3:n4,mm2,'-', n3:n4,mm3,'r.-',n3:n4,GLMF4,'k.-');
hold on
h_line = line([nn2 nn2],[0 ylim(2)]);
set(h_line,'linewidth',5,'color','k','linestyle','--');
title("time series of 6 sensors far away from thalamus close to 2s")
figure
plot(f*0.499:f*0.501,zef.measurements(58,f*0.499:f*0.501));


%% deterministic feature on time domain(non-linear)
%6 sensors close thalamus close to 0.5s
dv1 = (m-zef.measurements(ind, n1-1:n2-1))*f;
dv2 =(zef.measurements(ind, n1+1:n2+1)-m)*f;
AS = (dv1+dv2)/2;
SH = abs(dv2-dv1);
vas = (zef.measurements(ind, n1-1:n2-1).^2+m.^2+zef.measurements(ind, n1+1:n2+1).^2)/3;
sas = (dv1.^2+dv2.^2)/2;
M = sas*f./(2*pi*vas);
C = sqrt(((dv2-dv1)*f).^2./sas-4*pi^2*M.^2)/(2*pi);
%6 sensors far away from thalamus close to 0.5s
dv12 = (mm-zef.measurements(59:64, n1-1:n2-1))*f;
dv22 =(zef.measurements(59:64, n1+1:n2+1)-mm)*f;
AS2 = (dv12+dv22)/2;
SH2 = abs(dv22-dv12);
vas2 = (zef.measurements(59:64, n1-1:n2-1).^2+mm.^2+zef.measurements(59:64, n1+1:n2+1).^2)/3;
sas2 = (dv12.^2+dv22.^2)/2;
M2 = sas2*f./(2*pi*vas2);
C2 = sqrt(((dv22-dv12)*f).^2./sas2-4*pi^2*M2.^2)/(2*pi);
for i = 1:length(ind)
    figure,
    title("6 sensors close to 0.5s")
    subplot(311)
    plot(n1:n2,AS(i,:),'b',n1:n2,AS2(i,:),'r');
    subplot(312)
    plot(n1:n2,SH(i,:),'b',n1:n2,SH2(i,:),'r');
    subplot(313)
    plot(n1:n2,M(i,:),'b',n1:n2,M2(i,:),'r');
    %subplot(224)
end

%6 sensors close thalamus close to 2s
dv13 = (m2-zef.measurements(ind, n3-1:n4-1))*f;
dv23 =(zef.measurements(ind, n3+1:n4+1)-m2)*f;
AS3 = (dv13+dv23)/2;
SH3 = abs(dv23-dv13);
vas3 = (zef.measurements(ind, n3-1:n4-1).^2+m2.^2+zef.measurements(ind, n3+1:n4+1).^2)/3;
sas3 = (dv13.^2+dv23.^2)/2;
M3 = sas3*f./(2*pi*vas3);
C3 = sqrt(((dv23-dv13)*f).^2./sas3-4*pi^2*M3.^2)/(2*pi);
%6 sensors far away from thalamus close to 2s
dv14 = (mm2-zef.measurements(59:64, n3-1:n4-1))*f;
dv24 =(zef.measurements(59:64, n3+1:n4+1)-mm2)*f;
AS4 = (dv14+dv24)/2;
SH4 = abs(dv24-dv14);
vas4 = (zef.measurements(59:64, n3-1:n4-1).^2+mm2.^2+zef.measurements(59:64, n3+1:n4+1).^2)/3;
sas4 = (dv14.^2+dv24.^2)/2;
M4 = sas4*f./(2*pi*vas4);
C4 = sqrt(((dv24-dv14)*f).^2./sas4-4*pi^2*M4.^2)/(2*pi);
for i = 1:length(ind)
    figure,
    title("6 sensors close to 2s")
    subplot(311)
    plot(n3:n4,AS3(i,:),'b',n3:n4,AS4(i,:),'r');
    subplot(312)
    plot(n3:n4,SH3(i,:),'b',n3:n4,SH4(i,:),'r');
    subplot(313)
    plot(n3:n4,M3(i,:),'b',n3:n4,M4(i,:),'r');
    %subplot(224)
end


%% deterministic feature on frequency domain through AR(non-linear)
%6 sensors close thalamus close to 0.5s
id = zeros(1,length(ind));
for i = 1:length(ind)
    display(['for channel:',num2str(ind(i))])
    a =[-0.9:0.1:-0.1,0.1:0.1:0.9];
    h = zeros(1,length(a));
    p = zeros(1,length(a));
    XNew = zeros(length(m),2,length(a));
    for n = 1:length(a)
        n
        %ARIMAX model
        mdl = regARIMA('AR',a(n),'MA',0,'ARLags',1,'Intercept',mean(m1),'Beta',0.8,'Variance',mean(GLMF));
        %The power density:
        rng(1);
        [ARIMAX,XNew(:,:,n)] = arima(mdl,'X',m(i,:)');
        ARIMAX
        %test whether there distribution are similar
        [h(1,n), p(1,n)] = kstest2(m(i,:)', XNew(:,:,n)*[1;-a(n)], 'alpha', 0.05)
        clear ARIMAX
    end
    if h(:)==1
        [~,id(i)] = min(p);
    else
        [~,id(i)]=min(p(find(h==0)));
    end
    Xt_1 = XNew(:,:,id(i))*[1;-a(id(i))];
    for j = 1:length(Xt_1)
        B(j) = a(id(i))^j/(1-a(id(i))^2);
    end
    figure,
    subplot(2,1,1)
    spectrogram(B);
    title(["0.5s autocovariance spectrogram of AR(1) with $phi being",num2str(a(id(i))),"for channel",num2str(ind(i))])
    subplot(2,1,2)
    spectrogram(Xt_1);
    title(["0.5s spectral density of AR(1) of with $phi being",num2str(a(id(i))),"for channel",num2str(ind(i))])
end
 
%6 sensors close thalamus close to 2s
id = zeros(1,length(ind));
for i = 1:length(ind)
    display(['for channel:',num2str(ind(i))])
    a =[-0.9:0.1:-0.1,0.1:0.1:0.9];
    h = zeros(1,length(a));
    p = zeros(1,length(a));
    XNew = zeros(length(m2),2,length(a));
    for n = 1:length(a)
        n
        %ARIMAX model
        mdl2 = regARIMA('AR',a(n),'MA',0,'ARLags',1,'Intercept',mean(m3),'Beta',0.8,'Variance',mean(GLMF3));
        %The power density:
        rng(1);
        [ARIMAX,XNew(:,:,n)] = arima(mdl2,'X',m2(i,:)');
        ARIMAX
        %test whether there distribution are similar
        [h(1,n), p(1,n)] = kstest2(m2(i,:)', XNew(:,:,n)*[1;-a(n)], 'alpha', 0.05)
        clear ARIMAX
    end
    if h(:)==1
        [~,id(i)] = min(p);
    else
        [~,id(i)]=min(p(find(h==0)));
    end
    Xt_2 = XNew(:,:,id(i))*[1;-a(id(i))];
    for j = 1:length(Xt_2)
        B2(j) = a(id(i))^j/(1-a(id(i))^2);
    end
    figure,
    subplot(2,1,1)
    spectrogram(B2);
    title(["2s autocovariance spectrogram of AR(1) with $phi being",num2str(a(id(i))),"for channel",num2str(ind(i))])
    subplot(2,1,2)
    spectrogram(Xt_2);
    title(["2s spectral density of AR(1) of with $phi being",num2str(a(id(i))),"for channel",num2str(ind(i))])
end