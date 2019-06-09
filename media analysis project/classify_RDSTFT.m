M = 64;
L = 8;
fs = 20000;
    N = size(EEG,2);
    t0 = 0.25*fs;
    t1 = 0.75*fs;
    t2 = 1.25*fs;
    t3 = 1.95*fs;
    winlen = 10;
    Nwin = floor(N/winlen);
    Noverlap = floor(Nwin/2);
    seg = M;
    overlap = 0.5;
    seglen = floor(N/(seg-(seg-1)*overlap));
    nff = max(Nwin,ceil(log2(Nwin)));
    %find electrode around thalamus
    ind = find(zef.sensors(:,1)>-20 & zef.sensors(:,1) < 50 & zef.sensors(:,2)>-40 & zef.sensors(:,2) < 40);
    %DSTFT with hamming
    for i = 1: length(ind)
        figure,
    %     [S, F, T] = spectrogram(EEG(ind(1),:),hamming(Nwin),Noverlap,nff,fs,'yaxis')
        subplot(311)        
        spectrogram(EEG(ind(i),:),hamming(seglen),floor(overlap*seglen),seglen,fs,'yaxis')
        title(['STFT of EEG channel',num2str(i),' on the whole time']);
        subplot(312)        
        spectrogram(EEG(ind(i),t0:t1),hamming(seglen),floor(overlap*seglen),seglen,fs,'yaxis')
        title(['STFT of EEG channel',num2str(i),' from ',num2str(t0/fs),' to ',num2str(t1/fs)]);
        subplot(313) 
        spectrogram(EEG(ind(i),t2:t3),hamming(seglen),floor(overlap*seglen),seglen,fs,'yaxis')    
        title(['STFT of EEG channel',num2str(i),' from ',num2str(t2/fs),' to ',num2str(t3/fs)]);
        a = 0.1:0.1:1;
        for ai = 1:length(a)
        %RSTFT
            FF(ai,:,:) = RD_STFT(EEG(ind(i),:), fs, M, L, seglen, a(ai));
        %BP
            lambda = 0.1:0.1:0.5;
            for j = 1: length(lambda)
                F(ai,:) = FF(:);
                b = EEG(ind(i),1:length(F));
                W = b./F(ai,:);
                BP = 0.5*(b-W.*F(ai,:)).^2+lambda(j)*abs(F(ai,:)-b);
                X = BP(:);
                target = zeros(size(BP));
                
            %classification
                % Random Forest
                t = templateTree('NumVariablesToSample','all',...
                    'PredictorSelection','interaction-curvature','Surrogate','on');
                 rng(1); % For reproducibility
                 Mdl_1(j) = fitrensemble(BP, reshape(target,size(BP)),'Method','Bag','NumLearningCycles', 200, 'Learners',t);
                 yHat_1(j) = oobPredict(Mdl(j));
                 R2_1(j) = corr(Mdl(j).Y,yHat(j))^2
                % SVM
                rng(1);
                SVMModel(j) = fitcsvm(BP,reshape(target,size(BP)));
                [~,score(j)] = predict(SVMModel(j),X);
                % MLP
                trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
                hiddenLayerSize = 20;
                net = fitnet(hiddenLayerSize,trainFcn);
                net = init(net);
                % Setup Division of Data for Training, Validation, Testing
                net.divideParam.trainRatio = 70/100;
                net.divideParam.valRatio = 15/100;
                net.divideParam.testRatio = 15/100;

                [net,tr(j)] = train(net, BP, target);
                Outputs = net(BP);
                trOut(j,:) = Outputs(tr(j).trainInd);
                vOut(j,:) = Outputs(tr(j).valInd);
                tsOut(j,:) = Outputs(tr(j).testInd);
                e = gsubtract(target,Outputs);
                performance = perform(net,target,Outputs)
                trTarg(j,:) = target .* tr(j).trainMask{1};
                vTarg(j,:) = target .* tr(j).valMask{1};
                tsTarg(j,:) = target .* tr(j).testMask{1};
                 trPerformance = perform(net,trTarg(j,:) ,Outputs)
                 vPerformance   = perform(net,vTarg(j,:),Outputs)
                 tsPerformance  = perform(net,tsTarg(j,:),Outputs)
                % View the Network
                view(net)
%                figure(j),
%                plotregression(trTarg(j,:), tsTarg(j,:), 'Train' , vOut(j,:), tsOut(j,:), 'Validation', tsTarg(j,:), tsOut(j,:), 'Testing')
                % Plots
                % Uncomment these lines to enable various plots.
                figure, plotperform(tr(j))
                figure, plottrainstate(tr(j))
                figure, ploterrhist(e)
                figure, plotregression(target,Outputs)
                figure, plotfit(net,BP,target)
            end
            [acc_1(ai),ind_1] = max(R2_1);
            Lambda_1(ai) = lambda(ind_1);
            [acc_2(ai),ind_2] = max(score);
            Lambda_2(ai) = lambda(ind_2);
            [acc_3(ai),ind_3] = max(tr.best_tperf);
            Lambda_3(ai) = lambda(ind_3);
        end
        display(['For Channal:', num2str(ind(i))]);
        [Acc_1,Ind_1] = max(acc_1);
        display(['best accuracy for RF: ',Acc_1,' with lambda: ',Lambda_1(Ind_1),' f sub-bands: ', RD_STFT(EEG(ind(i),:), fs, M, L, seglen, a(Ind_1))])
        [Acc_2,Ind_2] = max(acc_2);
        display(['best accuracy for SVM: ',Acc_2,' with lambda: ',Lambda_2(Ind_2),' f sub-bands: ', RD_STFT(EEG(ind(i),:), fs, M, L, seglen, a(Ind_2))])
        [Acc_3,Ind_3] = max(acc_3);
        display(['best accuracy for MLP: ',Acc_3,' with lambda: ',Lambda_3(Ind_3),' f sub-bands: ', RD_STFT(EEG(ind(i),:), fs, M, L, seglen, a(Ind_3))])
    end
