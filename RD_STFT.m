function FF = RD_STFT(EEG, fs, M, L, seglen, a)
    %RDSTFT
    for l = 1:L
        for j = 1:  M/2
            x1 = EEG(seglen*(j-1)+1:seglen*(j-1)+seglen);
            if j == M/2
                x2 = repmat(0,[1,seglen]);
            else
                x2 = [repmat(0,[1,seglen/2]),EEG(seglen*j+seglen/2+1:seglen*j+seglen)];
            end
            tao(j,:) =(seglen*(j-1)+seglen-(x1+x2)+1)/seglen;
            g(j,:) = 0.54-0.46*cos(2*pi*tao(j,:));
            e(j,:) = exp(-2*pi*j*tao(j,:)/M);
            z = tao(j,:).*g(j,:).*e(j,:);
            B = (z-a)./(1-a*z);
            A = (1-a*z).*B.^j;
            phi(j) = sqrt(1-abs(a)^2)/sqrt(A*A');
        end
        for j = 1: M/2
            RF(j,:) = tao(j,:).*g(j,:)*phi(j);
        end
        figure,
        pspectrum(RF(:),fs,'spectrogram');
        f = zeros(1,size(g,2)*M/2/L);
        for s = 1:M/2/L;
            G = g((l-1)*M/2/L+s,:);
            f = f+repmat(RF((l-1)*M/2/L+s,:)*phi((l-1)*M/2/L+s)./G,[1,M/2/L]);
        end
        ff(l,:) = f;
    end
    F = zeros(1,size(ff,2));
    figure,
    for l = 1:L
        F = F + ff(l,:);
        FF(l,:) = F;
        subplot(L,1,l)
        plot(F)
    end
    figure,
    for l = 1:L
        subplot(L,1,l)
        plot(EEG(seglen*M/2/L*(l-1)+1:seglen*M/2/L*(l-1)+seglen*M/2/L))
    end
end
    
    
    

    