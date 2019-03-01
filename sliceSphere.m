function img1=sliceSphere(name,dist_,ROI, color)
img = imread(name);
[m,n,l]=size(img);
img1 = zeros(m,n);
for c = 1:3
    img1 = img(:,:,c);
%dist_=2;
    [y, x, C]=impixel(img1);
    img1=double(img1);
    N = size(C, 1);
    for k = 1:N
        if ((x(k)-ROI)*(x(k)+ROI-n)>0) || ((y(k)-ROI)*(y(k)+ROI-n)>0)
            img(x(k),y(k))=0;
            break
        else
            for i=x(k)-ROI:x(k)+ROI
                for j=y(k)-ROI:y(k)+ROI
                    if (img1(i,j)-C(k))^2 < dist_^2
                        img1(i,j)=color;
                    end
                end
            end
        end
    end
%    img1 = 255*gray2bin(img1, 'qam', 256);
    img(:,:,c)=uint8(img1);
end
imshow(img);
%img=uint8(img);