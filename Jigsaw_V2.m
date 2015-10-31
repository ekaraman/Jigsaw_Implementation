clear all;

% Set image and log directories
date = datestr(now,30);
logFile = ['log_', date, '.txt'];

curDir = pwd;
image = 'C:\Users\KARAMAN\Google Drive\RESEARCH\Jigsaw_Epitome\Jigsaw_Implementation\star_128.png';
logDir = 'C:\Users\KARAMAN\Google Drive\RESEARCH\Jigsaw_Epitome\Jigsaw_Implementation\Logs';
jMeanDir = 'C:\Users\KARAMAN\Google Drive\RESEARCH\Jigsaw_Epitome\Jigsaw_Implementation\jMeanImages';
jVarDir = 'C:\Users\KARAMAN\Google Drive\RESEARCH\Jigsaw_Epitome\Jigsaw_Implementation\jVarImages';
reconsDir = 'C:\Users\KARAMAN\Google Drive\RESEARCH\Jigsaw_Epitome\Jigsaw_Implementation\ReconstructedImage_woNoise';
reconsNoisyDir = 'C:\Users\KARAMAN\Google Drive\RESEARCH\Jigsaw_Epitome\Jigsaw_Implementation\ReconstructedImage_Noisy';

%Open log file
fileID = fopen([logDir,'\',logFile],'w');
fprintf(fileID,'%s\n',date);

% Read image
I = imread(image);
IDouble = double(I) / 255;

%Get sizes of image
ISizeRGB = size(I); %128x128x3
ISize2D = size(I(:,:,1)); %128x128
ISize1D = ISizeRGB (1);

%Draw original image
%figure, image(I), title('Original Image');

%Set size of Jigsaw Matrix
j1D = 32;
jSize = [3 j1D j1D];

%Initialize jigsaw mean and jigsaw variance
jMean = zeros(jSize) - 1;
jMeanInt = zeros(jSize) - 1;
jVar = zeros(jSize);

%Initialize reconstructed images 
reconsImage = zeros (ISizeRGB); %128x128x3
reconsNoisyImage = zeros (ISizeRGB); %128x128x3

%Set constants
beta = 1;
mean_0 = 0.5;

%Calculate mean and variance of image elements in order to find a and b
%constants
pixelMean = zeros (1,3);
pixelStd = zeros (1,3);
pixelVar = zeros (1,3);

% pixelMeanInt = zeros (1,3);
% pixelStdInt = zeros (1,3);
% pixelVarInt = zeros (1,3);

b = zeros(1,1,3);
a = zeros(1,1,3);

for i = 1 : 3
    pixelMean (1,i) = mean2(IDouble(:,:,i));
    pixelStd (1,i) = std2 (IDouble(:,:,i));
    pixelVar (1,i)= pixelStd (1,i) ^ 2;
    
%     pixelMeanInt (1,i) = mean2(I(:,:,i));
%     pixelStdInt (1,i) = std2 (I(:,:,i));
%     pixelVarInt (1,i) = pixelStdInt(1,i) ^ 2;
    
    %Initialize Gamma distribution parameters
    b(1,1,i) = 3 * (1 / pixelVar(1,i));
    a(1,1,i)= b(1,1,i)^2;
end

%Initialize Jigsaw Variance
for i = 1 : 3
    jVar(i,:,:) = b(i) / a(i);
end

%Initialize Jigsaw Mean and making sure to keep the pixels between 0 and 1
done = false;
while(~done)
    done = true;
    
    for i = 1 : 3
        idx = jMean (i,:,:) < 0 | jMean (i,:,:) > 1;
        N = sum(idx(:));
        
        if(N > 0)
            done = false;
            %jMean(i,idx) = randn(N,1) .* pixelStd(1,i) + pixelMean(1,i);
            jMean(i,idx) = random('norm', pixelMean(i), pixelStd(i));
        end
    end
end

%Initialize between 0 - 255 RGB
% while(~done)
%     done = true;
%     
%     for i = 1 : 3
%         idx = jMean (i,:,:) < 0 | jMean (i,:,:) > 255;
%         N = sum(idx(:));
%         
%         if(N > 0)
%             done = false;
%             %jMean(i,idx) = randn(N,1) .* pixelStd(1,i) + pixelMean(1,i);
%             jMean(i,idx) = random('norm', pixelMeanInt(i), pixelStdInt(i));
%         end
%     end
% end

%Convert jMean to unscaled RGB
jMeanInt = int32(floor(jMean .* 255));

%Convert [3 32 32] jigsaw to [32 32 3] jigsaw
jMean = permute(jMean,[2 3 1]);
jMeanInt = permute(jMeanInt,[2 3 1]);
jVar = permute(jVar,[2 3 1]);
jSize = [j1D j1D 3];

%Find label size
labelSize = int32((j1D * j1D));
pixelSize = int32(ISize1D^2);

%Set 4 connected grid of image pixels. 4 connected grid defined in
%Isize1DxIsize1D matrix (eg. if image size is 128x128, 4 conected grid 
%defined in 16384x16384 matrix) because alpha expansion grap cut code
%define neigberhood of pixels as this way.
[r,c] = size(I(:,:,1));                     %# Get the matrix size
diagVec1 = repmat([ones(c-1,1); 0],r,1);    %# Make the first diagonal vector
                                            %# (for horizontal connections)
diagVec1 = diagVec1(1:end-1);               %# Remove the last value
diagVec2 = ones(c*(r-1),1);                 %# Make the second diagonal vector
                                            %#   (for vertical connections)
adj = diag(diagVec1,1)+...                  %# Add the diagonals to a zero matrix
      diag(diagVec2,c);
adj = adj+adj.';                            %# Add the matrix to a transposed
                                            %# copy of itself to make it
                                            %# symmetric
%Get upper triangular parrt of adj matrix 
triuAdj = triu(adj);

%Set weights of neighbourhood edges
w = 50000;
triuAdj = triuAdj .* w;

%Alpha expansion code uses sparse matrix
Sadj = sparse(triuAdj);

%Convert image to 1x(ISize1DxISize1D)x3 matrix for graph cut code
%Eg. if image size is 128x128x3, converted matrix equals to 1x16384x3
I1DArray = int32(reshape (I,[1,pixelSize,3]));

%Set Label offset matrix
offset = zeros(labelSize,2);

%set offset values for I(1,1)
%Namely, finding offset values of assigning pixel(1,1) each jigsaw pixel.
index = 1; 
for i = 1 : j1D
    for j = 1 : j1D
        offset (index,1) = 1 - i;
        offset (index,2) = 1 - j;
        index = index + 1;
    end
end

%Test offset matrix has same offset value;
% disp('start  to check labels');
% for i = 1 : labelSize
%     for j = 1 : labelSize
%         if ((offset(i,1)==offset(j,1)) && (offset(i,2) == offset (j,2)) && (i ~= j))
%             disp('label error');
%             i
%             j
%             break
%         end
%     end
% end
% disp('end of cheeck label');
% i

%Initialize energy
E_old = 1;
E_new = 0;
em = 1;
%Beginning of EM algorithm
while (E_new <=  E_old)
    
    fprintf(fileID,'%s\n',['###############   EM iteration nu:   ',num2str(em),'    #################']);
    
    %Set data cost matrix
    fprintf(fileID,'%s\n','setting data cost matrix');
    dataCost = int32(zeros (labelSize, pixelSize));
    for i = 1 : labelSize
        i
        %fprintf(fileID,'%s\n',['Label=',num2str(i)]);
        for j = 1 : pixelSize
            %Convert 1D pixel to image 2D index
            [IX,IY] = ind2sub(ISize2D,j);
            %Convert offset value to jigsaw index
            jX = mod ((IX - offset (i,1)),j1D);
            if (jX == 0) 
                jX = j1D;
            end
            jY = mod ((IY - offset (i,2)),j1D);
            if (jY == 0) 
                jY = j1D;
            end
            for k = 1 : 3   
                if (jX < 0 || jX > j1D || jY < 0 || jY > j1D)
                    fprintf(fileID,'%s\n','index error');
                end
                dataCost(i,j) = (I1DArray(1,j,k) - jMeanInt(jX,jY,k))^2 + dataCost(i,j);
            end
        end
    end
    
    %Alpha expansion graphcut uses int32
    fprintf(fileID,'%s\n','setting data cost matrix finished');
    
    %Create graph cut handle
    fprintf(fileID,'%s\n','Create grap cut handle');
    h = GCO_Create(pixelSize,labelSize);
    
    %Set data cost matrix for alpha expansion graph cut
    fprintf(fileID,'%s\n','setting data cost matrix');
    GCO_SetDataCost(h,dataCost);
    
    %Since we use Pott's model we dont call GCO_SetSmoothCost
    
    %Setting Neighborhood relation
    fprintf(fileID,'%s\n','setting neighborhood relation matrix');
    GCO_SetNeighbors(h,Sadj);
    
    %Start expanssion so as to assign labels
    fprintf(fileID,'%s\n','Expansion step begin');
    GCO_Expansion(h);
    
    %Assign optimized label values for each pixel to label matrix (16384x1)
    fprintf(fileID,'%s\n','Set labels');
    label = GCO_GetLabeling(h);
    
    %Get optimized energy
    fprintf(fileID,'%s\n','Get optimized energy');
    [E_new D S] = GCO_ComputeEnergy(h);
    fprintf(fileID,'%s\n','Energy computed.');
    fprintf(fileID,'%s\n',['Total Energy = ', num2str(E_new)]);
    fprintf(fileID,'%s\n',['Data Cost Energy = ', num2str(D)]);
    fprintf(fileID,'%s\n',['Smooth Cost Energy = ', num2str(S)]);
    
    %Check While loop terminate case
    if (em == 1)
        E_old = E_new;
    else
        if (E_new <= E_old)
            E_old = E_new;
        else
            fprintf(fileID,'%s\n','#################################');
            fprintf(fileID,'%s\n','##########    CONVERGED      ##########');
            fprintf(fileID,'%s\n',['#######  EM = ', num2str(em), '   ########']);
        end
    end
    
    %Update Jigsaw mean and variance step
    %jigsawLabel(:,:,1) = how many times this label assigned to a pixel
    %jigsawLabel(:,:,2) = total value of all assigned pixels to this jigsaw
    %pixel
    %jigsawLabel(:,:,3) = sum of squares of assigned pixel values
    fprintf(fileID,'%s\n','Updating jMean and jVar');
    jigsawLabel = zeros (j1D,j1D);
    jigsawAssignedPixel = zeros (j1D,j1D,3);
    jigsawAssignedPixel2 = zeros (j1D,j1D,3);
    for i = 1 : pixelSize
        %Convert 1D pixel to image 2D index
        [IX,IY] = ind2sub(ISize2D,i);
        zX = mod((IX - offset(label(i),1)),j1D);
        if (zX == 0) 
            zX = j1D;
        end
        zY = mod((IY - offset(label(i),2)),j1D);
        if (zY == 0) 
            zY = j1D;
        end
        jigsawLabel(zX,zY) = jigsawLabel(zX,zY) + 1;
        for j = 1 :3
            jigsawAssignedPixel(zX,zY,j) = jigsawAssignedPixel(zX,zY,j) + IDouble(IX,IY,j);
            jigsawAssignedPixel2(zX,zY,j) = jigsawAssignedPixel2(zX,zY,j) + (IDouble(IX,IY,j)^2);
        end
    end
    
    %Update Jigsaw mean and variance
    for i = 1 : j1D
        for j = 1 : j1D
            for k = 1 : 3
                jMean(i,j,k) = (((beta * mean_0) + jigsawAssignedPixel(i,j,k)) / (beta + jigsawLabel(i,j)));
                jVar(i,j,k) = ((b(:,:,k) + (beta * mean_0^2) - ((beta + jigsawLabel(i,j)) * jMean(i,j,k)^2) + jigsawAssignedPixel2(i,j,k) ) / (a(:,:,k) + jigsawLabel(i,j)));
            end
        end
    end
    %update jMeanInt
    jMeanInt = floor(jMean .* 255);
    fprintf(fileID,'%s\n','jMean and jVar is updated');
    
    fprintf(fileID,'%s\n','Starting to reconstruct image');
    for i =  1 : pixelSize
        %Convert 1D pixel to image 2D index
        [IX,IY] = ind2sub(ISize2D,i);
        %Convert offset value to jigsaw index
        jX = mod ((IX - offset (label(i),1)),j1D);
        if (jX == 0) 
            jX = j1D;
        end
        jY = mod ((IY - offset (label(i),2)),j1D);
        if (jY == 0) 
            jY = j1D;
        end
        for j = 1 : 3
            reconsImage(IX,IY,j) = jMean(jX,jY,j);
            reconsNoisyImage(IX,IY,j) = imnoise(reconsImage(IX,IY,j),'localvar',jVar(jX,jY,j));
        end 
    end
    fprintf(fileID,'%s\n','End of reconstruction');
    
    %Start logging
    fprintf(fileID,'%s\n','Saving reconsImage image...');
    cd (reconsDir);
    imwrite(reconsImage,[num2str(em),'.png']);
    fprintf(fileID,'%s\n','ReconsImage saved.');
    
    fprintf(fileID,'%s\n','Saving reconsNoisyImage...');
    cd (reconsNoisyDir);
    imwrite(reconsNoisyImage,[num2str(em),'.png']);
    fprintf(fileID,'%s\n','ReconsNoisyImage saved.');
    
    fprintf(fileID,'%s\n','Saving jMean image...');
    cd (jMeanDir);
    imwrite(jMean,[num2str(em),'.png']);
    fprintf(fileID,'%s\n','jMean saved.');
    
    fprintf(fileID,'%s\n','Saving jVar image...');
    cd (jVarDir);
    imwrite(jVar,[num2str(em),'.png']);
    fprintf(fileID,'%s\n','jVar saved.');
    
    cd(pwd);
    
    fprintf(fileID,'%s\n',['###############   EM iteration nu:   ',num2str(em),'  finished #################']);
    
    em = em + 1;
    
    %End of logging
end

fprintf(fileID,'%s\n','Job completed, pls check reconstructed image.');

fclose(fileID);