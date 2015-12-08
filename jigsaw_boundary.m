load jigsaw.mat

image = 'C:\Users\KARAMAN\Google Drive\RESEARCH\Jigsaw_Epitome\Jigsaw_Implementation\star_128.png';

I = imread(image);
fig = figure('name','Jigsaw Segmentation of Patches');
imshow(I)
hold on
%Ceonvert 1D label to 2D matrix
offsetLabel = reshape (label,[128,128]);

binranges=1:1024;
 
bincounts = histc(label,binranges);

assignedLabels = find (bincounts);

labelSize = size (assignedLabels);

labelSize = labelSize(1,1);

boundaryAll = zeros(128,128);

for i = 1 : labelSize    
    connComp = offsetLabel == assignedLabels(i);
    B = bwboundaries(connComp,4);
    for k = 1:length(B)
        boundary = B{k};
        plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth',2)
    end
end

name = ('Jigsaw_Segmented_Patches.png');
saveas(fig, name, 'png');