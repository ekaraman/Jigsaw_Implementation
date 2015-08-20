clear all;

% read in an image
x = double(imread('C:\Users\KARAMAN\Google Drive\RESEARCH\Jigsaw_Epitome\Jigsaw_Implementation\128_Dog.png'))/255;

% the size of the jigsaw 
jSize = [32 32];

%Initialize jigsaw mean annd variance matrices
jMean = zeros(jSize);
jVar = zeros(jSize);
