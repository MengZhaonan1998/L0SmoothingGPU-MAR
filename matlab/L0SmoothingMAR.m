% L0SmoothingMAR.m is a matlab implementation of L0Smoothing for MAR.
% The Code is created based on the method described in the following paper 
% [1] "Image Smoothing via L0 Gradient Minimization", Li Xu, Cewu Lu, Yi Xu, Jiaya Jia, ACM Transactions on Graphics, 
% SIGGRAPH Asia 2011), 2011. 
% Steps:
%   1. Read a 768*768*1 32float raw data as a input
%   2. Process the input data L0Smoothing_MAR
%   3. Output the result

% Step 1
fin=fopen('../TestData/test_recon_768_768.raw');
I=fread(fin,'float=>float'); 
Im=reshape(I,[768,768]);

% Step 2
lambda = 2e-3;
kappa = 2.0;
S=L0Smoothing_MAR(Im, lambda, kappa);

% Step 3
fout=fopen('../TestData/output_test_recon_768_768.raw','w+');
fwrite(fout, S, 'float');
fclose(fout);

function S = L0Smoothing_MAR(Im, lambda, kappa)
%L0Smooth - Image Smoothing via L0 Gradient Minimization
%   S = L0Smooth_MAR(Im, lambda, kappa) performs L0 graidient smoothing of input
%   image Im, with smoothness weight lambda and rate kappa.
%
%   Paras: 
%   @Im    : Input 32float raw image.
%   @lambda: Smoothing parameter controlling the degree of smooth. (See [1]) 
%   @kappa : Parameter that controls the rate. (See [1])
%            Small kappa results in more iteratioins and with sharper edges.   

if ~exist('kappa','var')
    kappa = 2.0;
end
if ~exist('lambda','var')
    lambda = 2e-2;
end
maxS = max(max(Im));
S = Im / maxS;                            
betamax = 1e5;
fx = [1, -1];
fy = [1; -1];
[N,M] = size(Im);
sizeI2D = [N,M];
otfFx = psf2otf(fx,sizeI2D);
otfFy = psf2otf(fy,sizeI2D);
Normin1 = fft2(S);
Denormin2 = abs(otfFx).^2 + abs(otfFy ).^2;
beta = 2*lambda;
while beta < betamax
    Denormin   = 1 + beta*Denormin2;                         
    % h-v subproblem
    h = [diff(S,1,2), S(:,1,:) - S(:,end,:)];
    v = [diff(S,1,1); S(1,:,:) - S(end,:,:)];
    t = (h.^2+v.^2)<lambda/beta;
    h(t)=0; v(t)=0;
    % S subproblem
    Normin2 = [h(:,end,:) - h(:, 1,:), -diff(h,1,2)];
    Normin2 = Normin2 + [v(end,:,:) - v(1, :,:); -diff(v,1,1)];
    FS = (Normin1 + beta*fft2(Normin2))./Denormin;
    S = real(ifft2(FS));
    beta = beta*kappa;
    fprintf('.');
end
S = maxS * S;
fprintf('\n');
end
