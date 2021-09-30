% 2021-09-20 by Dushan N. Wadduwage
% main code
clear all; clc

%% model parameters
pram              = f_pram_init();
pram.DataGt       = '../../_Data/SRS2AFM/new_dataset/train_dev/gt/';
pram.DataSrs      = '../../_Data/SRS2AFM/new_dataset/train_dev/srs/';
pram.DataGt_val   = '../../_Data/SRS2AFM/new_dataset/test_dev/gt/';
pram.DataSrs_val  = '../../_Data/SRS2AFM/new_dataset/test_dev/srs/';

%% read data
imds_gt           = imageDatastore(pram.DataGt     );
imds_srs          = imageDatastore(pram.DataSrs    );
imds_gt_val       = imageDatastore(pram.DataGt_val );
imds_srs_val      = imageDatastore(pram.DataSrs_val);
    
imds_tr           = combine(imds_srs,imds_gt);
imds_val          = combine(imds_srs_val,imds_gt_val);
%% generate inv model
lgraph            = f_gen_inv(pram);

%% train inv model
pram.miniBatchSize  = 1; 
trOptions           = f_set_training_options(pram,imds_val);
net                 = trainNetwork(imds_tr,lgraph,trOptions);

%% test inv model





