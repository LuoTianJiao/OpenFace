clear

if(isunix)
    executable = '"../../build/bin/FeatureExtraction"';
else
    executable = '"../../x64/Release/FeatureExtraction.exe"';
end

output = 'yt_features/';

if(~exist(output, 'file'))
    mkdir(output)
end
    
if(exist([getenv('USERPROFILE') '/Dropbox/AAM/test data/'], 'file'))
    database_root = [getenv('USERPROFILE') '/Dropbox/AAM/test data/'];    
elseif(exist('D:/Dropbox/Dropbox/AAM/test data/', 'file'))
    database_root = 'D:/Dropbox/Dropbox/AAM/test data/';
elseif(exist('F:/Dropbox/AAM/test data/', 'file'))
    database_root = 'F:/Dropbox/AAM/test data/';
elseif(exist('/media/tadas/5E08AE0D08ADE3ED/Dropbox/AAM/test data/', 'file'))
    database_root = '/media/tadas/5E08AE0D08ADE3ED/Dropbox/AAM/test data/';
else
    database_root = '/multicomp/datasets/';
end

database_root = [database_root, '/ytceleb/'];

in_vids = dir([database_root '/*.avi']);

command = sprintf('%s -2Dfp -out_dir "%s" ', executable, output);
% add all videos to single argument list (so as not to load the model anew
% for every video)
for i=1:numel(in_vids)
    
    in_file_name = [database_root, '/', in_vids(i).name];        
    
    command = cat(2, command, [' -f "' in_file_name '" ']);                     
end

if(isunix)
    unix(command, '-echo')
else
    dos(command);
end

%%
output = 'yt_features_clm/';

if(~exist(output, 'file'))
    mkdir(output)
end

command = sprintf('%s -2Dfp -out_dir "%s" -mloc model/main_clm_general.txt ', executable, output);

% add all videos to single argument list (so as not to load the model anew
% for every video)
for i=1:numel(in_vids)    
    in_file_name = [database_root, '/', in_vids(i).name];            
    command = cat(2, command, [' -f "' in_file_name '"']);                     
end

if(isunix)
    unix(command, '-echo')
else
    dos(command);
end
%% evaluating yt datasets
d_loc = 'yt_features/';
d_loc_clm = 'yt_features_clm/';

files_yt = dir([d_loc, '/*.csv']);
preds_all = [];
preds_all_clm = [];
gts_all = [];
for i = 1:numel(files_yt)
    [~, name, ~] = fileparts(files_yt(i).name);
    pred_landmarks = dlmread([d_loc, files_yt(i).name], ',', 1, 0);
    pred_landmarks = pred_landmarks(:,5:end);
    
    xs = pred_landmarks(:, 1:end/2);
    ys = pred_landmarks(:, end/2+1:end);
    pred_landmarks = zeros([size(xs,2), 2, size(xs,1)]);
    pred_landmarks(:,1,:) = xs';
    pred_landmarks(:,2,:) = ys';
    
    pred_landmarks_clm = dlmread([d_loc_clm,  files_yt(i).name], ',', 1, 0);
    pred_landmarks_clm = pred_landmarks_clm(:,5:end);
    
    xs = pred_landmarks_clm(:, 1:end/2);
    ys = pred_landmarks_clm(:, end/2+1:end);
    pred_landmarks_clm = zeros([size(xs,2), 2, size(xs,1)]);
    pred_landmarks_clm(:,1,:) = xs';
    pred_landmarks_clm(:,2,:) = ys';    
    
    load([database_root, name, '.mat']);
    preds_all = cat(3, preds_all, pred_landmarks);
    preds_all_clm = cat(3, preds_all_clm, pred_landmarks_clm);
    gts_all = cat(3, gts_all, labels);
end

%%
[clnf_error, err_pp_clnf] = compute_error( gts_all - 1.5,  preds_all);
[clm_error, err_pp_clm] = compute_error( gts_all - 1.5,  preds_all_clm);

filename = sprintf('results/fps_yt');
save(filename);

% Also save them in a reasonable .txt format for easy comparison
f = fopen('results/fps_yt.txt', 'w');
fprintf(f, 'Model, mean,  median\n');
fprintf(f, 'OpenFace (CLNF):  %.4f,   %.4f\n', mean(clnf_error), median(clnf_error));
fprintf(f, 'CLM:   %.4f,   %.4f\n', mean(clm_error), median(clm_error));

fclose(f);
clear 'f'