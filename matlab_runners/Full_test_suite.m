% This is sort of the unit test for the whole module (needs datasets)
% Will take several hours to run all
clear
tic
%% Head pose
cd('Head Pose Experiments');
run_head_pose_tests_OpenFace;
assert(median(all_errors_biwi_OF(:)) < 2.8);
assert(median(all_errors_bu_OF(:)) < 2.2);
assert(median(all_errors_ict_OF(:)) < 2.1);
cd('../');

%% Features
cd('Feature Point Experiments');
run_OpenFace_feature_point_tests_300W;
assert(median(err_clnf) < 0.041);
assert(median(err_clnf_wild) < 0.041);
run_yt_dataset;
assert(median(clnf_error) < 0.053);
cd('../');

%% AUs
cd('Action Unit Experiments');
run_AU_prediction_Bosphorus
assert(mean(cccs_reg) > 0.56);
assert(mean(f1s_class) > 0.49);

run_AU_prediction_BP4D
assert(mean(ints_cccs) > 0.6);
assert(mean(f1s_class) > 0.6);

run_AU_prediction_DISFA
assert(mean(au_res) > 0.7);

run_AU_prediction_SEMAINE
assert(mean(f1s) > 0.41);

run_AU_prediction_FERA2011
assert(mean(au_res) > 0.5);

cd('../');

%% Gaze
cd('Gaze Experiments');
extract_mpii_gaze_test
assert(mean_error < 9.6)
assert(median_error < 9.0)
cd('../');

%% Demos
clear;
close all;
cd('Demos');
run_demo_images;
run_demo_videos;
run_demo_video_multi;
run_demo_align_size;
run_test_img_seq;
feature_extraction_demo_vid;
feature_extraction_demo_img_seq;
gaze_extraction_demo_vid;
cd('../');
toc