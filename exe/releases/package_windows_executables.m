clear;
version = '0.3.0';

out_x86 = sprintf('OpenFace_%s_win_x86', version);
out_x64 = sprintf('OpenFace_%s_win_x64', version);

mkdir(out_x86);
mkdir(out_x64);

in_x86 = '../../Release/';
in_x64 = '../../x64/Release/';

% Copy models
copyfile([in_x86, 'AU_predictors'], [out_x86, '/AU_predictors'])
copyfile([in_x86, 'classifiers'], [out_x86, '/classifiers'])
copyfile([in_x86, 'model'], [out_x86, '/model'])

copyfile([in_x64, 'AU_predictors'], [out_x64, '/AU_predictors'])
copyfile([in_x64, 'classifiers'], [out_x64, '/classifiers'])
copyfile([in_x64, 'model'], [out_x64, '/model'])

%% Copy libraries
libs_x86 = dir([in_x86, '*.lib'])';

for lib = libs_x86
   
    copyfile([in_x86, '/', lib.name], [out_x86, '/', lib.name])
    
end

libs_x64 = dir([in_x64, '*.lib'])';

for lib = libs_x64
   
    copyfile([in_x64, '/', lib.name], [out_x64, '/', lib.name])
    
end

%% Copy dlls
dlls_x86 = dir([in_x86, '*.dll'])';

for dll = dlls_x86
   
    copyfile([in_x86, '/', dll.name], [out_x86, '/', dll.name])
    
end

dlls_x64 = dir([in_x64, '*.dll'])';

for dll = dlls_x64
   
    copyfile([in_x64, '/', dll.name], [out_x64, '/', dll.name])
    
end

%% Copy exe's
exes_x86 = dir([in_x86, '*.exe'])';

for exe = exes_x86
   
    copyfile([in_x86, '/', exe.name], [out_x86, '/', exe.name])
    
end

exes_x64 = dir([in_x64, '*.exe'])';

for exe = exes_x64
   
    copyfile([in_x64, '/', exe.name], [out_x64, '/', exe.name])
    
end

%% Copy license and copyright
copyfile('../../Copyright.txt', [out_x86, '/Copyright.txt']);
copyfile('../../OpenFace-license.txt', [out_x86, '/OpenFace-license.txt']);

copyfile('../../Copyright.txt', [out_x64, '/Copyright.txt']);
copyfile('../../OpenFace-license.txt', [out_x64, '/OpenFace-license.txt']);
