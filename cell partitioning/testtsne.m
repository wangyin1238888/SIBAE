function testtsne()
disp("tsone.R")
Rscript =  'D:\R\R-4.3.1\bin\Rscript.exe';
RscriptFileName = ' ./fengtsne.R ';
filefolder = 'intermediateFiles';
eval([' system([', '''', Rscript, RscriptFileName, '''', ' filefolder]);']);
end
