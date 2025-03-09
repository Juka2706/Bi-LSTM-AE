set CONDAPATH=C:\Users\Fabia\anaconda3
rem Define here the name of the environment
set ENVNAME=ML_TF210

rem The following command activates the base environment.
rem call C:\ProgramData\Miniconda3\Scripts\activate.bat C:\ProgramData\Miniconda3
rem if %ENVNAME%==base (set ENVPATH=%CONDAPATH%) else (set ENVPATH=%CONDAPATH%\envs\%ENVNAME%)
set ENVPATH=C:\Users\Fabia\anaconda3\envs\%ENVNAME%
rem Activate the conda environment
rem Using call is required here, see: https://stackoverflow.com/questions/24678144/conda-environments-and-bat-files
call %CONDAPATH%\Scripts\activate.bat %ENVPATH%

uvicorn server:app

call conda deactivate