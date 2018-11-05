REM This batch script will set up an appropriate Python environment for CUDA GPU programming under Windows.
REM The last line launches a CMD prompt.  This can be any environment however.
REM If you wish to use an IDE such as Spyder or Jupyter Notebook, just change the last line to "spyder"
REM or "jupyter-notebook". 

call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
call "C:\Users\%username%\Anaconda2\Scripts\activate.bat" C:\Users\%username%\Anaconda2
cmd
