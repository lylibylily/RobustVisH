@echo off
setlocal enabledelayedexpansion

REM 设置Python解释器和脚本路径
set PYTHON_EXE=D:/Software/radioconda/python.exe
set SCRIPT=D:/GRC310/gmsk_haptic.py

REM 读取文件列表
for /f "tokens=1,2 delims= " %%A in (D:/GRC310/haptic_local.csv) do (
    set input_file=%%A
    set output_file=%%B
    echo Running !PYTHON_EXE! !SCRIPT! -i !input_file! -o !output_file!
    !PYTHON_EXE! !SCRIPT! -i !input_file! -o !output_file!
)

endlocal