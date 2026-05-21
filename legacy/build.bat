@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo [1/4] 正在准备清理旧环境...
rd /s /q build dist
del /f /q Gobang_AI_Studio.spec

echo [2/4] 正在激活虚拟环境并执行封装...
call .\env\Scripts\activate

:: --onedir: 生成一个文件夹模式
:: --contents-directory "lib": 将所有 DLL 和库文件放入 lib 子目录，保持根目录整洁
:: --add-data: 包含资源文件
pyinstaller --noconsole --onedir ^
    --contents-directory "lib" ^
    --icon=logo.ico ^
    --add-data "logo.ico;." ^
    --add-data "bg.jpg;." ^
    --add-data "pbrain-Yixin2018.exe;." ^
    --name "Gobang_AI_Studio" ^
    main.py

echo [3/4] 正在为 RTX 50 系列应用 CUDA DLL 补丁...
:: 将预览版 PyTorch 的核心库拷贝到 lib 文件夹中
set DEST_LIB=.\dist\Gobang_AI_Studio\lib
if exist ".\env\Lib\site-packages\torch\lib" (
    xcopy /y ".\env\Lib\site-packages\torch\lib\*.dll" "!DEST_LIB!\"
    echo ✅ 核心 DLL 补丁已应用到 lib 目录。
)

echo [4/4] 完成！
echo ============================================================
echo 发布包已生成在: dist\Gobang_AI_Studio\
echo 根目录下只有 Gobang_AI_Studio.exe 和资源文件，其余在 lib 文件夹中。
echo ============================================================
pause