# -*- coding: utf-8 -*-
import os
import subprocess
import time

# 必须和 task_gen.py 里的名字一模一样
EXE_NAME = "pbrain-Yixin2018.exe"

def check():
    print("=== 🔍 引擎诊断工具 ===")
    
    # 1. 检查当前工作目录
    cwd = os.getcwd()
    print(f"📂 当前工作目录: {cwd}")
    
    # 2. 检查文件是否存在
    file_path = os.path.join(cwd, EXE_NAME)
    if os.path.exists(file_path):
        print(f"✅ 找到文件: {file_path}")
    else:
        print(f"❌ 找不到文件! 程序希望它在: {file_path}")
        print("📋 当前目录下的文件有:")
        for f in os.listdir(cwd):
            print(f"   - {f}")
        return

    # 3. 尝试启动引擎
    print("\n🚀 正在尝试启动引擎...")
    try:
        # 模拟 task_gen.py 的启动方式
        proc = subprocess.Popen(
            file_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        
        # 发送简单的指令看看有没有反应
        print("   -> 发送 'START 15' 指令...")
        proc.stdin.write("START 15\n")
        proc.stdin.flush()
        
        # 等待回显
        time.sleep(1.0)
        if proc.poll() is not None:
            print(f"❌ 引擎启动后立即退出了 (Exit Code: {proc.returncode})")
            stderr_out = proc.stderr.read()
            if stderr_out:
                print(f"   错误信息: {stderr_out}")
            else:
                print("   没有错误信息，可能是缺少 DLL 或被杀毒软件拦截。")
        else:
            print("✅ 引擎启动成功！正在运行中。")
            proc.kill()
            
    except Exception as e:
        print(f"❌ 启动发生异常: {e}")

if __name__ == "__main__":
    check()
    input("\n按回车键退出...")