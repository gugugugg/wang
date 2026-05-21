# -*- coding: utf-8 -*-
import os, sys, torch, numpy as np, time, subprocess, multiprocessing, gc

# [优化] 彻底屏蔽子进程中的 Pygame 欢迎信息，保持日志清洁
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

def print_diag(stage, msg, status="INFO"):
    colors = {"INFO": "\033[94m", "PASS": "\033[92m", "FAIL": "\033[91m", "WARN": "\033[93m", "END": "\033[0m"}
    print(f"{colors[status]}[{stage}] {msg}{colors['END']}")

# 延迟导入以防止主进程污染
def import_core():
    try:
        from model import GobangDualHead
        from search_engine import AlphaBetaOptimizer
        from game_env import GobangEnv
        import task_gen, task_convert
        from config import BOARD_SIZE
        return GobangDualHead, AlphaBetaOptimizer, GobangEnv, task_gen, task_convert, BOARD_SIZE
    except ImportError as e:
        print_diag("IMPORT", f"模块导入失败: {e}", "FAIL")
        sys.exit(1)

def test_hardware():
    print_diag("STEP 2", "开始硬件与 CUDA 测试...")
    if not torch.cuda.is_available():
        print_diag("CUDA", "未检测到 GPU，将使用 CPU 运行", "WARN")
        return True
    device = torch.device("cuda")
    try:
        dummy = torch.randn(10, 10).to(device)
        print_diag("CUDA", f"检测到设备: {torch.cuda.get_device_name(0)}", "PASS")
        return True
    except Exception as e:
        print_diag("CUDA", f"显存访问失败: {e}", "FAIL")
        return False

def test_engine_health():
    """专门诊断专家引擎的可执行状态"""
    print_diag("STEP 6", "检查专家引擎健康度...")
    exe_name = "pbrain-Yixin2018.exe"
    exe_path = os.path.join(os.getcwd(), exe_name)
    
    # 1. 文件存在检查
    if not os.path.exists(exe_path):
        print_diag("ENGINE", f"找不到引擎文件: {exe_path}", "FAIL")
        return False
    
    # 2. 权限与可执行检查
    try:
        # 尝试静默启动引擎并立即杀死，测试其是否能运行
        flags = 0x08000000 if os.name == 'nt' else 0
        proc = subprocess.Popen([exe_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, creationflags=flags)
        proc.terminate()
        print_diag("ENGINE", "引擎文件存在且具备执行权限", "PASS")
    except Exception as e:
        print_diag("ENGINE", f"引擎无法启动 (可能是缺少 DLL 或权限不足): {e}", "FAIL")
        return False
    return True

def test_pipeline_deep_scan():
    """深度扫描流水线每一阶段"""
    print_diag("STEP 7", "开始流水线压力测试 (Deep Scan)...")
    GobangDualHead, AlphaBetaOptimizer, GobangEnv, task_gen, task_convert, BOARD_SIZE = import_core()
    
    # 1. 清理环境
    if os.path.exists("raw_sgf_data"):
        import shutil
        try:
            shutil.rmtree("raw_sgf_data")
            print_diag("PIPELINE", "旧数据目录已清理")
        except: pass

    # 2. 尝试生成 (增加超时容错)
    print_diag("PIPELINE", "正在请求 Yixin 生成样本 (Timeout: 500ms)...")
    try:
        # 测试时适当放宽超时，防止笔记本因节能模式降频导致失败
        task_gen.run_generation_task("2") 
    except Exception as e:
        print_diag("PIPELINE", f"生成任务抛出异常: {e}", "FAIL")
        return False

    # 3. 检查目录产生
    if not os.path.exists("raw_sgf_data"):
        print_diag("PIPELINE", "失败原因: 生成任务结束，但 raw_sgf_data 目录未创建", "FAIL")
        return False

    # 4. 检查文件有效性
    files = os.listdir("raw_sgf_data")
    if not files:
        print_diag("PIPELINE", "失败原因: 目录已创建，但没有生成任何 .sgf 文件 (引擎可能超时或逻辑中断)", "FAIL")
        # 诊断：查看系统日志
        if os.path.exists("system_log.txt"):
            print_diag("DIAG", "正在读取 system_log.txt 最后 3 行:")
            with open("system_log.txt", "r", encoding="utf-8") as f:
                print("".join(f.readlines()[-3:]))
        return False
    
    print_diag("PIPELINE", f"成功捕捉到 {len(files)} 个原始样本", "PASS")

    # 5. 转换测试
    print_diag("PIPELINE", "开始执行格式转换 (moves|result)...")
    conv_count = task_convert.run_convert_task()
    if conv_count <= 0:
        print_diag("PIPELINE", "失败原因: SGF 文件存在，但转换器未能解析出有效内容 (正则匹配失败)", "FAIL")
        return False
    
    print_diag("PIPELINE", f"转换流水线完整通过: {conv_count} 局数据已入库", "PASS")
    return True

def run_all():
    # 延迟加载核心组件
    GobangDualHead, AlphaBetaOptimizer, GobangEnv, task_gen, task_convert, BOARD_SIZE = import_core()
    
    print("="*60)
    print("      望 (Genshin AI Gobang) 深度诊断系统 [AAAI Standard]      ")
    print("="*60)
    
    results = []
    results.append(("硬件环境", test_hardware()))
    
    # 模型契约测试
    print_diag("STEP 3", "执行模型 I/O 契约验证...")
    try:
        model = GobangDualHead(num_filters=16)
        dummy = torch.zeros((1, 3, 15, 15))
        p, v = model(dummy)
        results.append(("模型契约", True))
        print_diag("MODEL", "Policy/Value 解构正常", "PASS")
    except Exception as e:
        results.append(("模型契约", False))
        print_diag("MODEL", f"契约破坏: {e}", "FAIL")

    # 随机开局测试
    print_diag("STEP 4", "验证随机开局协议...")
    env = GobangEnv()
    stones = [np.count_nonzero(env.reset(randomize_opening=True)) for _ in range(5)]
    res = sum(stones) > 0
    results.append(("随机开局", res))
    print_diag("ENV", f"随机开局统计: {stones}", "PASS" if res else "FAIL")

    # 引擎与流水线深度扫描
    if test_engine_health():
        results.append(("引擎健康", True))
        results.append(("流水线", test_pipeline_deep_scan()))
    else:
        results.append(("引擎健康", False))
        results.append(("流水线", False))

    print("\n" + "="*60)
    print("                最终诊断汇总                ")
    print("="*60)
    all_ok = True
    for name, res in results:
        status = "🟢 [PASS]" if res else "🔴 [FAIL]"
        print(f"{name.ljust(25)} {status}")
        if not res: all_ok = False
    
    if all_ok:
        print("\n🎉 系统核心链路已打通，可以开始 AAAI 规模的蒸馏实验。")
    else:
        print("\n🆘 警告: 关键链路存在故障，请参照上方红色报错信息修复。")
    print("="*60)

if __name__ == "__main__":
    multiprocessing.freeze_support() # 必须
    run_all()