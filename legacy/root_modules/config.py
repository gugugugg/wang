# -*- coding: utf-8 -*-
import pygame

# === Dimensions ===
DESIGN_WIDTH = 1200
DESIGN_HEIGHT = 800
SIDEBAR_WIDTH = 200
CONTENT_START_X = SIDEBAR_WIDTH + 20
BOARD_SIZE = 15
BOARD_PIXEL_SIZE = 600

# === Resources ===
BACKGROUND_IMAGE_PATH = "bg.jpg"
FONT_MAIN = "Microsoft YaHei" 

# === Colors ===
COLOR_BG_DARK = (30, 30, 30)
COLOR_SIDEBAR_BG = (20, 20, 20, 230)
COLOR_TEXT_MAIN = (240, 240, 240)
COLOR_TEXT_DIM = (160, 160, 160)
COLOR_ACCENT = (0, 120, 215)
COLOR_DANGER = (220, 50, 50)
COLOR_SUCCESS = (46, 204, 113)
COLOR_WARNING = (243, 156, 18)
COLOR_BOARD = (222, 184, 135)

COLOR_INPUT_BG = (50, 50, 50)
COLOR_INPUT_BORDER_ACTIVE = (0, 120, 215)
COLOR_INPUT_BORDER_INACTIVE = (80, 80, 80)
COLOR_TOGGLE_ON = (46, 204, 113) 
COLOR_TOGGLE_OFF = (80, 80, 80) 

# === Language Pack (Unicode Safe) ===
LANGS = {
    "CN": {
        "title": "Genshin AI Gobang", 
        "nav_distill": "蒸馏模式", 
        "nav_train": "训练配置", 
        "nav_monitor": "实时监控", 
        "nav_eval": "智能评估", 
        "nav_hva": "人机对战", 
        "nav_ava": "机机对战", 
        "nav_bg": "切换背景", 
        "lbl_filters": "模型规模", 
        "lbl_episodes": "单轮步数", 
        "lbl_loops": "自动循环", 
        "lbl_save": "存档前缀", 
        "lbl_distill_games": "单轮生成局数", 
        "lbl_distill_cycles": "蒸馏循环次数", 
        "lbl_distill_targets": "选择训练目标", 
        "lbl_path_common": "选择模型", 
        "lbl_path_black": "黑方模型", 
        "lbl_path_white": "白方模型", 
        "btn_path": "选择文件夹", 
        "btn_browse": "浏览...", 
        "btn_start_train": "开始训练", 
        "btn_start_distill": "启动蒸馏流水线", 
        "btn_stop": "停止任务", 
        "btn_export": "导出报表", 
        "btn_start_eval": "开始评估", 
        "btn_start_hva": "开始对战", 
        "btn_start_ava": "开始观战", 
        "btn_clear": "清空棋盘", # [新增]
        "status_ready": "系统就绪", 
        "msg_win": "胜利!", 
        "msg_loss": "失败!", 
        "msg_tie": "平局", 
    },
    "EN": {
        "title": "Genshin AI Gobang",
        "nav_distill": "Distillation",
        "nav_train": "Training",
        "nav_monitor": "Monitor",
        "nav_eval": "Evaluation",
        "nav_hva": "Human vs AI",
        "nav_ava": "AI vs AI",
        "nav_bg": "Background",
        "lbl_filters": "Filters",
        "lbl_episodes": "Steps/Loop",
        "lbl_loops": "Auto Loops",
        "lbl_save": "Save Prefix",
        "lbl_distill_games": "Games/Cycle",
        "lbl_distill_cycles": "Total Cycles",
        "lbl_distill_targets": "Target Models",
        "lbl_path_common": "Model Path",
        "lbl_path_black": "Black Model",
        "lbl_path_white": "White Model",
        "btn_path": "Set Folder",
        "btn_browse": "Browse...",
        "btn_start_train": "Start Training",
        "btn_start_distill": "Start Distillation",
        "btn_stop": "Stop Task",
        "btn_export": "Export Report",
        "btn_start_eval": "Run Eval",
        "btn_start_hva": "Start Game",
        "btn_start_ava": "Watch Game",
        "btn_clear": "Clear Board", # [Added]
        "status_ready": "System Ready",
        "msg_win": "You Win!",
        "msg_loss": "You Lose!",
        "msg_tie": "Draw Game",
    },
    "JP": {
        "title": "Genshin AI Gobang", 
        "nav_distill": "蒸留モード", 
        "nav_train": "トレーニング", 
        "nav_monitor": "リアルタイム監視", 
        "nav_eval": "性能評価", 
        "nav_hva": "人間 vs AI", 
        "nav_ava": "AI vs AI", 
        "nav_bg": "背景変更", 
        "lbl_filters": "モデル規模", 
        "lbl_episodes": "ステップ/回", 
        "lbl_loops": "自動ループ", 
        "lbl_save": "ファイル名", 
        "lbl_distill_games": "生成数/回", 
        "lbl_distill_cycles": "ループ回数", 
        "lbl_distill_targets": "対象モデル", 
        "lbl_path_common": "モデル選択", 
        "lbl_path_black": "黒 (AI)", 
        "lbl_path_white": "白 (AI)", 
        "btn_path": "フォルダ選択", 
        "btn_browse": "参照...", 
        "btn_start_train": "学習開始", 
        "btn_start_distill": "蒸留を開始", 
        "btn_stop": "停止", 
        "btn_export": "レポート出力", 
        "btn_start_eval": "評価開始", 
        "btn_start_hva": "対戦開始", 
        "btn_start_ava": "観戦開始", 
        "btn_clear": "盤面クリア", # [Added]
        "status_ready": "準備完了", 
        "msg_win": "勝利!", 
        "msg_loss": "敗北...", 
        "msg_tie": "引き分け", 
    }
}