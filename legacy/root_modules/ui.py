# -*- coding: utf-8 -*-
import pygame
import numpy as np
import math
from config import *

# --- 1. 核心绘图辅助函数 (适配 main.py 第11行导入) ---
def draw_smooth_rect(screen, color, rect, radius=10, width=0, alpha=255):
    """ 绘制抗锯齿圆角矩形，用于提升 UI 整体质感 """
    rect = pygame.Rect(rect)
    color = pygame.Color(color)
    # 创建双倍大小的 Surface 以实现平滑缩放抗锯齿
    shape_surf = pygame.Surface((rect.width * 2, rect.height * 2), pygame.SRCALPHA)
    pygame.draw.rect(shape_surf, color, shape_surf.get_rect(), width * 2, border_radius=radius * 2)
    smooth_surf = pygame.transform.smoothscale(shape_surf, (rect.width, rect.height))
    if alpha < 255: 
        smooth_surf.set_alpha(alpha)
    screen.blit(smooth_surf, rect.topleft)

# --- 2. 现代按钮组件 (适配 main.py 第80-160行) ---
class ModernButton:
    """ 
    适配 main.py 的按钮类。
    支持 text_key 自动翻译及 callback 回调逻辑。
    """
    def __init__(self, x, y, w, h, text_key, callback=None, color=COLOR_ACCENT):
        self.rect = pygame.Rect(x, y, w, h)
        self.text_key = text_key
        self.callback = callback
        self.base_color = color
        self.is_hovered = False

    def update(self, m_pos):
        """ 适配 main.py 427行：更新悬停状态 """
        self.is_hovered = self.rect.collidepoint(m_pos)

    def handle_event(self, event, v_m_pos):
        """ 适配 main.py 506行：处理点击并执行回调 """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(v_m_pos) and self.callback:
                self.callback()
                return True
        return False

    def draw(self, screen, font, lang="CN"):
        # 悬停时颜色稍微变亮
        c = [min(255, v + 30) for v in self.base_color] if self.is_hovered else self.base_color
        draw_smooth_rect(screen, c, self.rect, radius=8)
        
        # 适配 config.py 中的 LANGS 语言包
        display_text = LANGS[lang].get(self.text_key, self.text_key)
        txt = font.render(display_text, True, COLOR_TEXT_MAIN)
        screen.blit(txt, txt.get_rect(center=self.rect.center))

# --- 3. 实验参数输入框 (适配 main.py 第90-104行) ---
class InputBox:
    """ 
    用于设置实验参数（Filters, Loops等）的文本框。
    支持从 main.py 获取数值。
    """
    def __init__(self, x, y, w, h, text='', label_key=''):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = str(text)
        self.label_key = label_key
        self.active = False

    def get_value(self):
        """ 适配 main.py 391行：返回输入内容 """
        return self.text

    def handle_event(self, event, v_m_pos):
        """ 适配 main.py 510行：处理点击激活与键盘输入 """
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(v_m_pos)
        
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                self.text += event.unicode

    def draw(self, screen, font, lang="CN"):
        # 绘制上方标签
        lbl_text = LANGS[lang].get(self.label_key, self.label_key)
        lbl = font.render(lbl_text, True, COLOR_TEXT_DIM)
        screen.blit(lbl, (self.rect.x, self.rect.y - 25))
        
        # 绘制框体与边框
        color = COLOR_INPUT_BORDER_ACTIVE if self.active else COLOR_INPUT_BORDER_INACTIVE
        draw_smooth_rect(screen, COLOR_INPUT_BG, self.rect, radius=5)
        pygame.draw.rect(screen, color, self.rect, 2, border_radius=5)
        
        # 渲染文字
        txt_surf = font.render(self.text, True, COLOR_TEXT_MAIN)
        screen.blit(txt_surf, (self.rect.x + 10, self.rect.y + 8))

# --- 4. 核心棋盘组件 (适配 main.py 第170行及512行) ---
class BoardWidget:
    """ 
    负责 15x15 棋盘的渲染与交互。
    """
    def __init__(self, x, y, size):
        self.rect = pygame.Rect(x, y, size, size)
        self.grid_size = size / 16  # 预留外边距
        self.board = np.zeros((15, 15))

    def update(self, board):
        """ 适配 main.py 440行：更新当前棋盘状态 """
        self.board = board

    def get_grid_pos(self, v_m_pos):
        """ 适配 main.py 512行：转换屏幕坐标为棋盘索引 """
        if not self.rect.collidepoint(v_m_pos): 
            return None
        ox, oy = v_m_pos[0] - self.rect.x, v_m_pos[1] - self.rect.y
        c = round((ox / self.grid_size) - 1)
        r = round((oy / self.grid_size) - 1)
        if 0 <= r < 15 and 0 <= c < 15:
            return (r, c)
        return None

    def draw(self, screen):
        # 1. 棋盘木质底色
        draw_smooth_rect(screen, COLOR_BOARD, self.rect, radius=10)
        
        # 2. 网格绘制
        for i in range(15):
            pos = self.grid_size * (i + 1)
            # 横线
            pygame.draw.line(screen, (60, 40, 20), 
                             (self.rect.x + self.grid_size, self.rect.y + pos), 
                             (self.rect.right - self.grid_size, self.rect.y + pos))
            # 纵线
            pygame.draw.line(screen, (60, 40, 20), 
                             (self.rect.x + pos, self.rect.y + self.grid_size), 
                             (self.rect.x + pos, self.rect.bottom - self.grid_size))
        
        # 3. 棋子绘制
        for r in range(15):
            for c in range(15):
                val = self.board[r, c]
                if val == 0: continue
                # 1: 黑子, -1: 白子
                color = (0, 0, 0) if val == 1 else (255, 255, 255)
                center = (int(self.rect.x + (c+1)*self.grid_size), 
                          int(self.rect.y + (r+1)*self.grid_size))
                pygame.draw.circle(screen, color, center, int(self.grid_size * 0.42))

# --- 5. 实验监控图表 (适配 main.py 第168-169行) ---
class DynamicChart:
    """ 
    实时渲染训练 Loss 或胜率曲线。
    """
    def __init__(self, x, y, w, h, title):
        self.rect = pygame.Rect(x, y, w, h)
        self.title = title
        self.data = []

    def set_data(self, data_list):
        """ 适配 main.py 441行：设置绘图数据 """
        self.data = data_list

    def draw(self, screen, title_font, status_font, m_pos):
        # 背景装饰
        draw_smooth_rect(screen, (20, 20, 20, 150), self.rect, radius=10)
        screen.blit(title_font.render(self.title, True, COLOR_TEXT_MAIN), 
                    (self.rect.x + 10, self.rect.y + 10))
        
        if len(self.data) > 1:
            # 只显示最近的 50 个数据点
            display_data = list(self.data)[-50:]
            max_v = max(display_data) if max(display_data) != 0 else 1
            min_v = min(display_data)
            
            pts = []
            for i, v in enumerate(display_data):
                # 坐标计算逻辑
                px = self.rect.x + (i * (self.rect.width / 50))
                py = self.rect.bottom - ((v - min_v) / (max_v - min_v + 1e-6) * (self.rect.height - 50)) - 10
                pts.append((px, py))
            
            if len(pts) > 1:
                pygame.draw.lines(screen, COLOR_ACCENT, False, pts, 2)

# --- 6. 实验进度条 (适配 main.py 第171行) ---
class ProgressBar:
    """ 
    展示蒸馏流水线进度的全局进度条。
    """
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)
        self.current_val = 0.0
        self.target_val = 0.0

    def set_target(self, val):
        """ 适配 main.py 482行：设置目标进度 """
        self.target_val = val

    def update_animation(self):
        """ 适配 main.py 486行：平滑动画更新 """
        self.current_val += (self.target_val - self.current_val) * 0.05

    def draw(self, screen, eta_str, font):
        """ 适配 main.py 487行：绘制进度条及 ETA 信息 """
        # 背景
        draw_smooth_rect(screen, (50, 50, 50), self.rect, radius=5)
        
        # 填充
        fill_w = int(self.rect.width * self.current_val)
        if fill_w > 0:
            fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_w, self.rect.height)
            draw_smooth_rect(screen, COLOR_SUCCESS, fill_rect, radius=5)
        
        # 文字显示
        txt = font.render(f"Progress: {int(self.current_val*100)}% | ETA: {eta_str}", 
                          True, COLOR_TEXT_MAIN)
        screen.blit(txt, txt.get_rect(center=self.rect.center))