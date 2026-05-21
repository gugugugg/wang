# -*- coding: utf-8 -*-
import os
import glob
import re


SOURCE_DIR = 'raw_sgf_data'
OUTPUT_FILE = 'human_games.txt'
BOARD_SIZE = 15


def run_convert_task():
    """
    Convert generated SGF files into training lines.

    Output format:
        move0 move1 move2 ...|result

    result is 1 for black win and -1 for white win.
    """
    if not os.path.exists(SOURCE_DIR):
        return 0

    files = glob.glob(os.path.join(SOURCE_DIR, "*.sgf"))
    if not files:
        return 0

    new_games = []

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            result = 0
            re_match = re.search(r'RE\[([BW])\+', content)

            if re_match:
                winner = re_match.group(1)
                result = 1 if winner == 'B' else -1
            else:
                try:
                    os.remove(file_path)
                except:
                    pass
                continue

            tokens = re.findall(r';[BW]\[([a-zA-Z0-9,]{2,5})\]', content)
            moves = []

            for t in tokens:
                try:
                    if ',' in t:
                        x, y = map(int, t.split(','))
                        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                            moves.append(str(y * BOARD_SIZE + x))
                    else:
                        col = ord(t[0].lower()) - 97
                        row = ord(t[1].lower()) - 97
                        if 0 <= col < BOARD_SIZE and 0 <= row < BOARD_SIZE:
                            moves.append(str(row * BOARD_SIZE + col))
                except:
                    continue

            if len(moves) > 5:
                line = " ".join(moves) + f"|{result}\n"
                new_games.append(line)

            try:
                os.remove(file_path)
            except:
                pass

        except Exception:
            continue

    if not new_games:
        return 0

    try:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.writelines(new_games)
    except:
        return 0

    return len(new_games)
