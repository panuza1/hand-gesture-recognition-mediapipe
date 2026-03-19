#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gesture_resolver.py  —  รวม gesture ของมือซ้าย + ขวา → combined action

วิธีใช้:
    resolver = GestureResolver()
    action = resolver.resolve("peace", "thumbs_up")  # → "volume_up"

เพิ่ม combo ใหม่:
    แก้ GESTURE_COMBOS ด้านล่าง
    key   = (left_gesture, right_gesture)   ← ชื่อต้องตรงกับ label ใน keypoint_classifier_label.csv
    value = ชื่อ action ที่คุณต้องการ
"""


# ─────────────────────────────────────────────────────────────────────────────
# Combo table
# แก้ตรงนี้เพื่อเพิ่ม / ลด gesture combinations ได้เลย
#
# label ที่มีใน repo เดิม (class 0-2):
#   "Open"     = มือเปิด
#   "Close"    = กำหมัด
#   "Pointer"  = ชี้นิ้ว
#
# "none" = ไม่พบมือข้างนั้นในเฟรม
# ─────────────────────────────────────────────────────────────────────────────

GESTURE_COMBOS: dict[tuple[str, str], str] = {

    # ── ไม่เจอมือเลย ───────────────────────────────────────────────────────
    ("none",    "none"):      "idle",

    # ── เจอมือเดียว ────────────────────────────────────────────────────────
    ("none",    "Open"):      "right_open",
    ("none",    "Close"):     "right_fist",
    ("none",    "Pointer"):   "right_point",
    ("Open",    "none"):      "left_open",
    ("Close",   "none"):      "left_fist",
    ("Pointer", "none"):      "left_point",

    # ── เจอ 2 มือ ──────────────────────────────────────────────────────────
    ("Open",    "Open"):      "both_open",
    ("Close",   "Close"):     "both_fist",
    ("Pointer", "Pointer"):   "both_point",
    ("Open",    "Close"):     "left_open_right_fist",
    ("Close",   "Open"):      "left_fist_right_open",
    ("Open",    "Pointer"):   "left_open_right_point",
    ("Pointer", "Open"):      "left_point_right_open",
    ("Close",   "Pointer"):   "left_fist_right_point",
    ("Pointer", "Close"):     "left_point_right_fist",

    # ── ตัวอย่าง custom combo สำหรับท่าใหม่ที่ train เพิ่ม ────────────────
    # ("peace",      "thumbs_up"): "level_up",
    # ("thumbs_up",  "thumbs_up"): "confirm",
    # ("thumbs_down","thumbs_down"):"cancel",
}


class GestureResolver:
    def __init__(self, combos: dict | None = None):
        """
        combos: ถ้าต้องการ override GESTURE_COMBOS จากภายนอก ส่ง dict เข้ามา
                ถ้าไม่ส่ง จะใช้ GESTURE_COMBOS ด้านบน
        """
        self._combos = combos if combos is not None else GESTURE_COMBOS

    def resolve(self, left: str, right: str) -> str:
        """
        รับ gesture ของมือซ้ายและมือขวา
        คืน action string ที่ตรงกับ combo table
        ถ้าไม่พบใน table คืน string บอก combo ที่ไม่รู้จัก
        """
        key = (left, right)
        return self._combos.get(key, f"unknown ({left} | {right})")

    def add_combo(self, left: str, right: str, action: str):
        """เพิ่ม combo ใหม่ตอน runtime โดยไม่ต้องแก้ไฟล์"""
        self._combos[(left, right)] = action

    def list_combos(self) -> list[tuple]:
        """แสดง combo ทั้งหมดที่ลงทะเบียนไว้"""
        return [(left, right, action)
                for (left, right), action in self._combos.items()]


# ─────────────────────────────────────────────────────────────────────────────
# ทดสอบง่ายๆ ถ้ารัน python gesture_resolver.py โดยตรง
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    resolver = GestureResolver()

    test_cases = [
        ("none",    "none"),
        ("none",    "Open"),
        ("Open",    "Close"),
        ("Close",   "Close"),
        ("Pointer", "Open"),
        ("peace",   "thumbs_up"),   # ยังไม่มีใน table → unknown
    ]

    print(f"{'Left':<14} {'Right':<14} {'Action'}")
    print("-" * 46)
    for left, right in test_cases:
        action = resolver.resolve(left, right)
        print(f"{left:<14} {right:<14} {action}")