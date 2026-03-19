#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.py  —  Two-Hand Gesture Recognition
ต่อยอดจาก: https://github.com/kinivi/hand-gesture-recognition-mediapipe

การเปลี่ยนแปลงจาก repo เดิม:
  - max_num_hands=2
  - แยก Left / Right ด้วย multi_handedness
  - cv2.flip() ก่อนส่ง MediaPipe (แก้ left/right สลับ)
  - เรียก GestureResolver เพื่อรวม 2 มือเป็น action เดียว
  - โหมด logging บันทึก keypoint แยก class ได้เหมือนเดิม (กด k → 0-9)
"""

import csv
import copy
import argparse
import itertools
from collections import Counter, deque

import cv2
import numpy as np
import mediapipe as mp

from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from utils.cvfpscalc import CvFpsCalc
from gesture_resolver import GestureResolver   # ไฟล์ใหม่ที่เพิ่มเข้ามา


# ─────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width",  type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence",  type=float, default=0.5)
    return parser.parse_args()


# ─────────────────────────────────────────────
# Landmark helpers  (เหมือน repo เดิมทุกอย่าง)
# ─────────────────────────────────────────────
def calc_landmark_list(image, landmarks):
    h, w = image.shape[:2]
    return [[min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)]
            for lm in landmarks.landmark]


def pre_process_landmark(landmark_list):
    """Normalize keypoints ให้ relative กับ wrist และ scale ด้วย max value"""
    temp = copy.deepcopy(landmark_list)
    base_x, base_y = temp[0]
    for pt in temp:
        pt[0] -= base_x
        pt[1] -= base_y
    flat = list(itertools.chain.from_iterable(temp))
    max_val = max(map(abs, flat)) or 1
    return [v / max_val for v in flat]   # 42 values


def draw_landmarks(image, landmark_list):
    """วาด keypoints และ connections บน frame"""
    connections = [
        (0,1),(1,2),(2,3),(3,4),        # thumb
        (0,5),(5,6),(6,7),(7,8),         # index
        (0,9),(9,10),(10,11),(11,12),    # middle
        (0,13),(13,14),(14,15),(15,16),  # ring
        (0,17),(17,18),(18,19),(19,20),  # pinky
        (5,9),(9,13),(13,17),            # palm
    ]
    # lines
    for start_idx, end_idx in connections:
        cv2.line(image,
                 tuple(landmark_list[start_idx]),
                 tuple(landmark_list[end_idx]),
                 (255, 255, 255), 2)
    # dots
    for idx, pt in enumerate(landmark_list):
        color = (0, 0, 255) if idx == 0 else (0, 255, 0)
        cv2.circle(image, tuple(pt), 5, color, -1)
        cv2.circle(image, tuple(pt), 5, (255, 255, 255), 1)
    return image


# ─────────────────────────────────────────────
# ส่วนที่เพิ่มใหม่: แยก Left / Right hand
# ─────────────────────────────────────────────
def get_hands_by_side(results):
    """
    คืน dict {"Left": landmarks_or_None, "Right": landmarks_or_None}
    MediaPipe ส่ง label "Left"/"Right" กลับมาใน multi_handedness
    (เพราะเราทำ flip ก่อน ค่านี้จะตรงกับมือจริงของผู้ใช้)
    """
    sides = {"Left": None, "Right": None}
    if not results.multi_hand_landmarks:
        return sides
    for hand_lm, hand_info in zip(results.multi_hand_landmarks,
                                  results.multi_handedness):
        label = hand_info.classification[0].label  # "Left" or "Right"
        sides[label] = hand_lm
    return sides


# ─────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────
def draw_info_text(image, handedness_label, gesture_text, brect=None):
    """แสดง label มือซ้าย/ขวา และชื่อ gesture ที่มุมบนซ้าย"""
    color = (255, 128, 0) if handedness_label == "Left" else (0, 128, 255)
    x_offset = 10 if handedness_label == "Left" else image.shape[1] // 2
    cv2.putText(image, f"{handedness_label}: {gesture_text}",
                (x_offset, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return image


def draw_combined_action(image, action_text):
    """แสดง combined action ด้านล่าง"""
    h = image.shape[0]
    cv2.rectangle(image, (0, h - 60), (image.shape[1], h), (50, 50, 50), -1)
    cv2.putText(image, f"Action: {action_text}",
                (20, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 200), 2, cv2.LINE_AA)
    return image


def draw_mode_info(image, mode, number):
    mode_names = {0: "Normal", 1: "Logging Keypoint"}
    mode_str = mode_names.get(mode, "")
    if mode == 1 and number >= 0:
        mode_str += f"  class={number}"
    cv2.putText(image, mode_str, (10, image.shape[0] - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
    return image


# ─────────────────────────────────────────────
# Data logging  (เหมือน repo เดิม)
# ─────────────────────────────────────────────
def logging_csv(number, landmark_list):
    if number < 0:
        return
    csv_path = "model/keypoint_classifier/keypoint.csv"
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *pre_process_landmark(landmark_list)])


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args = get_args()

    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    mp_hands_module = mp.solutions.hands
    hands = mp_hands_module.Hands(
        static_image_mode=False,
        max_num_hands=2,                              # ← เปลี่ยนจาก 1 เป็น 2
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    # โหลด classifier เดิมของ repo (ใช้ร่วมกันทั้ง 2 มือ)
    keypoint_classifier = KeyPointClassifier()
    with open("model/keypoint_classifier/keypoint_classifier_label.csv",
              encoding="utf-8-sig") as f:
        keypoint_labels = [row[0] for row in csv.reader(f)]

    resolver = GestureResolver()   # combine (left, right) → action
    fps_calc = CvFpsCalc(buffer_len=10)

    mode   = 0   # 0=normal, 1=logging
    number = -1  # class id ที่กำลัง log

    while True:
        fps = fps_calc.get()
        key = cv2.waitKey(10)
        if key == 27:   # ESC
            break

        # Mode switching
        if key == ord("k"):
            mode = 1 if mode != 1 else 0
            number = -1
        if mode == 1 and ord("0") <= key <= ord("9"):
            number = key - ord("0")

        ret, frame = cap.read()
        if not ret:
            break

        # ── FLIP ก่อนเสมอ ──────────────────────────────────────────────
        # สำคัญมาก: ทำให้ "Left" ของ MediaPipe = มือซ้ายจริงของผู้ใช้
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ── แยก Left / Right ───────────────────────────────────────────
        hand_sides = get_hands_by_side(results)

        left_gesture  = "none"
        right_gesture = "none"

        for side_label, hand_lm in hand_sides.items():
            if hand_lm is None:
                continue

            landmark_list = calc_landmark_list(image, hand_lm)
            processed     = pre_process_landmark(landmark_list)

            # บันทึก training data ถ้าอยู่ใน logging mode
            if mode == 1 and number >= 0:
                logging_csv(number, landmark_list)

            # Classify
            gesture_id   = keypoint_classifier(processed)
            gesture_name = keypoint_labels[gesture_id]

            # วาด keypoints
            image = draw_landmarks(image, landmark_list)
            image = draw_info_text(image, side_label, gesture_name)

            if side_label == "Left":
                left_gesture  = gesture_name
            else:
                right_gesture = gesture_name

        # ── รวม 2 มือ → action ─────────────────────────────────────────
        action = resolver.resolve(left_gesture, right_gesture)
        image  = draw_combined_action(image, action)

        # ── Debug / Mode info ──────────────────────────────────────────
        image = draw_mode_info(image, mode, number)
        cv2.putText(image, f"FPS: {fps:.1f}", (image.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

        cv2.imshow("Two-Hand Gesture Recognition", image)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()