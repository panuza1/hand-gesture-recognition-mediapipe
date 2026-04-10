"""
hand_tracker.py  (patched)
--------------------------

"""

import copy
import itertools
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np


# ── types ─────────────────────────────────────────────────────────────────────

@dataclass
class HandState:
    """Snapshot of one hand's state for a single frame (after smoothing)."""
    side: str                          # 'Left' | 'Right'
    gesture: str = 'no_hand'
    raw_gesture: str = 'no_hand'
    confidence: float = 0.0           # MediaPipe handedness confidence
    classifier_score: float = 0.0     # real softmax score from classifier
    landmarks_raw: Optional[object] = None   # mediapipe NormalizedLandmarkList
    keypoints: Optional[List[float]] = None  # pre-processed 42-float list
    visible: bool = False


@dataclass
class FrameResult:
    """Full result for one processed frame."""
    left: HandState = field(default_factory=lambda: HandState('Left'))
    right: HandState = field(default_factory=lambda: HandState('Right'))

    @property
    def both_visible(self) -> bool:
        return self.left.visible and self.right.visible

    @property
    def status(self) -> str:
        """'both' | 'left_only' | 'right_only' | 'none'"""
        if self.left.visible and self.right.visible:
            return 'both'
        if self.left.visible:
            return 'left_only'
        if self.right.visible:
            return 'right_only'
        return 'none'


# ── smoothing buffer ──────────────────────────────────────────────────────────

class GestureBuffer:
    """
    Majority-vote buffer over the last `size` frames.

    """

    def __init__(self, size: int = 7):
        self._buf: deque = deque(maxlen=size)

    def update(self, gesture: str) -> str:
        self._buf.append(gesture)
        return self.stable

    @property
    def stable(self) -> str:
        if not self._buf:
            return 'no_hand'
        return Counter(self._buf).most_common(1)[0][0]

    def reset(self):
        self._buf.clear()


# ── pre-processing (identical to original repo) ───────────────────────────────

def _calc_landmark_list(image: np.ndarray, landmarks) -> List[List[int]]:
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        lx = min(int(landmark.x * image_width), image_width - 1)
        ly = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([lx, ly])
    return landmark_point


def _pre_process_landmark(landmark_list: List[List[int]]) -> List[float]:
    """Normalize to wrist-relative, flatten, range-normalize to [-1, 1]."""
    temp = copy.deepcopy(landmark_list)
    base_x, base_y = temp[0][0], temp[0][1]
    for pt in temp:
        pt[0] -= base_x
        pt[1] -= base_y
    flat = list(itertools.chain.from_iterable(temp))
    max_val = max(map(abs, flat)) or 1
    return [v / max_val for v in flat]


# ── main tracker class ────────────────────────────────────────────────────────

class HandTracker:
    """
    Usage
    -----
    tracker = HandTracker(classifier, label_list, buffer_size=7)

    while cap.isOpened():
        ret, frame = cap.read()
        flipped = cv2.flip(frame, 1)

        #  pass BOTH frames
        #   unflipped_frame=frame  → MediaPipe runs on this (correct L/R labels)
        #   first arg flipped      → landmark pixel coords mapped here (correct draw pos)
        result: FrameResult = tracker.process(flipped, unflipped_frame=frame)
    """

    NO_HAND_LABEL = 'no_hand'

    def __init__(
        self,
        keypoint_classifier,
        classifier_labels: List[str],
        buffer_size: int = 7,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 2,
    ):
        self._clf = keypoint_classifier
        self._labels = classifier_labels
        self._buffers: Dict[str, GestureBuffer] = {
            'Left':  GestureBuffer(buffer_size),
            'Right': GestureBuffer(buffer_size),
        }

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(
        self,
        frame: np.ndarray,
        unflipped_frame: Optional[np.ndarray] = None,
    ) -> FrameResult:
        """
        Process one frame and return a FrameResult.

          If unflipped_frame is provided, MediaPipe detection runs on it so
          Left/Right labels are correct.  Landmark pixel coords are still
          mapped onto `frame` (the flipped display frame) so draw positions
          are correct on screen.
        """
        result = FrameResult()

        detection_frame = unflipped_frame if unflipped_frame is not None else frame
        rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
        mp_result = self._hands.process(rgb)

        detected_sides = set()

        if mp_result.multi_hand_landmarks and mp_result.multi_handedness:
            for hand_landmarks, handedness in zip(
                mp_result.multi_hand_landmarks,
                mp_result.multi_handedness,
            ):
                classification = handedness.classification[0]
                side = classification.label        # 'Left' or 'Right' — now correct
                mp_confidence = classification.score

                detected_sides.add(side)

                # pixel coords mapped onto the flipped frame (matches screen)
                lm_list = _calc_landmark_list(frame, hand_landmarks)
                keypoints = _pre_process_landmark(lm_list)

                # Bug 3: clf now returns (index, score) tuple from patched classifier
                clf_result = self._clf(keypoints)
                if isinstance(clf_result, (tuple, list)):
                    clf_id, clf_score = clf_result[0], clf_result[1]
                else:
                    # fallback for unpatched classifier
                    clf_id, clf_score = clf_result, 1.0

                raw_gesture = self._labels[clf_id]
                stable_gesture = self._buffers[side].update(raw_gesture)

                state = HandState(
                    side=side,
                    gesture=stable_gesture,
                    raw_gesture=raw_gesture,
                    confidence=float(mp_confidence),
                    classifier_score=float(clf_score),
                    landmarks_raw=hand_landmarks,
                    keypoints=keypoints,
                    visible=True,
                )

                if side == 'Left':
                    result.left = state
                else:
                    result.right = state

        for side in ('Left', 'Right'):
            if side not in detected_sides:
                self._buffers[side].reset()
                state = HandState(
                    side=side,
                    gesture=self.NO_HAND_LABEL,
                    visible=False,
                )
                if side == 'Left':
                    result.left = state
                else:
                    result.right = state

        return result

    def close(self):
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()