"""
mqtt_pub.py  (patched)
----------------------
"""

import json
import time
import logging
import threading
from typing import Optional

import paho.mqtt.client as mqtt

from hand_tracker import FrameResult, HandState

logger = logging.getLogger(__name__)


class MQTTPublisher:
    """
    Thin MQTT wrapper. Publishes FrameResult from HandTracker.

    Topics:
      robot/hands/left/gesture   → JSON payload
      robot/hands/right/gesture  → JSON payload
      robot/hands/status         → "both" | "left_only" | "right_only" | "none"
    """

    BASE_TOPIC = 'robot/hands'

    def __init__(
        self,
        broker: str = 'localhost',
        port: int = 1883,
        client_id: str = 'hand_gesture_publisher',
        qos: int = 1,
        publish_keypoints: bool = False,
        only_on_change: bool = True,
        connect_timeout: float = 3.0,   #  max seconds to wait for connection
    ):
        self._broker = broker
        self._port = port
        self._qos = qos
        self._publish_keypoints = publish_keypoints
        self._only_on_change = only_on_change
        self._connect_timeout = connect_timeout

        # Event to block until _on_connect fires
        self._connected_event = threading.Event()
        self._connected = False

        # use CallbackAPIVersion.VERSION1 for paho-mqtt v2+
        try:
            self._client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION1,
                client_id=client_id,
            )
        except AttributeError:
            # paho-mqtt < 2.0 — old API, no CallbackAPIVersion
            self._client = mqtt.Client(client_id=client_id)

        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        self._last_gesture: dict = {'Left': None, 'Right': None}
        self._last_status: Optional[str] = None

    # ── connection ─────────────────────────────────────────────────────────────

    def connect(self):
        self._client.connect(self._broker, self._port, keepalive=60)
        self._client.loop_start()
        # wait until _on_connect fires (or timeout)
        connected = self._connected_event.wait(timeout=self._connect_timeout)
        if not connected:
            logger.warning(
                f'[MQTT] connection to {self._broker}:{self._port} timed out '
                f'after {self._connect_timeout}s — publish calls will be dropped'
            )

    def disconnect(self):
        self._client.loop_stop()
        self._client.disconnect()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            self._connected_event.set()     # unblock connect()
            logger.info(f'[MQTT] connected to {self._broker}:{self._port}')
        else:
            logger.warning(f'[MQTT] connect failed rc={rc}')

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        logger.info('[MQTT] disconnected')

    # ── publish ────────────────────────────────────────────────────────────────

    def publish(self, frame_result: FrameResult):
        """Call once per processed frame."""
        if not self._connected:
            return

        for state in (frame_result.left, frame_result.right):
            self._publish_hand(state)

        status = frame_result.status
        if status != self._last_status:
            self._client.publish(
                f'{self.BASE_TOPIC}/status',
                status,
                qos=self._qos,
                retain=True,
            )
            self._last_status = status

    def _publish_hand(self, state: HandState):
        side_lower = state.side.lower()
        topic = f'{self.BASE_TOPIC}/{side_lower}/gesture'

        if self._only_on_change and state.gesture == self._last_gesture[state.side]:
            return

        payload = {
            'gesture':    state.gesture,
            'confidence': round(state.confidence, 3),
            'clf_score':  round(state.classifier_score, 3),
            'visible':    state.visible,
            'timestamp':  round(time.time(), 3),
        }

        if self._publish_keypoints and state.keypoints:
            payload['keypoints'] = [round(v, 4) for v in state.keypoints]

        self._client.publish(topic, json.dumps(payload), qos=self._qos)
        self._last_gesture[state.side] = state.gesture