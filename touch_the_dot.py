"""
Touch the Dot Game: A game where players use hand tracking to touch dots that appear on the screen.
"""

import sys
import random  # Standard library imports first
import threading
import time
from typing import List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
import cv2
import numpy as np
import mediapipe as mp


class Modes(Enum):
    """
    Enum class with game modes: easy, medium and hard.
    """

    EASY = 50
    MEDIUM = 25
    HARD = 5


@ dataclass
class Shapes:
    """
    Data class with sizes of frame and side panel.
    """

    frame_height = 480
    side_panel_width = 280


COUNTDOWN = 60


class HandDetector:
    """
    Class for detecting hands in a video frame using MediaPipe.
    """

    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)
        self.mp_drawing_utils = mp.solutions.drawing_utils

    def process_frame(self, frame: np.ndarray) -> Any:
        """
        Process a frame to detect hands.
        """
        # pylint: disable=no-member
        return self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def draw_landmarks(self, frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Draw hand landmarks on a frame.
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing_utils.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame

    @ staticmethod
    def get_index_tip_coordinates(frame: np.ndarray, results: Any) -> Optional[Tuple[int, int]]:
        """
        Retrieve the coordinates of the index finger tip from hand landmarks.
        """
        index_tip = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                height, width, _ = frame.shape
                index_tip_coord = (int(hand_landmarks.landmark[8].x * width),
                                   int(hand_landmarks.landmark[8].y * height))
                index_tip.append(index_tip_coord)
        return index_tip


class TouchGame:
    """
    Class representing the Touch the Dot game.
    """

    def __init__(self, difficulty: str = Modes.EASY.value) -> None:
        self.difficulty = difficulty
        self.detector = HandDetector()
        self.circle_found = True
        self.game_over = False
        self.circle_coords: Optional[Tuple[int, int]] = None
        self.timer = f"{COUNTDOWN}"
        self.resize_factor = 1

    def countdown(self, game_time: int) -> None:
        """
        Countdown timer.
        """
        while game_time > 0 and not self.game_over:
            _, secs = divmod(game_time, 60)
            self.timer = f"{secs:02d}"
            time.sleep(1)
            game_time -= 1

        self.game_over = True
        print("\nTime is up!")

    @staticmethod
    def draw_random_circle(frame: np.ndarray) -> Tuple[int, int]:
        """
        Draw a random circle on the frame.
        """
        max_y, max_x, _ = frame.shape
        coord1 = random.randint(10, max_y - 10)
        coord2 = random.randint(10, max_x - 10)
        return (coord2, coord1)

    def check_touch(self, index_tip: Optional[Tuple[int, int]]) -> bool:
        """
        Check if the index finger tip is touching the circle.
        """
        if index_tip and self.circle_coords:
            for index_tip_hand in index_tip:
                distance = np.sqrt((index_tip_hand[0] - self.circle_coords[0]) ** 2 +
                                   (index_tip_hand[1] - self.circle_coords[1]) ** 2)
                if distance < self.difficulty:
                    return True
        return False

    @staticmethod
    def add_text(frame: np.ndarray, text: str, position: Tuple[int, int]) -> None:
        """
        Write text onto a frame at a certain postion.
        """
        # pylint: disable=no-member
        cv2.putText(frame, text, position, 5, 1.5, (255, 255, 255), 2)

    def find_index_tip(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find the tip of the index finger.
        """
        results = self.detector.process_frame(frame)
        frame = self.detector.draw_landmarks(frame, results)
        index_tip = self.detector.get_index_tip_coordinates(frame, results)
        return index_tip

    def _resize_frame(self, side_panel: np.ndarray, key: int) -> np.ndarray:
        """
        Resize frame.
        """
        # pylint: disable=no-member
        if key == ord('1'):
            self.resize_factor = 1
            side_panel = cv2.resize(
                side_panel, (Shapes.side_panel_width, Shapes.frame_height))
        elif key == ord('2'):
            self.resize_factor = 1.5
            side_panel = cv2.resize(
                side_panel, (Shapes.side_panel_width,
                             int(Shapes.frame_height * self.resize_factor)))
        return side_panel

    def _set_difficulty(self, key: int) -> int:
        """
        Set game difficulty.
        """
        if key == ord('e'):
            self.difficulty = Modes.EASY.value
        elif key == ord('m'):
            self.difficulty = Modes.MEDIUM.value
        elif key == ord('h'):
            self.difficulty = Modes.HARD.value

    def game_loop(self) -> None:
        """
        The main game loop.
        """
        counter = 0
        timer_thread = threading.Thread(
            target=self.countdown, args=(COUNTDOWN,))
        timer_thread.start()
        # pylint: disable=no-member
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Cannot open camera")
            sys.exit()

        side_panel = np.full(
            (Shapes.frame_height, Shapes.side_panel_width, 3), (0, 0, 0), dtype=np.uint8)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if self.resize_factor != 1:
                frame = cv2.resize(
                    frame, (int(frame.shape[1] * self.resize_factor),
                            int(frame.shape[0] * self.resize_factor)))

            if self.circle_found:
                self.circle_coords = self.draw_random_circle(frame)
                self.circle_found = False

            index_tip = self.find_index_tip(frame)

            touch_detected = self.check_touch(index_tip)

            if self.circle_coords:
                # pylint: disable=no-member
                cv2.circle(frame, self.circle_coords, 5, (255, 0, 0), -1)

            side_panel[:] = (0, 0, 0)
            self.add_text(side_panel, f"Time: {self.timer}", (10, 35))
            self.add_text(side_panel, f"Count: {counter}", (10, 70))
            self.add_text(
                side_panel, f"Mode: {Modes(self.difficulty).name}", (10, side_panel.shape[0] - 20))

            full_game = np.hstack((frame, side_panel))
            cv2.imshow("Frame", full_game)

            if touch_detected:
                self.circle_found = True
                counter += 1

            if self.game_over:
                print(f"Final score: {counter}")
                break

            key = cv2.waitKey(1) & 0xFF
            # pylint: disable=no-member
            if key == ord('e') or key == ord('m') or key == ord('h'):
                self._set_difficulty(key)
            elif key == ord('1') or key == ord('2'):
                side_panel = self._resize_frame(side_panel, key)
            elif key == ord('q'):
                break

        cap.release()
        # pylint: disable=no-member
        cv2.destroyAllWindows()

        timer_thread.join()


if __name__ == "__main__":
    game = TouchGame()
    game.game_loop()
