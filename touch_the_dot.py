"""
Touch the Dot Game: A game where players use hand tracking to touch dots that appear on the screen.
"""

import sys
import random  # Standard library imports first
import threading
import time
from typing import Tuple, Optional, Any
import cv2
import numpy as np
import mediapipe as mp

EASY_MODE = 50
MEDIUM_MODE = 25
HARD_MODE = 5

COUNTDOWN = 60


class HandDetector:
    """
    Class for detecting hands in a video frame using MediaPipe.
    """

    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
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

    @staticmethod
    def get_index_tip_coordinates(frame: np.ndarray, results: Any) \
            -> Optional[Tuple[int, int]]:
        """
        Retrieve the coordinates of the index finger tip from hand landmarks.
        """
        index_tip = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                height, width, _ = frame.shape
                index_tip = (int(hand_landmarks.landmark[8].x * width),
                             int(hand_landmarks.landmark[8].y * height))
        return index_tip


class TouchGame:
    """
    Class representing the Touch the Dot game.
    """

    def __init__(self, difficulty: str = EASY_MODE) -> None:
        self.difficulty = difficulty
        self.counter = 0
        self.detector = HandDetector()
        self.circle_found = True
        self.circle_coords: Optional[Tuple[int, int]] = None
        self.game_over = False
        self.timer = f"{COUNTDOWN}"

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
            distance = np.sqrt((index_tip[0] - self.circle_coords[0]) ** 2 +
                               (index_tip[1] - self.circle_coords[1]) ** 2)
            return distance < self.difficulty
        return False

    def game_loop(self) -> None:
        """
        The main game loop.
        """
        timer_thread = threading.Thread(
            target=self.countdown, args=(COUNTDOWN,))
        timer_thread.start()
        # pylint: disable=no-member
        cap = cv2.VideoCapture(2)

        if not cap.isOpened():
            print("Cannot open camera")
            sys.exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if self.circle_found:
                self.circle_coords = self.draw_random_circle(frame)
                self.circle_found = False

            results = self.detector.process_frame(frame)
            frame = self.detector.draw_landmarks(frame, results)
            index_tip = self.detector.get_index_tip_coordinates(frame, results)

            touch_detected = self.check_touch(index_tip)

            if self.circle_coords:
                # pylint: disable=no-member
                cv2.circle(frame, self.circle_coords, 5, (255, 0, 0), -1)

            # pylint: disable=no-member
            cv2.putText(frame, str(self.timer),
                        (frame.shape[1] - 70, 45), 5, 2, (0, 0, 255), 2)
            # pylint: disable=no-member
            cv2.putText(frame, str(self.counter),
                        (15, 45), 5, 2, (0, 0, 255), 2)
            # pylint: disable=no-member
            cv2.imshow("Frame", frame)

            if touch_detected:
                self.circle_found = True
                self.counter += 1

            if self.game_over:
                print(f"Final score: {self.counter}")
                break

            # pylint: disable=no-member
            if cv2.waitKey(1) == ord('e'):
                self.difficulty = EASY_MODE
            # pylint: disable=no-member
            elif cv2.waitKey(1) == ord('m'):
                self.difficulty = MEDIUM_MODE
            # pylint: disable=no-member
            elif cv2.waitKey(1) == ord('h'):
                self.difficulty = HARD_MODE
            # pylint: disable=no-member
            elif cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        # pylint: disable=no-member
        cv2.destroyAllWindows()

        timer_thread.join()


if __name__ == "__main__":
    game = TouchGame()
    game.game_loop()
