"""
Touch the Dot Game: A game where players use hand tracking to touch dots that appear on the screen.
"""

import sys
import random
import threading
import time
from typing import List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
import cv2
import numpy as np
import mediapipe as mp

# pylint: disable=E1101


class Modes(Enum):
    """
    Enum class with game modes: easy, medium and hard.
    """

    EASY = 50
    MEDIUM = 25
    HARD = 5


@dataclass
class Shapes:
    """
    Data class with sizes of frame and side panel.
    """

    frame_height = 480
    side_panel_width = 280


COUNTDOWN = 10


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
        Processes a video frame to detect hands and extract landmark data using MediaPipe.

        Args:
            frame (np.ndarray): The current frame from the video capture device.

        Returns:
            Any: The processed results containing hand landmark information.
        """
        return self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    @staticmethod
    def draw_landmarks(frame: np.ndarray, results: Any) -> np.ndarray:
        """
        Draws hand landmarks on the video frame using the results from MediaPipe hand tracking.

        Args:
            frame (np.ndarray): The current frame from the video capture device to draw
            landmarks on.
            results (Any): The results from hand tracking that contain landmark data.

        Returns:
            np.ndarray: The frame with hand landmarks drawn.
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                tip_x = int(hand_landmarks.landmark[8].x * frame.shape[1])
                tip_y = int(hand_landmarks.landmark[8].y * frame.shape[0])
                cv2.circle(frame, (tip_x, tip_y), 5, (0, 255, 0), -1)
        return frame

    @staticmethod
    def get_index_tip_coordinates(
        frame: np.ndarray, results: Any
    ) -> Optional[Tuple[int, int]]:
        """
        Retrieves the coordinates of the index finger tips detected in the frame.

        Args:
            frame (np.ndarray): The current frame from the video capture device.
            results (Any): The results containing hand landmark data.

        Returns:
            Optional[List[Tuple[int, int]]]: A list of tuples containing the x and y coordinates of
            index finger tips, or None if no hands are detected.
        """
        index_tip = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                height, width, _ = frame.shape
                index_tip_coord = (
                    int(hand_landmarks.landmark[8].x * width),
                    int(hand_landmarks.landmark[8].y * height),
                )
                index_tip.append(index_tip_coord)
        return index_tip


# pylint: disable=R0903
class ExternalGameAttributes:
    """
    Class to hold external game attributes like difficulty and resize factor.
    """

    def __init__(self, difficulty: int = Modes.EASY.value, resize_factor: float = 1):
        self.difficulty = difficulty
        self.resize_factor = resize_factor


class TouchGame:
    """
    Class representing the Touch the Dot game.
    """

    def __init__(self) -> None:
        self.attributes = ExternalGameAttributes()
        self.detector = HandDetector()
        self.circle_coords: Optional[Tuple[int, int]] = None
        self.timer = f"{COUNTDOWN}"
        self.game_started = False
        self.circle_found = True
        self.game_over = False

    def countdown(self, game_time: int) -> None:
        """
        Countdown timer for the game's duration.

        Args:
            game_time (int): Total game time in seconds.
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
        Draw a random circle within the frame.

        Args:
            frame (np.ndarray): The current video frame.

        Returns:
            Tuple[int, int]: Coordinates of the circle's center.
        """
        max_y, max_x, _ = frame.shape
        coord1 = random.randint(10, max_y - 10)
        coord2 = random.randint(10, max_x - 10)
        return (coord2, coord1)

    def check_touch(self, index_tip: Optional[Tuple[int, int]]) -> bool:
        """
        Check if the index finger tip is touching the circle.

        Args:
            index_tip (Optional[Tuple[int, int]]): Coordinates of the index finger tip.

        Returns:
            bool: True if touching the circle, False otherwise.
        """
        if index_tip and self.circle_coords:
            for index_tip_hand in index_tip:
                distance = np.sqrt(
                    (index_tip_hand[0] - self.circle_coords[0]) ** 2
                    + (index_tip_hand[1] - self.circle_coords[1]) ** 2
                )
                if distance < self.attributes.difficulty:
                    return True
        return False

    @staticmethod
    def add_text(frame: np.ndarray, text: str, position: Tuple[int, int]) -> None:
        """
        Write text onto a frame at a specified position.

        Args:
            frame (np.ndarray): Frame on which text is to be written.
            text (str): Text to write.
            position (Tuple[int, int]): Coordinates for the text placement.
        """
        cv2.putText(frame, text, position, 5, 1.5, (255, 255, 255), 2)

    def find_index_tip(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find the tip of the index finger using the hand detector.

        Args:
            frame (np.ndarray): The current frame from video capture.

        Returns:
            List[Tuple[int, int]]: A list of coordinates for index finger tips detected.
        """
        results = self.detector.process_frame(frame)
        frame = self.detector.draw_landmarks(frame, results)
        index_tip = self.detector.get_index_tip_coordinates(frame, results)
        return index_tip

    def _resize_frame(self, side_panel: np.ndarray, key: int) -> np.ndarray:
        """
        Resize the game frame based on user input to change scale.

        Args:
            side_panel (np.ndarray): The current side panel.
            key (int): Keyboard input used to determine scaling.

        Returns:
            np.ndarray: The resized side panel.
        """
        if key == ord("1"):
            self.attributes.resize_factor = 1
            side_panel = cv2.resize(
                side_panel, (Shapes.side_panel_width, Shapes.frame_height)
            )
        elif key == ord("2"):
            self.attributes.resize_factor = 1.5
            side_panel = cv2.resize(
                side_panel,
                (
                    Shapes.side_panel_width,
                    int(Shapes.frame_height * self.attributes.resize_factor),
                ),
            )
        return side_panel

    def _set_difficulty(self, key: int) -> int:
        """
        Set the game difficulty based on a key press.

        Args:
            key (int): Key pressed to set the difficulty level.
        """
        if key == ord("e"):
            self.attributes.difficulty = Modes.EASY.value
        elif key == ord("m"):
            self.attributes.difficulty = Modes.MEDIUM.value
        elif key == ord("h"):
            self.attributes.difficulty = Modes.HARD.value

    def handle_key_input(
        self, key: int, side_panel: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Handles key inputs to control game functions like starting, changing difficulty,
        resizing the frame, and exiting.

        Args:
            key (int): The key pressed by the user.
            side_panel (np.ndarray): The side panel part of the game UI.

        Returns:
            Optional[np.ndarray]: The potentially resized side panel or None if the game
            is to be exited.
        """
        if key == ord("s") and not self.game_started:
            self.start_game()
        elif key in {ord("e"), ord("m"), ord("h")}:
            self._set_difficulty(key)
        elif key in {ord("1"), ord("2")}:
            return self._resize_frame(side_panel, key)
        elif key == ord("q"):
            self.game_over = True
            return None
        return side_panel

    def start_game(self) -> None:
        """
        Starts the game by setting the game_started flag to True and starting the timer thread.
        """
        self.game_started = True
        timer_thread = threading.Thread(target=self.countdown, args=(COUNTDOWN,))
        if not timer_thread.is_alive():
            timer_thread.start()

    def process_frame_actions(self, frame: np.ndarray) -> int:
        """
        Processes frame actions when the game has started, checking for touches
        and managing the game state.

        Args:
            frame (np.ndarray): The current frame from the video capture.
            side_panel (np.ndarray): The side panel used for displaying
            game information.

        Returns:
            int: The increment to the score counter if a touch is detected.
        """
        if self.game_started:
            if self.circle_found:
                self.circle_coords = self.draw_random_circle(frame)
                self.circle_found = False

            if self.circle_coords:
                cv2.circle(frame, self.circle_coords, 5, (255, 0, 0), -1)

            index_tip = self.find_index_tip(frame)
            touch_detected = self.check_touch(index_tip)

            if touch_detected:
                self.circle_found = True
                return 1
        return 0

    def game_loop(self) -> None:
        """
        Main game loop that manages the overall game operations including capturing
        frames, processing actions, and handling UI updates.
        """
        counter = 0
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            sys.exit()

        side_panel = np.full(
            (Shapes.frame_height, Shapes.side_panel_width, 3), (0, 0, 0), dtype=np.uint8
        )

        while True:
            if self.game_over:
                break
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            if self.attributes.resize_factor != 1:
                frame = cv2.resize(
                    frame,
                    (
                        int(frame.shape[1] * self.attributes.resize_factor),
                        int(frame.shape[0] * self.attributes.resize_factor),
                    ),
                )

            if not self.game_started:
                self.add_text(frame, "Press 's' to start", (50, frame.shape[0] // 2))

            counter += self.process_frame_actions(frame)

            side_panel[:] = (0, 0, 0)
            self.add_text(side_panel, f"Time: {self.timer}", (10, 35))
            self.add_text(side_panel, f"Count: {counter}", (10, 70))
            self.add_text(
                side_panel,
                f"Mode: {Modes(self.attributes.difficulty).name}",
                (10, side_panel.shape[0] - 20),
            )

            full_game = np.hstack((frame, side_panel))
            cv2.imshow("Frame", full_game)

            key = cv2.waitKey(1) & 0xFF
            new_side_panel = self.handle_key_input(key, side_panel)
            if new_side_panel is None:
                break

            side_panel = new_side_panel

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    game = TouchGame()
    game.game_loop()
