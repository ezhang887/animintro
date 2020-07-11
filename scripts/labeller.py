import vlc
import os
import signal
import glob
import argparse
import sys
import time
import threading
import logging
import shutil

from reprint import output
from pynput import keyboard
from pathlib import Path


class Labeller:
    def __init__(self, output_folder, logging_folder):
        # setup keyboard listener
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

        # labels
        self.intro_start = None
        self.intro_end = None
        self.outro_start = None
        self.outro_end = None

        # constants
        self.skip_time = 5000  # 5 sec
        self.end_margin = 500  # 0.5 sec
        self.thread_period = 0.1  # 0.1 sec
        self.numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]

        self.output_folder = output_folder
        self.logging_folder = logging_folder

        # setup signal handler
        signal.signal(signal.SIGINT, self._sigint_handler)

        self.displayer = output(initial_len=5, interval=0)
        self.output_buffer = self.displayer.__enter__()
        self.status = ""

    def run(self, video_path):
        basename = Path(video_path).stem
        self.status = f"Starting labeller for {basename}..."
        self.logger = self._setup_logger(f"{basename}.log")
        self.logger.info(self.status)

        # start playing video
        self.player = vlc.MediaPlayer(video_path)
        self.player.play()
        time.sleep(0.1)  # hack

        if not self.player.is_playing():
            return

        self.running = True
        self.end_time = self.player.get_media().get_duration()

        # start watchdog
        watchdog = threading.Thread(target=self._watchdog)
        watchdog.start()

        # wait until the watchdog has exited
        watchdog.join()

        # and cleanup
        self.player.stop()
        self._save_video_files(basename)

    def __del__(self):
        self.listener.stop()
        self.displayer.__exit__()

    def _setup_logger(self, filename):
        handler = logging.FileHandler(os.path.join(self.logging_folder, filename))
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)

        logger = logging.getLogger(filename)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        return logger

    def _validate_labels(self):
        # TODO: maybe check if the timestamps are correct (like intro_end > intro_start,
        # intro_end < outtro_start), etc
        return None not in {
            self.intro_start,
            self.intro_end,
            self.outro_start,
            self.outro_end,
        }

    def _save_video_files(self, basename):
        output_fname = f"{os.path.join(self.output_folder, basename)}.label"
        with open(output_fname, "w") as f:
            f.write(f"{self.intro_start}\n")
            f.write(f"{self.intro_end}\n")
            f.write(f"{self.outro_start}\n")
            f.write(f"{self.outro_end}\n")
        self.logger.info(f"Saved labels to {output_fname}...")

        self.intro_start = None
        self.intro_end = None
        self.outro_start = None
        self.outro_end = None

    def _sigint_handler(self, signum, frame):
        self.running = False
        self.listener.stop()
        sys.exit(0)

    def _watchdog(self):
        while self.running:
            self._print()
            if self.player.get_time() >= self.end_time - self.end_margin:
                if self.player.get_state() == vlc.State.Playing:
                    self.logger.warn(
                        "Hit end of video, pausing so vlc doesn't get stuck..."
                    )
                    self.player.pause()
            time.sleep(self.thread_period)

    def _on_press(self, key):
        k = ""
        try:
            k = key.char
        except AttributeError:
            pass

        t = self.player.get_time()

        # toggle pause, skip forward/back
        if k == "p":
            self.player.pause()
        elif k == "l":
            new_t = min(self.end_time - self.end_margin, t + self.skip_time)
            self.player.set_time(new_t)
        elif k == "k":
            new_t = max(0, t - self.skip_time)
            self.player.set_time(new_t)

        # move to next video
        elif k == "n":
            if self._validate_labels():
                self.running = False
            else:
                self.status = "Labels not valid! Not moving on..."
                self.logger.warn(self.status)

        # set labels
        elif k == "q":
            self.status = f"Saving intro_start as {t}..."
            self.logger.info(self.status)
            self.intro_start = t
        elif k == "w":
            self.status = f"Saving intro_end as {t}..."
            self.logger.info(self.status)
            self.intro_end = t
        elif k == "e":
            self.status = f"Saving outro_start as {t}..."
            self.logger.info(self.status)
            self.outro_start = t
        elif k == "r":
            self.status = f"Saving outro_end as {t}..."
            self.logger.info(self.status)
            self.outro_end = t

        # jump to labelled times
        elif k == "a":
            if self.intro_start:
                self.player.set_time(self.intro_start)
        elif k == "s":
            if self.intro_end:
                self.player.set_time(self.intro_end)
        elif k == "d":
            if self.outro_start:
                self.player.set_time(self.outro_start)
        elif k == "f":
            if self.outro_end:
                self.player.set_time(self.outro_end)

        # jump to video position
        elif k in self.numbers:
            pos = 0.1 * int(k)
            self.player.set_position(pos)
            if self.player.get_state != vlc.State.Playing:
                self.player.play()

    def _print(self):
        self.output_buffer[0] = self.status
        self.output_buffer[1] = f"Intro Start: {self.intro_start}"
        self.output_buffer[2] = f"Intro End: {self.intro_end}"
        self.output_buffer[3] = f"Outro Start: {self.outro_start}"
        self.output_buffer[4] = f"Outro End: {self.outro_end}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_folder")
    parser.add_argument("label_folder")
    parser.add_argument("--log_folder", default="logs")
    args = parser.parse_args()

    global_logger = logging.getLogger(__name__)

    video_folder = args.video_folder
    label_folder = args.label_folder

    label_files = glob.glob(os.path.join(label_folder, "*.label"))
    label_files = [Path(f).stem for f in label_files]
    video_files = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))
    video_files = [f for f in video_files if not Path(f).stem in label_files]

    log_folder = args.log_folder
    # If the logs folder already exists, move that folder to somewhere else
    if os.path.exists(log_folder):
        new_logs_path = f"{log_folder}_{time.strftime('%Y%m%d-%H%M%S')}"
        global_logger.warn(
            f"{log_folder} already exists! Moving that to {new_logs_path}"
        )
        shutil.move(log_folder, new_logs_path)
    os.makedirs(log_folder)

    labeller = Labeller(label_folder, log_folder)

    for file_name in video_files:
        labeller.run(file_name)
