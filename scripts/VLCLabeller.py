import vlc
import os
import signal
import glob
import argparse
import sys
import time
import threading

from reprint import output
from pynput import keyboard
from pathlib import Path


class Labeller:

    def __init__(self, output_folder):
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

        # setup signal handler
        signal.signal(signal.SIGINT, self._sigint_handler)

        self._fancy_out = output(initial_len=5, interval=0)
        self.fancy_out = self._fancy_out.__enter__()
        self.fancy_log = ''

    def run(self, video_path):
        basename = Path(video_path).stem
        self.fancy_log = f"Starting labeller for {basename}..."

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
                self.fancy_log = "Labels not valid! Not moving on..."

        # set labels
        elif k == "q":
            self.fancy_log = f"Saving intro_start as {t}..."
            self.intro_start = t
        elif k == "w":
            self.fancy_log = f"Saving intro_end as {t}..."
            self.intro_end = t
        elif k == "e":
            self.fancy_log = f"Saving outro_start as {t}..."
            self.outro_start = t
        elif k == "r":
            self.fancy_log = f"Saving outro_end as {t}..."
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
        self.fancy_out[0] = self.fancy_log
        self.fancy_out[1] = f"Intro Start: {self.intro_start}"
        self.fancy_out[2] = f"Intro End: {self.intro_end}"
        self.fancy_out[3] = f"Outro Start: {self.outro_start}"
        self.fancy_out[4] = f"Outro End: {self.outro_end}"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("video_folder")
    parser.add_argument("label_folder")
    args = parser.parse_args()

    video_folder = args.video_folder
    label_folder = args.label_folder

    label_files = glob.glob(os.path.join(label_folder, "*.label"))
    label_files = [Path(f).stem for f in label_files]
    video_files = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))
    video_files = [f for f in video_files if not Path(f).stem in label_files]

    labeller = Labeller(label_folder)
    for file_name in video_files:
        labeller.run(file_name)
