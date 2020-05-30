import vlc
import os
import signal
import glob
import argparse
import sys
import time
import threading

from pynput import keyboard

class Labeller():

    def __init__(self):
        # setup keyboard listener
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

        self.intro_start = None
        self.intro_end = None
        self.outro_start = None
        self.outro_end = None
 
        self.skip_time = 5000 # 5 sec
        self.end_margin = 500 # 0.5 sec
        self.thread_period = 0.1 # 0.1 sec
        self.numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

        # setup watchdog and signal handler
        signal.signal(signal.SIGINT, self._sigint_handler)

    def run(self, video_path):
        # start playing video
        self.player = vlc.MediaPlayer(video_path)
        self.player.play()
        self.running = True
        time.sleep(0.1) # hack
        self.end_time = self.player.get_media().get_duration()

        # start watchdog
        watchdog = threading.Thread(target=self._watchdog)
        watchdog.start()

        # wait until the watchdog has exited
        watchdog.join()

        # and cleanup
        self.player.stop()

    def _sigint_handler(self, signum, frame):
        self.running = False
        self.listener.stop()
        sys.exit(0)

    def _watchdog(self):
        while self.running:
            if self.player.get_time() >= self.end_time - self.end_margin:
                if self.player.get_state() == vlc.State.Playing:
                    self.player.pause()
            time.sleep(self.thread_period)

    def _on_press(self, key):
        k = ''
        try:
            k = key.char
        except AttributeError:
            pass

        t = self.player.get_time()

        # toggle pause, skip forward/back
        if k == 'p':
            self.player.pause()
        elif k == 'l':
            new_t = min(self.end_time - self.end_margin, t + self.skip_time)
            self.player.set_time(new_t)
        elif k == 'k':
            new_t = max(0, t - self.skip_time)
            self.player.set_time(new_t)
        # move to next video
        elif k == 'n':
            # TODO: maybe add some checks to make sure all the labels were set b4 moving on
            self.running = False

        # set labels
        elif k == 'q':
            self.intro_start = t
        elif k == 'w':
            self.intro_end = t
        elif k == 'e':
            self.outro_start = t
        elif k == 'r':
            self.outro_end = t

        # jump to labelled times
        elif k == 'a':
            if self.intro_start:
                self.player.set_time(self.intro_start)
        elif k == 's':
            if self.intro_end:
                self.player.set_time(self.intro_end)
        elif k == 'd':
            if self.outro_start:
                self.player.set_time(self.outro_start)
        elif k == 'f':
            if self.outro_end:
                self.player.set_time(self.outro_end)

        # jump to video position
        elif k in self.numbers:
            pos = 0.1 * int(k)
            self.player.set_position(pos)
            if self.player.get_state != vlc.State.Playing:
                self.player.play()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_folder")
    #parser.add_argument("output_folder")
    args = parser.parse_args()

    video_folder = args.video_folder
    video_files = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))

    labeller = Labeller()
    for f in video_files:
        labeller.run(f)
