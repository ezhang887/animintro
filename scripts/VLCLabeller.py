import vlc
import signal
import sys
import time
import threading

from pynput import keyboard

class Labeller():

    def __init__(self, video):

        self.video = video

        self.player = vlc.MediaPlayer(video)
        self.player.play()

        listener = keyboard.Listener(
                on_press=self._on_press)
        listener.start()

        self.intro_start = None
        self.intro_end = None
        self.outro_start = None
        self.outro_end = None
 
        time.sleep(0.1) # hack
        self.end_time = self.player.get_media().get_duration()
        
        self.skip_time = 5000 # 5 sec
        self.end_margin = 500 #0.5 sec
        self.thread_period = 0.1 #0.1 sec
        self.numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

        self.running = True
        signal.signal(signal.SIGINT, self._sigint_handler)

        t1 = threading.Thread(target=self._watchdog)
        t1.start()
        t1.join()
        self.player.stop()

    def _sigint_handler(self, signum, frame):
        self.running = False

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

        # toggle pause, skip forward/back
        if k == 'p':
            self.player.pause()
        elif k == 'l':
            t = self.player.get_time()
            new_t = min(self.end_time - self.end_margin, t + self.skip_time)
            self.player.set_time(new_t)
        elif k == 'k':
            t = self.player.get_time()
            new_t = max(0, t - self.skip_time)
            self.player.set_time(new_t)

        # set labels
        elif k == 'q':
            self.intro_start = self.player.get_time()
        elif k == 'w':
            self.intro_end = self.player.get_time()
        elif k == 'e':
            self.outro_start = self.player.get_time()
        elif k == 'r':
            self.outro_end = self.player.get_time()

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

Labeller("/home/ezhang/anime/video_folder/anime.mp4")
