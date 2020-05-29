import vlc
from pynput import keyboard

class Labeller():

    def __init__(self, video):
        self.player = vlc.MediaPlayer(video)
        self.player.play()

        listener = keyboard.Listener(
                on_press=self.on_press)
        listener.start()

        self.intro_start = None
        self.intro_end = None
        self.outro_start = None
        self.outro_end = None

        self.end_time = self.player.get_length() #TODO - add functionality
        print(self.end_time)
        self.skip_time = 5000

        # self.player.set_rate(0.5)

        self.numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

    def on_press(self, key):
        k = ''
        try:
            k = key.char
        except AttributeError:
            pass

        if k == 'p':
            self.player.pause()
        elif k == 'l':
            t = self.player.get_time()
            #new_t = min(self.end_time, t + self.skip_time)
            new_t = t + self.skip_time
            self.player.set_time(new_t)
        elif k == 'k':
            t = self.player.get_time()
            new_t = max(0, t - self.skip_time)
            self.player.set_time(new_t)

        elif k == 'q':
            self.intro_start = self.player.get_time()
        elif k == 'w':
            self.intro_end = self.player.get_time()
        elif k == 'e':
            self.outro_start = self.player.get_time()
        elif k == 'r':
            self.outro_end = self.player.get_time()

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

        elif k in self.numbers:
            pos = 0.1 * int(k)
            self.player.set_position(pos)

Labeller("test.mp4")
while True:
    continue

