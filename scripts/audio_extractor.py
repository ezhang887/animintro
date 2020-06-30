import os
import glob
import argparse
import subprocess

from pathlib import Path


class AudioExtractor:
    def __init__(self, audio_folder):
        self.audio_folder = audio_folder

    def extract(self, video_path):

        basename = Path(video_path).stem
        audio_file = "%s%s.wav" % (self.audio_folder, basename)
        command = "ffmpeg -i %s -ab 160k -ac 2 -ar 44100 -vn %s" % (
            video_path,
            audio_file,
        )

        subprocess.call(command, shell=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("video_folder")
    parser.add_argument("audio_folder")
    args = parser.parse_args()

    video_folder = args.video_folder
    audio_folder = args.audio_folder

    audio_files = glob.glob(os.path.join(audio_folder, "*.wav"))
    audio_files = [Path(f).stem for f in audio_files]
    video_files = glob.glob(os.path.join(video_folder, "*.mp4"))
    video_files = [f for f in video_files if Path(f).stem not in audio_files]

    audio_extractor = AudioExtractor(audio_folder)
    for file_name in video_files:
        audio_extractor.extract(file_name)
