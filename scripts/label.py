import argparse
import datetime
import json
import sys
import os
import glob
import logging
from pathlib import Path
import cv2

"""
q -> quit
n -> next frame (can hold)
p -> prev frame (can hold)
1 -> mark intro start
2 -> mark intro end
3 -> mark outro start
4 -> mark outro end
[Enter] -> next video
"""


VALID_KEYS = {e: ord(e) for e in ["q", "p", "n", "1", "2", "3", "4"]}
VALID_KEYS["Enter"] = 13


def create_label_dict():
    return {
        "intro_start": None,
        "intro_end": None,
        "outro_start": None,
        "outro_end": None,
    }


def label_video(video_path):
    res = create_label_dict()

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        _, frame = cap.read()
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.imshow("frame", frame)

        key = cv2.waitKey(30)
        while not key in VALID_KEYS.values():
            key = cv2.waitKey(30)

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_minutes = str(datetime.timedelta(seconds=timestamp_ms / 1000))

        if key == VALID_KEYS["q"]:
            sys.exit(0)

        elif key == VALID_KEYS["p"] and frame_num > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 2)

        elif key == VALID_KEYS["1"]:
            res["intro_start"] = timestamp_minutes

        elif key == VALID_KEYS["2"]:
            res["intro_end"] = timestamp_minutes

        elif key == VALID_KEYS["3"]:
            res["outro_start"] = timestamp_minutes

        elif key == VALID_KEYS["4"]:
            res["outro_end"] = timestamp_minutes

        elif key == VALID_KEYS["Enter"]:
            if None in res.values():
                logging.warn(
                    "Not all the timestamps are marked! Please mark everything before pressing Enter."
                )
            else:
                break

    return res


def main(args):
    video_folder = args.video_folder
    output_folder = args.output_folder

    for mp4 in sorted(glob.glob(os.path.join(video_folder, "*.mp4"))):
        basename = Path(mp4).stem
        logging.info(f"Labeling video {basename}...")
        result = label_video(mp4)

        output_path = os.path.join(output_folder, f"{basename}.json")
        logging.info(f"Saving labels at {output_path}...")
        with open(output_path, "w") as output_file:
            json.dump(result, output_file)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("video_folder")
    parser.add_argument("output_folder")
    args = parser.parse_args()

    main(args)
