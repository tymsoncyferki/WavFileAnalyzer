import os

class Config:

    MAIN_CLASSES = [
        "one", "two", "three", "four",
        "five", "six", "seven", "eight", "nine",
    ]

    BASEDIR = os.environ.get("BASEDIR", "/home/tymek/.cache/kagglehub/datasets/neehakurelli/google-speech-commands/versions/1")
    