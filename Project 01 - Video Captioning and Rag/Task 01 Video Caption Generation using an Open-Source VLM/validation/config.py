"""
Central configuration for the caption/detection validation pipeline.

This is the only file you should need to edit between runs: point it at your
video folder and captions CSV, and fill in the ground truth for each video.
"""

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("MOONDREAM_API_KEY")

VIDEO_DIR = "Videos"
CAPTIONS_CSV = "output_captions.csv"

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

ANNOTATED_FRAMES_DIR = "annotated_frames"
REPORT_CSV = "validation_report.csv"
REPORT_EXCEL = "validation_report.xlsx"

# Single combined report produced by run_pipeline.py (captioning + validation together)
COMBINED_REPORT_CSV = "video_report.csv"
COMBINED_REPORT_EXCEL = "video_report.xlsx"

FRAMES_PER_VIDEO = 3

# What is ACTUALLY shown in each video. Keys must exactly match filenames in
# VIDEO_DIR. Values are a LIST of expected labels - a video can show more than
# one relevant subject (e.g. a technician working on a generator). Each label
# must be a key in SYNONYMS and DETECTION_QUERY below (or a plain keyword).
#
# NOTE: the keys below for the newly added videos are my best guess at the
# real filenames, converted back from the underscore-separated names they
# were uploaded with. Double check these against your actual Videos folder -
# if a key doesn't match exactly, that video will silently be skipped by
# validation instead of erroring.
GROUND_TRUTH = {
    "Close-up shot of insulators in substation.mp4": ["substation"],
    "Drone footage of cell tower at sunset.mp4": ["cell tower"],
    "Electricians working at substation.mp4": ["technician", "substation"],
    "Equipment maintenance by Technician.mp4": ["technician", "machine"],
    "Industrial equipment.mp4": ["robotic arm", "conveyor belt"],
    "Men working on electrical grids.mp4": ["technician", "utility tower"],
    "Technician working on battery installation at a power plant site.mp4": ["technician", "battery"],
    "Telecom antenna.mp4": ["cell tower"],
    "Tilt shot of a solar panel below a light tower.mp4": ["solar panel"],
    "Transformers at the electrical substation.mp4": ["substation"],

    # --- newly added videos ---
    "Aerial footage showcasing a modern industrial power plant with numerous structures.mp4": ["substation", "utility tower"],
    "Aerial view capturing emissions from an industrial power plant.mp4": ["power plant"],
    "Stunning aerial view of a large radio telescope under a clear blue sky.mp4": ["radio telescope"],
    "Team of workers securing an electrical line with safety gear outdoors.mp4": ["technician", "utility tower"],
    "Utility worker repairing powerlines using a bucket lift, ensuring electricity safety.mp4": ["technician", "utility tower"],
    "Worker installing solar panels on a rooftop, promoting renewable energy.mp4": ["technician", "solar panel"],
    "Workers in hardhats operating an aerial lift truck at a construction site.mp4": ["technician"],
}

# Alternate phrasings that should still count as a correct identification of
# the ground truth category, for the text-based (caption) validator.
SYNONYMS = {
    "generator": ["generator", "diesel engine", "engine", "genset", "power unit", "industrial machine"],
    "battery": ["battery", "battery bank", "battery pack", "battery rack", "electrical box", "utility cabinet"],
    "cell tower": ["cell tower", "telecom tower", "communication tower", "antenna", "mast", "cellular"],
    "substation": ["substation", "transformer", "insulator", "electrical grid", "power grid"],
    "utility tower": ["utility tower", "utility pole", "transmission tower", "power line", "electrical grid"],
    "solar panel": ["solar panel", "solar", "photovoltaic"],
    "technician": ["technician", "worker", "electrician", "person", "engineer"],

    # --- newly added categories ---
    "robotic arm": ["robotic arm", "industrial arm", "automation arm", "robot"],
    "conveyor belt": ["conveyor belt", "conveyor", "assembly line"],
    "power plant": ["power plant", "power station", "industrial plant", "cooling tower", "factory", "plant"],
    "radio telescope": ["radio telescope", "satellite dish", "dish antenna", "parabolic dish", "observatory"],
    "machine": ["machine", "equipment", "machinery", "industrial equipment"],
}

# Search phrase used against the Moondream detect() endpoint for each ground
# truth category (open-vocabulary - free text, not a fixed class list).
DETECTION_QUERY = {
    "generator": "generator",
    "battery": "battery",
    "cell tower": "cell tower",
    "substation": "transformer",
    "utility tower": "utility tower",
    "solar panel": "solar panel",
    "technician": "person",

    # --- newly added categories ---
    "robotic arm": "robotic arm",
    "conveyor belt": "conveyor belt",
    "power plant": "power plant",
    "radio telescope": "radio telescope dish",
    "machine": "machine",
}

CLASS_COLORS = {
    "generator": "#E74C3C",
    "battery": "#8E44AD",
    "cell tower": "#2980B9",
    "substation": "#16A085",
    "utility tower": "#2C3E50",
    "solar panel": "#27AE60",
    "technician": "#C0392B",

    # --- newly added categories ---
    "robotic arm": "#F39C12",
    "conveyor belt": "#D35400",
    "power plant": "#7F8C8D",
    "radio telescope": "#34495E",
    "machine": "#95A5A6",
}