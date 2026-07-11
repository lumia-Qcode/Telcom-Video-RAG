"""
Central configuration for the caption/detection validation pipeline.

This is the only file you should need to edit between runs: point it at your
video folder and captions CSV, and fill in the ground truth for each video.
"""

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("MOONDREAM_API_KEY")

VIDEO_DIR = "../Videos"
CAPTIONS_CSV = "../output_captions.csv"

ANNOTATED_FRAMES_DIR = "annotated_frames"
REPORT_CSV = "validation_report.csv"
REPORT_EXCEL = "validation_report.xlsx"

FRAMES_PER_VIDEO = 3

# What is ACTUALLY shown in each video. Keys must exactly match filenames in
# VIDEO_DIR. Values must be keys in SYNONYMS below (or any plain keyword).
GROUND_TRUTH = {
    "Close-up shot of insulators in substation.mp4": "substation",
    "Drone footage of cell tower at sunset.mp4": "cell tower",
    "Electricians working at substation.mp4": "technician",
    "Equipment maintenance by Technician.mp4": "technician",
    "Industrial equipment.mp4": "robotic arm",
    "Men working on electrical grids.mp4": "utility tower",
    "Technician working on battery installation at a power plant site.mp4": "battery",
    "Telecom antenna.mp4": "cell tower",
    "Tilt shot of a solar panel below a light tower.mp4": "solar panel",
    "Transformers at the electrical substation.mp4": "substation",
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
    "robotic arm": ["robotic arm", "industrial robot", "automated arm", "mechanical arm"],
    "technician": ["technician", "worker", "electrician", "person", "engineer"],
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
}

CLASS_COLORS = {
    "generator": "#E74C3C",
    "battery": "#8E44AD",
    "cell tower": "#2980B9",
    "substation": "#16A085",
    "utility tower": "#2C3E50",
    "solar panel": "#27AE60",
    "technician": "#C0392B",
}
