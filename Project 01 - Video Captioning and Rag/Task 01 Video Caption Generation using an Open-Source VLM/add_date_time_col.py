"""
Adds 'Date' and 'Time' columns to video_report.csv.

Your videos don't have real capture timestamps (they're stock clips), so
this generates plausible synthetic values instead: spread across the last
30 days, during typical field-work hours (8 AM - 6 PM). This is enough to
let the RAG pipeline support time-range queries like "last week" or
"between 4 and 5 PM" - just be aware these dates/times are placeholders,
not real inspection timestamps. If you get real capture dates later
(e.g. from video file metadata), re-run with USE_REAL_METADATA = True
below instead.

Run from the project root (same folder as video_report.csv):
    python add_datetime_columns.py
"""

import csv
import random
from datetime import datetime, timedelta

CSV_PATH = "video_report.csv"
OUTPUT_PATH = "video_report.csv"  # overwrites in place; back up first if unsure

# Set a fixed seed so re-running this script (e.g. after adding more videos)
# doesn't reshuffle the dates already assigned to existing rows.
random.seed(42)

TODAY = datetime(2026, 7, 14)
DAYS_BACK = 30
WORK_START_HOUR = 8
WORK_END_HOUR = 18


def random_datetime():
    days_offset = random.randint(0, DAYS_BACK)
    dt_date = TODAY - timedelta(days=days_offset)
    hour = random.randint(WORK_START_HOUR, WORK_END_HOUR - 1)
    minute = random.choice([0, 15, 30, 45])
    return dt_date.replace(hour=hour, minute=minute)


def add_datetime_columns():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print(f"'{CSV_PATH}' is empty.")
        return

    header = rows[0]
    if "Date" in header or "Time" in header:
        print("Date/Time columns already exist - remove them first if you want to regenerate.")
        return

    header.extend(["Date", "Time"])
    new_rows = [header]

    updated_count = 0
    in_video_rows = True
    for row in rows[1:]:
        if not row or not row[0].strip():
            # Blank row marks the start of the appended metrics summary
            # section - stop adding Date/Time to everything from here on.
            in_video_rows = False
            new_rows.append(row)
            continue

        if not in_video_rows:
            new_rows.append(row)
            continue

        dt = random_datetime()
        row.extend([dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")])
        new_rows.append(row)
        updated_count += 1

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

    print(f"Added Date/Time columns to {updated_count} video rows in '{OUTPUT_PATH}'.")
    print("Note: these are synthetic placeholder timestamps, not real capture times.")


if __name__ == "__main__":
    add_datetime_columns()