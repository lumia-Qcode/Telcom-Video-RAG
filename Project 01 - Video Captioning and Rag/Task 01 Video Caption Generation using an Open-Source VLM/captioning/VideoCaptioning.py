import os
import csv
import cv2
from PIL import Image
import moondream as md
from dotenv import load_dotenv

# ---------------- CONFIG ----------------
load_dotenv()
API_KEY = os.getenv("MOONDREAM_API_KEY")
VIDEO_DIR = "Videos"
OUTPUT_FILE = "output_captions.csv"
FRAMES_PER_VIDEO = 3

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def load_model():
    print("Connecting to Moondream Cloud...")
    model = md.vl(api_key=API_KEY)
    print("Connected.\n")
    return model


def extract_frames(video_path, num_frames=3):
    """Extract evenly spaced frames from a video as PIL Images."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = [int(total * (i + 1) / (num_frames + 1)) for i in range(num_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def merge_captions(captions_list):
    """Combine per-frame captions into one caption, removing exact duplicates."""
    seen = set()
    merged = []
    for cap in captions_list:
        if cap not in seen:
            merged.append(cap)
            seen.add(cap)
    return " ".join(merged)


def pick_best_caption(captions_list):
    """Frames from the same clip often describe the same scene in slightly different
    words. Rather than concatenate near-duplicate phrasing (repetitive/unprofessional),
    pick the single most descriptive answer - using length as a proxy for detail -
    as the representative short caption for the whole video."""
    if not captions_list:
        return ""
    return max(captions_list, key=len).strip()


def generate_infrastructure_captions(model, frames):
    """
    Note: Moondream's query()/caption() API accepts ONE image per call, not a list -
    so we query each frame individually, then combine the per-frame answers.
    """
    detailed_prompt = (
        "Describe this frame in detail. Identify any people present, describing their "
        "clothing, physical appearance, what they are doing, and any specific tools or "
        "equipment they are handling. Explicitly look for and describe any infrastructure, "
        "including generators, power plants, utility towers, cell towers, battery banks, "
        "or electrical components. Describe the surrounding environment, weather conditions, "
        "and geography accurately."
    )
    short_prompt = (
        "Summarize the main activity, the primary subjects, and key infrastructure elements "
        "(such as towers, power equipment, or machinery) seen in this frame in one descriptive sentence."
    )
    frame_diagnostic_prompt = (
        "List the main objects, infrastructure assets, and human actions visible in this "
        "specific frame as a clear, straightforward sentence."
    )

    detailed_answers = []
    short_answers = []
    frame_captions = []

    for frame in frames:
        detailed_res = model.query(frame, detailed_prompt)
        detailed_answers.append(detailed_res["answer"].strip())

        short_res = model.query(frame, short_prompt)
        short_answers.append(short_res["answer"].strip())

        f_res = model.query(frame, frame_diagnostic_prompt)
        frame_captions.append(f_res["answer"].strip())

    final_caption = merge_captions(detailed_answers)
    output_caption = pick_best_caption(short_answers)

    return final_caption, output_caption, frame_captions


def main():
    if API_KEY is None:
        print("Please set your Moondream API key in the MOONDREAM_API_KEY environment variable.")
        return

    if not os.path.isdir(VIDEO_DIR):
        print(f"Folder '{VIDEO_DIR}/' not found. Create it and add your video clips.")
        return

    video_files = sorted(
        f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(VIDEO_EXTENSIONS)
    )
    if not video_files:
        print(f"No videos found in '{VIDEO_DIR}/'.")
        return

    model = load_model()
    results = []

    for vfile in video_files:
        path = os.path.join(VIDEO_DIR, vfile)
        print(f"Processing: {vfile}")

        frames = extract_frames(path, FRAMES_PER_VIDEO)
        if not frames:
            print(f"  Could not read frames from {vfile}, skipping.\n")
            continue

        final_caption, output_caption, frame_captions = generate_infrastructure_captions(model, frames)

        for i, f_cap in enumerate(frame_captions):
            print(f"  Frame {i + 1}/{len(frames)}: {f_cap}")

        results.append({
            "video": vfile,
            "frame_captions": frame_captions,
            "final_caption": final_caption,
            "output_caption": output_caption,
        })
        print(f"  -> Detailed Analysis: {final_caption}")
        print(f"  -> Summary Caption: {output_caption}\n")

    # Save to CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Video", "Detailed Analysis", "Summary_Caption"] + [
            f"Frame {i+1} Assets" for i in range(FRAMES_PER_VIDEO)
        ]
        writer.writerow(header)
        for r in results:
            row = [r["video"], r["final_caption"], r["output_caption"]] + r["frame_captions"]
            writer.writerow(row)

    print(f"Done. Captions for {len(results)} video(s) saved to '{OUTPUT_FILE}'.")

    print_terminal_table(results)

    pdf_path = "output_captions.pdf"
    write_captions_pdf(results, pdf_path)
    print(f"Readable PDF report saved to '{pdf_path}'.")


def print_terminal_table(results):
    """Print a simple formatted table of Video | Summary Caption to the terminal."""
    if not results:
        return

    col1_width = max(len(r["video"]) for r in results) + 2
    col1_width = max(col1_width, len("Video") + 2)
    col2_width = 80

    print("\n" + "=" * (col1_width + col2_width + 3))
    print(f"{'Video'.ljust(col1_width)} | {'Summary Caption'}")
    print("-" * (col1_width + col2_width + 3))

    for r in results:
        text = r["output_caption"]
        wrapped_lines = [text[i:i + col2_width] for i in range(0, len(text), col2_width)] or [""]
        print(f"{r['video'].ljust(col1_width)} | {wrapped_lines[0]}")
        for line in wrapped_lines[1:]:
            print(f"{''.ljust(col1_width)} | {line}")
        print("-" * (col1_width + col2_width + 3))


def write_captions_pdf(results, pdf_path):
    """Build a human-readable PDF report from the same data written to the CSV."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

    if not results:
        return

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    video_style = ParagraphStyle(
        "VideoHeading", parent=styles["Heading2"],
        spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#1a1a1a"),
    )
    label_style = ParagraphStyle(
        "Label", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=10, spaceBefore=6, spaceAfter=2,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=14, spaceAfter=4,
    )
    short_style = ParagraphStyle(
        "Short", parent=styles["Normal"],
        fontSize=10, leading=14, spaceAfter=4,
        textColor=colors.HexColor("#0b5d1e"),
    )
    frame_style = ParagraphStyle(
        "Frame", parent=styles["Normal"],
        fontSize=9, leading=13, leftIndent=12, spaceAfter=2,
        textColor=colors.HexColor("#333333"),
    )

    doc = SimpleDocTemplate(
        pdf_path, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )

    story = []
    story.append(Paragraph("Infrastructure Video Analysis Report", title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Total videos processed: {len(results)}", body_style))
    story.append(Spacer(1, 12))

    for idx, r in enumerate(results):
        story.append(Paragraph(r["video"], video_style))

        story.append(Paragraph("Summary Caption:", label_style))
        story.append(Paragraph(r["output_caption"], short_style))

        story.append(Paragraph("Detailed Analysis & Environment Breakdown:", label_style))
        story.append(Paragraph(r["final_caption"], body_style))

        if r["frame_captions"]:
            story.append(Paragraph("Individual Frame Asset Logs:", label_style))
            for i, cap in enumerate(r["frame_captions"]):
                story.append(Paragraph(f"Frame {i + 1}: {cap}", frame_style))

        if idx < len(results) - 1:
            story.append(Spacer(1, 10))

    doc.build(story)


if __name__ == "__main__":
    main()