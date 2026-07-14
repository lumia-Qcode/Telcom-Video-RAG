"""
Flask app: serves the minimal frontend (static/index.html), a JSON API for
querying the video-report RAG pipeline, and the raw video files so they can
be played back inline next to each answer.

Run:
    python app.py
Then open http://localhost:5000
"""

import os
from urllib.parse import quote

from flask import Flask, abort, jsonify, request, send_from_directory

import config
from rag_engine import answer_query

app = Flask(__name__, static_folder="static", static_url_path="")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/videos/<path:filename>")
def serve_video(filename):
    """Streams a video file (supports HTTP range requests, so seeking/scrubbing
    in the <video> player works). Only serves files that live directly inside
    config.VIDEOS_DIR - no path traversal outside it."""
    full_path = os.path.join(config.VIDEOS_DIR, filename)
    if not os.path.isfile(full_path):
        abort(404, description=f"'{filename}' not found in {config.VIDEOS_DIR}")
    return send_from_directory(config.VIDEOS_DIR, filename, conditional=True)


@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json(silent=True) or {}
    q = (data.get("query") or "").strip()
    if not q:
        return jsonify({"error": "Missing 'query' field."}), 400

    try:
        result = answer_query(q)
    except Exception as e:  # noqa: BLE001 - surface a clean error to the frontend
        return jsonify({"error": str(e)}), 500

    sources = []
    for d in result.sources:
        video_exists = os.path.isfile(os.path.join(config.VIDEOS_DIR, d.video))
        sources.append(
            {
                "video": d.video,
                "date": d.metadata.get("date"),
                "time": d.metadata.get("time"),
                "labels": d.metadata.get("expected_labels"),
                "classification": d.metadata.get("classification"),
                "summary_caption": d.metadata.get("summary_caption"),
                "video_url": f"/videos/{quote(d.video)}" if video_exists else None,
            }
        )

    return jsonify({"answer": result.answer, "sources": sources})


if __name__ == "__main__":
    if not os.path.isdir(config.VIDEOS_DIR):
        print(f"NOTE: '{config.VIDEOS_DIR}' doesn't exist yet - video playback will be "
              f"unavailable until you add your video files there (or set VIDEOS_DIR in .env).")
    app.run(debug=True, port=5000)
