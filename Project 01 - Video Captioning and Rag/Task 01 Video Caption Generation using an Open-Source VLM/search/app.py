"""
Flask app: serves the search page and the semantic search API.

Run:
    python app.py

Then open http://127.0.0.1:5000 in a browser.

Make sure you've run index_builder.py at least once before starting this,
otherwise the vector database will be empty.
"""

from flask import Flask, jsonify, render_template, request, send_from_directory

import config
from search_service import build_default_search_service

app = Flask(__name__)
search_service = build_default_search_service()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"results": []})

    results = search_service.search(query)
    return jsonify({
        "results": [
            {"video": r.video, "caption": r.summary_caption, "score": r.score}
            for r in results
        ]
    })


@app.route("/videos/<path:filename>")
def serve_video(filename):
    return send_from_directory(config.VIDEO_DIR, filename)


if __name__ == "__main__":
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=True)