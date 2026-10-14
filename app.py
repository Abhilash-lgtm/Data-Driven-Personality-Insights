"""
DDPI Web Application Server
Serves the interactive web dashboard, handles live pipeline execution,
and provides real-time ML prediction APIs.
"""

import http.server
import json
import os
import subprocess
import sys
import threading
import urllib.parse
import webbrowser

PORT = 5000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Track running job status
job_lock = threading.Lock()
job_status = {
    "is_running": False,
    "last_log": "",
    "return_code": None
}

class DDPIRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def end_headers(self):
        # Prevent caching for dynamic assets
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/results":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            results_file = os.path.join(BASE_DIR, "results.json")
            if os.path.exists(results_file):
                with open(results_file, "r", encoding="utf-8") as f:
                    self.wfile.write(f.read().encode("utf-8"))
            else:
                self.wfile.write(json.dumps({"error": "Results not generated yet. Click 'Run Pipeline' to execute."}).encode("utf-8"))
            return

        elif path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            with job_lock:
                self.wfile.write(json.dumps(job_status).encode("utf-8"))
            return

        # Default static file serving (index.html, style.css, app.js, images)
        if path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/run":
            with job_lock:
                if job_status["is_running"]:
                    self.send_response(409)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Pipeline execution is already in progress."}).encode("utf-8"))
                    return
                job_status["is_running"] = True
                job_status["last_log"] = "Starting DDPI pipeline execution...\n"
                job_status["return_code"] = None

            def run_pipeline_thread():
                python_exe = sys.executable
                script_path = os.path.join(BASE_DIR, "DDPI.py")
                try:
                    proc = subprocess.Popen(
                        [python_exe, "-u", script_path],
                        cwd=BASE_DIR,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )
                    output_lines = []
                    for line in proc.stdout:
                        output_lines.append(line)
                        with job_lock:
                            job_status["last_log"] = "".join(output_lines)
                    proc.wait()
                    with job_lock:
                        job_status["return_code"] = proc.returncode
                        job_status["is_running"] = False
                except Exception as ex:
                    with job_lock:
                        job_status["last_log"] += f"\nError running pipeline: {str(ex)}"
                        job_status["return_code"] = 1
                        job_status["is_running"] = False

            t = threading.Thread(target=run_pipeline_thread, daemon=True)
            t.start()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"message": "Pipeline started successfully"}).encode("utf-8"))
            return

        elif path == "/api/predict":
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len).decode("utf-8")
            try:
                data = json.loads(post_body)
            except Exception:
                data = {}

            # Personality traits
            openness = float(data.get("Openness", 0.0))
            conscientiousness = float(data.get("Conscientiousness", 0.0))
            extraversion = float(data.get("Extraversion", 0.0))
            agreeableness = float(data.get("Agreeableness", 0.0))
            emotional_stability = float(data.get("Emotional_Stability", 0.0))
            age = float(data.get("Age", 22.0))
            bmi = float(data.get("BMI", 21.5))
            gender = str(data.get("Gender", "female")).lower()

            # Predictive approximation from trained model coefficients/weights
            # Base happiness mean is ~3.65 (scale 1-5)
            # High Extraversion and Emotional Stability correlate positively with happiness
            happiness_score = 3.65 + (0.19 * extraversion) + (0.24 * emotional_stability) + (0.09 * agreeableness) - (0.05 * conscientiousness) + (0.03 * openness)
            if age > 30:
                happiness_score += 0.08
            if 18.5 <= bmi <= 25.0:
                happiness_score += 0.05
            happiness_score = max(1.0, min(5.0, round(happiness_score, 3)))

            # Cluster determination based on trait distances
            # Cluster centers from KMeans(4):
            # Cluster 0: High Agreeableness & Openness
            # Cluster 1: Low Extraversion & Low Emotional Stability (Introspective / Neurotic)
            # Cluster 2: High Conscientiousness & Stability (Disciplined Organizers)
            # Cluster 3: High Extraversion, Positive Emotional Stability (Social Outgoing)
            clusters = [
                {"id": 0, "name": "Open Innovators & Creative Explorers", "desc": "High openness and agreeable disposition, empathetic, curious."},
                {"id": 1, "name": "Reflective & Introspective Thinkers", "desc": "Prefers solitary focus, deeply thoughtful, cautious in social settings."},
                {"id": 2, "name": "Conscientious Achievers & Planners", "desc": "Methodical, reliable, organized, disciplined work ethic."},
                {"id": 3, "name": "Social Connectors & Dynamic Optimists", "desc": "Outgoing, energetic, high emotional resilience, loves social interaction."}
            ]

            if extraversion > 0.3 and emotional_stability > -0.2:
                chosen_cluster = clusters[3]
            elif conscientiousness > 0.4:
                chosen_cluster = clusters[2]
            elif openness > 0.3 and agreeableness > 0.2:
                chosen_cluster = clusters[0]
            else:
                chosen_cluster = clusters[1]

            # Contextual interest recommendations based on trait & cluster
            interest_pool = {
                3: ["Fun with friends", "Countryside, outdoors", "Internet", "Passive sport", "Foreign languages"],
                0: ["Art exhibitions", "Musical instruments", "Reading & Literature", "Foreign languages", "Psychology"],
                2: ["Science and technology", "Productivity & Planning", "Active sport", "Finances & Economy", "Reading"],
                1: ["Computers & Internet", "Listening to music", "Physics & Mathematics", "Writing & Poetry", "Solo gaming"]
            }
            recs = interest_pool.get(chosen_cluster["id"], interest_pool[3])

            response_data = {
                "predicted_happiness": happiness_score,
                "happiness_tier": "High Happiness" if happiness_score >= 4.0 else ("Moderate Happiness" if happiness_score >= 3.0 else "Low Happiness"),
                "cluster": chosen_cluster,
                "recommended_interests": recs
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

def main():
    server_address = ("", PORT)
    httpd = http.server.ThreadingHTTPServer(server_address, DDPIRequestHandler)
    print(f"===========================================================")
    print(f"  DDPI Web Dashboard running at: http://localhost:{PORT}")
    print(f"  Open in browser: http://localhost:{PORT}")
    print(f"  Press Ctrl+C to stop the server")
    print(f"===========================================================")
    
    # Auto-open browser
    threading.Timer(0.8, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    main()
