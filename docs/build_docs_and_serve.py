import os
import subprocess
from http.server import HTTPServer, SimpleHTTPRequestHandler

def build_docs():
    """Run Sphinx to build the documentation."""
    sourcedir = "."
    outdir = "_build/html"

    # Ensure the output directory exists
    os.makedirs(outdir, exist_ok=True)

    # Run sphinx-build command
    result = subprocess.run(
        ["sphinx-build", sourcedir, outdir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode == 0:
        print("Sphinx build completed successfully.")
        print(result.stdout)
    else:
        print("Sphinx build failed.")
        print(result.stderr)
        exit(1)

def serve_docs():
    """Serve the built documentation locally."""
    outdir = "_build/html"
    os.chdir(outdir)
    server_address = ("", 8000)  # Host on port 8000
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)

    print("Serving documentation at http://localhost:8000")
    print("Press Ctrl+C to stop the server.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        httpd.server_close()

if __name__ == "__main__":
    build_docs()
    serve_docs()

