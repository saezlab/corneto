import argparse
import os
import shutil
import subprocess
from http.server import HTTPServer, SimpleHTTPRequestHandler


def build_docs(force=False, stop_on_warning=False, clean=False):
    """Run Sphinx to build the documentation, with options to force execution, treat warnings as errors,
    and clean the build directory before building.

    When stop_on_warning is enabled, the -W flag is added to treat warnings as errors.
    Note: With Sphinx 8.1 and above, --keep-going is always enabled; this means that the build
    runs to completion and exits with status 1 if warnings are encountered.
    """
    sourcedir = "."
    outdir = "_build/html"

    # Remove existing build directory if clean is True.
    if clean and os.path.exists(outdir):
        print(f"Cleaning the build directory: {outdir}")
        shutil.rmtree(outdir)

    # Ensure the output directory exists
    os.makedirs(outdir, exist_ok=True)

    # Prepare sphinx-build command
    sphinx_cmd = ["sphinx-build", sourcedir, outdir]

    if force:
        sphinx_cmd.extend(["-D", "nb_execution_mode=force"])  # Override execution mode

    if stop_on_warning:
        # The -W flag tells Sphinx to treat warnings as errors.
        sphinx_cmd.append("-W")

    # Run sphinx-build command
    result = subprocess.run(
        sphinx_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if result.returncode == 0:
        print("Sphinx build completed successfully.")
        print(result.stdout)
    else:
        print("Sphinx build failed.")
        print(result.stderr)
        if stop_on_warning:
            print("Stopping execution due to warnings treated as errors in the build.")
            exit(1)


def serve_docs():
    """Serve the built documentation locally."""
    outdir = "_build/html"
    if not os.path.exists(outdir):
        print("Error: Documentation has not been built. Please run the build first.")
        exit(1)

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
    parser = argparse.ArgumentParser(
        description="Build and serve Sphinx documentation."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution of Jupyter notebooks during build",
    )
    parser.add_argument(
        "--stop-on-warning",
        action="store_true",
        help="Treat warnings as errors: exit if any warnings occur during the build",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing content in _build/html before building",
    )
    args = parser.parse_args()

    build_docs(force=args.force, stop_on_warning=args.stop_on_warning, clean=args.clean)

    # Optionally, check that the documentation was built by verifying the existence of the index file.
    index_file = os.path.join("_build", "html", "index.html")
    if args.stop_on_warning and not os.path.exists(index_file):
        print("Error: Build failed due to warnings treated as errors. Exiting.")
        exit(1)

    serve_docs()
