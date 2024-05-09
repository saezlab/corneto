def dot_vizjs_html(
    dot_input,
    container_id=None,
    viz_js_url=None,
    full_render_js_url=None,
    vizjs_version=None,
):
    import base64
    import uuid

    # Generate a random container ID if none is provided
    if container_id is None:
        container_id = f"container-{uuid.uuid4()}"

    dot_string = ""

    # Determine if input is a file path or a Graphviz object
    if isinstance(dot_input, str):
        try:
            with open(dot_input, "r") as file:
                dot_string = file.read()
        except IOError as e:
            print(f"Error reading DOT file: {e}")
            return
    else:
        try:
            dot_string = dot_input.source
        except AttributeError:
            print("Provided object is not a valid Graphviz object.")
            return

    # Base64 encode the DOT content to safely embed it in HTML/JavaScript
    dot_string_base64 = base64.b64encode(dot_string.encode()).decode("utf-8")
    if vizjs_version is not None:
        vizjs_version = f"@{vizjs_version}"
    else:
        vizjs_version = ""
    # Setting default URLs if custom URLs are not provided
    if not viz_js_url:
        viz_js_url = f"https://unpkg.com/viz.js{vizjs_version}/viz.js"
    if not full_render_js_url:
        full_render_js_url = f"https://unpkg.com/viz.js{vizjs_version}/full.render.js"

    return f"""
    <div id='{container_id}'></div> <!-- Container to display the graph -->
    <script>
    function loadScript(url, callback) {{
        var script = document.createElement('script');
        script.type = 'text/javascript';
        script.src = url;
        script.onreadystatechange = callback;
        script.onload = callback;
        document.head.appendChild(script);
    }}

    function renderGraph(encodedDot) {{
        loadScript('{viz_js_url}', function() {{
            loadScript('{full_render_js_url}', function() {{
                var viz = new Viz();
                var dotString = atob(encodedDot);

                viz.renderSVGElement(dotString)
                    .then(function(element) {{
                        document.getElementById('{container_id}').appendChild(element);
                    }})
                    .catch(error => {{
                        console.error('Error rendering graph:', error);
                    }});
            }});
        }});
    }}

    // Execute the rendering function with the base64-encoded DOT string
    renderGraph("{dot_string_base64}");
    </script>
    """


def dot_vizjs(
    dot_input,
    container_id=None,
    viz_js_url=None,
    full_render_js_url=None,
    vizjs_version=None,
):
    html_code = dot_vizjs_html(
        dot_input,
        container_id=container_id,
        viz_js_url=viz_js_url,
        full_render_js_url=full_render_js_url,
        vizjs_version=vizjs_version,
    )

    try:
        from IPython.display import HTML, display

        display(HTML(html_code))

    except ImportError as e:
        raise ImportError(
            "IPython is not installed but is required for displaying the output "
            "directly in Jupyter notebooks. Please install IPython with "
            "`pip install ipython`."
        ) from e
