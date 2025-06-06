def _is_url(url):
    from urllib.parse import urlparse

    if isinstance(url, str):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False
    return False


def _download(url_file):
    import pathlib

    if not _is_url(url_file):
        raise ValueError("Invalid url")
    import os
    import tempfile
    from urllib.request import urlopen

    ext = pathlib.Path(url_file).suffix
    path = os.path.join(tempfile.mkdtemp(), "file" + ext)
    with urlopen(url_file) as rsp, open(path, "wb") as output:
        output.write(rsp.read())
    return path
