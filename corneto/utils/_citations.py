"""Citation utilities for corneto methods.

This module provides utilities for handling and displaying citations in BibTeX format.
"""

import os
import re
from pathlib import Path
from typing import Dict, List

from IPython.display import HTML, display


def unescape_latex(text: str) -> str:
    """Convert common LaTeX escaped symbols and accented characters into their Unicode equivalents.

    Recognized patterns include commands with optional accent parts, for example:

      - {\aa}   -> å
      - {\"a}   -> ä
      - {\'e}   -> é
      - {\\`o}  -> ò
      - {\\^u}  -> û
      - {\\~n}  -> ñ

    Args:
        text: A string possibly containing LaTeX escapes.

    Returns:
        A string with known LaTeX escapes replaced by their Unicode characters.
    """
    if not text:
        return text

    # Mapping for commands without an accent
    basic_mapping = {
        ("", "aa"): "å",
        ("", "AA"): "Å",
        ("", "o"): "ø",
        ("", "O"): "Ø",
        ("", "ss"): "ß",
    }

    # Mapping for accented letters.
    accent_mapping = {
        ('"', "a"): "ä",
        ('"', "A"): "Ä",
        ('"', "o"): "ö",
        ('"', "O"): "Ö",
        ('"', "u"): "ü",
        ('"', "U"): "Ü",
        ("'", "a"): "á",
        ("'", "A"): "Á",
        ("'", "e"): "é",
        ("'", "E"): "É",
        ("'", "i"): "í",
        ("'", "I"): "Í",
        ("'", "o"): "ó",
        ("'", "O"): "Ó",
        ("'", "u"): "ú",
        ("'", "U"): "Ú",
        ("`", "a"): "à",
        ("`", "A"): "À",
        ("`", "e"): "è",
        ("`", "E"): "È",
        ("`", "i"): "ì",
        ("`", "I"): "Ì",
        ("`", "o"): "ò",
        ("`", "O"): "Ò",
        ("`", "u"): "ù",
        ("`", "U"): "Ù",
        ("^", "a"): "â",
        ("^", "A"): "Â",
        ("^", "e"): "ê",
        ("^", "E"): "Ê",
        ("^", "i"): "î",
        ("^", "I"): "Î",
        ("^", "o"): "ô",
        ("^", "O"): "Ô",
        ("^", "u"): "û",
        ("^", "U"): "Û",
        ("~", "a"): "ã",
        ("~", "A"): "Ã",
        ("~", "n"): "ñ",
        ("~", "N"): "Ñ",
        ("~", "o"): "õ",
        ("~", "O"): "Õ",
        # Extend with further mappings as required.
    }

    # Combined mapping used for replacement.
    mapping = {}
    mapping.update(basic_mapping)
    mapping.update(accent_mapping)

    # Regex to match LaTeX escapes of the form {\\<accent?><command>}.
    # The first group captures an optional accent symbol,
    # and the second group captures one or more letters/digits.
    pattern = re.compile(r"{\\([\"'`\^~]?)([a-zA-Z0-9]+)}")

    def replace_match(match: re.Match) -> str:
        accent = match.group(1)  # Could be empty, or one of: ", ', `, ^, ~
        command = match.group(2)
        # Look up in the mapping based on the tuple key.
        return mapping.get((accent, command), match.group(0))  # default: leave unchanged

    # Replace all occurrences in the text.
    return pattern.sub(replace_match, text)


def get_bibtex_from_keys(keys: List[str]) -> str:
    """Get BibTeX entries for the specified keys.

    Args:
        keys: List of citation keys to retrieve.

    Returns:
        A string containing the BibTeX entries.
    """
    if not keys:
        return ""

    # Path to the BibTeX file.
    module_dir = Path(__file__).parent.parent
    bib_path = os.path.join(module_dir, "resources", "citations.bib")

    if not os.path.exists(bib_path):
        raise FileNotFoundError(f"Citations file not found at {bib_path}")

    with open(bib_path, "r", encoding="utf-8") as f:
        bibtex_content = f.read()

    # Extract entries matching the provided keys.
    entries = []
    seen_entries = set()
    for key in keys:
        # A robust approach to match BibTeX entries with balanced braces.
        entry_pattern = re.compile(f"@\\w+{{{key},", re.DOTALL)
        start_match = entry_pattern.search(bibtex_content)

        if start_match:
            start_pos = start_match.start()
            pos = start_match.end() - 1  # Start from character after the opening '{'
            brace_count = 1
            entry_end = -1

            while pos < len(bibtex_content) and brace_count > 0:
                if bibtex_content[pos] == "{":
                    brace_count += 1
                elif bibtex_content[pos] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        entry_end = pos + 1  # Include the closing brace.
                        break
                pos += 1

            if entry_end > 0:
                entry = bibtex_content[start_pos:entry_end]
                if entry not in seen_entries:
                    entries.append(entry)
                    seen_entries.add(entry)

    return "\n\n".join(entries)


def parse_bibtex(bibtex_str: str) -> List[Dict[str, str]]:
    """Parse BibTeX entries into structured dictionaries.

    Args:
        bibtex_str: String containing one or more BibTeX entries.

    Returns:
        List of dictionaries, each containing parsed citation information.
    """
    if not bibtex_str or not bibtex_str.strip():
        return []

    # Split the string at the start of each BibTeX entry.
    entries = re.split(r"(?=@\w+\{)", bibtex_str.strip())
    parsed = []

    for entry in entries:
        if not entry.strip():
            continue

        citation = {}
        entry_type_match = re.match(r"@(\w+){([^,]+),", entry)
        if not entry_type_match:
            continue

        citation["type"] = entry_type_match.group(1)
        citation["id"] = entry_type_match.group(2)

        # Extract all field-value pairs.
        fields = re.findall(r'(\w+)\s*=\s*(?:{((?:[^{}]|{[^{}]*})+)}|"([^"]*)")', entry, re.DOTALL)

        for field, brace_value, quote_value in fields:
            # Select value from braces (if present) or from quotes.
            value = (brace_value if brace_value else quote_value).strip().replace("\n", " ")
            # Unescape any LaTeX symbols before storing.
            citation[field.strip().lower()] = unescape_latex(value)

        parsed.append(citation)

    return parsed


def format_authors(author_str: str) -> str:
    """Format author names for display.

    Converts names in "Lastname, Firstname" into "Firstname Lastname"
    and unescapes any LaTeX symbols.

    Args:
        author_str: A string with author names separated by 'and'.

    Returns:
        A formatted string of author names.
    """
    if not author_str:
        return "Unknown Author"

    authors = [a.strip() for a in author_str.split(" and ")]
    formatted = []

    for name in authors:
        parts = name.split(",")
        if len(parts) == 2:
            # Reformat name and unescape LaTeX symbols.
            formatted.append(unescape_latex(f"{parts[1].strip()} {parts[0].strip()}"))
        else:
            formatted.append(unescape_latex(name.strip()))

    return ", ".join(formatted)


def render_references_html(citations: List[Dict[str, str]]) -> str:
    """Render citations as HTML.

    Args:
        citations: List of citation dictionaries.

    Returns:
        HTML string representing the citations.
    """
    html_entries = []
    for c in citations:
        author = format_authors(c.get("author", "Unknown Author"))
        title = unescape_latex(c.get("title", "Untitled"))
        year = c.get("year", "n.d.")
        extra = []

        if "journal" in c:
            extra.append(f"<em>{unescape_latex(c['journal'])}</em>")
        elif "booktitle" in c:
            extra.append(f"<em>{unescape_latex(c['booktitle'])}</em>")
        elif "publisher" in c:
            extra.append(f"<em>{unescape_latex(c['publisher'])}</em>")

        if "volume" in c:
            extra.append(f"Vol. {c['volume']}")
        if "number" in c:
            extra.append(f"No. {c['number']}")
        if "pages" in c:
            extra.append(f"pp. {c['pages']}")

        citation_str = f"<strong>{author}</strong>. <em>{title}</em>. {', '.join(extra)} ({year})."
        html_entries.append(f"<li>{citation_str}</li>")

    return f"<ul>{''.join(html_entries)}</ul>"


def format_references_plaintext(keys: List[str]) -> str:
    """Format citations as plain text for __repr__ methods and other non-HTML contexts.

    Args:
        keys: List of citation keys to format.

    Returns:
        A formatted string with one citation per line, or a message if no citations are found.
    """
    if not keys:
        return ""

    result = ""
    bibtex = get_bibtex_from_keys(keys)

    if bibtex:
        citations = parse_bibtex(bibtex)
        for c in citations:
            author = format_authors(c.get("author", "Unknown Author"))
            title = unescape_latex(c.get("title", "Untitled"))
            year = c.get("year", "n.d.")
            extra = []

            if "journal" in c:
                extra.append(f"{unescape_latex(c['journal'])}")
            elif "booktitle" in c:
                extra.append(f"{unescape_latex(c['booktitle'])}")
            elif "publisher" in c:
                extra.append(f"{unescape_latex(c['publisher'])}")

            if "volume" in c:
                extra.append(f"Vol. {c['volume']}")
            if "number" in c:
                extra.append(f"No. {c['number']}")
            if "pages" in c:
                extra.append(f"pp. {c['pages']}")

            citation_str = f"\n - {author}. {title}. {', '.join(extra)} ({year})."
            result += citation_str
    else:
        # Handle the case where citations are not found in the BibTeX database
        for key in keys:
            result += f"\n - {key} (Citation not found in BibTeX database)"

    return result


def show_references(keys: List[str]) -> None:
    """Display formatted citations in a Jupyter notebook.

    Args:
        keys: List of citation keys to display.
    """
    bibtex = get_bibtex_from_keys(keys)
    if not bibtex:
        display(HTML("<p>No citations available.</p>"))
        return

    citations = parse_bibtex(bibtex)
    html = render_references_html(citations)
    display(HTML(html))


def show_bibtex(keys: List[str]) -> None:
    """Display raw BibTeX entries in a nicely formatted and styled HTML block."""
    bibtex = get_bibtex_from_keys(keys)
    if not bibtex:
        display(HTML("<p>No BibTeX entries available.</p>"))
