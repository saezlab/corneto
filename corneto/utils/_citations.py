"""Citation utilities for corneto methods.

This module provides utilities for handling and displaying citations in BibTeX format.
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from IPython.display import HTML, display


def get_bibtex_from_keys(keys: List[str]) -> str:
    """Get BibTeX entries for the specified keys.

    Args:
        keys: List of citation keys to retrieve.

    Returns:
        A string containing the BibTeX entries.
    """
    if not keys:
        return ""

    # Path to the BibTeX file
    module_dir = Path(__file__).parent.parent
    bib_path = os.path.join(module_dir, "resources", "citations.bib")

    if not os.path.exists(bib_path):
        raise FileNotFoundError(f"Citations file not found at {bib_path}")

    with open(bib_path, "r") as f:
        bibtex_content = f.read()

    # Extract entries matching the provided keys
    entries = []
    # Process each key, ensuring no duplicates
    seen_entries = set()
    for key in keys:
        # We need a more robust approach to match BibTeX entries with balanced braces
        # Find the start of the entry
        entry_pattern = re.compile(f"@\\w+{{{key},", re.DOTALL)
        start_match = entry_pattern.search(bibtex_content)

        if start_match:
            start_pos = start_match.start()
            # Find the matching closing brace by counting open/close braces
            pos = start_match.end() - 1  # Start from the position after the opening '{'
            brace_count = 1
            entry_end = -1

            while pos < len(bibtex_content) and brace_count > 0:
                if bibtex_content[pos] == "{":
                    brace_count += 1
                elif bibtex_content[pos] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        entry_end = pos + 1  # Include the closing brace
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
        bibtex_str: String containing one or more BibTeX entries

    Returns:
        List of dictionaries, each containing parsed citation information
    """
    if not bibtex_str or not bibtex_str.strip():
        return []

    # Split the string at the beginning of each entry (where @ occurs)
    entries = re.split(r"(?=@\w+\{)", bibtex_str.strip())
    parsed = []

    for entry in entries:
        if not entry.strip():
            continue

        # Extract entry type and key
        citation = {}
        entry_type_match = re.match(r"@(\w+){([^,]+),", entry)
        if not entry_type_match:
            continue

        citation["type"] = entry_type_match.group(1)
        citation["id"] = entry_type_match.group(2)

        # Extract all field-value pairs using a more robust pattern
        # This handles nested braces and both quote and brace delimited values
        fields = re.findall(
            r'(\w+)\s*=\s*(?:{((?:[^{}]|{[^{}]*})+)}|"([^"]*)")', entry, re.DOTALL
        )

        for field, brace_value, quote_value in fields:
            # Use the value from braces if present, otherwise use the quoted value
            value = brace_value if brace_value else quote_value
            citation[field.strip().lower()] = value.strip().replace("\n", " ")

        parsed.append(citation)

    return parsed


def format_authors(author_str: str) -> str:
    """Format author names for display.

    Handles multiple authors separated by 'and', and formats names
    from 'Lastname, Firstname' to 'Firstname Lastname'.

    Args:
        author_str: String containing author names separated by 'and'

    Returns:
        Formatted string with proper author display names
    """
    if not author_str:
        return "Unknown Author"

    # Split by ' and ' to handle multiple authors
    authors = [a.strip() for a in author_str.split(" and ")]
    formatted = []

    for name in authors:
        # Handle 'Lastname, Firstname' format
        parts = name.split(",")
        if len(parts) == 2:
            # Convert 'Lastname, Firstname' to 'Firstname Lastname'
            formatted.append(f"{parts[1].strip()} {parts[0].strip()}")
        else:
            # Keep as is for other formats
            formatted.append(name.strip())

    # Join all authors with commas
    return ", ".join(formatted)


def render_citations_html(citations: List[Dict[str, str]]) -> str:
    """Render citations as HTML.

    Args:
        citations: List of citation dictionaries

    Returns:
        HTML string representing the citations
    """
    html_entries = []
    for c in citations:
        author = format_authors(c.get("author", "Unknown Author"))
        title = c.get("title", "Untitled")
        year = c.get("year", "n.d.")
        extra = []

        if "journal" in c:
            extra.append(f"<em>{c['journal']}</em>")
        elif "booktitle" in c:
            extra.append(f"<em>{c['booktitle']}</em>")
        elif "publisher" in c:
            extra.append(f"<em>{c['publisher']}</em>")

        if "volume" in c:
            extra.append(f"Vol. {c['volume']}")
        if "number" in c:
            extra.append(f"No. {c['number']}")
        if "pages" in c:
            extra.append(f"pp. {c['pages']}")

        citation_str = f"<strong>{author}</strong>. <em>{title}</em>. {', '.join(extra)} ({year})."
        html_entries.append(f"<li>{citation_str}</li>")

    return f"<ul>{''.join(html_entries)}</ul>"


def show_citations(keys: List[str]) -> None:
    """Display formatted citations in a Jupyter notebook.
    
    Args:
        keys: List of citation keys to display
    """
    bibtex = get_bibtex_from_keys(keys)
    if not bibtex:
        display(HTML("<p>No citations available.</p>"))
        return

    citations = parse_bibtex(bibtex)
    html = render_citations_html(citations)
    display(HTML(html))


def show_bibtex(keys: List[str]) -> None:
    """Display raw BibTeX entries in a formatted block for easy copying.
    
    Args:
        keys: List of citation keys to display
    """
    bibtex = get_bibtex_from_keys(keys)
    if not bibtex:
        display(HTML("<p>No BibTeX entries available.</p>"))
        return

    # Display raw BibTeX in a formatted code block for easy copy-paste
    escaped_bibtex = bibtex.strip().replace("<", "&lt;").replace(">", "&gt;")
    formatted_bibtex = f"""
    <div style='
        background: #f5f5f5;
        padding: 1em;
        border-radius: 5px;
        overflow-x: auto;
        font-family: monospace;
        font-size: 0.9em;
        line-height: 1.5em;
        white-space: pre-wrap;
    '>
        <code style='
            display: block;
            white-space: pre-wrap;
            text-indent: -2em;
            padding-left: 2em;
        '>{escaped_bibtex}</code>
    </div>
    """
    display(HTML(formatted_bibtex))
