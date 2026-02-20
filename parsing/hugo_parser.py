import re
from pathlib import Path

import frontmatter
from langchain_core.documents import Document


def clean_hugo_shortcodes(text: str) -> str:
    """
    Translates Fuzzball Hugo shortcodes into standard Markdown
    so the LLM and Embedding models can understand them natively.
    """
    # 1. Remove the opening wrappers we don't care about
    text = re.sub(r'\{\{<\s*tabs\s+"[^"]*"\s*>\}\}', "", text)
    text = re.sub(r'\{\{<\s*tab\s+"select"\s*>\}\}', "", text)

    # 2. Convert useful tabs to standard markdown headers (e.g. ### macOS)
    text = re.sub(r'\{\{<\s*tab\s+"([^"]+)"\s*>\}\}', r"### \1", text)

    # 3. Convert hints to standard markdown blockquotes (e.g. > **note:**)
    text = re.sub(r"\{\{<\s*hint\s+type=(\w+)\s*>\}\}", r"> **\1:** ", text)

    # 4. Remove all closing tags like {{< /tab >}}, {{< /tabs >}}, {{< /hint >}}
    text = re.sub(r"\{\{<\s*/\w+\s*>\}\}", "", text)

    # 5. Clean up excess blank lines left over from removing shortcodes
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def parse_hugo_file(file_path: str | Path) -> Document | None:
    """
    Reads a Hugo markdown file, extracts YAML frontmatter into metadata,
    cleans the shortcodes, and returns a LangChain Document.
    """
    path = Path(file_path)

    # Ignore _index.md files if they are just structural and have no real content,
    # but for now we will parse everything. (You can tweak this later).

    try:
        # frontmatter.load safely separates the YAML header from the content
        post = frontmatter.load(path)

        # 1. Get raw markdown content and clean it
        raw_content = post.content
        clean_content = clean_hugo_shortcodes(raw_content)

        # If the file has no content after cleaning, skip it
        if not clean_content:
            return None

        # 2. Extract metadata safely
        metadata = {
            "source": str(path),
            "title": post.metadata.get("title", path.stem),
            "author": post.metadata.get("params", {}).get("author", "Unknown"),
            "weight": post.metadata.get("weight", 0),
        }

        # 3. Return a LangChain Document ready for chunking
        return Document(page_content=clean_content, metadata=metadata)

    except Exception as e:
        print(f"Failed to parse {path}: {e}")
        return None


# Quick test block: if you run this file directly, it will test the parser
if __name__ == "__main__":
    # Create a dummy test file based on your example to verify it works
    test_file = Path("test_hugo.md")
    test_file.write_text("""---
title: Installing the CLI
params:
  author: David Godlove
---
{{< tabs "OS" >}}
{{< tab "RPM" >}}
Install via RPM.
{{< hint type=note >}}
You need a token.
{{< /hint >}}
{{< /tab >}}
{{< /tabs >}}""")

    doc = parse_hugo_file(test_file)
    print("--- PARSED METADATA ---")
    print(doc.metadata)
    print("\n--- CLEANED CONTENT ---")
    print(doc.page_content)

    # Clean up test file
    test_file.unlink()
