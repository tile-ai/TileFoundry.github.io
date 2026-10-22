"""Send links that leave docs/ to the TileFoundry source on GitHub.

Upstream pages link to repository files outside docs/, such as
../CONTRIBUTING.md. The link works on GitHub, but the site has no page for it,
so `mkdocs build --strict` stops. Each such link is rewritten to the same file
at the TileFoundry ref being rendered, so it opens what the page describes.
"""

import os
import posixpath
import re

# Where the upstream repository keeps the pages this site renders.
DOCS_DIR = "docs"

# The target of an inline Markdown link, up to an anchor, a title or the ")".
LINK_TARGET = re.compile(r"(\]\()([^)\s#]+)")
URL_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")


def on_page_markdown(markdown, page, config, files):
    # The publish workflow names the ref it renders; a local preview has none
    # and links to main.
    ref = os.environ.get("TILEFOUNDRY_REF") or "main"
    blob = f"{config.repo_url.rstrip('/')}/blob/{ref}/"
    page_dir = posixpath.dirname(page.file.src_uri)

    def rewrite(match):
        target = match.group(2)
        if URL_SCHEME.match(target) or target.startswith("/"):
            return match.group(0)
        path = posixpath.normpath(posixpath.join(DOCS_DIR, page_dir, target))
        # A link inside docs/ is the site's own; one above the repository root
        # is broken everywhere, and strict mode should still say so.
        if path == DOCS_DIR or path.startswith((DOCS_DIR + "/", "..")):
            return match.group(0)
        return match.group(1) + blob + path

    return LINK_TARGET.sub(rewrite, markdown)
