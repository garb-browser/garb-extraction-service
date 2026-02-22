"""
GARB Extraction Service
Extracts article content from web pages for eye tracking research.
Uses a cascade of extraction algorithms for high-quality content extraction:
  1. Trafilatura (primary) - Best overall recall and modern web structure handling
  2. Mozilla Readability (fallback) - Reliable for simpler page structures
Preserves formatting, images, and metadata while removing ads/navigation.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from readability import Document
from bs4 import BeautifulSoup
import os
import re
from urllib.parse import urljoin, urlparse
from datetime import datetime
import hashlib
import requests
import trafilatura
from trafilatura.settings import use_config

# Configure trafilatura for optimal extraction
TRAFILATURA_CONFIG = use_config()
TRAFILATURA_CONFIG.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")
TRAFILATURA_CONFIG.set("DEFAULT", "MIN_OUTPUT_SIZE", "100")

# Minimum word count to consider extraction successful
MIN_WORDS_THRESHOLD = 50

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Optional API key authentication
EXTRACT_API_KEY = os.environ.get('EXTRACT_API_KEY', '')


def check_api_key():
    """Check API key if EXTRACT_API_KEY is set. Returns error response or None."""
    if not EXTRACT_API_KEY:
        return None  # No key configured, allow all requests
    provided_key = request.headers.get('X-API-Key', '')
    if provided_key != EXTRACT_API_KEY:
        return jsonify({'error': 'Invalid or missing API key'}), 401
    return None

# Simple in-memory cache (URL hash -> extracted content)
extraction_cache = {}
CACHE_MAX_SIZE = 100

# User agent for fetching URLs
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'


def get_cache_key(url):
    """Generate cache key from URL."""
    return hashlib.md5(url.encode()).hexdigest()


MAX_CONTENT_SIZE = 10 * 1024 * 1024  # 10MB


def fetch_url(url):
    """Fetch URL content with proper headers. Enforces a 10MB size limit."""
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    response = requests.get(url, headers=headers, timeout=15, stream=True)
    response.raise_for_status()

    # Check Content-Length header first
    content_length = response.headers.get('Content-Length')
    if content_length and int(content_length) > MAX_CONTENT_SIZE:
        response.close()
        raise ValueError(f'Response too large: {int(content_length)} bytes exceeds {MAX_CONTENT_SIZE} byte limit')

    # Read in chunks to enforce limit even without Content-Length
    chunks = []
    total = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        total += len(chunk)
        if total > MAX_CONTENT_SIZE:
            response.close()
            raise ValueError(f'Response too large: exceeds {MAX_CONTENT_SIZE} byte limit')
        chunks.append(chunk)

    content = b''.join(chunks)
    encoding = response.encoding or response.apparent_encoding or 'utf-8'
    return content.decode(encoding, errors='replace')


def resolve_image_url(src, base_url):
    """Convert relative image URLs to absolute URLs."""
    if not src:
        return None
    if src.startswith('data:'):
        return src  # Data URLs are already complete
    if src.startswith('//'):
        return 'https:' + src
    if src.startswith('/') or not src.startswith('http'):
        return urljoin(base_url, src)
    return src


def normalize_image_url(src):
    """Normalize image URL for deduplication."""
    if not src:
        return ''
    # Remove query parameters and fragments for comparison
    parsed = urlparse(src)
    path = parsed.path

    # Special handling for Wikipedia/Wikimedia images
    # URLs like: /wikipedia/commons/thumb/a/ab/Image.jpg/220px-Image.jpg
    # Should normalize to the base image: /wikipedia/commons/a/ab/Image.jpg
    # Check both by domain AND by path pattern (for relative URLs)
    is_wikimedia = (
        'wikimedia.org' in parsed.netloc or
        'wikipedia.org' in parsed.netloc or
        '/wikipedia/' in path or
        '/commons/' in path or
        'upload.wikimedia' in src.lower()
    )

    if is_wikimedia and '/thumb/' in path:
        # Extract base path: /commons/thumb/a/ab/Image.jpg/220px-Image.jpg -> /commons/a/ab/Image.jpg
        parts = path.split('/thumb/')
        if len(parts) == 2:
            prefix = parts[0]  # /wikipedia/commons
            rest = parts[1]    # a/ab/Image.jpg/220px-Image.jpg
            # Remove the last segment (size variant like 220px-Image.jpg)
            rest_parts = rest.rsplit('/', 1)
            if len(rest_parts) == 2 and re.match(r'\d+px-', rest_parts[1]):
                path = f"{prefix}/{rest_parts[0]}"  # /wikipedia/commons/a/ab/Image.jpg

    # For Wikimedia, also extract just the filename for better deduplication
    # because the same image can appear via different paths
    if is_wikimedia:
        # Extract the actual filename (last meaningful part before size suffix)
        # e.g., "Image.jpg" from various path formats
        filename_match = re.search(r'/([^/]+\.(jpg|jpeg|png|gif|svg|webp))', path, re.I)
        if filename_match:
            # Use just the filename as the key (more aggressive deduplication)
            return f"wikimedia:{filename_match.group(1).lower()}"

    # Keep just scheme, netloc, and normalized path
    normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
    # Normalize scheme
    if normalized.startswith('//'):
        normalized = 'https:' + normalized
    elif normalized.startswith(':/'):
        normalized = 'https' + normalized
    return normalized.lower()


def is_valid_content_image(src, alt=''):
    """Check if an image is likely content (not ad/tracking/logo)."""
    if not src:
        return False

    src_lower = src.lower()

    # Skip tracking pixels and ads
    skip_patterns = [
        'pixel', 'tracker', 'beacon', 'analytics',
        'advertisement', 'sponsor', 'promo',
        '1x1', 'spacer', 'blank', 'transparent',
        'facebook.com/tr', 'doubleclick', 'googlesyndication',
        'amazon-adsystem', 'adservice'
    ]

    for pattern in skip_patterns:
        if pattern in src_lower:
            return False

    # Skip very small images (likely icons/buttons) based on filename hints
    if any(x in src_lower for x in ['icon', 'button', 'arrow', 'sprite']):
        return False

    return True


def clean_text(text):
    """Clean extracted text: normalize whitespace, preserve proper spacing."""
    # Replace multiple whitespace (including newlines) with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def is_citation_link(href, link_text):
    """
    Check if a link is a Wikipedia citation reference that should be filtered out.
    Returns True if the link looks like a citation (e.g., [10], [23]) pointing to #cite_note.
    """
    if not href or not link_text:
        return False

    href_lower = href.lower()
    text_stripped = link_text.strip()

    # Check if URL contains citation anchor patterns
    is_cite_url = (
        '#cite_note' in href_lower or
        '#cite_ref' in href_lower or
        '#ref-' in href_lower or
        '#note-' in href_lower or
        '#endnote' in href_lower or
        '#footnote' in href_lower
    )

    # Check if link text looks like a citation number: [1], [23], 1, 23, etc.
    is_cite_text = bool(re.match(r'^\[?\d+\]?$', text_stripped))

    # Also catch superscript-style citations like "10" that link to citations
    if is_cite_url and is_cite_text:
        return True

    # If URL is clearly a citation link, filter it regardless of text
    if is_cite_url and len(text_stripped) <= 5:
        return True

    return False


def preprocess_html_for_readability(html, url=''):
    """
    Pre-process HTML to remove clutter before Readability extraction.
    Targets Wikipedia-specific elements, infoboxes, navigation, and other noise.
    """
    soup = BeautifulSoup(html, 'lxml')

    # Elements to remove completely (Wikipedia and general clutter)
    selectors_to_remove = [
        # Wikipedia-specific
        '.infobox',           # Wikipedia infoboxes (the metadata tables)
        '.sidebar',           # Sidebar boxes
        '.navbox',            # Navigation boxes at bottom
        '.navbox-styles',     # Navigation box styles
        '.mbox',              # Message boxes
        '.ambox',             # Article message boxes
        '.tmbox',             # Talk page message boxes
        '.ombox',             # Other message boxes
        '.cmbox',             # Category message boxes
        '.fmbox',             # Footer message boxes
        '.imbox',             # Image message boxes
        '.dmbox',             # Disambiguation message boxes
        '.sistersitebox',     # Sister project boxes
        '.portalbox',         # Portal boxes
        '.vertical-navbox',   # Vertical navigation
        '.authority-control', # Authority control section
        '.catlinks',          # Category links
        '.mw-editsection',    # Edit section links
        '.reference',         # Reference numbers [1], [2], etc.
        '.references',        # References section
        '.reflist',           # Reference list
        '.refbegin',          # Reference section beginning
        '.citation',          # Citation elements
        '#toc',               # Table of contents
        '.toc',               # Table of contents class
        '.mw-headline-anchor', # Headline anchors
        '.mw-jump-link',      # Jump links
        '.noprint',           # No-print elements
        '.metadata',          # Metadata elements
        '.stub',              # Stub notices
        '.plainlinks',        # Plain links boxes
        '.wikitable.sortable', # Sortable data tables (often not main content)
        '.shortdescription',  # Wikipedia short description
        '.page-header',       # Page header areas
        '#siteSub',           # "From Wikipedia" text
        '#contentSub',        # Content sub-header
        '.mw-indicators',     # Page status indicators
        '.mw-body-header',    # Body header
        '#mw-fr-revisiontag', # Revision tags
        '.sister-wikipedia',  # Sister wiki links
        '.spoken-wikipedia',  # Audio article links
        '.geo',               # Geographic coordinates
        '.geo-dec',           # Decimal coordinates
        '.geo-dms',           # DMS coordinates

        # General website clutter
        '.advertisement',
        '.ad-container',
        '.social-share',
        '.share-buttons',
        '.related-posts',
        '.recommended',
        '.newsletter-signup',
        '.popup',
        '.modal',
        '.cookie-notice',
        '.cookie-banner',
        'aside',              # Sidebar content
        '[role="complementary"]',  # Complementary content
        '[role="navigation"]',     # Navigation content
        '.nav',
        '.navigation',
        '.menu',
        '.breadcrumb',
        '.breadcrumbs',
        'footer',
        '.footer',
        '.site-footer',
        '.comments',
        '.comment-section',
        '#comments',
    ]

    for selector in selectors_to_remove:
        try:
            for element in list(soup.select(selector)):
                if hasattr(element, 'decompose'):
                    element.decompose()
        except Exception:
            pass  # Skip invalid selectors

    # Remove tables that look like metadata (have few cells but lots of labels)
    for table in list(soup.find_all('table')):
        # Skip if not a proper element
        if not hasattr(table, 'find_all'):
            continue

        # Check if it's a metadata-style table (lots of th elements, few td)
        th_count = len(table.find_all('th'))
        td_count = len(table.find_all('td'))
        tr_count = len(table.find_all('tr'))

        # If table has many rows with header cells (like key-value metadata), remove it
        if tr_count > 5 and th_count > tr_count * 0.4:
            table.decompose()
            continue

        # Check if table contains mostly short label-value pairs
        cells = table.find_all(['td', 'th'])
        if cells:
            avg_text_length = sum(len(c.get_text(strip=True)) for c in cells) / len(cells)
            # Metadata tables typically have short cell content
            if tr_count > 10 and avg_text_length < 30:
                table.decompose()

    return str(soup)


def extract_text_with_links(element, base_url=''):
    """
    Extract text from element while preserving hyperlinks.
    Returns dict with 'text' (plain text) and 'html' (text with links preserved).
    """
    # Clone the element to avoid modifying original
    from copy import copy

    result_parts = []
    plain_parts = []

    for child in element.children:
        if isinstance(child, str):
            # Text node
            cleaned = re.sub(r'\s+', ' ', child)
            if cleaned.strip():
                result_parts.append(cleaned)
                plain_parts.append(cleaned)
        elif child.name == 'a':
            # Hyperlink - preserve it (unless it's a citation)
            href = child.get('href', '')
            link_text = clean_text(child.get_text(separator=' '))

            # For citation links, keep the number but don't make it a hyperlink
            if is_citation_link(href, link_text):
                if link_text:
                    result_parts.append(link_text)  # Just the number, no link marker
                    plain_parts.append(link_text)
                continue

            if href and not href.startswith('#') and not href.startswith('javascript:'):
                # Resolve relative URLs
                if href.startswith('/') or not href.startswith('http'):
                    href = urljoin(base_url, href)
                if link_text:
                    # Format: [[link_text|url]]
                    result_parts.append(f'[[{link_text}|{href}]]')
                    plain_parts.append(link_text)
            else:
                # Invalid or anchor link - just get text
                if link_text:
                    result_parts.append(link_text)
                    plain_parts.append(link_text)
        elif child.name in ['strong', 'b', 'em', 'i', 'span']:
            # Inline elements - recursively process
            nested = extract_text_with_links(child, base_url)
            result_parts.append(nested['html'])
            plain_parts.append(nested['text'])
        else:
            # Other elements - just get text
            text = clean_text(child.get_text(separator=' '))
            if text:
                result_parts.append(text)
                plain_parts.append(text)

    html_result = ' '.join(result_parts)
    text_result = ' '.join(plain_parts)

    # Clean up extra spaces
    html_result = re.sub(r'\s+', ' ', html_result).strip()
    text_result = re.sub(r'\s+', ' ', text_result).strip()

    return {'html': html_result, 'text': text_result}


def extract_with_readability(html, url=''):
    """
    Extract article content using Mozilla Readability algorithm.
    Returns structured content with images preserved.
    """
    # Pre-process HTML to remove clutter (infoboxes, navigation, etc.)
    cleaned_html = preprocess_html_for_readability(html, url)

    # Parse with Readability
    doc = Document(cleaned_html, url=url)

    # Get the cleaned article HTML
    article_html = doc.summary()
    title = doc.title()

    # Parse the article HTML
    soup = BeautifulSoup(article_html, 'lxml')

    # Also parse original HTML for metadata and images that might be missed
    original_soup = BeautifulSoup(html, 'lxml')

    # Extract structured content
    formatted_parts = []
    images = []
    seen_images = set()  # Uses normalized URLs for deduplication

    def add_image_if_new(src, alt='', img_type='inline', caption=''):
        """Add image if not already seen (using normalized URL)."""
        if not src:
            return False
        normalized = normalize_image_url(src)
        if normalized and normalized not in seen_images and is_valid_content_image(src):
            seen_images.add(normalized)
            formatted_parts.append({
                'type': 'image',
                'src': src,
                'alt': alt or caption,
                'caption': caption if caption else None
            })
            images.append({'src': src, 'type': img_type, 'alt': alt or caption})
            return True
        return False

    # Process all elements in order
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                                   'p', 'blockquote', 'ul', 'ol',
                                   'figure', 'img', 'table']):
        tag_name = element.name

        if tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            text = clean_text(element.get_text(separator=' '))
            if text and len(text) > 2:  # Skip very short headings (likely noise)
                # Skip headings that look like infobox labels
                skip_headings = [
                    'motto', 'type', 'established', 'location', 'campus',
                    'endowment', 'budget', 'president', 'provost', 'students',
                    'undergraduates', 'postgraduates', 'academic staff',
                    'administrative staff', 'website', 'nickname', 'colors',
                    'sporting affiliations', 'academic affiliations',
                    'motto in english', 'coordinates'
                ]
                if text.lower() in skip_headings:
                    continue

                level = int(tag_name[1])
                formatted_parts.append({
                    'type': 'heading',
                    'level': level,
                    'text': text
                })

        elif tag_name == 'p':
            # Check for images inside paragraph
            for img in element.find_all('img'):
                src = resolve_image_url(img.get('src', ''), url)
                add_image_if_new(src, img.get('alt', ''))

            # Extract text with links preserved
            extracted = extract_text_with_links(element, url)
            text = extracted['text']
            if text and len(text) > 15:  # Skip very short paragraphs
                # Skip paragraphs that look like metadata (key|value patterns, coordinates, etc.)
                pipe_count = text.count('|')
                if pipe_count > 3 and pipe_count > len(text) / 50:
                    continue  # Too many pipe separators - likely metadata
                if re.match(r'^[\d°′″.NSEW\s,/-]+$', text):
                    continue  # Looks like coordinates
                if text.startswith('Coordinates:') or text.startswith('Location:'):
                    continue  # Skip coordinate/location metadata

                formatted_parts.append({
                    'type': 'paragraph',
                    'text': text,
                    'html': extracted['html']  # Includes link markers [[text|url]]
                })

        elif tag_name == 'blockquote':
            text = clean_text(element.get_text(separator=' '))
            if text:
                formatted_parts.append({
                    'type': 'quote',
                    'text': text
                })

        elif tag_name in ['ul', 'ol']:
            items = []
            for li in element.find_all('li', recursive=False):
                item_text = clean_text(li.get_text(separator=' '))
                if item_text:
                    items.append(item_text)
            if items:
                formatted_parts.append({
                    'type': 'list',
                    'ordered': tag_name == 'ol',
                    'items': items
                })

        elif tag_name == 'figure':
            img = element.find('img')
            if img:
                src = resolve_image_url(img.get('src', ''), url)
                caption = ''
                figcaption = element.find('figcaption')
                if figcaption:
                    caption = clean_text(figcaption.get_text(separator=' '))
                add_image_if_new(src, img.get('alt', ''), 'inline', caption)

        elif tag_name == 'img':
            src = resolve_image_url(element.get('src', ''), url)
            add_image_if_new(src, element.get('alt', ''))

        elif tag_name == 'table':
            # Skip tables that look like metadata/infoboxes
            table_classes = ' '.join(element.get('class', []))
            if any(x in table_classes for x in ['infobox', 'metadata', 'sidebar', 'navbox', 'mbox']):
                continue

            # Check if it's a key-value style table (skip those)
            th_count = len(element.find_all('th'))
            td_count = len(element.find_all('td'))
            tr_count = len(element.find_all('tr'))

            # Skip tables that are mostly headers (metadata style)
            if tr_count > 0 and th_count > tr_count * 0.3:
                continue

            # Extract table as text only if it seems like content
            rows = []
            for tr in element.find_all('tr'):
                cells = [clean_text(td.get_text(separator=' ')) for td in tr.find_all(['td', 'th'])]
                if cells and len(''.join(cells)) > 10:  # Skip rows with very little content
                    rows.append(' | '.join(cells))
            if rows and len(rows) >= 2:  # Only include tables with actual content
                formatted_parts.append({
                    'type': 'paragraph',
                    'text': '\n'.join(rows)
                })

    # Try to find main/top image from original page if not found
    if not images:
        # Look for og:image or twitter:image
        og_image = original_soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            src = resolve_image_url(og_image['content'], url)
            if src and is_valid_content_image(src):
                images.append({'src': src, 'type': 'top', 'alt': ''})

        # Look for article main image
        if not images:
            article_img = original_soup.select_one('article img, .article img, .post img, main img')
            if article_img:
                src = resolve_image_url(article_img.get('src', ''), url)
                if src and is_valid_content_image(src):
                    images.append({'src': src, 'type': 'top', 'alt': article_img.get('alt', '')})

    # Build plain text version
    plain_text = soup.get_text(separator=' ', strip=True)
    # Clean up excessive whitespace while preserving paragraph breaks
    plain_text = re.sub(r'\s+', ' ', plain_text).strip()

    return {
        'title': title,
        'formatted': formatted_parts,
        'plain': plain_text,
        'images': images
    }


def extract_metadata(html, url=''):
    """Extract article metadata from HTML."""
    soup = BeautifulSoup(html, 'lxml')
    metadata = {}

    # Extract author
    author_meta = soup.find('meta', attrs={'name': re.compile(r'author', re.I)})
    if author_meta and author_meta.get('content'):
        metadata['authors'] = [author_meta['content']]
    else:
        # Try JSON-LD
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                import json
                data = json.loads(script.string)
                if isinstance(data, dict) and 'author' in data:
                    author = data['author']
                    if isinstance(author, dict):
                        metadata['authors'] = [author.get('name', '')]
                    elif isinstance(author, str):
                        metadata['authors'] = [author]
                    break
            except:
                pass

    # Extract publish date
    date_meta = soup.find('meta', attrs={'property': re.compile(r'published_time|date', re.I)})
    if date_meta and date_meta.get('content'):
        metadata['publish_date'] = date_meta['content']
    else:
        time_tag = soup.find('time', attrs={'datetime': True})
        if time_tag:
            metadata['publish_date'] = time_tag['datetime']

    # Extract description
    desc_meta = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
    if desc_meta and desc_meta.get('content'):
        metadata['description'] = desc_meta['content']

    # Extract domain
    if url:
        parsed = urlparse(url)
        metadata['domain'] = parsed.netloc

    return metadata


def extract_with_trafilatura(html, url=''):
    """
    Extract article content using Trafilatura.
    Trafilatura has better recall than Readability for complex page structures.
    Returns structured content in the same format as extract_with_readability.
    """
    try:
        # Extract main text with trafilatura
        # include_images=True to get image references
        # include_links=True to preserve hyperlinks
        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            include_images=True,
            include_links=True,
            output_format='xml',  # XML gives us structured content
            config=TRAFILATURA_CONFIG
        )

        if not extracted:
            print("Trafilatura: No content extracted")
            return None

        # Also extract as plain text for word count check
        plain_text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            output_format='txt',
            config=TRAFILATURA_CONFIG
        ) or ''

        # Check if we got enough content
        word_count = len(plain_text.split())
        if word_count < MIN_WORDS_THRESHOLD:
            print(f"Trafilatura: Insufficient content ({word_count} words < {MIN_WORDS_THRESHOLD})")
            return None

        # Parse the XML output to build structured content
        from xml.etree import ElementTree as ET
        try:
            root = ET.fromstring(f"<root>{extracted}</root>")
        except ET.ParseError:
            # If XML parsing fails, fall back to plain text
            print("Trafilatura: XML parsing failed, using plain text")
            root = None

        formatted_parts = []
        images = []
        seen_images = set()

        # Parse original HTML for images and title
        original_soup = BeautifulSoup(html, 'lxml')

        # Get title from metadata
        title = ''
        title_meta = original_soup.find('meta', property='og:title')
        if title_meta and title_meta.get('content'):
            title = title_meta['content']
        if not title:
            title_tag = original_soup.find('title')
            if title_tag:
                title = clean_text(title_tag.get_text())

        if root is not None:
            # Helper function to get ALL text from an element including nested children
            def get_full_text(elem):
                """Get all text content from element including nested elements."""
                return clean_text(''.join(elem.itertext()))

            def get_text_with_links(elem, base_url=''):
                """
                Get text from element with links preserved as [[text|url]] markers.
                Similar to extract_text_with_links but for XML elements.
                Returns dict with 'text' (plain) and 'html' (with link markers).
                """
                result_parts = []  # With link markers
                plain_parts = []   # Plain text

                def process_element(el):
                    # Handle element's direct text
                    if el.text:
                        text = el.text.strip()
                        if text:
                            result_parts.append(text)
                            plain_parts.append(text)

                    # Process children
                    for child in el:
                        child_tag = child.tag.lower() if child.tag else ''

                        if child_tag == 'ref':
                            # Link element - extract href and text
                            href = child.get('target') or child.get('href') or ''
                            link_text = ''.join(child.itertext()).strip()

                            # For citation links, keep the number but don't make it a hyperlink
                            if is_citation_link(href, link_text):
                                if link_text:
                                    result_parts.append(link_text)  # Just the number, no link marker
                                    plain_parts.append(link_text)
                                continue

                            if href and link_text:
                                # Resolve relative URLs
                                if href.startswith('/') or (not href.startswith('http') and not href.startswith('//')):
                                    href = urljoin(base_url, href)
                                # Add as link marker
                                result_parts.append(f'[[{link_text}|{href}]]')
                                plain_parts.append(link_text)
                            elif link_text:
                                result_parts.append(link_text)
                                plain_parts.append(link_text)
                        elif child_tag in ['hi', 'emph', 'b', 'i', 'strong', 'em']:
                            # Inline formatting - recursively process
                            nested = get_text_with_links(child, base_url)
                            result_parts.append(nested['html'])
                            plain_parts.append(nested['text'])
                        else:
                            # Other elements - get their text
                            child_text = ''.join(child.itertext()).strip()
                            if child_text:
                                result_parts.append(child_text)
                                plain_parts.append(child_text)

                        # Handle tail text (text after child element)
                        if child.tail:
                            tail = child.tail.strip()
                            if tail:
                                result_parts.append(tail)
                                plain_parts.append(tail)

                process_element(elem)

                html_result = ' '.join(result_parts)
                text_result = ' '.join(plain_parts)

                # Clean up extra spaces
                html_result = clean_text(html_result)
                text_result = clean_text(text_result)

                return {'html': html_result, 'text': text_result}

            # Track processed elements to avoid duplicates
            processed_elements = set()

            # Process XML elements - only handle top-level content elements
            for element in root.iter():
                # Skip if already processed (as part of a parent)
                if id(element) in processed_elements:
                    continue

                tag = element.tag.lower() if element.tag else ''

                # Skip inline elements - they're processed as part of their parent
                if tag in ['hi', 'ref', 'a', 'link', 'item', 'cell', 'row']:
                    continue

                if tag == 'head':
                    # Heading element - get full text including any nested formatting
                    text = get_full_text(element)
                    if text and len(text) > 2:
                        formatted_parts.append({
                            'type': 'heading',
                            'level': 2,
                            'text': text
                        })
                        # Mark all children as processed
                        for child in element.iter():
                            processed_elements.add(id(child))

                elif tag == 'p':
                    # Paragraph - get text with links preserved
                    extracted = get_text_with_links(element, url)
                    text = extracted['text']
                    html_with_links = extracted['html']
                    if text and len(text) > 15:
                        formatted_parts.append({
                            'type': 'paragraph',
                            'text': text,
                            'html': html_with_links  # Contains [[text|url]] link markers
                        })
                        # Mark all children as processed
                        for child in element.iter():
                            processed_elements.add(id(child))

                elif tag == 'quote':
                    # Blockquote - get full text
                    text = get_full_text(element)
                    if text:
                        formatted_parts.append({
                            'type': 'quote',
                            'text': text
                        })
                        for child in element.iter():
                            processed_elements.add(id(child))

                elif tag == 'list':
                    # List items - get full text for each item
                    items = []
                    for item in element.findall('.//item'):
                        item_text = get_full_text(item)
                        if item_text:
                            items.append(item_text)
                        processed_elements.add(id(item))
                    if items:
                        formatted_parts.append({
                            'type': 'list',
                            'ordered': False,
                            'items': items
                        })
                    for child in element.iter():
                        processed_elements.add(id(child))

                elif tag == 'graphic':
                    # Image reference
                    src = element.get('src', '')
                    if src:
                        src = resolve_image_url(src, url)
                        normalized = normalize_image_url(src)
                        if src and normalized not in seen_images and is_valid_content_image(src):
                            seen_images.add(normalized)
                            alt = element.get('alt', '') or element.get('title', '')
                            formatted_parts.append({
                                'type': 'image',
                                'src': src,
                                'alt': alt
                            })
                            images.append({'src': src, 'type': 'inline', 'alt': alt})
                    processed_elements.add(id(element))
        else:
            # Fallback: split plain text into paragraphs
            paragraphs = [p.strip() for p in plain_text.split('\n\n') if p.strip()]
            for para in paragraphs:
                if len(para) > 15:
                    formatted_parts.append({
                        'type': 'paragraph',
                        'text': para,
                        'html': para
                    })

        # Try to find images from original page if none found
        if not images:
            og_image = original_soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                src = resolve_image_url(og_image['content'], url)
                if src and is_valid_content_image(src):
                    images.append({'src': src, 'type': 'top', 'alt': ''})

            if not images:
                article_img = original_soup.select_one('article img, .article img, main img')
                if article_img:
                    src = resolve_image_url(article_img.get('src', ''), url)
                    if src and is_valid_content_image(src):
                        images.append({'src': src, 'type': 'top', 'alt': article_img.get('alt', '')})

        print(f"Trafilatura: Extracted {len(formatted_parts)} content blocks, {word_count} words")

        return {
            'title': title,
            'formatted': formatted_parts,
            'plain': plain_text,
            'images': images,
            'extractor': 'trafilatura'
        }

    except Exception as e:
        print(f"Trafilatura extraction error: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_with_cascade(html, url=''):
    """
    Extract content using a cascade of extractors.
    Tries Trafilatura first (better recall), falls back to Readability.
    Returns the best result along with which extractor was used.
    """
    # Try Trafilatura first (better for complex page structures)
    print(f"Attempting extraction with Trafilatura...")
    result = extract_with_trafilatura(html, url)

    if result and len(result.get('formatted', [])) >= 3:
        # Trafilatura succeeded with sufficient content
        print(f"Trafilatura extraction successful")
        return result

    # Fall back to Readability
    print(f"Falling back to Readability...")
    readability_result = extract_with_readability(html, url)
    readability_result['extractor'] = 'readability'

    # If Trafilatura got some content, compare and use the better one
    if result and result.get('plain'):
        trafilatura_words = len(result['plain'].split())
        readability_words = len(readability_result.get('plain', '').split())

        # Use whichever got more content (with a small bias toward Trafilatura)
        if trafilatura_words > readability_words * 0.8:
            print(f"Using Trafilatura result ({trafilatura_words} words vs {readability_words})")
            return result
        else:
            print(f"Using Readability result ({readability_words} words vs {trafilatura_words})")
            return readability_result

    print(f"Using Readability result")
    return readability_result


@app.route('/', methods=['GET', 'POST', 'OPTIONS'])
@cross_origin()
def content_extractor():
    if request.method == 'GET':
        return "<h1>GARB Extraction Service - Running</h1><p>POST a URL or JSON with raw HTML to extract article content. Using Trafilatura (primary) with Readability fallback for optimal extraction.</p>"

    if request.method == 'POST':
        # Check API key authentication
        auth_error = check_api_key()
        if auth_error:
            return auth_error

        try:
            # Check if it's JSON (raw HTML mode) or plain text (URL mode)
            content_type = request.content_type or ''

            if 'application/json' in content_type:
                # JSON mode: { url: "...", html: "..." }
                data = request.get_json(silent=True)
                if data is None:
                    return jsonify({'error': 'Invalid JSON body'}), 400
                if not isinstance(data, dict):
                    return jsonify({'error': 'Request body must be a JSON object'}), 400
                if 'url' not in data and 'html' not in data:
                    return jsonify({'error': "Request must include 'url' or 'html' field"}), 400
                url = data.get('url', '')
                raw_html = data.get('html', '')
                use_raw_html = bool(raw_html)
            else:
                # Plain text mode: URL only
                url = str(request.data, encoding='utf-8').strip()
                raw_html = ''
                use_raw_html = False

            if not url and not raw_html:
                return jsonify({'error': 'No URL or HTML provided'}), 400

            # Check cache first (only for URL mode without raw HTML)
            cache_key = get_cache_key(url) if url else None
            if cache_key and cache_key in extraction_cache and not use_raw_html:
                print(f"Cache hit for: {url[:50]}...")
                return jsonify(extraction_cache[cache_key])

            print(f"Extracting: {url[:80] if url else 'raw HTML'}..." + (" (from raw HTML)" if use_raw_html else ""))
            start_time = datetime.now()

            # Get HTML content
            html = raw_html if use_raw_html else None

            if not html:
                try:
                    html = fetch_url(url)
                except requests.exceptions.HTTPError as e:
                    status_code = e.response.status_code if e.response else 0
                    if status_code == 429:
                        return jsonify({
                            'error': 'Website is rate limiting. Try sending raw HTML instead.',
                            'error_code': 'RATE_LIMITED',
                            'title': 'Rate Limited',
                            'content': '',
                            'formatted_content': []
                        }), 200
                    elif status_code == 403:
                        return jsonify({
                            'error': 'Website blocked access. Try sending raw HTML instead.',
                            'error_code': 'FORBIDDEN',
                            'title': 'Access Denied',
                            'content': '',
                            'formatted_content': []
                        }), 200
                    else:
                        raise
                except Exception as fetch_error:
                    error_msg = str(fetch_error)
                    if '429' in error_msg or 'rate' in error_msg.lower():
                        return jsonify({
                            'error': 'Website is rate limiting. Try sending raw HTML instead.',
                            'error_code': 'RATE_LIMITED',
                            'title': 'Rate Limited',
                            'content': '',
                            'formatted_content': []
                        }), 200
                    raise

            # Extract content using cascade (Trafilatura -> Readability)
            content_data = extract_with_cascade(html, url)
            metadata = extract_metadata(html, url)

            if not content_data['plain']:
                return jsonify({
                    'error': 'Could not extract content from this page',
                    'title': content_data['title'] or 'Unknown',
                    'content': '',
                    'formatted_content': []
                }), 200

            # Build response - with final deduplication of images
            images = content_data['images']
            formatted_content = content_data['formatted']

            # Final deduplication pass for images in formatted_content
            # This catches any duplicates that slipped through earlier checks
            seen_img_keys = set()
            deduplicated_formatted = []
            for block in formatted_content:
                if block.get('type') == 'image':
                    img_src = block.get('src', '')
                    img_key = normalize_image_url(img_src)
                    if img_key and img_key in seen_img_keys:
                        continue  # Skip duplicate image
                    seen_img_keys.add(img_key)
                deduplicated_formatted.append(block)

            # Also deduplicate the images array
            seen_img_keys_list = set()
            deduplicated_images = []
            for img in images:
                img_key = normalize_image_url(img.get('src', ''))
                if img_key and img_key in seen_img_keys_list:
                    continue
                seen_img_keys_list.add(img_key)
                deduplicated_images.append(img)

            extractor_used = content_data.get('extractor', 'unknown')
            response_data = {
                'title': content_data['title'] or 'Untitled Article',
                'img_src': deduplicated_images[0]['src'] if deduplicated_images else '',
                'content': content_data['plain'],  # Backward compatibility
                'formatted_content': deduplicated_formatted,  # Use deduplicated version
                'images': deduplicated_images,  # Use deduplicated version
                'metadata': metadata,
                'url': url,
                'extraction_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                'extractor': extractor_used  # Track which extractor was used
            }

            # Cache the result (only for URL mode)
            if cache_key and not use_raw_html:
                if len(extraction_cache) >= CACHE_MAX_SIZE:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(extraction_cache))
                    del extraction_cache[oldest_key]
                extraction_cache[cache_key] = response_data

            print(f"Extracted in {response_data['extraction_time_ms']:.0f}ms using {extractor_used}: {content_data['title'][:50] if content_data['title'] else 'Unknown'}...")

            return jsonify(response_data)

        except Exception as e:
            print(f"Extraction error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'error': str(e),
                'title': 'Extraction Failed',
                'content': '',
                'formatted_content': []
            }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        'status': 'healthy',
        'engine': 'trafilatura-readability-cascade',
        'primary': 'trafilatura',
        'fallback': 'readability',
        'cache_size': len(extraction_cache),
        'cache_max': CACHE_MAX_SIZE
    })


@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear the extraction cache."""
    auth_error = check_api_key()
    if auth_error:
        return auth_error
    extraction_cache.clear()
    return jsonify({'status': 'cache cleared'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9000))
    print(f"Starting GARB Extraction Service on port {port}")
    print("Using Trafilatura (primary) with Readability (fallback) for content extraction")
    app.run(host='0.0.0.0', port=port)
