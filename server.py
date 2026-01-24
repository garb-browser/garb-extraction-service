"""
GARB Extraction Service
Extracts article content from web pages for eye tracking research.
Uses Mozilla Readability algorithm for high-quality content extraction.
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

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Simple in-memory cache (URL hash -> extracted content)
extraction_cache = {}
CACHE_MAX_SIZE = 100

# User agent for fetching URLs
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'


def get_cache_key(url):
    """Generate cache key from URL."""
    return hashlib.md5(url.encode()).hexdigest()


def fetch_url(url):
    """Fetch URL content with proper headers."""
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return response.text


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
            # Hyperlink - preserve it
            href = child.get('href', '')
            if href and not href.startswith('#') and not href.startswith('javascript:'):
                # Resolve relative URLs
                if href.startswith('/') or not href.startswith('http'):
                    href = urljoin(base_url, href)
                link_text = clean_text(child.get_text(separator=' '))
                if link_text:
                    # Format: [[link_text|url]]
                    result_parts.append(f'[[{link_text}|{href}]]')
                    plain_parts.append(link_text)
            else:
                # Invalid or anchor link - just get text
                link_text = clean_text(child.get_text(separator=' '))
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
    seen_images = set()

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
                if src and src not in seen_images and is_valid_content_image(src):
                    seen_images.add(src)
                    formatted_parts.append({
                        'type': 'image',
                        'src': src,
                        'alt': img.get('alt', '')
                    })
                    images.append({'src': src, 'type': 'inline', 'alt': img.get('alt', '')})

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
                if src and src not in seen_images and is_valid_content_image(src):
                    seen_images.add(src)
                    caption = ''
                    figcaption = element.find('figcaption')
                    if figcaption:
                        caption = clean_text(figcaption.get_text(separator=' '))
                    formatted_parts.append({
                        'type': 'image',
                        'src': src,
                        'alt': img.get('alt', '') or caption,
                        'caption': caption
                    })
                    images.append({'src': src, 'type': 'inline', 'alt': caption})

        elif tag_name == 'img':
            src = resolve_image_url(element.get('src', ''), url)
            if src and src not in seen_images and is_valid_content_image(src):
                seen_images.add(src)
                formatted_parts.append({
                    'type': 'image',
                    'src': src,
                    'alt': element.get('alt', '')
                })
                images.append({'src': src, 'type': 'inline', 'alt': element.get('alt', '')})

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


@app.route('/', methods=['GET', 'POST', 'OPTIONS'])
@cross_origin()
def content_extractor():
    if request.method == 'GET':
        return "<h1>GARB Extraction Service - Running</h1><p>POST a URL or JSON with raw HTML to extract article content. Now using Mozilla Readability for better extraction!</p>"

    if request.method == 'POST':
        try:
            # Check if it's JSON (raw HTML mode) or plain text (URL mode)
            content_type = request.content_type or ''

            if 'application/json' in content_type:
                # JSON mode: { url: "...", html: "..." }
                data = request.get_json()
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

            # Extract content using Readability
            content_data = extract_with_readability(html, url)
            metadata = extract_metadata(html, url)

            if not content_data['plain']:
                return jsonify({
                    'error': 'Could not extract content from this page',
                    'title': content_data['title'] or 'Unknown',
                    'content': '',
                    'formatted_content': []
                }), 200

            # Build response
            images = content_data['images']
            response_data = {
                'title': content_data['title'] or 'Untitled Article',
                'img_src': images[0]['src'] if images else '',
                'content': content_data['plain'],  # Backward compatibility
                'formatted_content': content_data['formatted'],
                'images': images,
                'metadata': metadata,
                'url': url,
                'extraction_time_ms': (datetime.now() - start_time).total_seconds() * 1000
            }

            # Cache the result (only for URL mode)
            if cache_key and not use_raw_html:
                if len(extraction_cache) >= CACHE_MAX_SIZE:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(extraction_cache))
                    del extraction_cache[oldest_key]
                extraction_cache[cache_key] = response_data

            print(f"Extracted in {response_data['extraction_time_ms']:.0f}ms: {content_data['title'][:50] if content_data['title'] else 'Unknown'}...")

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
        'engine': 'mozilla-readability',
        'cache_size': len(extraction_cache),
        'cache_max': CACHE_MAX_SIZE
    })


@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear the extraction cache."""
    extraction_cache.clear()
    return jsonify({'status': 'cache cleared'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9000))
    print(f"Starting GARB Extraction Service on port {port}")
    print("Using Mozilla Readability for content extraction")
    app.run(host='0.0.0.0', port=port)
