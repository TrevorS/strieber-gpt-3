# Page Reader Redesign - LLM-Optimized Tool Interface

## Overview

Complete redesign of the Jina Reader MCP tool to follow best practices for LLM tool consumption, inspired by how my `web_search` tool structures its results. Optimized for `gpt-oss-120b`, not just Claude.

## Key Changes

### 1. **Removed Jina API Fallback** ✅

- **Before**: Hybrid approach with Jina API fallback
- **After**: 100% local processing, no external dependencies
- **Benefit**: Simpler architecture, no rate limits, completely free

### 2. **Structured Data Output** ✅

**Before** (jina_reader.py):
```python
return "# Article Title\n\nMarkdown content..."  # Plain string
```

**After** (page_reader.py):
```python
return {
    "content": "Markdown content...",
    "metadata": {
        "title": "Article Title",
        "author": "John Doe",
        "word_count": 1234,
        "reading_time_minutes": 5,
        "published_date": "2025-01-15",
        ...
    },
    "links": [
        {"text": "Link text", "url": "https://...", "is_external": true},
        ...
    ],
    "images": [
        {"url": "https://...", "alt": "Image alt text", "title": "..."},
        ...
    ],
    "sections": ["Heading 1", "Heading 2", ...]
}
```

**Benefit**: LLM can easily parse, reference, and work with structured data

### 3. **Mozilla Readability Integration** ✅

- **Added**: `readabilipy` library (same tech as Jina Reader uses)
- **Purpose**: Clean content extraction before markdown conversion
- **Benefit**: Removes ads, navigation, clutter at HTML level (not markdown level)

### 4. **New Tool Shapes** ✅

**Before** (single tool):
- `jina_fetch_page(url, remove_images, gather_links, timeout, bypass_cache)`

**After** (multiple focused tools):
- `fetch_page(url, include_links, include_images, timeout)` - Full structured output
- `fetch_page_text(url, timeout)` - Simple text-only (backward compat)
- `get_page_links(url, external_only, timeout)` - Just extract links
- `get_page_info()` - Tool capabilities

**Benefit**: Each tool has a clear purpose, easier for LLM to choose

### 5. **Better Metadata Extraction** ✅

Extracts from multiple sources:
- **Title**: `<title>`, readability, Open Graph
- **Author**: `<meta name="author">`, `article:author`, readability
- **Date**: `article:published_time`, `<meta name="date">`
- **Site Name**: `og:site_name`
- **Language**: `<html lang="...">`
- **Excerpt**: `<meta name="description">`, `og:description`
- **Word Count**: Calculated from plain text
- **Reading Time**: Estimated at 200 WPM

**Benefit**: Rich context for LLM to understand the content

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  fetch_page(url)                                        │
│                                                          │
│  1. Playwright Fetcher (8005)                           │
│     └─> Headless Chromium renders JavaScript           │
│         Returns: Raw HTML (100-500KB typical)           │
│                                                          │
│  2. Mozilla Readability (Python)                        │
│     └─> Extracts main content, removes clutter          │
│         Returns: Clean HTML + metadata                  │
│                                                          │
│  3. BeautifulSoup Extraction                            │
│     └─> Parse HTML, extract links/images/sections       │
│         Returns: Structured metadata                    │
│                                                          │
│  4. ReaderLM-v2 Conversion (8004)                       │
│     └─> Clean HTML → Markdown                           │
│         Returns: Formatted markdown text                │
│                                                          │
│  5. Structured Assembly                                 │
│     └─> Combine all data into PageContent dataclass     │
│         Returns: {content, metadata, links, images}     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Code Structure

### Data Classes (Type Safety)

```python
@dataclass
class PageMetadata:
    url: str
    title: Optional[str]
    author: Optional[str]
    site_name: Optional[str]
    published_date: Optional[str]
    word_count: int
    reading_time_minutes: int
    language: Optional[str]
    excerpt: Optional[str]

@dataclass
class PageLink:
    text: str
    url: str
    is_external: bool

@dataclass
class PageImage:
    url: str
    alt: Optional[str]
    title: Optional[str]

@dataclass
class PageContent:
    content: str
    metadata: PageMetadata
    links: List[PageLink]
    images: List[PageImage]
    sections: List[str]
```

### Processing Pipeline

```python
async def process_page(url, timeout, include_links, include_images):
    # 1. Fetch
    html = await fetch_html_with_playwright(url, timeout)

    # 2. Parse
    soup = BeautifulSoup(html, "lxml")

    # 3. Extract clean content
    readability_data = extract_with_readability(html, url)

    # 4. Extract metadata
    metadata = extract_metadata(soup, url, readability_data)

    # 5. Extract links/images (conditional)
    links = extract_links(soup, url) if include_links else []
    images = extract_images(soup) if include_images else []

    # 6. Extract structure
    sections = extract_sections(soup)

    # 7. Convert to markdown
    markdown = await convert_html_to_markdown(clean_html, url)

    # 8. Assemble structured result
    return PageContent(...)
```

## Benefits for gpt-oss-120b

### 1. **Consistent Tool Format**

All tools (`web_search`, `page_reader`) return dicts with similar structure:
```python
{
    "text" or "content": "...",  # Main text
    "sources" or "metadata": {...},  # Structured data
    ...
}
```

LLM learns one pattern, applies to all tools.

### 2. **Explicit Structure**

Instead of parsing markdown strings, LLM gets explicit fields:
- `metadata.title` - No need to extract from markdown header
- `metadata.word_count` - Immediate access, no counting
- `links[].is_external` - Pre-computed, no URL comparison needed

### 3. **Flexible Usage**

```python
# Full analysis
result = fetch_page(url, include_links=True, include_images=True)
# Access: result['metadata']['author'], result['links'], etc.

# Quick summary
text = fetch_page_text(url)
# Access: Just markdown string

# Link discovery
links = get_page_links(url, external_only=True)
# Access: links['links']
```

LLM can choose the right tool for the task.

### 4. **Reduced Token Usage**

**Before** (jina_reader):
```
# Article Title
Source: https://...
---
![image](...)
![image](...)
Content with many images...
```

**After** (page_reader):
```python
{
    "content": "# Article Title\n\nContent...",  # No image clutter
    "images": [...]  # Separate, optional
}
```

Markdown is cleaner, images are optional structured data.

## Performance Comparison

| Metric | Before (jina_reader.py) | After (page_reader.py) |
|--------|-------------------------|------------------------|
| **Output Format** | Plain string | Structured dict |
| **Metadata** | Embedded in markdown | Explicit fields |
| **Links** | All in markdown | Optional structured |
| **Images** | Inline markdown | Optional structured |
| **Content Extraction** | ReaderLM only | Readability + ReaderLM |
| **Ad Removal** | ReaderLM | Readability (better) |
| **API Dependency** | Optional fallback | None (100% local) |
| **Tool Count** | 3 tools | 4 tools (more focused) |

## Examples

### Example 1: Full Structured Analysis

```python
result = fetch_page(
    "https://blog.example.com/article",
    include_links=True,
    include_images=True
)

# Access structured data
print(result['metadata']['title'])        # "Article Title"
print(result['metadata']['author'])       # "John Doe"
print(result['metadata']['word_count'])   # 1234
print(f"{result['metadata']['reading_time_minutes']} min read")  # "5 min read"

# Process links
external_links = [l for l in result['links'] if l['is_external']]
print(f"Found {len(external_links)} external references")

# Work with content
content = result['content']  # Clean markdown
```

### Example 2: Quick Content Extraction

```python
text = fetch_page_text("https://docs.python.org/3/tutorial/")

# Just markdown string, ready to use
print(text)
# Output:
# # Python Tutorial
#
# Source: https://docs.python.org/3/tutorial/
#
# ---
#
# Python is an easy to learn...
```

### Example 3: Link Discovery

```python
links = get_page_links("https://news.ycombinator.com")

print(f"Found {links['link_count']} links")
for link in links['links'][:5]:
    print(f"- {link['text']}: {link['url']}")
```

## Migration Notes

### For Users

**Old way** (jina_reader):
```python
content = jina_fetch_page("https://example.com")
# Returns: "# Title\n\nContent..."
```

**New way** (page_reader):
```python
# Option 1: Get structure
result = fetch_page("https://example.com")
content = result['content']
title = result['metadata']['title']

# Option 2: Get simple text (backward compat)
content = fetch_page_text("https://example.com")
# Returns: "# Title\n\nSource: ...\n\n---\n\nContent..."
```

### Configuration Changes

**Removed** from `.env`:
```bash
USE_LOCAL_READER=true  # Always local now
JINA_API_KEY=...       # No longer needed
```

**Docker Compose**:
- Service renamed: `mcp-jina-reader` → `mcp-page-reader`
- Module: `jina_reader.py` → `page_reader.py`
- No environment variables needed (100% local)

## Dependencies Added

```
readabilipy==0.3.0      # Mozilla Readability (pure Python mode)
beautifulsoup4==4.12.3  # HTML parsing
lxml==5.1.0             # Fast XML/HTML parser backend
```

No Node.js required (readabilipy uses pure-Python mode).

## Future Enhancements

Possible improvements:
- [ ] Cache processed pages (Redis/sqlite)
- [ ] Batch processing (multiple URLs at once)
- [ ] PDF support (via pdf2html + ReaderLM)
- [ ] Screenshot capture (Playwright can do this)
- [ ] Custom CSS selector extraction
- [ ] Content diff (compare versions of same page)
- [ ] Summarization mode (truncate content to X words)

## References

- **ReadabiliPy**: https://github.com/alan-turing-institute/ReadabiliPy
- **Mozilla Readability**: https://github.com/mozilla/readability
- **ReaderLM-v2**: https://huggingface.co/jinaai/ReaderLM-v2
- **BeautifulSoup**: https://www.crummy.com/software/BeautifulSoup/
- **Playwright**: https://playwright.dev/python/
