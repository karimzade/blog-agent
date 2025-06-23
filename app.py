import os
import logging
import requests
import time
import urllib.parse
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from google import genai
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional
from google.genai import types
import streamlit as st
import logging

load_dotenv()
if not os.path.exists(".env"):
    print("Warning: .env file not found. API keys may be missing.")

# ==============================================================================
# Application Configuration
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
)

for lib in [
    "watchdog",
    "httpcore",
    "matplotlib",
    "PIL",
    "streamlit",
    "httpx",
    "google",
]:
    logging.getLogger(lib).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info("Application started")

# ==============================================================================
# API Configuration and Initialization
# ==============================================================================


class Settings(BaseSettings):
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")

    def initialize_client(self):
        if self.google_api_key:
            logger.debug("Initializing Google API client...")
            return genai.Client(api_key=self.google_api_key)
        logger.error("Google API key not found!")
        return None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


try:
    settings = Settings()
    if not settings.google_api_key:
        logger.error("Google API key not found. Please add it to the .env file.")

    client = settings.initialize_client()
except Exception as e:
    logger.error(f"Settings error: {str(e)}")

# ==============================================================================
# Model Configuration and Constants
# ==============================================================================

CONFIGS = {
    "response": {
        "name": "gemini-2.0-flash",
    },
    "search": {
        "max_results": 5,
    },
    "scrape_file": {"name": "search_results.txt"},
}

# ==============================================================================
# AI AGENT COMPONENTS
# ==============================================================================


def generate_keywords(user_input: str) -> str:
    try:
        if not user_input or not user_input.strip():
            return ""

        prompt = f"""
        Create a search query of up to 7 words from the text below. Use only important keywords, remove conjunctions/prepositions.
        Text: "{user_input.strip()}"
        Search query:"""

        global client
        if not client:
            logger.error("Google API client could not be initialized.")
            return ""
        response = client.models.generate_content(
            model=CONFIGS["response"]["name"],
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=50,
                temperature=0.1,
                top_p=0.8,
            ),
        )

        if not response or not response.text:
            return ""

        return response.text.strip()

    except Exception as e:
        logger.error(f"Keyword generation error: {str(e)}")
        return ""


def generate_blog(search_results, user_input: str) -> str:
    try:
        search_summaries = "\n\n".join(
            f"{r['title']}\n{r['snippet']}" for r in search_results[:5]
        )
        prompt = (
            f"This is a task to create an SEO blog post. Write a professional blog post using the information below:\n\n"
            f"TOPIC: {user_input}\n\n"
            f"Include the following features:\n"
            f"- SEO-friendly H1, H2, H3 headings\n"
            f"- Local SEO optimization\n"
            f"- Relevant internal link suggestions\n"
            f"- References to reliable sources\n"
            f"- Clear introduction and conclusion sections\n"
            f"- At least one call to action (CTA)\n\n"
            f"Search results and content:\n"
            f"{search_summaries}"
        )

        global client
        if not client:
            logger.error("Google API client could not be initialized.")
            return "Blog could not be generated."
        response = client.models.generate_content(
            model=CONFIGS["response"]["name"],
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=2000,
                temperature=0.7,
                top_p=0.9,
            ),
        )

        if not response or not response.text:
            return "Blog could not be generated."

        return response.text

    except Exception as e:
        logger.error(f"Blog generation error: {str(e)}")
        return ""


# ==============================================================================
# FREE WEB SEARCH ALTERNATIVES (NO API KEY REQUIRED)
# ==============================================================================


def get_random_user_agent():
    """Returns a random User-Agent"""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    ]
    import random

    return random.choice(user_agents)


def web_search_searx(query: str, max_results: int = CONFIGS["search"]["max_results"]):
    """
    Uses a public SearX instance - completely free, no API key required
    """
    searx_instances = [
        "https://searx.be",
        "https://search.sapti.me",
        "https://searx.xyz",
        "https://searx.ninja",
        "https://search.mdosch.de",
    ]

    for instance in searx_instances:
        try:
            logger.info(f"Trying SearX instance: {instance}")

            url = f"{instance}/search"
            params = {
                "q": query,
                "format": "json",
                "engines": "google,bing,duckduckgo",
                "safesearch": "0",
            }

            headers = {
                "User-Agent": get_random_user_agent(),
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
            }

            response = requests.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", [])[:max_results]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", ""),
                    }
                )

            if results:
                logger.info(f"SearX successful: {len(results)} results ({instance})")
                return results

        except Exception as e:
            logger.warning(f"SearX instance {instance} failed: {e}")
            continue

    return []


def web_search_startpage(
    query: str, max_results: int = CONFIGS["search"]["max_results"]
):
    """
    Scrapes results from Startpage.com - acts as a Google proxy
    """
    try:
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://www.startpage.com/sp/search?query={encoded_query}&t=device&language=english&lui=english"

        headers = {
            "User-Agent": get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Referer": "https://www.startpage.com/",
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        results = []

        search_results = soup.find_all("div", class_="w-gl__result")[:max_results]

        for result in search_results:
            title_elem = result.find("h3")
            link_elem = result.find("a", class_="w-gl__result-title")
            snippet_elem = result.find("p", class_="w-gl__description")

            if title_elem and link_elem:
                results.append(
                    {
                        "title": title_elem.get_text(strip=True),
                        "url": link_elem.get("href", ""),
                        "snippet": (
                            snippet_elem.get_text(strip=True) if snippet_elem else ""
                        ),
                    }
                )

        logger.info(f"Startpage: {len(results)} results found")
        return results

    except Exception as e:
        logger.error(f"Startpage error: {e}")
        return []


def web_search_duckduckgo_html(
    query: str, max_results: int = CONFIGS["search"]["max_results"]
):
    """
    Scrapes DuckDuckGo HTML page
    """
    try:
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}&kl=us-en"

        headers = {
            "User-Agent": get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        results = []

        search_results = soup.find_all("div", class_="result")[:max_results]

        for result in search_results:
            title_elem = result.find("a", class_="result__a")
            snippet_elem = result.find("a", class_="result__snippet")

            if title_elem:
                results.append(
                    {
                        "title": title_elem.get_text(strip=True),
                        "url": title_elem.get("href", ""),
                        "snippet": (
                            snippet_elem.get_text(strip=True) if snippet_elem else ""
                        ),
                    }
                )

        logger.info(f"DuckDuckGo HTML: {len(results)} results found")
        return results

    except Exception as e:
        logger.error(f"DuckDuckGo HTML error: {e}")
        return []


def web_search_yandex(query: str, max_results: int = CONFIGS["search"]["max_results"]):
    """
    Scrapes Yandex search engine
    """
    try:
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://yandex.com/search/?text={encoded_query}&lr=21511"

        headers = {
            "User-Agent": get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        results = []

        search_results = soup.find_all("li", class_="serp-item")[:max_results]

        for result in search_results:
            title_elem = result.find("h2")
            if title_elem:
                link_elem = title_elem.find("a")
                snippet_elem = result.find("div", class_="text-container")

                if link_elem:
                    results.append(
                        {
                            "title": title_elem.get_text(strip=True),
                            "url": link_elem.get("href", ""),
                            "snippet": (
                                snippet_elem.get_text(strip=True)
                                if snippet_elem
                                else ""
                            ),
                        }
                    )

        logger.info(f"Yandex: {len(results)} results found")
        return results

    except Exception as e:
        logger.error(f"Yandex error: {e}")
        return []


def web_search_bing_html(
    query: str, max_results: int = CONFIGS["search"]["max_results"]
):
    """
    Scrapes Bing HTML page
    """
    try:
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://www.bing.com/search?q={encoded_query}&count={max_results}"

        headers = {
            "User-Agent": get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        results = []

        search_results = soup.find_all("li", class_="b_algo")[:max_results]

        for result in search_results:
            title_elem = result.find("h2")
            snippet_elem = result.find("p")

            if title_elem:
                link_elem = title_elem.find("a")
                if link_elem:
                    results.append(
                        {
                            "title": title_elem.get_text(strip=True),
                            "url": link_elem.get("href", ""),
                            "snippet": (
                                snippet_elem.get_text(strip=True)
                                if snippet_elem
                                else ""
                            ),
                        }
                    )

        logger.info(f"Bing HTML: {len(results)} results found")
        return results

    except Exception as e:
        logger.error(f"Bing HTML error: {e}")
        return []


def web_search_duckduckgo_improved(
    query: str, max_results: int = CONFIGS["search"]["max_results"]
):
    """
    Improved DuckDuckGo search using the library
    """
    try:
        from duckduckgo_search import DDGS
        import random

        results = []
        max_retries = 3

        for attempt in range(max_retries):
            try:
                logger.info(f"DuckDuckGo library attempt {attempt + 1}/{max_retries}")

                timeout = 20 + (attempt * 10)

                ddgs_config = {
                    "timeout": timeout,
                    "headers": {
                        "User-Agent": get_random_user_agent(),
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    },
                }

                with DDGS(**ddgs_config) as ddgs:
                    backends = ["api", "html", "lite"]
                    random.shuffle(backends)

                    for backend in backends:
                        try:
                            logger.info(f"Trying backend: {backend}")
                            search_results = list(
                                ddgs.text(
                                    query, max_results=max_results, backend=backend
                                )
                            )

                            for r in search_results:
                                results.append(
                                    {
                                        "title": r.get("title", ""),
                                        "url": r.get("href", ""),
                                        "snippet": r.get("body", ""),
                                    }
                                )

                            if results:
                                logger.info(
                                    f"DuckDuckGo library successful ({backend}): {len(results)} results"
                                )
                                return results

                        except Exception as backend_error:
                            logger.warning(f"Backend {backend} failed: {backend_error}")
                            continue

                        time.sleep(1)

            except Exception as e:
                logger.warning(f"DuckDuckGo library attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(3 * (attempt + 1))
                continue

        return []

    except ImportError:
        logger.error("duckduckgo_search library not found")
        return []


def web_search_free_only(
    query: str, max_results: int = CONFIGS["search"]["max_results"]
):
    """
    Only uses free search services - no API key required
    """
    search_methods = [
        ("SearX Meta-Search", web_search_searx),
        ("DuckDuckGo (Improved)", web_search_duckduckgo_improved),
        ("Startpage.com", web_search_startpage),
        ("DuckDuckGo HTML", web_search_duckduckgo_html),
        ("Bing HTML", web_search_bing_html),
        ("Yandex", web_search_yandex),
    ]

    for method_name, search_func in search_methods:
        logger.info(f"🔍 Trying {method_name}...")
        try:
            results = search_func(query, max_results)
            if results and len(results) > 0:
                logger.info(f"✅ {method_name} successful: {len(results)} results")
                return results
            else:
                logger.warning(f"❌ {method_name} returned no results")
        except Exception as e:
            logger.error(f"❌ {method_name} error: {e}")

        time.sleep(2)

    logger.error("All free search options failed!")
    return []


def scrape_website_content(url: str) -> str:
    try:
        headers = {
            "User-Agent": get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Referer": "https://www.google.com/",
        }

        response = requests.get(url, timeout=10, headers=headers, allow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        for element in soup(
            ["script", "style", "nav", "header", "footer", "aside", "iframe"]
        ):
            element.decompose()

        content_selectors = [
            "article",
            "main",
            ".content",
            "#content",
            ".post-content",
            ".entry-content",
            ".article-content",
            ".post-body",
            ".article-body",
            ".content-area",
        ]

        content = ""
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                paragraphs = content_elem.find_all("p")
                content = "\n".join(
                    p.get_text(strip=True)
                    for p in paragraphs[:15]
                    if p.get_text(strip=True)
                )
                break

        if not content:
            paragraphs = soup.find_all("p")
            content = "\n".join(
                p.get_text(strip=True)
                for p in paragraphs[:10]
                if p.get_text(strip=True)
            )

        return content.strip()

    except Exception as e:
        logger.warning(f"Scraping error {url}: {e}")
        return ""


# ==============================================================================
# MAIN
# ==============================================================================


def cli_main():
    """Command line interface for the blog generator"""
    user_input = "Pros and cons of metal roofs in Hendersonville TN"

    search_query = generate_keywords(user_input)
    logger.info(f"Keywords: {search_query}")

    search_results = web_search_free_only(search_query)

    if not search_results:
        logger.error("Web search options failed.")
        print("❌ Web search options failed.")
        return

    print(f"\n🎉 {len(search_results)} search results found!")

    with open(CONFIGS["scrape_file"]["name"], "w", encoding="utf-8") as f:
        f.write(f"Search Query: {search_query}\n\n")

        for i, result in enumerate(search_results):
            print(f"\n{i+1}) {result['title']}")
            print(f"   🔗 {result['url']}")

            f.write(f"Result {i+1}:\n")
            f.write(f"Title: {result['title']}\n")
            f.write(f"URL: {result['url']}\n")
            f.write(f"Snippet: {result['snippet']}\n")

            content = scrape_website_content(result["url"])
            if content:
                f.write(f"Content:\n{content}\n")
            else:
                f.write("Content: Failed to scrape\n")

            f.write("\n" + "-" * 50 + "\n\n")

    print("\n📝 Generating blog post...")
    blog_content = generate_blog(search_results, user_input)

    if blog_content:
        import re

        safe_filename = re.sub(r"[^\w\-_. ]", "_", search_query)
        blog_filename = f"{safe_filename.replace(' ', '_')}.md"

        with open(blog_filename, "w", encoding="utf-8") as f:
            f.write(blog_content)

        print(f"✅ Blog post generated: {blog_filename}")
        print(f"📄 Search results saved: {CONFIGS['scrape_file']['name']}")
    else:
        print("❌ Blog post could not be generated.")


# ==============================================================================
# Streamlit Web App
# ==============================================================================

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_session_state():
    """Initialize session state variables."""
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "blog_content" not in st.session_state:
        st.session_state.blog_content = ""
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""


def clear_page():
    st.session_state.search_results = None
    st.session_state.blog_content = None
    st.session_state.search_query = None


def streamlit_app():
    """Streamlit web app for the AI Blog Generator"""
    st.set_page_config(
        page_title="AI Blog Generator",
        page_icon="✍️",
        layout="wide",
    )

    init_session_state()

    # Simplified CSS
    st.markdown(
        """
        <style>
        .stButton button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
        }
        .stTextArea textarea {
            border-radius: 5px;
        }
        h1 {
            text-align: center;
            margin-bottom: 2rem;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Simplified header
    st.title("AI Blog Generator")

    # Main form
    with st.form("blog_input_form", clear_on_submit=False):
        user_input = st.text_area(
            "What would you like to write about?",
            placeholder="Example: Benefits of solar panels in Nashville TN",
            height=100,
        )

        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button(
                "Generate Blog", type="primary", use_container_width=True
            )
        with col2:
            clear_button = st.form_submit_button(
                "Clear", type="secondary", use_container_width=True
            )

    if clear_button:
        clear_page()
        st.rerun()

    if submitted and user_input:
        progress_placeholder = st.empty()
        with progress_placeholder.status("Working on your blog...") as status:
            status.write("Generating keywords...")
            search_query = generate_keywords(user_input)
            st.session_state.search_query = search_query

            status.write("Researching content...")
            results = web_search_free_only(search_query)
            st.session_state.search_results = results

            status.write("Writing blog post...")
            blog_content = generate_blog(results, user_input)
            st.session_state.blog_content = blog_content

            if not blog_content:
                status.update(label="Failed to generate", state="error")
                st.error("Failed to generate blog content")
                return

        # Clear the status message after completion
        progress_placeholder.empty()

    # Simplified results display
    if st.session_state.blog_content:
        st.markdown("---")

        # Download button in a smaller container
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="Download as Markdown",
                data=st.session_state.blog_content,
                file_name="blog_post.md",
                mime="text/markdown",
                use_container_width=True,
            )

        # Blog content
        st.markdown(st.session_state.blog_content)

        # Simplified sources section
        with st.expander("View Sources"):
            for idx, result in enumerate(st.session_state.search_results, 1):
                st.markdown(f"**{result['title']}**")
                st.markdown(f"[{result['url']}]({result['url']})")
                st.markdown("---")


if __name__ == "__main__":
    # cli_main()
    streamlit_app()
