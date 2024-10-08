import streamlit as st
from langchain_community.document_loaders import SitemapLoader
from fake_useragent import UserAgent

ua = UserAgent()


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    try:
        loader = SitemapLoader(url)
        loader.requests_per_second = 3
        loader.headers = {"User-Agent": ua.random}
        docs = loader.load()
        return docs
    except Exception:
        return []


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🌐",
)

st.title("SiteGPT")

st.markdown(
    """
  Ask questions about the content of a website.

  Start by writing the URL of the website on the sidebar.
"""
)

with st.sidebar:
    url = st.text_input(
        "Write down a sitemap URL",
        placeholder="https://example.com/sitemap.xml",
    )

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a sitemap URL.")
    else:
        docs = load_website(url)
        if docs:
            st.write(docs)
        else:
            st.error(
                "Failed to load documents from the sitemap. Please check the"
                " URL and try again."
            )
