import streamlit as st
from langchain_community.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1,
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question.
    If you can't just say you don't know, don't make anything up.

    Then, give a score to the answer between 0 and 5.
    If the answer answers the user question the score should be high,
    else it should be low.
    Make sure to always include the answer's score even if it's 0.
    Context: {context}

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!
    Question: {question}
"""
)

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers
            to answer the user's question.
            Use the answers that have the highest score (more helpful)
            and favor the most recent ones.
            Cite sources and return the sources of the answers as they are,
            do not change them.
            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm

    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
            }
            for doc in docs
        ],
    }


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\n\nSource:{answer['source']}\n\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")


@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    try:
        loader = SitemapLoader(
            url,
            filter_urls=[
                r"^(.*\/en-gb\/).*",
            ],
            parsing_function=parse_page,
        )
        loader.requests_per_second = 3
        docs = loader.load_and_split(text_splitter=splitter)
        vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
        return vector_store.as_retriever()
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
        retriever = load_website(url)
        query = st.text_input("Ask a question to the website.")
        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )

            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))  # noqa: W605
