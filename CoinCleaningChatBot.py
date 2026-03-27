import streamlit as st
from openai import OpenAI
# import certifi
# import os
import pinecone
from pinecone import Pinecone, ServerlessSpec

# os.environ["SSL_CERT_FILE"] = certifi.where()
st.set_page_config(page_title="Coin Cleaning Consultant", page_icon="🪙")
st.title("🪙 Coin Cleaning Consultant")

@st.cache_resource
def init_clients():
    openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("ancient-coin-cleaning")
    return openai_client, index

client, index = init_clients()

CHAT_MODEL      = "gpt-5-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K           = 10 # Chunks to retrieve per query
SYSTEM_PROMPT   = """You are an expert ancient coin cleaning and artifact conservation assistant.
Answer questions using the context passages provided below each user message.
If the context doesn't contain enough information to answer confidently, ask
the user for clarification rather than guessing. Always cite which source
(document name and author) informed your answer. Don't ask for photos.

With silver coins, never suggest harsh mechanical tools.
With silvered coins, coins that are silver plated, always emphasize gentle care.
With Bronze coins, acids, electrolysis, etc are last resorts."""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "assistant",
            "content": "Welcome! I'm your coin cleaning and artifact conservation consultant. "
                       "Ask me anything about cleaning ancient coins, treating patinas, "
                       "removing corrosion, or conserving metal artifacts."
        }
    ]

# ── RAG retrieval ──────────────────────────────────────────────────────────────

def retrieve_context(query: str, top_k: int = TOP_K) -> tuple[str, list[dict]]:
    """
    Embeds the query and retrieves the top-k most relevant chunks from Pinecone,
    guaranteeing representation from at least 2 distinct source documents.

    Strategy:
      1. Fetch top_k results normally.
      2. Check how many distinct source documents are represented.
      3. If only 1 document appears, fetch additional chunks and inject the
         highest-scoring chunk from a second document, replacing the
         lowest-scoring chunk from the dominant document to keep total
         context size stable.
    """
    query_embedding = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL
    ).data[0].embedding

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    matches = results.matches

    # ── Enforce minimum 2-source diversity ────────────────────────────────────

    represented_docs = {m.metadata["source_file"] for m in matches}

    if len(represented_docs) < 2:
        # Fetch a larger pool and find the best chunk from a different document
        extended_results = index.query(
            vector=query_embedding,
            top_k=top_k * 6,   # cast a wider net
            include_metadata=True
        )
        # Find the highest-scoring match from any doc not already represented
        backup_match = next(
            (m for m in extended_results.matches
             if m.metadata["source_file"] not in represented_docs),
            None
        )
        if backup_match:
            # Drop the lowest-scoring match from the dominant doc to make room
            matches = sorted(matches, key=lambda m: m.score, reverse=True)
            matches[-1] = backup_match

    # ── Build context and source list ─────────────────────────────────────────

    # Re-sort by score descending so best chunks lead the context
    matches = sorted(matches, key=lambda m: m.score, reverse=True)

    context_blocks = []
    sources = []
    for match in matches:
        meta = match.metadata
        context_blocks.append(
            f"[Source: {meta['doc_name']} by {meta['author']}]\n{meta['text']}"
        )
        sources.append({
            "doc_name": meta["doc_name"],
            "author":   meta["author"],
            "score":    round(match.score, 3),
        })

    context_str = "\n\n---\n\n".join(context_blocks)
    return context_str, sources

def build_rag_user_message(user_query: str, context: str) -> str:
    """
    Wraps the user's query with retrieved context so the model
    can ground its answer in the source material.
    """
    return (
        f"CONTEXT FROM KNOWLEDGE BASE:\n\n{context}\n\n"
        f"---\n\nUSER QUESTION:\n{user_query}"
    )

for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    # Strip the injected context before displaying — show only the clean query
    display_content = message.get("display_content", message["content"])
    with st.chat_message(message["role"]):
        st.markdown(display_content)

if prompt := st.chat_input("Ask about coin cleaning or artifact conservation...", max_chars=1000):

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve relevant context from Pinecone
    with st.spinner("Searching knowledge base..."):
        context, sources = retrieve_context(prompt)

    # Build the context-enriched message for the LLM
    rag_message = build_rag_user_message(prompt, context)

    # Append to history with both the RAG-enriched content (sent to LLM)
    # and the clean display_content (shown in the UI)
    st.session_state.messages.append({
        "role":            "user",
        "content":         rag_message,   # what the LLM sees
        "display_content": prompt,        # what the user sees in the chat
    })

    # Generate and stream the response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_response += delta
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

        # Show source attribution in an expander below the response
        if sources:
            with st.expander("📚 Sources consulted", expanded=False):
                seen = set()
                for s in sources:
                    key = (s["doc_name"], s["author"])
                    if key not in seen:
                        st.markdown(f"- **{s['doc_name']}** — *{s['author']}* (relevance: {s['score']})")
                        seen.add(key)

    # Append assistant response to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_response,
    })