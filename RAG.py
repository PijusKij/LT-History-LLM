import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# === Load the CSV and clean text ===
csv_path = "/Users/pijuskijauskas/Desktop/lthist_llm/lthist_wiki.csv"
df = pd.read_csv(csv_path)
texts = df['Text'].fillna('').astype(str).tolist()

# === Load new embedding model: BGE Large ===
embedder = SentenceTransformer("BAAI/bge-large-en")
embedder.max_seq_length = 512  # recommended for this model

# === Add retrieval instruction and embed texts ===
instruction = "Represent this document for retrieval: "
texts_with_instruction = [instruction + t for t in texts]
print("Embedding documents with BGE Large...")
embeddings = embedder.encode(texts_with_instruction, batch_size=32, show_progress_bar=True, convert_to_numpy=True).astype("float32")

# === Build and save FAISS index ===
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, "wikipedia_index.faiss")

# === Save metadata (title, text, url) ===
df[['Title', 'Text', 'URL']].to_csv("wikipedia_metadata.csv", index=False)
print("Indexing complete. Saved FAISS index and metadata.")

# === Reload FAISS and metadata for querying ===
index = faiss.read_index("wikipedia_index.faiss")
df = pd.read_csv("wikipedia_metadata.csv")

# === Reload embedder for query encoding ===
embedder = SentenceTransformer("BAAI/bge-large-en")
embedder.max_seq_length = 512

# === Load Mistral-7B-Instruct generator ===
model_id = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    load_in_4bit=True  # requires bitsandbytes
)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# === Helper: Build prompt for Mistral
def build_prompt(context, query):
    return f"""[INST] You are an expert in Lithuania's history. Use only the context below to answer the user's question. 
If the answer is not in the context, respond: "The answer is not in the provided context."

Context:
{context}

Question: {query}
Answer: [/INST]"""

# === RAG function ===
def rag_query(query, top_k=3, max_context_chars=1500, return_sources=True):
    # Embed the query
    query_instruction = "Represent this question for retrieving relevant documents: "
    query_embedding = embedder.encode([query_instruction + query], convert_to_numpy=True).astype("float32")

    # FAISS search
    _, indices = index.search(query_embedding, top_k)
    selected_rows = df.iloc[indices[0]]

    # Build context
    docs = selected_rows["Text"].tolist()
    context = "\n\n".join(docs)[:max_context_chars]

    # Build and send prompt to Mistral
    prompt = build_prompt(context, query)
    response = generator(prompt, max_new_tokens=200, do_sample=False, temperature=0)[0]["generated_text"]
    answer = response.split("[/INST]")[-1].strip()

    # Get source links
    sources = selected_rows[["Title", "URL"]].values.tolist()
    if return_sources:
        return answer, sources
    else:
        return answer

# === Example usage ===
if __name__ == "__main__":
    user_query = "What was the first president of the independent lithuania"
    answer, source_links = rag_query(user_query)

    print("\n🧠 Answer:\n", answer)
    print("\n🔗 Sources:")
    for i, (title, url) in enumerate(source_links, 1):
        print(f"{i}. {title} - {url}")
