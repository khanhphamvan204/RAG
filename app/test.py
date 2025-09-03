from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},  # dùng CPU
    encode_kwargs={'normalize_embeddings': True}
)

# Test thử
emb = embedding_model.embed_query("Test")
print(len(emb), emb[:5])
