import os
import uuid

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "insurance_policy"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # text-embedding-3-small 的维度



def connect_milvus():
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )
    print("✅ Milvus connected")


def create_collection_if_not_exists():
    if utility.has_collection(COLLECTION_NAME):
        print("ℹ️ Collection already exists")
        return Collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="page", dtype=DataType.INT64),
    ]

    schema = CollectionSchema(fields, description="PDF Knowledge Base")

    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema
    )

    # 创建索引（HNSW）
    index_params = {
        "metric_type": "IP",
        "index_type": "HNSW",
        "params": {
            "M": 8,
            "efConstruction": 64
        }
    }

    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )

    print("✅ Collection created")
    return collection


def load_and_split_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    docs = splitter.split_documents(documents)
    return docs


def insert_documents(collection: Collection, docs):
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    texts = [doc.page_content for doc in docs]
    embeddings = embedding_model.embed_documents(texts)

    ids = [str(uuid.uuid4()) for _ in texts]
    sources = [doc.metadata.get("source", "unknown") for doc in docs]
    pages = [doc.metadata.get("page", 0) for doc in docs]

    data = [
        ids,
        embeddings,
        texts,
        sources,
        pages
    ]

    collection.insert(data)
    collection.flush()

    print(f"✅ Inserted {len(texts)} chunks")



def search(collection: Collection, query: str, top_k: int = 3):
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    query_vector = embedding_model.embed_query(query)

    collection.load()

    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"ef": 64}},
        limit=top_k,
        output_fields=["text", "source", "page"]
    )

    return results[0]



def answer_question(collection: Collection, question: str):
    search_results = search(collection, question, top_k=3)

    context = "\n\n".join(
        [hit.entity.get("text") for hit in search_results]
    )

    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = f"""
    基于以下知识库内容回答问题：

    {context}

    问题：
    {question}
    """

    response = llm.invoke(prompt)
    return response.content



def build_knowledge_base(pdf_path: str):
    connect_milvus()
    collection = create_collection_if_not_exists()

    docs = load_and_split_pdf(pdf_path)
    insert_documents(collection, docs)

    print("🎉 Knowledge base ready!")


if __name__ == "__main__":
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    build_knowledge_base("data/Insurance_Handbook_removed.pdf")
