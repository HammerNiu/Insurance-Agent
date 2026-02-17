from pymilvus import connections, Collection
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PG_DB_URL = os.getenv("PG_DB_URL")
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
COLLECTION_NAME = "insurance_customers"
DIMENSION = 1536

connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

policy_collection = Collection("insurance_policy")
customer_collection = Collection("insurance_customers")

policy_collection.load()
customer_collection.load()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


@tool
def search_policy(query: str, top_k: int = 3) -> str:
    """Inquiring insurance clauses, insurance manuals, and insurance policy contents"""
    query_vector = embedding_model.embed_query(query)

    results = policy_collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"ef": 64}},
        limit=top_k,
        output_fields=["text"]
    )

    hits = results[0]
    return "\n\n".join(hit.entity.get("text") for hit in hits)


@tool
def search_customer(query: str, top_k: int = 3) -> str:
    """Inquiring customer information, including customer ID, name, policy type, and remarks"""

    query_vector = embedding_model.embed_query(query)

    results = customer_collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"ef": 64}},
        limit=top_k,
        output_fields=["customer_id", "customer_name", "policy_types", "metadata"]
    )

    hits = results[0]

    context = "\n\n".join([
        f"""
Customer ID: {hit.entity.get('customer_id')}
Name: {hit.entity.get('customer_name')}
Policy Types: {hit.entity.get('policy_types')}
Metadata: {hit.entity.get('metadata')}
"""
        for hit in hits
    ])

    return context


llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

agent = create_agent(
    llm,
    tools=[search_policy, search_customer],
)

if __name__ == "__main__":
    messages = []
    while True:
        question = input("\n Question: ")
        if question == "q":
            break

        messages.append(HumanMessage(content=question))
        response = agent.invoke({
            "messages": messages
        })

        answer = response["messages"][-1]
        print("\n Agent Answer：\n", answer.content)

        messages.append(answer)
