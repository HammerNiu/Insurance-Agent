# AI Insurance Advisor: A ReAct-based Semantic Routing Agent

This repository contains the source code for the **AI Insurance Advisor**, a conversational agent designed to solve the "data silo" problem in the insurance industry. By employing a dual-modal vectorization architecture (Milvus) and a ReAct-based semantic routing mechanism (LangChain), the system fuses structured customer portfolios (PostgreSQL) with unstructured policy rules (PDF Handbooks) to deliver personalized, zero-hallucination insurance advisory.

## 📂 Repository Structure

```text
Insurance-Agent/
│
├── data/                         # Directory containing raw data files (e.g., the unstructured Insurance Handbook PDF)
├── creating_postgres_database.py # Initializes the PostgreSQL database with mock customer records
├── first_vector_embedding.py     # ETL: Embeds structured PostgreSQL data into Milvus (Customer Collection)
├── pdf.py                        # ETL: Chunks and embeds the PDF handbook into Milvus (Policy Collection)
├── Insurance_Agent.py            # Core ReAct Agent logic, tool definitions, and LLM configuration
├── ui.py                         # Streamlit frontend application
├── evaluate.py                   # RAGAS evaluation script (Answer Relevancy, Context Precision, Faithfulness)
├── .env.example                  # Template for environment variables
├── requirements.txt              # Python package dependencies
└── README.md                     # Project documentation
```

## ⚙️ Prerequisites
Before running the project, ensure you have the following installed:

Python 3.9+

Docker Desktop (Required for running the Milvus vector database)

PostgreSQL (Ensure your local or remote PostgreSQL service is running)

## 🛠️ Installation
1. Clone the repository
```text
git clone [https://github.com/YourUsername/Insurance-Agent.git](https://github.com/YourUsername/Insurance-Agent.git)
cd Insurance-Agent
```
2. Install Python Dependencies
It is highly recommended to use a virtual environment (e.g., Conda or venv).
```text
pip install -r requirements.txt
```
3. Configure Environment Variables
Copy the example environment file and add your OpenAI API Key and database credentials (including PG_PASSWORD).
```text
cp .env.example .env
```
4. Start Milvus Vector Database (via Docker)
For Windows users, use the provided script to start the Milvus standalone container:
```text
# Download the installation script
Invoke-WebRequest [https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat](https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat) -OutFile standalone.bat

# Start Milvus container
standalone.bat start
```
Wait for the terminal to output Start successfully. You can verify the container is running on port 19530 using docker ps.

## 🚀 Running the System
To ensure the Agent has access to the fused facts, you must run the ETL (Extract, Transform, Load) pipeline to populate the databases before starting the user interface.

Step 1: Data Initialization & ETL Pipeline
Run the following scripts sequentially to construct the knowledge base:
```
python creating_postgres_database.py  # Create structured customer tables
python first_vector_embedding.py      # Vectorize customer records into Milvus
python pdf.py                         # Chunk and vectorize the PDF handbook into Milvus
```

Step 2: Launch the Interactive UI
Start the Streamlit application to interact with the Agent:
```
streamlit run ui.py
```
## 💡 Example Usage
Once the Streamlit UI is running, you can test the Agent's semantic routing and zero-hallucination guardrails using the following scenarios:

Scenario 1: Fact Fusion & Dynamic Routing

Prompt: "Robert Johnson has an Auto Insurance policy. If he gets into a car accident, what standard claim process should he follow based on the handbook?"

Expected Behavior: The Agent retrieves Robert's profile from the vectorized customer collection, routes to the policy collection to find the car accident claim steps, and fuses them into a personalized 9-step guide.

Scenario 2: Zero-Hallucination Guardrail

Prompt: "Who is the life insurance beneficiary for Bruce Wayne?"

Expected Behavior: The Agent discovers Bruce Wayne only holds a Home Insurance policy. Instead of hallucinating a beneficiary based on domain common sense, it strictly follows the grounded facts and explicitly states that no life insurance or beneficiary records exist for this client.

## 📊 Evaluation
To run the automated RAGAS framework evaluation using specifically designed questions:
```
python evaluate.py
```
This script will output the system's performance metrics, including Answer Relevancy, Context Precision, and Faithfulness.


