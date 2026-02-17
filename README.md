# Insurance-Agent
The insurance agent is designed to create an intelligent insurance advisory platform that leverages advanced language models and knowledge retrieval systems to provide specialized, domain-specific insurance expertise. 

# Milvus Installation via Docker

```powershell
# Download the installation script
Invoke-WebRequest https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat -OutFile standalone.bat

# Start Milvus container
standalone.bat start
Wait for Milvus starting...
Start successfully.

# Verify Milvus is running (port 19530 should be listening)
docker ps
```

# Usage
```powershell
python creating_postgres_database.py
python first_vector_embedding.py
python pdf.py
python Insurance_Agent.py
```
