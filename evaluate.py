import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from datasets import Dataset

# Import the core agent and retrieval tools from your backend
from Insurance_Agent import agent, search_policy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# Load environment variables (API keys, DB credentials)
load_dotenv()

# ==============================================================================
# 1. DEFINE THE EVALUATION DATASET (GROUND TRUTH)
# A curated list of queries spanning both structured database intent and 
# unstructured policy handbook intent.
# ==============================================================================
questions = [
    "What are the two basic coverages included in a personal automobile insurance policy?",
    "What does homeowners insurance typically cover?",
    "What is the definition of a deductible in an insurance policy?",
    "What is the primary purpose of liability insurance?",
    "What is Loss of Use coverage in a homeowners policy?",
    "What should a policyholder do immediately after an auto accident according to standard claim processes?",
    "Does standard homeowners insurance cover flood damage?",
    "What is the difference between actual cash value and replacement cost?"
]

ground_truths = [
    "A personal automobile policy generally includes liability coverage (for damage caused to others) and physical damage coverage (for damage to the policyholder's own car).",
    "Homeowners insurance provides coverage for the structure of the home, personal belongings, liability protection, and additional living expenses.",
    "A deductible is the amount of loss paid by the policyholder before the insurance policy starts paying.",
    "Liability insurance is designed to protect the policyholder against financial loss if they are legally responsible for injuring someone else or damaging their property.",
    "Loss of Use coverage pays for additional living expenses, such as hotel bills or restaurant meals, if your home is destroyed or becomes uninhabitable.",
    "The policyholder should notify the police, exchange information with other drivers, document the scene, and promptly notify their insurance company to start the claims process.",
    "No, standard homeowners insurance policies typically do not cover flood damage; a separate flood insurance policy is required.",
    "Actual cash value pays the depreciated value of the item, while replacement cost pays the amount needed to replace the item with a new one of similar kind and quality."
]

answers = []
contexts = []

print("🚀 Starting Automated RAG Evaluation Pipeline...\n")

# ==============================================================================
# 2. AUTOMATED INFERENCE & CONTEXT RETRIEVAL
# ==============================================================================
for q in questions:
    print(f"Testing Query: {q}")
    
    # Step A: Isolate the retrieval context for RAGAS precision scoring
    try:
        retrieved = search_policy.invoke({"query": q})
        context_text = str(retrieved)
    except Exception as e:
        context_text = "Missing Context"
        
    # Step B: Invoke the LangChain Agent for the synthesized final answer
    try:
        # Utilizing the Tool Calling architecture with memory disabled/bypassed for stateless eval
        response = agent.invoke({"messages": [{"role": "user", "content": q}]})
        answer = response["messages"][-1].content
    except Exception as e:
        answer = "Error generating response"

    answers.append(answer)
    contexts.append([context_text])

# ==============================================================================
# 3. HUGGINGFACE DATASET FORMATTING
# ==============================================================================
dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

print("\n Invoking LLM-as-a-Judge (RAGAS) for multi-dimensional scoring...\n")

# Initialize the evaluator models (GPT-4o-mini for judging, text-embedding-3-small for vector math)
evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
evaluator_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Execute the evaluation across three core metrics
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    raise_exceptions=False
)

df = result.to_pandas()

# ==============================================================================
# 4. REPORT GENERATION & VISUALIZATION
# ==============================================================================
print("\n" + "="*50)
print("📊 FINAL RAG EVALUATION REPORT")
print("="*50)

# Calculate the mean scores across all queries
avg_scores = df[['faithfulness', 'answer_relevancy', 'context_precision']].mean()

print(f"• Faithfulness (Factuality):    {avg_scores['faithfulness']:.4f}")
print(f"• Answer Relevancy (On-topic):  {avg_scores['answer_relevancy']:.4f}")
print(f"• Context Precision (Search):   {avg_scores['context_precision']:.4f}")
print("="*50)

# Save detailed results to CSV for academic transparency
df.to_csv("evaluation_results.csv", index=False)



# Plotting the evaluation metrics
plt.figure(figsize=(8, 5))
# Professional color palette (Emerald, Blue, Violet)
colors = ['#10b981', '#3b82f6', '#8b5cf6'] 
bars = plt.bar(avg_scores.index.str.replace('_', ' ').str.title(), avg_scores.values, color=colors)

plt.ylim(0, 1.1)
plt.ylabel('Score (0.0 to 1.0)')
plt.title('RAG System Performance Metrics (RAGAS Framework)')

# Annotate the bars with exact numeric values
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
# Save the figure as a high-resolution PNG
plt.savefig('ragas_evaluation_metrics.png', dpi=300)

print("\n Visualization saved successfully as 'ragas_evaluation_metrics.png'.")