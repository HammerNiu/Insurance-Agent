from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)

from dotenv import load_dotenv
from datasets import Dataset

from Insurance_Agent import agent, search_policy

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

questions = [

# Life Insurance
"What Is The Difference Between Basic Life Insurance And AD&D?",
"What Is The Best Permanent Life Insurance Policy?",

# Auto Insurance
"What Is Personal Automobile Insurance?",
"What Does Liability Auto Insurance Cover?",

# Home Insurance
"What Does Homeowners Insurance Cover?",
"What Is Loss Of Use Coverage?",

# # Health Insurance
# "What Does Short Term Health Insurance Cover?",
# "How Is Health Insurance Premium Calculated?",
# "Can Full Time Students Get Health Insurance?",

]

ground_truths = [

# Life Insurance
"Basic life insurance covers death from any cause, while AD&D only pays for accidental death or dismemberment.",
"The best permanent life insurance policy is one that fits the policyholder’s financial needs and remains affordable long term.",

# Auto Insurance
"Personal automobile insurance covers privately owned vehicles and provides liability and physical damage protection.",
"Liability auto insurance pays for bodily injury and property damage caused to others in an accident.",

# Home Insurance
"Homeowners insurance provides coverage for the home structure, personal belongings, liability protection and additional living expenses.",
"Loss of use coverage pays for additional living expenses if your home becomes uninhabitable.",

# # Health Insurance
# "Short-term health insurance covers major medical expenses such as ER visits, urgent care, doctor visits, prescription drugs, and hospital services.",
# "Under the Affordable Care Act, individual and small-group health insurance use community rating based on age.",
# "Yes. Students can get individual health insurance or stay on their parents' plan until age 26.",

]

answers = []
contexts = []

print("Running evaluation queries...\n")

for q in questions:

    print("Question:", q)

    # retrieval
    retrieved = search_policy.invoke({"query": q})

    context_text = str(retrieved)

    # agent answer
    response = agent.invoke({
        "messages": [{"role": "user", "content": q}]
    })

    answer = response["messages"][-1].content

    print("Answer:", answer[:120], "...\n")

    answers.append(answer)

    # contexts: list[list[str]]
    contexts.append([context_text])


dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})


print("\nRunning RAGAS evaluation...\n")

# LLM evaluator
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision
    ],
    llm=llm,
    embeddings=embeddings,
    raise_exceptions=False
)

print("\n========== RAG Evaluation ==========\n")

print(result)
