from Insurance_Agent import agent

# Test the evaluate_response tool
test_query = "What is Bruce Wayne's property address?"
test_response = "Bruce Wayne's property address is 123 Main St, Gotham City."

print("Testing evaluate_response tool:")
try:
    result = agent.run(f"Evaluate this response: Query: {test_query} Response: {test_response}")
    print(f"Evaluation Result: {result}")
except Exception as e:
    print(f"Error: {e}")

print("\nTool added successfully. The agent can now evaluate responses.")