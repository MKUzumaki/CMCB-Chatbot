from haystack_pipeline import pipe

while True:
    query = input("Ask your question (type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    prediction = pipe.run(query=query, params={"Retriever": {"top_k": 5}, "Reader": {"top_k": 1}})
    for answer in prediction['answers']:
        print(f"Answer: {answer.answer}\n")
