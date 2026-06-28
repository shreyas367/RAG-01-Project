from src.rag import ask_question

while True:

    query = input("\nAsk: ")

    if query.lower() == "exit":
        break

    result = ask_question(query)

    print("\nANSWER:\n")
    print(result["answer"])

    print("\nSources:\n")

    for source in result["sources"]:

        print(
            f"📄 {source['document']}  |  Page {source['page']}"
        )