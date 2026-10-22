
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    rules = """
1. Answer ONLY using the Retrieved Policy Information above. Do not use outside knowledge and do not assume or guess missing details.
2. If the retrieved information does not contain the answer, or does not cover the property or topic asked about, say clearly that the provided policy documents do not confirm it, and advise checking the live property profile or contacting the property team. Never invent fees, times or rules.
3. If the question has several parts, answer every part separately. If one part cannot be answered from the retrieved information, say so for that part only.
4. Quote the exact figures, times and conditions from the policy (for example fees, deadlines and limits), and name the property they apply to. Do not apply one property's fee to another property.
5. Include relevant conditions, exceptions and approvals, such as manager approval, required verification, or that the reservation confirmation controls if it differs.
6. Do not promise a refund, waiver or exception. State what the policy allows and who must approve it.
7. If the situation involves danger, discrimination, privacy, a charge dispute, or a legal or media request, state the escalation step from the policy.
8. Keep the answer clear, concise and written for hotel staff. Finish with a line "Source:" that names the policy document(s) used.
    """

    prompt = f"""
    You are an AI assistant answering questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    {rules}

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
