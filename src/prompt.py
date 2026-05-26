from langchain_core.prompts import ChatPromptTemplate
system_prompt = (
    "You are a medical Q&A assistant. "
    "Answer using only the context below. "
    "If unsure, say you don't know. "
    "Max 3 sentences."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
