import os
import re
from dotenv import load_dotenv
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Load env
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

DB_FAISS_PATH = "vectorstore/db_faiss"

# Cache vectorstore
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Set up custom prompt
def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Load HuggingFace model
def load_chat_llm(repo_id):
    base_llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        task="text-generation"
    )
    return ChatHuggingFace(llm=base_llm)

def clean_response(text):
    return re.sub(r"</?\|?im_.*?\|?>|<s>|</s>", "", text).strip()

# Main UI
def main():
    st.set_page_config(page_title="MedBuddy - AI Medical Assistant", page_icon="💊")
    st.title("💊 MedBuddy: Your AI Medical Assistant")
    st.markdown("Ask any medical question, and I’ll do my best to answer using trusted context. 👩‍⚕️")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    prompt = st.chat_input("Type your medical question here...")

    if prompt:
        # Show user message
        st.chat_message("user").markdown(f"👤 **You:** {prompt}")
        st.session_state.messages.append({'role': 'user', 'content': f"👤 **You:** {prompt}"})

        # Prompt template
        CUSTOM_PROMPT_TEMPLATE = """
        You are a helpful medical assistant. Use only the information provided in the context to answer the user's question clearly, factually, and in 4–6 informative bullet points.

        If the answer is not found in the context, say: "The context does not contain enough information to answer this question."

        <context>
        {context}
        </context>

        Question: {question}

        Answer:
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("❌ Error loading vectorstore.")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_chat_llm(HUGGINGFACE_REPO_ID),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # Run query
            response = qa_chain.invoke({"query": prompt})
            result = clean_response(response["result"])
            sources = response["source_documents"]

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(f"🤖 **MedBuddy:**\n\n{result}")

                with st.expander("📄 View Source Documents"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source {i+1}**: Page {doc.metadata.get('page_label', 'N/A')}")
                        st.write(doc.page_content)

            st.session_state.messages.append({'role': 'assistant', 'content': f"🤖 **MedBuddy:**\n\n{result}"})

        except Exception as e:
            st.error(f"🚨 Error: {str(e)}")

if __name__ == "__main__":
    main()
