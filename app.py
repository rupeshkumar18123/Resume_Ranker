import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings

#load env file
load_dotenv()

#llm 

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),          # resource URL
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), # your model deployment name
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),                  # key1 or key2
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),          # e.g. 2024‑06‑01‑preview
    model_name="gpt-4o",
    temperature=0.3,
)


#embedding model
embedding_model= AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("embedding_AZURE_OPENAI_API_BASE"),          # resource URL
    azure_deployment=os.getenv("embedding_AZURE_OPENAI_DEPLOYMENT_NAME"), # your model deployment name
    api_key=os.getenv("embedding_AZURE_OPENAI_API_KEY"),                  # key1 or key2
    api_version=os.getenv("embedding_AZURE_OPENAI_API_VERSION"),          # e.g. 2024‑06‑01‑preview

)

#streamlit ui
st.set_page_config(page_title="Resume Ranker", page_icon=":robot_face:" , layout="wide")
st.title("Resume Ranker using Azure OpenAI")

#job description input
job_input_method = st.radio("Select Job Description input method:", ["Type manually", "Upload JD PDF"])

job_description = ""

if job_input_method == "Type manually":
    job_description = st.text_area("Enter Job Description:", height=200)
elif job_input_method == "Upload JD PDF":
    uploaded_file = st.file_uploader("Upload Job Description PDF", type=["pdf"] , key="jd")
    if uploaded_file is not None:
        jd_reader = PdfReader(uploaded_file)
        for page in jd_reader.pages:
            job_description += page.extract_text()

#resume file input
resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"], key="resume", accept_multiple_files=True)

#process button
if st.button("🚀 Rank Resumes"):
    if not job_description.strip():
        st.error("Please provide a Job Description.")
        st.stop()
    elif not resume_file:
        st.error("Please upload at least one Resume PDF.")
        st.stop()
    #promt template
    prompt_template = PromptTemplate(
        input_variables=["resume","jd"],
        template="""You are a professional recruiter. Compare the following resume against the job description.
        Job Description
        {jd}
        Resume:
        {resume}
        
        Return only a score (out of 100) and a short explanation.
        Format strictly like:
        Score: <number>
        Reason: <brief reason>
        """

    )

    chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )

    results = []
    for file in resume_file:
        resume_reader = PdfReader(file)
        resume_text = "\n".join(page.extract_text() for page in resume_reader.pages if page.extract_text())

        try:
            response = chain.invoke({"resume": resume_text, "jd": job_description})
            response_text = response.get("text") if isinstance(response, dict) else response

            score = 0
            reason = "Could not extract score."

            for line in response_text.splitlines():
                if "score:" in line.lower():
                    try:
                        score = int(line.lower().split("score:")[1].strip().split()[0])
                    except:
                        score = 0
                if "reason:" in line.lower():
                    reason = line.split(":", 1)[1].strip()

            results.append((file.name, score, reason))

        except Exception as e:
            results.append((file.name, 0, f"Error: {str(e)}"))
        # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)

    st.subheader("📊 Ranked Resumes")
    for i, (name, score, reason) in enumerate(results, 1):
        st.markdown(f"**{i}. {name}** - Score: {score}")
        st.markdown(f"_Reason_: {reason}\n")
