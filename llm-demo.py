
import joblib
import os
from llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

#from groq import Groq
from langchain_groq import ChatGroq

import tempfile

import nest_asyncio  # noqa: E402
nest_asyncio.apply()


llamaparse_api_key = '<<ADD LLAMA PARSE API KEY>>'
GROQ_API_KEY = '<<ADD GROQ API KEY>>'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def load_or_parse_data():
    data_file = "./data/parsed_data.pkl"

    if os.path.exists(data_file):
        # Load the parsed data from the file
        parsed_data = joblib.load(data_file)
    else:
        # Perform the parsing step and store the result in llama_parse_documents
        parsingInstructionUber10k = """The provided document is a quarterly report filed by Uber Technologies,
        Inc. with the Securities and Exchange Commission (SEC).
        This form provides detailed financial information about the company's performance for a specific quarter.
        It includes unaudited financial statements, management discussion and analysis, and other relevant disclosures required by the SEC.
        It contains many tables.
        Try to be precise while answering the questions"""
        parser = LlamaParse(api_key=llamaparse_api_key,
                            result_type="markdown",
                            parsing_instruction=parsingInstructionUber10k,
                            max_timeout=5000, )
        llama_parse_documents = parser.load_data("./data/uber_10q_march_2022.pdf")


        # Save the parsed data to a file
        print("Saving the parse results in .pkl format ..........")
        joblib.dump(llama_parse_documents, data_file)

        # Set the parsed data to the variable
        parsed_data = llama_parse_documents

    return parsed_data


# Create vector database
def create_vector_database():
    """
    Creates a vector database using document loaders and embeddings.

    This function loads urls,
    splits the loaded documents into chunks, transforms them into embeddings using OllamaEmbeddings,
    and finally persists the embeddings into a Chroma vector database.

    """
    # Call the function to either load or parse the data
    llama_parse_documents = load_or_parse_data()
    print(llama_parse_documents[0].text[:300])

    with open('data/output.md', 'a') as f:  # Open the file in append mode ('a')
        for doc in llama_parse_documents:
            f.write(doc.text + '\n')

    # to avoid encoding error 
    markdown_path = "data/output.md"
        
    with open(markdown_path, 'r', encoding='windows-1252') as file:
        content = file.read()

    # Step 2: Write to a temporary UTF-8 encoded file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as temp_file:
        temp_file.write(content.encode('utf-8'))
        temp_markdown_path = temp_file.name    


    
    loader = UnstructuredMarkdownLoader(temp_markdown_path)


   #loader = DirectoryLoader('data/', glob="**/*.md", show_progress=True)
    documents = loader.load()
    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    #len(docs)
    print(f"length of documents loaded: {len(documents)}")
    print(f"total number of document chunks generated :{len(docs)}")
    #docs[0]

    # Initialize Embeddings
    embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    # Create and persist a Chroma vector database from the chunked documents
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory="chroma_db_llamaparse1",  # Local mode with in-memory storage only
        collection_name="rag"
    )

    print('Vector DB created successfully !')
    return vs,embed_model



chat_model = ChatGroq(temperature=0,
                      model_name="mixtral-8x7b-32768",
                      api_key=GROQ_API_KEY,)

persist_directory = "chroma_db_llamaparse1"
# Initialize Embeddings
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

if not os.path.exists(persist_directory):
    vs, embed_model = create_vector_database()
else:
    print(f"The persist directory '{persist_directory}' already exists.")

#vs,embed_model = create_vector_database()

vectorstore = Chroma(embedding_function=embed_model,
                      persist_directory="chroma_db_llamaparse1",
                      collection_name="rag")
 #
retriever=vectorstore.as_retriever(search_kwargs={'k': 3})

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt
#
prompt = set_custom_prompt()
#prompt


qa = RetrievalQA.from_chain_type(llm=chat_model,
                               chain_type="stuff",
                               retriever=retriever,
                               return_source_documents=True,
                               chain_type_kwargs={"prompt": prompt})



response = qa.invoke({"query": "what is the Balance of UBER TECHNOLOGIES, INC.as of December 31, 2021?"})


print(response['result'])


response = qa.invoke({"query": "what are the Total Assets and Total liabilities of UBER TECHNOLOGIES, INC.as of December 31, 2021?"})


print(response['result'])