# import requests
# from bs4 import BeautifulSoup

# url = "https://chloewang-200.github.io/"

# response = requests.get(url)

# # Use BeautifulSoup to parse the HTML content
# soup = BeautifulSoup(response.text, 'lxml')

# # Extracts all text from the webpage
# text = soup.get_text(separator=' ', strip=True)

# with open('output.txt', 'w', encoding='utf-8') as file:
#     file.write(text)
import os
import json
from langchain.embeddings import OpenAIEmbeddings

from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Now you can access your OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

with open('chloeData.txt', 'r') as file:
    file_content = file.read()

# split into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1500, # amount of characters!
    chunk_overlap=500,
    length_function=len
)

chunks = text_splitter.split_text(file_content)

embeddings_model = OpenAIEmbeddings(api_key=openai_api_key)
embeddings = embeddings_model.embed_documents(chunks)

text_embedding_pairs = [
    {"text": chunks[i], "embedding": embeddings[i]}
    for i in range(len(chunks))
]

with open('text_embedding_pairs.json', 'w', encoding='utf-8') as f:
    json.dump(text_embedding_pairs, f, ensure_ascii=False, indent=4)
# for i in range(len(chunks)):
#     print(PURPLE + BOLD + f'CHUNK: {i}' + RESET)
#     print("\""+ chunks[i] + "\"\n")



