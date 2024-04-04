from flask import Response, session
import requests
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai
from langchain_openai import OpenAIEmbeddings

# Make sure to set the OPENAI_API_KEY environment variable in a .env file (create if does not exist) - see .env.example

class OpenAI:

    # def __init__(self):
    #     print("initializing openAI")

    #     with open('text_embedding_pairs.json', 'r', encoding='utf-8') as file:
    #         self.data = json.load(file)

    #     # Separate texts and embeddings
    #     self.texts = [item['text'] for item in self.data]
    #     self.embeddings = np.array([item['embedding'] for item in self.data])
    
    def generate_embedding(question):
        embeddings_model = OpenAIEmbeddings()
        question_embedding = np.array(embeddings_model.embed_query(question))
        return question_embedding
    
    def compare_embeddings(user_embedding, embeddings):
        # Assuming `stored_embeddings` is a numpy array of embeddings loaded previously
        similarities = cosine_similarity([user_embedding], embeddings)
        top_indices = np.argsort(similarities[0])[-4:][::-1]
        return top_indices
    
    def get_related_texts(texts, top_indices):
        # Fetch related texts for the top indices
        return [texts[i] for i in top_indices]
    
    @staticmethod
    def create_chat_body(body, stream=False):
        # Text messages are stored inside request body using the Deep Chat JSON format:
        # https://deepchat.dev/docs/connect
        print("creating chat body")
        user_messages = [msg for msg in body["messages"] if msg["role"] != "ai" and msg["role"] != "system"]
        context_messages = []
        with open('text_embedding_pairs.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        print("file loaded")
        embeddings = np.array([item['embedding'] for item in data])
        texts = [item['text'] for item in data]
        # find top 4 messages for all user messages
        print("finding contexts")
        
        embeddings_model = OpenAIEmbeddings()
        for user_msg in user_messages:
            # Generate embedding for the user's message text
            # user_embedding = generate_embedding(user_msg["content"])
            
            session['conversation_history'].append(f"User: {user_msg['text']}")
            session.modified = True
            print("model loaded")
            print(user_msg)
            question_embedding = np.array(embeddings_model.embed_query(user_msg["text"]))
            print("question")
            # Find the indices of the top 4 most similar embeddings
            # top_indices = compare_embeddings(user_embedding, embeddings)
            similarities = cosine_similarity([question_embedding], embeddings)
            top_indices = np.argsort(similarities[0])[-4:][::-1]
            # Retrieve the related texts based on the top_indices
            related_texts = [texts[i] for i in top_indices]
            # Combine related texts as context and add to context_messages
            context = " ".join(related_texts)
            context_messages.append({"role": "system", "content": context})
        # print("context", context)
        # print("contexts found")
        system_message = {
            "role": "system", 
            "content":  ("You are ChloeBot, an AI version of Chloe, a college student at CMU passionate about making fun things with art "
                         "and technology. You like texting with emojis"
                         "You're known for being friendly, approachable, and always ready to share your "
                         "experiences and insights. When answering questions, use a casual and concise tone and speak "
                         "in the first person to reflect your personal views and experiences. "
                         "you'll act as Chloe and you're chattimg with your friends. "
                         "Now, a friend is asking you a question:")
        }
        context_message = {
            "role": "assistant", 
            "content": (f"Use this information "
                        f"as context to answer the user question:\n\n{related_texts}.\n"
                        "Please stick to this context when answering the question. \
                                            Do not use any other knowledge you might have" )
        }

        user_assistant_messages = [
            {
                "role": "assistant" if message["role"] == "ai" else "user",
                "content": message["text"]
            } for message in body["messages"] if message["role"] != "system"
        ]

        # Combine the system message with user/assistant messages
        chat_body = {
            "model": "gpt-3.5-turbo",
            "messages": [system_message, context_message] + user_assistant_messages,
            "max_tokens": 100,  # Example token limit
            "temperature": 0.5,
            # "frequency_penalty": 1.0,
           
        }
        print("chat_body", chat_body)
        # chat_body = {
        #     "messages": [
        #         {
        #             "role": "system", "content":  "You are Chloe, a college student passionate\
        #                 about making fun things with art and technology. Your journey has taken \
        #                     you from design to coding and beyond. You're known for being friendly,\
        #                           approachable, and always ready to share your experiences and insights.\
        #                               When answering questions, use a casual tone and speak in the first \
        #                                 person to reflect your personal views and experiences. Remember, \
        #                                     you're helping friends learn more about you and your interests.\
        #                                         Now, a friend is asking you a question:"
        #         },
        #         {
        #             "role": "assistant" if message["role"] == "ai" else message["role"],
        #             "content": message["text"]
        #         } for message in body["messages"]
        #     ],
        #     "model": "gpt-3.5-turbo",
        # }
        if stream:
            chat_body["stream"] = True
        return chat_body

    def chat(self, body):
        print("chatting", body)
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.getenv("OPENAI_API_KEY")
        }
        print("here")
        chat_body = self.create_chat_body(body)
        print("chat_body", chat_body)
        # response = openai.Completion.create(
        #     model=chat_body['model'],
        #     prompt=[{"role": message["role"], "content": message["content"]} for message in chat_body['messages']],
        #     max_tokens=1000,
        #     temperature=0.7,  # Adjust based on how creative you want the responses to be
        #     n=1,
        #     stop=None,  # You can define stop sequences if needed
        #     user="user-id-or-session-id"  # Use a consistent user or session ID if tracking conversation state
        # )
        # print("response is", response)
        # result = response.choices[0]['message']['content']
        

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", json=chat_body, headers=headers)
        json_response = response.json()
        if "error" in json_response:
            raise Exception(json_response["error"]["message"])
        result = json_response["choices"][0]["message"]["content"]
        
        # Sends response back to Deep Chat using the Response format:
        # https://deepchat.dev/docs/connect/#Response
        return {"text": result}

    def chat_stream(self, body):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.getenv("OPENAI_API_KEY")
        }
        chat_body = self.create_chat_body(body, stream=True)
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", json=chat_body, headers=headers, stream=True)

        def generate():
            # increase chunk size if getting errors for long messages
            for chunk in response.iter_content(chunk_size=2048):
                if chunk:
                    if not(chunk.decode().strip().startswith("data")):
                        errorMessage = json.loads(chunk.decode())["error"]["message"]
                        print("Error in the retrieved stream chunk:", errorMessage)
                        # this exception is not caught, however it signals to the user that there was an error
                        raise Exception(errorMessage)
                    lines = chunk.decode().split("\n")
                    filtered_lines = list(
                        filter(lambda line: line.strip(), lines))
                    for line in filtered_lines:
                        data = line.replace("data:", "").replace(
                            "[DONE]", "").replace("data: [DONE]", "").strip()
                        if data:
                            try:
                                result = json.loads(data)
                                content = result["choices"][0].get("delta", {}).get("content", "")
                                # Sends response back to Deep Chat using the Response format:
                                # https://deepchat.dev/docs/connect/#Response
                                yield "data: {}\n\n".format(json.dumps({"text": content}))
                            except json.JSONDecodeError:
                                # Incomplete JSON string, continue accumulating lines
                                pass
        return Response(generate(), mimetype="text/event-stream")

    # By default - the OpenAI API will accept 1024x1024 png images, however other dimensions/formats can sometimes work by default
    # You can use an example image here: https://github.com/OvidijusParsiunas/deep-chat/blob/main/example-servers/ui/assets/example-image.png
    def image_variation(self, files):
        url = "https://api.openai.com/v1/images/variations"
        headers = {
            "Authorization": "Bearer " + os.getenv("OPENAI_API_KEY")
        }
        # Files are stored inside a files object
        # https://deepchat.dev/docs/connect
        image_file = files[0]
        form = {
            "image": (image_file.filename, image_file.read(), image_file.mimetype)
        }
        response = requests.post(url, files=form, headers=headers)
        json_response = response.json()
        if "error" in json_response:
            raise Exception(json_response["error"]["message"])
        # Sends response back to Deep Chat using the Response format:
        # https://deepchat.dev/docs/connect/#Response
        return {"files": [{"type": "image", "src": json_response["data"][0]["url"]}]}
