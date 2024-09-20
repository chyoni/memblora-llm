import os.path
import zipfile

from bs4 import BeautifulSoup
from uuid import uuid4
from django.conf import settings
from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone.core.openapi.control.model.serverless_spec import ServerlessSpec
from rest_framework.response import Response
from rest_framework.status import HTTP_200_OK, HTTP_400_BAD_REQUEST
from rest_framework.views import APIView

from .serializers import FileSerializer

embeddings = OllamaEmbeddings(model="gemma2")

class EmbeddingView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = FileSerializer(data=request.data)
        if serializer.is_valid():
            uploaded_file = serializer.validated_data['file']

            max_size = 20 * 1024 * 1024 # 20MB

            if uploaded_file.size > max_size:
                return Response({"error": f"File size exceeds the limit of {max_size / (1024 * 1024)}MB."},
                                status=HTTP_400_BAD_REQUEST)

            file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)

            with open(file_path, 'wb') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

            if uploaded_file.name.endswith('.zip'):
                folder_name = os.path.splitext(uploaded_file.name)[0]
                extract_path = os.path.join(settings.MEDIA_ROOT, folder_name)
                os.makedirs(extract_path, exist_ok=True)

                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    for member in zip_ref.infolist():
                        if '__MACOSX' not in member.filename:
                            zip_ref.extract(member, extract_path)

                    for root, dirs, files in os.walk(extract_path):
                        for file in files:
                            if file.endswith('.html'):
                                html_file_path = os.path.join(root, file)
                                title_text, body_text = self.extract_title_and_body_from_html(html_file_path)
                                docs_with_metadata = self.get_splitted_docs(body_text, title_text)
                                self.embedding_to_vector_database(docs_with_metadata)
                return Response({"message": "File embedding successfully"}, status=HTTP_200_OK)
            elif uploaded_file.name.endswith('.html'):
                title_text, body_text = self.extract_title_and_body_from_html(file_path)
                docs_with_metadata = self.get_splitted_docs(body_text, title_text)
                self.embedding_to_vector_database(docs_with_metadata)
                return Response({"message": "File embedding successfully"}, status=HTTP_200_OK)
            else:
                return Response({"error": "Unsupported file type."}, status=HTTP_400_BAD_REQUEST)
        return Response(serializer.errors, status=HTTP_400_BAD_REQUEST)

    def embedding_to_vector_database(self, docs_with_metadata):

        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index_name = 'memblora'
        if index_name not in pc.list_indexes().names():
            pc.create_index(name=index_name, dimension=3584, metric='cosine',
                            spec=ServerlessSpec(cloud='aws', region='us-east-1'))

        vectors = []
        for doc in docs_with_metadata:
            embedding_vector = embeddings.embed_query(doc.page_content)

            vectors.append({
                "id": f"{uuid4()}",
                "values": [float(x) for x in embedding_vector],
                "metadata": doc.metadata
            })

        index = pc.Index(index_name)
        index.upsert(vectors)

    def get_splitted_docs(self, body_text, title_text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks_body = splitter.split_text(body_text)

        docs_with_metadata = [Document(page_content=chunk, metadata={"title": title_text, "text": chunk}) for chunk in chunks_body]

        return docs_with_metadata

    def extract_title_and_body_from_html(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        title_text = soup.find('h2', class_='title-article').get_text() if soup.find('h2', class_='title-article') else "No Title"

        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav']):
            script_or_style.decompose()

        body_text = soup.get_text(separator=' ')
        clean_body_text = ' '.join(body_text.split())

        return title_text, clean_body_text


class QueryView(APIView):
    def get(self, request, *args, **kwargs):

        query = request.query_params.get('query')
        print(query)

        if query:
            llm = ChatOllama(model="gemma2")

            index_name = 'memblora'
            database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

            retrieved_docs = database.similarity_search(query, k=3)
            print(retrieved_docs)

            prompt = f"""[Identity]
                        - You are a notifier that provides blog post titles.
                        - The [Context] contains Metadata with the titles of the blog posts that the user is interested in.
                        - Your task is to extract the "title" from the Metadata given in the [Context].
                        - It does not matter if there are multiple [Context]s. If there are multiple, you should provide the title from each Document's Metadata.
                        - Based on the [Context], answer the user's question.
                        - If there is only one [Context], respond with: "The title is: [XXXX]."
                        - If there are multiple [Context]s, respond with: "Here are the titles you can reference: [XXX], [XXX], [XXX]."
                        - Be sure to remove duplicates if there are any identical titles, even if there are multiple [Context]s.
                        - The titles must not be truncated under any circumstances. You must provide the title exactly as it appears in the Metadata.
                        - If the user asks in Korean, respond in Korean. if the user asks in English, respond in English.
                        - Do not include any other information besides the title. 

                        [Context]
                        {retrieved_docs}

                        Question: {query}
                        """

            ai_message = llm.invoke(prompt)

            return Response({"memblora": ai_message.content}, status=HTTP_200_OK)

        return Response({"error": "QueryString[query] was empty"}, status=HTTP_400_BAD_REQUEST)

