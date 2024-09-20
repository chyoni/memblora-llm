# Memblora App (Memblora LLM)

### What is memblora ?
- 새로 배운 내용, 트러블 슈팅, 레퍼런스 등등을 위해 작성한 블로그 포스팅
- 그러나, 포스팅이 가면 갈수록 많아지고 100개, 200개, 300개가 넘어가니 내가 작성한 글이 어떤 글인지 찾기가 어려워졌다.
- 이럴때, 내가 작성한 글을 주제나 관련 내용만 살짝 알려주면 포스팅 제목을 알려주는 ChatBot이 있다면?

### Environments
- Python, Django
- LangChain
- Pinecone
- Ollama, Gemma

### How to Use?
- 개별 HTML 파일 또는 HTML 파일을 압축한 Zip 파일을 업로드 하면 해당 파일의 문서 데이터를 벡터 데이터베이스에 임베딩
- 벡터 데이터베이스에 유사도 검색을 통해 질문과 관련된 블로그 포스팅을 찾는다.
- Gemma가 찾은 블로그 제목을 질문자에게 알려준다.

