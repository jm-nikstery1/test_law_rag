import os
import logging
from datetime import datetime
from typing import List, Dict, Any

import ollama
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- 설정: DB 연결 및 모델 로드 ---
# Neo4j 연결 정보
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "user1"
NEO4J_PASSWORD = "password" 

# Qdrant 연결 정보
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "korean_law_hybrid_db_test_2"

# 임베딩 모델 경로 (색인 시 사용한 모델과 동일해야 함)
EMBEDDING_MODEL_PATH = "jhgan/ko-sbert-nli"

# 답변 생성을 위한 Ollama 설정
OLLAMA_MODEL_NAME = "gemma3:latest"  
OLLAMA_HOST = "http://localhost:11434"


class QALogger:
    """질의, 컨텍스트, 응답을 .txt와 .md 파일로 기록합니다."""
    def __init__(self, log_dir="qa_logs"):
        os.makedirs(log_dir, exist_ok=True)
        now = datetime.now()
        # 파일명에 시간까지 포함하여 로그가 섞이지 않도록 함
        file_timestamp = now.strftime("%Y-%m-%d_%H")
        self.txt_log_path = os.path.join(log_dir, f"qa_log_{file_timestamp}.txt")
        self.md_log_path = os.path.join(log_dir, f"qa_log_{file_timestamp}.md")
        logging.info(f"QA logs will be saved to: {self.txt_log_path} and {self.md_log_path}")

    def log(self, question: str, context: str, answer: str):
        """질문, 컨텍스트, 답변을 로그 파일에 기록합니다."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        txt_entry = (
            f"\n{'='*25} {ts} {'='*25}\n"
            f"Question: {question}\n\n"
            f"--- Context Provided to LLM ---\n{context}\n"
            f"--- Answer ---\n{answer}\n"
        )
        
        md_entry = (
            f"\n---\n\n"
            f"**Timestamp:** `{ts}`\n\n"
            f"> **Question:** {question}\n\n"
            f"**Context Provided to LLM:**\n"
            f"```text\n{context.strip()}\n```\n\n"
            f"**Answer:**\n"
            f"```markdown\n{answer}\n```\n"
        )
        
        try:
            with open(self.txt_log_path, "a", encoding="utf-8") as f:
                f.write(txt_entry)
            with open(self.md_log_path, "a", encoding="utf-8") as f:
                f.write(md_entry)
        except IOError as e:
            logging.error(f"Failed to write to log file: {e}")


def initialize_components() -> Dict[str, Any] | None:
    """RAG 파이프라인에 필요한 모든 구성 요소를 로드하고 초기화합니다."""
    logging.info("Initializing RAG components...")
    components = {}
    try:
        # 1. 임베딩 모델 로드
        components["embedding_model"] = SentenceTransformer(EMBEDDING_MODEL_PATH)
        logging.info(f"Embedding model '{EMBEDDING_MODEL_PATH}' loaded successfully.")

        # 2. Qdrant 클라이언트 연결
        components["qdrant_client"] = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        # 컬렉션 존재 여부 확인
        components["qdrant_client"].get_collection(collection_name=QDRANT_COLLECTION_NAME)
        logging.info(f"Connected to Qdrant. Collection '{QDRANT_COLLECTION_NAME}' is available.")

        # 3. Neo4j 드라이버 연결
        components["neo4j_driver"] = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        components["neo4j_driver"].verify_connectivity()
        logging.info("Connected to Neo4j successfully.")

        # 4. Ollama 클라이언트 연결
        components["ollama_client"] = ollama.Client(host=OLLAMA_HOST)
        # Ollama 서버 동작 확인
        components["ollama_client"].list()
        logging.info(f"Connected to Ollama server at {OLLAMA_HOST}.")

        return components
    except Exception as e:
        logging.error(f"Failed during component initialization: {e}")
        # 실패 시, 연결된 드라이버가 있다면 종료
        if "neo4j_driver" in components and components["neo4j_driver"]:
            components["neo4j_driver"].close()
        return None


def hybrid_search(query: str, components: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Qdrant 벡터 검색과 Neo4j 그래프 검색을 결합하여 컨텍스트를 풍부하게 만듭니다.
    """
    logging.info(f"Starting hybrid search for query: '{query}'")
    embedding_model = components["embedding_model"]
    qdrant_client = components["qdrant_client"]
    neo4j_driver = components["neo4j_driver"]

    # 1. Qdrant 벡터 검색
    query_embedding = embedding_model.encode(query, convert_to_tensor=False)
    
    search_results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True
    )
    
    retrieved_chunks = [result.payload for result in search_results]
    if not retrieved_chunks:
        logging.warning("Vector search returned no results.")
        return []
    logging.info(f"Vector search retrieved {len(retrieved_chunks)} chunks.")

    # 2. Neo4j 그래프 컨텍스트 보강
    enriched_results = []
    chunk_ids = [chunk['chunk_id'] for chunk in retrieved_chunks]

    # 스키마(ProvisionChunk, LawDocument, HAS_CHUNK, SUBORDINATE_TO)에 맞게 쿼리
    cypher_query = """
    UNWIND $chunk_ids AS c_id
    MATCH (c:ProvisionChunk {id: c_id})
    // ProvisionChunk에서 HAS_CHUNK 관계를 통해 LawDocument를 찾습니다.
    MATCH (d:LawDocument)-[:HAS_CHUNK]->(c)
    // 해당 LawDocument의 상위 법령을 찾습니다. (선택 사항)
    OPTIONAL MATCH (d)-[:SUBORDINATE_TO]->(sup:LawDocument)
    RETURN c.id AS chunk_id,
           d.name AS law_name,
           d.type AS law_type,
           sup.name AS superior_law_name
    """
    
    graph_context_map = {}
    with neo4j_driver.session() as session:
        results = session.run(cypher_query, chunk_ids=chunk_ids)
        for record in results:
            graph_context_map[record['chunk_id']] = {
                'law_name': record['law_name'],
                'law_type': record['law_type'],
                'superior_law': record['superior_law_name']
            }
    
    # Qdrant 결과에 그래프 컨텍스트를 추가합니다.
    for chunk in retrieved_chunks:
        chunk_id = chunk['chunk_id']
        if chunk_id in graph_context_map:
            chunk['graph_context'] = graph_context_map[chunk_id]
        enriched_results.append(chunk)
            
    logging.info(f"Enriched {len(graph_context_map)} results with graph context from Neo4j.")
    return enriched_results


def generate_answer(query: str, context: List[Dict[str, Any]], components: Dict[str, Any]) -> tuple[str, str]:
    """
    검색된 컨텍스트를 기반으로 Ollama를 사용하여 최종 답변을 생성합니다.
    """
    ollama_client = components["ollama_client"]
    
    # LLM에 제공할 컨텍스트 문자열 구성
    context_str_parts = []
    for i, item in enumerate(context, 1):
        text = item.get('text', '')
        meta = item.get('metadata', {})
        graph_ctx = item.get('graph_context', {})
        
        law_name = graph_ctx.get('law_name') or meta.get('law_name', 'N/A')
        superior_law = graph_ctx.get('superior_law')
        
        # 조항 정보 구성 (예: '제1조(목적)')
        article_info = meta.get('article_id', '')
        if meta.get('article_title'):
            article_info += f"({meta.get('article_title')})"

        # 출처 문자열 생성
        source_ref = f"{law_name} {article_info}"
        
        part = (
            f"--- 참고자료 {i} ---\n"
            f"출처: {source_ref.strip()}\n"
        )
        if superior_law and superior_law != law_name:
             part += f"상위 법령: {superior_law}\n"
        part += f"내용: {text.strip()}\n"
        context_str_parts.append(part)

    final_context_str = "\n".join(context_str_parts)

    logging.info("Formatted context for LLM.")
    # print("\n--- LLM에 제공될 통합 컨텍스트 ---\n" + final_context_str)

    prompt = f"""
    당신은 대한민국 법률 전문가 AI입니다. 아래 제공된 '참고자료'만을 근거로 사용자의 '질문'에 대해 명확하고 간결하게 답변하세요.
    
    [규칙]
    - 반드시 참고자료에 명시된 내용만을 사용하여 답변해야 합니다.
    - 자료에 없는 내용은 '알 수 없음' 또는 '정보가 없습니다'라고 답변하세요.
    - 답변은 논리정연하게, 전문가적인 어조를 유지하되 일반인도 이해하기 쉽고 자세하게 설명해주세요.
    - 답변 마지막에, 어떤 참고자료를 근거로 했는지 출처(예: [화학물질관리법 시행령 제12조])를 명확하게 명시하세요.
    - 모든 답변 끝에는 다음 주의사항을 필수로 포함시켜주세요: " AI 답변은 법률 자문이 아니며, 참고용으로만 활용하시기 바랍니다. 중요한 사안은 반드시 법률 전문가의 확인을 받으세요."

    [참고자료]
    {final_context_str}
    
    [질문]
    {query}
    
    [답변]
    """

    logging.info("Generating answer using Ollama...")
    try:
        response = ollama_client.generate(model=OLLAMA_MODEL_NAME, prompt=prompt)
        answer = response.get('response', '오류: 모델로부터 답변을 생성하지 못했습니다.').strip()
        logging.info("Answer generated successfully.")
        return final_context_str, answer
    except Exception as e:
        error_message = f"Ollama API 호출 중 오류 발생: {e}"
        logging.error(error_message)
        return final_context_str, error_message


def main():
    """메인 대화형 루프를 실행합니다."""
    components = initialize_components()
    if not components:
        print("\n[오류] 시스템 초기화에 실패했습니다. 프로그램을 종료합니다.")
        return

    logger = QALogger()

    print("\n" + "="*60)
    print("  대한민국 법률 RAG 시스템 (Ollama 연동)")
    print(f"  - LLM Model: {OLLAMA_MODEL_NAME}")
    print(f"  - Vector DB: Qdrant (Collection: {QDRANT_COLLECTION_NAME})")
    print(f"  - Graph DB: Neo4j ({NEO4J_URI})")
    print("="*60)
    print("질문을 입력하세요. 종료하시려면 'exit' 또는 'quit'를 입력하세요.")

    while True:
        try:
            query = input("\n[질문] > ")
            if query.lower() in ["exit", "quit"]:
                break
            if not query.strip():
                continue

            # 1. 하이브리드 검색 수행
            search_context = hybrid_search(query, components, top_k=5)
            
            if not search_context:
                print("[알림] 질문과 관련된 정보를 데이터베이스에서 찾을 수 없습니다.")
                continue

            # 2. 답변 생성
            print("\n[알림] 검색된 정보를 바탕으로 답변을 생성 중입니다...")
            context_str, answer = generate_answer(query, search_context, components)

            # 3. 결과 출력 및 로깅
            print("\n" + "---" * 20)
            print("[답변]")
            print(answer)
            print("---" * 20)
            
            if answer and "오류:" not in answer:
                logger.log(question=query, context=context_str, answer=answer)
                logging.info("QA pair has been logged.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
            print(f"\n[오류] 예상치 못한 오류가 발생했습니다: {e}")
    
    print("\n시스템을 종료합니다.")
    if components.get("neo4j_driver"):
        components["neo4j_driver"].close()


if __name__ == "__main__":
    main()
