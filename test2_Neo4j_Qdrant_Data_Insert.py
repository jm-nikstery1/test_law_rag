import json
import os
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List, Dict, Any

# --- 설정: DB 연결, 모델, 입력 파일 정보 ---
# 사용자의 환경에 맞게 수정
# Neo4j 연결 정보
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "user1"
NEO4J_PASSWORD = "password"

# Qdrant 연결 정보
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "korean_law_hybrid_db_test_2"

# 사용할 임베딩 모델 (Hugging Face)
EMBEDDING_MODEL = 'jhgan/ko-sbert-nli'

# 이전 단계에서 생성된 최종 JSON 파일
INPUT_JSON_FILE = "law_chunks_final_exmaple_test1.json"

# --- 지식 그래프 구축 (Neo4j) ---
class LawKnowledgeGraph:
    """
    Neo4j 데이터베이스에 법률 지식 그래프를 생성하고 관리합니다.
    """
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            print("Neo4j에 성공적으로 연결되었습니다.")
        except Exception as e:
            print(f"Neo4j 연결 실패: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()
            print("Neo4j 연결이 종료되었습니다.")

    def cleanup_database(self):
        """스크립트 재실행을 위해 기존 데이터를 모두 삭제합니다."""
        if not self.driver: return
        with self.driver.session() as session:
            print("  - 기존 Neo4j 데이터를 정리합니다...")
            session.run("MATCH (n) DETACH DELETE n")
            print("  - 데이터 정리 완료.")

    def create_constraints(self):
        """데이터 무결성과 쿼리 성능을 위해 제약 조건을 생성합니다."""
        if not self.driver: return
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:LawDocument) REQUIRE d.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:ProvisionChunk) REQUIRE c.id IS UNIQUE")
            print("  - Neo4j 제약조건이 생성되었거나 이미 존재합니다.")

    def insert_data_from_json(self, chunks: List[Dict[str, Any]]):
        """JSON 청크 데이터를 기반으로 노드와 관계를 생성합니다."""
        if not self.driver: return
        
        with self.driver.session() as session:
            # 1. LawDocument 노드 생성
            law_docs = {chunk['metadata']['law_name']: chunk['metadata'] for chunk in chunks}
            print(f"  - {len(law_docs)}개의 LawDocument 노드를 생성/업데이트합니다.")
            for name, meta in tqdm(law_docs.items(), desc="LawDocument 노드 처리 중"):
                session.run("""
                    MERGE (d:LawDocument {name: $name})
                    SET d.type = $type,
                        d.enforcement_date = $enforcement_date,
                        d.source_file = $source_file,
                        d.hierarchy_path = $hierarchy_path
                """, name=name, type=meta.get('law_type'), enforcement_date=meta.get('enforcement_date'),
                       source_file=meta.get('source_file'), hierarchy_path="/".join(meta.get('hierarchy', [])))

            # 2. 법률 간 상하 관계(SUBORDINATE_TO) 생성
            print("  - 법률 간 상하 관계를 생성합니다.")
            for name, meta in law_docs.items():
                hierarchy = meta.get('hierarchy', [])
                if hierarchy:
                    parent_law_name_pattern = hierarchy[-1]
                    for potential_parent_name in law_docs:
                        if parent_law_name_pattern in potential_parent_name:
                            session.run("""
                                MATCH (sub:LawDocument {name: $sub_name})
                                MATCH (sup:LawDocument {name: $sup_name})
                                MERGE (sub)-[:SUBORDINATE_TO]->(sup)
                            """, sub_name=name, sup_name=potential_parent_name)
                            break
            
            # 3. ProvisionChunk 노드 및 관계 생성
            print(f"  - {len(chunks)}개의 ProvisionChunk 노드와 관계를 생성합니다.")
            for chunk in tqdm(chunks, desc="ProvisionChunk 노드 처리 중"):
                meta = chunk['metadata']
                params = {
                    'chunk_id': chunk['chunk_id'], 'text': chunk['text'], 'law_name': meta['law_name'],
                    'chapter': meta.get('chapter'), 'article_id': meta.get('article_id'),
                    'article_title': meta.get('article_title'), 'clause_num': meta.get('clause_num')
                }
                session.run("""
                    MATCH (d:LawDocument {name: $law_name})
                    MERGE (c:ProvisionChunk {id: $chunk_id})
                    SET c.text = $text,
                        c.chapter = $chapter,
                        c.article = $article_id + ' (' + $article_title + ')',
                        c.clause = $clause_num
                    MERGE (d)-[:HAS_CHUNK]->(c)
                """, params)


# --- 벡터 임베딩 및 저장소 색인 (Qdrant) ---
class VectorIndexer:
    """텍스트를 임베딩하고 Qdrant 벡터 저장소에 색인합니다."""
    def __init__(self, host, port, model_name):
        try:
            self.client = QdrantClient(host=host, port=port, timeout=60)
            print(f"\nQdrant에 성공적으로 연결되었습니다 ({host}:{port}).")
            self.model = SentenceTransformer(model_name)
            print(f"임베딩 모델 '{model_name}'을 로드했습니다.")
        except Exception as e:
            print(f"Qdrant 또는 임베딩 모델 초기화 실패: {e}")
            self.client = None
            self.model = None

    def setup_collection(self, collection_name: str):
        """Qdrant에 컬렉션을 생성합니다. (존재하지 않을 경우)"""
        if not self.client or not self.model: return
        
        vector_size = self.model.get_sentence_embedding_dimension()
        try:
            collections = self.client.get_collections().collections
            if collection_name not in {c.name for c in collections}:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                )
                print(f"  - Qdrant 컬렉션 '{collection_name}'이(가) 새로 생성되었습니다.")
            else:
                print(f"  - 기존 Qdrant 컬렉션 '{collection_name}'을(를) 사용합니다.")
        except Exception as e:
            print(f"  - Qdrant 컬렉션 확인/생성 중 오류: {e}")

    def index_data(self, collection_name: str, chunks: List[Dict[str, Any]]):
        """ 청크 데이터를 배치로 나누어 임베딩하고 Qdrant에 업로드합니다."""
        if not self.client or not self.model: return
        
        print(f"  - {len(chunks)}개 청크에 대한 임베딩을 생성하고 Qdrant에 저장합니다.")
        
        embeddings = self.model.encode(
            [chunk['text'] for chunk in chunks],
            show_progress_bar=True,
            batch_size=32
        )
        
        # 타임아웃 방지를 위해 배치(batch) 처리 추가
        batch_size = 100  # 한 번에 업로드할 포인트 수
        print(f"  - Qdrant에 데이터를 {batch_size}개씩 나누어 저장합니다.")
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Qdrant에 데이터 저장 중"):
            batch_end = i + batch_size
            batch_chunks = chunks[i:batch_end]
            batch_embeddings = embeddings[i:batch_end]
            
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=chunk['chunk_id'],
                        vector=embedding.tolist(),
                        payload=chunk
                    )
                    for chunk, embedding in zip(batch_chunks, batch_embeddings)
                ],
                wait=True
            )
        print(f"  - Qdrant에 데이터 저장 완료.")


# --- 메인 실행 로직 ---
def main():
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"입력 파일 '{INPUT_JSON_FILE}'을 찾을 수 없습니다.")
        return
    with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    print(f"'{INPUT_JSON_FILE}'에서 {len(chunks_data)}개의 청크를 로드했습니다.")

    # 1. Neo4j 지식 그래프 구축
    print("\n--- 1단계: 지식 그래프 구축 (Neo4j) ---")
    kg = LawKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    if kg.driver:
        kg.cleanup_database()
        kg.create_constraints()
        kg.insert_data_from_json(chunks_data)
        kg.close()

    # 2. Qdrant 벡터 저장소 색인
    print("\n--- 2단계: 벡터 저장소 색인 (Qdrant) ---")
    indexer = VectorIndexer(QDRANT_HOST, QDRANT_PORT, EMBEDDING_MODEL)
    if indexer.client and indexer.model:
        indexer.setup_collection(QDRANT_COLLECTION_NAME)
        indexer.index_data(QDRANT_COLLECTION_NAME, chunks_data)

    print("\n\n모든 데이터베이스 적재 작업이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    main()
