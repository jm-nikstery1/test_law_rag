import os
import time
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    # context_utilization 제거
    LLMContextPrecisionWithoutReference,  # 대체 메트릭 추가
    AnswerCorrectness,  # 추가 메트릭
    ContextEntityRecall  # 추가 메트릭
    
)
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from typing import List, Dict, Any

from tqdm import tqdm

# --- 0. 설정: DB, 모델, 평가 파일 정보 ---
# Neo4j 연결 정보
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "user1"
NEO4J_PASSWORD = "password"

# Qdrant 연결 정보
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "korean_law_hybrid_db_test_2"

# 사용할 임베딩 및 LLM 모델
EMBEDDING_MODEL = 'jhgan/ko-sbert-nli'
OLLAMA_MODEL = "gemma3:latest" 

EVAL_DATA_FILE = "evaluation_dataset_law_test_1/ragas_eval_data_ollama_gemma3_50_test1.json"

# --- 하이브리드 RAG 파이프라인 정의 ---

class HybridRAGSystem:
    """
    Qdrant (Vector)와 Neo4j (Graph)를 결합한 하이브리드 RAG 시스템
    """
    def __init__(self, qdrant_host, qdrant_port, collection_name, 
                 neo4j_uri, neo4j_user, neo4j_password,
                 embedding_model, ollama_model):
        try:
            self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
            self.embedding_model = SentenceTransformer(embedding_model)
            self.collection_name = collection_name
            print("Vector Retriever (Qdrant)가 성공적으로 연결되었습니다.")
            self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            self.neo4j_driver.verify_connectivity()
            print("Graph Retriever (Neo4j)가 성공적으로 연결되었습니다.")
            self.generator_llm = Ollama(model=ollama_model)
            print(f"Generator가 Ollama 모델 '{ollama_model}'을 사용하도록 설정되었습니다.")
        except Exception as e:
            print(f"RAG 시스템 초기화 실패: {e}")
            raise

    def close(self):
        if self.neo4j_driver:
            self.neo4j_driver.close()

    def retrieve(self, question: str, top_k: int = 3, graph_k: int = 2) -> List[str]:
        query_vector = self.embedding_model.encode(question).tolist()
        qdrant_results = self.qdrant_client.search(
            collection_name=self.collection_name, query_vector=query_vector, limit=top_k
        )
        initial_contexts = [hit.payload for hit in qdrant_results]
        graph_contexts = []
        with self.neo4j_driver.session() as session:
            for payload in initial_contexts:
                if payload and 'metadata' in payload and 'law_name' in payload['metadata']:
                    law_name = payload['metadata']['law_name']
                    cypher_query = """
                    MATCH (d:LawDocument {name: $law_name})-[:SUBORDINATE_TO*0..]->(parent)
                    MATCH (parent)-[:HAS_CHUNK]->(chunk)
                    RETURN chunk.text AS text LIMIT $limit
                    """
                    results = session.run(cypher_query, law_name=law_name, limit=graph_k)
                    for record in results:
                        graph_contexts.append(record["text"])
        final_context_texts = [p['text'] for p in initial_contexts if 'text' in p]
        final_context_texts.extend(graph_contexts)
        return list(dict.fromkeys(final_context_texts))

    def generate(self, question: str, contexts: List[str]) -> str:
        if not contexts: return "관련 정보를 찾지 못했습니다."
        context_str = "\n\n".join(contexts)
        prompt = f"""
        당신은 한국 법률 전문가입니다. 오직 제공된 컨텍스트에만 근거하여 다음 질문에 답하세요.
        답변은 질문의 핵심에 대해 직접적이고 간결해야 합니다.
        질문과 직접적으로 관련 없는 부가 정보나 배경 설명은 포함하지 마십시오.
        참고 정보에 내용이 없다면, 정보를 찾을 수 없다고 답변하세요. 절대로 내용을 지어내지 마세요.

        [참고 정보]
        ---
        {context_str}
        ---
        
        [질문]
        {question}
        
        [답변]
        """
        try:
            return self.generator_llm.invoke(prompt)
        except Exception as e:
            print(f"Ollama 답변 생성 중 오류 발생: {e}")
            return "답변 생성에 실패했습니다."

    def ask(self, question: str) -> Dict[str, Any]:
        retrieved_contexts = self.retrieve(question)
        answer = self.generate(question, retrieved_contexts)
        return {"answer": answer, "contexts": retrieved_contexts}


# --- 메인 평가 실행 로직 ---
def main():
    if not os.path.exists(EVAL_DATA_FILE):
        print(f"평가 파일 '{EVAL_DATA_FILE}'을 찾을 수 없습니다."); return
    
    eval_df = pd.read_json(EVAL_DATA_FILE)
    print(f"'{EVAL_DATA_FILE}'에서 {len(eval_df)}개의 평가 데이터를 로드했습니다.")

    rag_system = None
    try:
        rag_system = HybridRAGSystem(
            QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME,
            NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
            EMBEDDING_MODEL, OLLAMA_MODEL
        )

        results = []
        print("\n--- 하이브리드 RAG 파이프라인을 실행하여 답변 및 컨텍스트를 수집합니다 ---")
        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="하이브리드 RAG 실행 중"):
            rag_output = rag_system.ask(row['question'])
            results.append({
                "question": row['question'], "ground_truth": row['ground_truth'],
                "answer": rag_output['answer'], "contexts": rag_output['contexts']
            })

        result_df = pd.DataFrame(results)
        eval_dataset = Dataset.from_pandas(result_df)
        
        # ### TimeoutError 해결을 위해 순차적 평가 및 대기 로직 추가 ###
        print("\n--- Ragas 평가를 시작합니다 (순차 실행 및 대기 시간 추가) ---")
        ragas_llm = Ollama(model=OLLAMA_MODEL)
        ragas_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # ### [평가 지표 수정] ###
        # 불안정한 context_precision을 제거하고, 검색 성능을 더 잘 측정할 수 있는
        # context_recall과 context_utilization을 추합니다.
        metrics = [
            faithfulness, 
            answer_relevancy, 
            context_recall,       # 검색된 컨텍스트가 정답 생성에 필요한 모든 정보를 포함하는지 측정
            #LLMContextPrecisionWithoutReference,  # context_utilization 대체
            AnswerCorrectness(),  # 클래스를 인스턴스화하여 사용
            #ContextEntityRecall, # 새로운 평가 지표
        ]

        all_scores = []
        for item in tqdm(eval_dataset, desc="Ragas 평가 중 (항목별)"):
            item_dataset = Dataset.from_dict({k: [v] for k, v in item.items()})
            try:
                score = evaluate(
                    dataset=item_dataset, 
                    metrics=metrics, 
                    llm=ragas_llm, 
                    embeddings=ragas_embeddings,
                )
                all_scores.append(score.to_pandas())
            except Exception as e:
                print(f"\n항목 평가 중 오류 발생: {item.get('question', 'N/A')}")
                print(f"오류 유형: {type(e).__name__}, 내용: {e}")
                failed_data = {k: [v] for k, v in item.items()}
                for m in metrics:
                    failed_data[m.name] = [None]
                all_scores.append(pd.DataFrame(failed_data))
            
            # Ollama 서버의 부하를 줄이기 위해 3초간 대기합니다.
            time.sleep(2)

        print("\n--- Ragas 평가 완료 ---")
        
        if all_scores:
            score_df = pd.concat(all_scores, ignore_index=True)
            
            final_scores = {}
            for m in metrics:
                valid_scores = pd.to_numeric(score_df[m.name], errors='coerce').dropna()
                if not valid_scores.empty:
                    final_scores[m.name] = valid_scores.mean()

            print("\n[ 최종 평가 결과 (하이브리드) ]"); print(final_scores)
            print("\n[ 상세 평가 결과 (DataFrame) ]"); print(score_df.head())
    
            output_filename = f"ragas_hybrid_evaluation_result_gemma3_4b_{len(eval_df)}q_metrics_updated_sequential_test_1.csv"
            score_df.to_csv(output_filename, index=False, encoding="utf-8-sig")
            print(f"\n평가 결과를 '{output_filename}' 파일로 저장했습니다.")
        else:
            print("처리된 평가 결과가 없습니다.")

    finally:
        if rag_system:
            rag_system.close()

if __name__ == "__main__":
    main()
