import docx
import re
import json
import os
import pathlib
import uuid
from transformers import AutoTokenizer
from typing import List, Dict, Any, Tuple, Optional

# --- 설정 (Configuration) ---
#TOKENIZER_PATH = "google_gemma-3-4b-it-tokenizer-path"
TOKENIZER_PATH = "/media/test1/새 볼륨2/google-gemma-3-4b-it/gemma-3-4b-it"
ROOT_DIR = "화학물질관리법_예시" # 법령 문서가 있는 루트 디렉토리

# 청킹 관련 설정
MAX_CHUNK_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 50

file_save_name = "final_exmaple_test1"

# --- 전역 토크나이저 초기화 ---
TOKENIZER: Optional[AutoTokenizer] = None
try:
    TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    print(f"성공적으로 토크나이저를 불러왔습니다: {TOKENIZER_PATH}")
except Exception as e:
    print(f"토크나이저 로딩 실패: {TOKENIZER_PATH}. 경로를 확인해주세요. 오류: {e}")
    TOKENIZER = None

# --- 문서 파싱 및 메타데이터 추출 ---
def parse_law_document(file_path: str) -> Tuple[Dict[str, Any], str]:
    """
    .docx 법령 문서를 파싱하고 파일명과 내용에서 메타데이터를 추출합니다.
    """
    try:
        document = docx.Document(file_path)
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        file_name = os.path.basename(file_path)
        
        # 파일명에서 메타데이터 추출
        meta_match = re.search(r"^(.*?)\((.*?)\)\(제?.*?\)\((.*?)\)", file_name)
        if meta_match:
            law_name, law_type, enforcement_date = meta_match.groups()
            metadata = {
                "law_name": law_name.strip(),
                "law_type": law_type.strip(),
                "enforcement_date": enforcement_date.replace('.docx', '').strip(),
                "source_file": file_name,
            }
        else:
            law_name = file_name.replace('.docx', '')
            metadata = {"law_name": law_name, "law_type": "유형분석필요", "enforcement_date": "날짜분석필요", "source_file": file_name}

        # 본문 시작점 탐지
        content_start_index = 0
        article_start_pattern = re.compile(r"^\s*제1조\s*\([^)]+\).+")
        for i, p_text in enumerate(paragraphs):
            if article_start_pattern.match(p_text.strip()):
                content_start_index = i
                break
        
        content = "\n".join(paragraphs[content_start_index:])
        print(f"  - '{metadata['law_name']}' 파싱 완료. 본문 시작 라인: {content_start_index + 1}")
        return metadata, content
    except Exception as e:
        print(f"  - 문서 파싱 오류: {e}")
        return {}, ""

# --- 고급 하이브리드 청킹 ---
def _split_text_with_overlap(text: str) -> List[str]:
    """긴 텍스트를 토크나이저 기반 슬라이딩 윈도우로 분할합니다."""
    if not TOKENIZER or len(TOKENIZER.encode(text)) <= MAX_CHUNK_TOKENS:
        return [text]
    
    tokens = TOKENIZER.encode(text)
    sub_chunks = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + MAX_CHUNK_TOKENS, len(tokens))
        sub_chunk_tokens = tokens[start_idx:end_idx]
        sub_chunks.append(TOKENIZER.decode(sub_chunk_tokens, skip_special_tokens=True))
        if end_idx == len(tokens):
            break
        start_idx += MAX_CHUNK_TOKENS - CHUNK_OVERLAP_TOKENS
    return sub_chunks

def advanced_hybrid_chunking(metadata: Dict[str, Any], content: str) -> List[Dict[str, Any]]:
    """'장-조-항' 구조를 인식하고 긴 '항'만 토큰 기반으로 재분할합니다."""
    final_chunks = []
    current_chapter = "정보 없음"
    current_article_title = "정보 없음"
    current_article_id = "정보 없음"
    
    # 정규식 패턴 정의
    patterns = {
        'chapter': re.compile(r"^제(\d+|[一-十]+)장\s+(.*)"),
        'article': re.compile(r"^(제\d+조(?:의\d+)?)\s?\((.*?)\)"),
    }

    # 전체 내용을 '조' 단위로 먼저 분할
    articles = re.split(r'(^제\d+조(?:의\d+)?\s?\([^)]+\))', content, flags=re.MULTILINE)
    
    for i in range(1, len(articles), 2):
        header = articles[i]
        body = articles[i+1]
        
        # '장' 정보 업데이트
        for line in body.split('\n'):
             match_chapter = patterns['chapter'].match(line.strip())
             if match_chapter:
                 current_chapter = f"제{match_chapter.group(1)}장 {match_chapter.group(2)}"
                 break

        match_article = patterns['article'].match(header.strip())
        if match_article:
            current_article_id = match_article.group(1)
            current_article_title = match_article.group(2)

        # '항' 단위로 분할 (①, ② ...)
        clauses = re.split(r'(\s*①\s*|\s*②\s*|\s*③\s*|\s*④\s*|\s*⑤\s*|\s*⑥\s*|\s*⑦\s*|\s*⑧\s*|\s*⑨\s*|\s*⑩\s*)', body)
        
        # '조'의 본문 (항이 없는 경우)
        if clauses[0].strip():
            chunk_text = f"{header.strip()}\n{clauses[0].strip()}"
            sub_chunks = _split_text_with_overlap(chunk_text)
            for part_idx, sub_chunk_text in enumerate(sub_chunks):
                chunk_meta = metadata.copy()
                chunk_meta.update({
                    "chapter": current_chapter,
                    "article_id": current_article_id,
                    "article_title": current_article_title,
                    "clause_num": "N/A",
                })
                final_chunks.append({"chunk_id": str(uuid.uuid4()), "text": sub_chunk_text, "metadata": chunk_meta})

        # '항' 단위 처리
        for j in range(1, len(clauses), 2):
            clause_num_str = clauses[j].strip()
            clause_text = clauses[j+1].strip()
            if not clause_text: continue

            full_clause_text = f"{clause_num_str} {clause_text}"
            sub_chunks = _split_text_with_overlap(full_clause_text)
            for part_idx, sub_chunk_text in enumerate(sub_chunks):
                chunk_meta = metadata.copy()
                chunk_meta.update({
                    "chapter": current_chapter,
                    "article_id": current_article_id,
                    "article_title": current_article_title,
                    "clause_num": clause_num_str,
                })
                final_chunks.append({"chunk_id": str(uuid.uuid4()), "text": sub_chunk_text, "metadata": chunk_meta})

    print(f"  - 고급 하이브리드 청킹으로 {len(final_chunks)}개 청크 생성 완료.")
    return final_chunks

# --- 지식 그래프 생성 (Neo4j 준비) ---
class LawKnowledgeGraph:
    """폴더 구조와 청크 메타데이터를 기반으로 Cypher 쿼리를 생성합니다."""
    def __init__(self):
        self.cypher_queries = []
        print("\n--- 지식 그래프 Cypher 쿼리 생성 시작 ---")

    def generate_cypher_from_chunks(self, chunks: List[Dict[str, Any]]):
        created_docs = set()
        for chunk in chunks:
            meta = chunk['metadata']
            law_name = meta["law_name"]
            
            # LawDocument 노드 생성 (폴더 구조 기반 계층 정보 포함)
            if law_name not in created_docs:
                hierarchy_path = "/".join(meta.get("hierarchy", []))
                query = (
                    f'MERGE (d:LawDocument {{name: "{law_name}"}}) '
                    f'ON CREATE SET d.type = "{meta["law_type"]}", '
                    f'd.enforcement_date = "{meta["enforcement_date"]}", '
                    f'd.source_file = "{meta["source_file"]}", '
                    f'd.hierarchy_path = "{hierarchy_path}"'
                )
                self.cypher_queries.append(query)
                created_docs.add(law_name)

            # ProvisionChunk 노드 생성
            chunk_id = chunk['chunk_id']
            escaped_text = chunk["text"].replace('"', '\\"').replace('\n', '\\n')
            query = (
                f'CREATE (c:ProvisionChunk {{'
                f'id: "{chunk_id}", '
                f'text: "{escaped_text}", '
                f'chapter: "{meta["chapter"]}", '
                f'article: "{meta["article_id"]} ({meta["article_title"]})", '
                f'clause: "{meta["clause_num"]}"'
                f'}})'
            )
            self.cypher_queries.append(query)

            # LawDocument -> ProvisionChunk 관계 생성
            query = (
                f'MATCH (d:LawDocument {{name: "{law_name}"}}), (c:ProvisionChunk {{id: "{chunk_id}"}}) '
                f'MERGE (d)-[:HAS_CHUNK]->(c)'
            )
            self.cypher_queries.append(query)

    def generate_hierarchy_relations(self, all_metadatas: List[Dict[str, Any]]):
        """폴더 구조(hierarchy)를 기반으로 법률 간 상하 관계를 생성합니다."""
        relations = set()
        for meta in all_metadatas:
            hierarchy = meta.get("hierarchy", [])
            law_name = meta["law_name"]
            if len(hierarchy) > 0:
                # 부모 폴더 이름이 부모 법률 이름이라고 가정
                # 예: /화학물질관리법/시행령/ -> 부모는 '화학물질관리법'
                parent_law_name = hierarchy[-1] 
                relations.add((law_name, parent_law_name))

        for sub_law, sup_law in relations:
             # 부모 법률 이름과 실제 문서 이름이 다를 수 있으므로, 가장 유사한 이름을 찾음
             # 간단한 예시로, 여기서는 부모 폴더 이름이 문서 이름에 포함되는 경우를 찾음
            parent_doc_name = None
            for m in all_metadatas:
                if sup_law in m['law_name']:
                    parent_doc_name = m['law_name']
                    break
            
            if parent_doc_name:
                query = (
                    f'MATCH (sub:LawDocument {{name: "{sub_law}"}}), (sup:LawDocument {{name: "{parent_doc_name}"}}) '
                    f'MERGE (sub)-[:SUBORDINATE_TO]->(sup)'
                )
                self.cypher_queries.append(query)

    def save_queries_to_file(self, file_path=f"law_chunks_{file_save_name}.cql"):
        with open(file_path, "w", encoding="utf-8") as f:
            for query in self.cypher_queries:
                f.write(query + ";\n")
        print(f"\n모든 Cypher 쿼리를 '{file_path}' 파일에 저장했습니다.")

# --- 메인 실행 로직 ---
def run_pipeline():
    if not TOKENIZER:
        print("\n실행 중단: 토크나이저를 불러올 수 없습니다.")
        return

    all_chunks = []
    all_metadatas = []
    
    print("\n--- 지능형 법령 문서 전처리 파이프라인 시작 ---")
    doc_files = sorted(list(pathlib.Path(ROOT_DIR).rglob('*.docx')))
    if not doc_files:
        print(f"[경고] '{ROOT_DIR}' 디렉토리에서 .docx 파일을 찾을 수 없습니다.")
        return

    for file_path in doc_files:
        print(f"\n파일 처리 중: {file_path}")
        metadata, content = parse_law_document(str(file_path))
        
        if content:
            # 폴더 구조에서 계층 정보 추출
            try:
                relative_path = file_path.relative_to(ROOT_DIR)
                metadata['hierarchy'] = list(relative_path.parts[:-1])
                print(f"  - 탐지된 폴더 계층: {metadata['hierarchy']}")
            except ValueError:
                metadata['hierarchy'] = []
            
            all_metadatas.append(metadata)
            chunks = advanced_hybrid_chunking(metadata, content)
            all_chunks.extend(chunks)

    print(f"\n--- 총 {len(all_chunks)}개 청크 생성 완료 ---")

    if all_chunks:
        kg = LawKnowledgeGraph()
        kg.generate_cypher_from_chunks(all_chunks)
        kg.generate_hierarchy_relations(all_metadatas)
        kg.save_queries_to_file()
        
        output_json_path = f"law_chunks_{file_save_name}.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        print(f"모든 청크 데이터를 '{output_json_path}' 파일에 저장했습니다.")

if __name__ == "__main__":
    run_pipeline()
