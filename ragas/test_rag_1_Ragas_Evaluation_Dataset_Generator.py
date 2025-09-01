import docx
import re
import json
import os
import pathlib
import uuid
import time
import ollama  
from transformers import AutoTokenizer
from typing import List, Dict, Any, Tuple, Optional

# --- 설정 (Configuration) ---
#TOKENIZER_PATH = "google_gemma-3-4b-it-tokenizer-path"
TOKENIZER_PATH = "/media/test1/새 볼륨2/google-gemma-3-4b-it/gemma-3-4b-it"
ROOT_DIR = "화학물질관리법_예시" # 법령 문서가 있는 루트 디렉토리
EVAL_DATA_DIR = "evaluation_dataset_law_test_1" # 생성된 평가 데이터를 저장할 폴더
MAX_CHUNKS_FOR_EVAL = 100 # 평가 데이터 생성을 위해 사용할 최대 청크 수 (비용/시간 관리)
OLLAMA_MODEL = "gemma3:latest" # 사용할 Ollama 모델 이름

# 청킹 관련 설정
MAX_CHUNK_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 50

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
        
        meta_match = re.search(r"^(.*?)\((.*?)\)\(제?.*?\)\((.*?)\)", file_name)
        if meta_match:
            law_name, law_type, enforcement_date = meta_match.groups()
            metadata = {
                "law_name": law_name.strip(), "law_type": law_type.strip(),
                "enforcement_date": enforcement_date.replace('.docx', '').strip(), "source_file": file_name,
            }
        else:
            law_name = file_name.replace('.docx', '')
            metadata = {"law_name": law_name, "law_type": "유형분석필요", "enforcement_date": "날짜분석필요", "source_file": file_name}

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
        if end_idx == len(tokens): break
        start_idx += MAX_CHUNK_TOKENS - CHUNK_OVERLAP_TOKENS
    return sub_chunks

def advanced_hybrid_chunking(metadata: Dict[str, Any], content: str) -> List[Dict[str, Any]]:
    """'장-조-항' 구조를 인식하고 긴 '항'만 토큰 기반으로 재분할합니다."""
    final_chunks = []
    current_chapter, current_article_title, current_article_id = "정보 없음", "정보 없음", "정보 없음"
    
    patterns = {
        'chapter': re.compile(r"^제(\d+|[一-十]+)장\s+(.*)"),
        'article': re.compile(r"^(제\d+조(?:의\d+)?)\s?\((.*?)\)")
        }
    articles = re.split(r'(^제\d+조(?:의\d+)?\s?\([^)]+\))', content, flags=re.MULTILINE)
    
    for i in range(1, len(articles), 2):
        header, body = articles[i], articles[i+1]
        
        for line in body.split('\n'):
             match_chapter = patterns['chapter'].match(line.strip())
             if match_chapter:
                 current_chapter = f"제{match_chapter.group(1)}장 {match_chapter.group(2)}"; break
        
        match_article = patterns['article'].match(header.strip())
        if match_article:
            current_article_id, current_article_title = match_article.group(1), match_article.group(2)

        clauses = re.split(r'(\s*①\s*|\s*②\s*|\s*③\s*|\s*④\s*|\s*⑤\s*|\s*⑥\s*|\s*⑦\s*|\s*⑧\s*|\s*⑨\s*|\s*⑩\s*)', body)
        
        def process_and_append_chunk(text, clause_num="N/A"):
            sub_chunks = _split_text_with_overlap(text)
            for sub_chunk_text in sub_chunks:
                chunk_meta = metadata.copy()
                chunk_meta.update({"chapter": current_chapter, "article_id": current_article_id,
                                   "article_title": current_article_title, "clause_num": clause_num})
                final_chunks.append({"chunk_id": str(uuid.uuid4()), "text": sub_chunk_text, "metadata": chunk_meta})

        if clauses[0].strip():
            process_and_append_chunk(f"{header.strip()}\n{clauses[0].strip()}")

        for j in range(1, len(clauses), 2):
            clause_num_str, clause_text = clauses[j].strip(), clauses[j+1].strip()
            if clause_text:
                process_and_append_chunk(f"{clause_num_str} {clause_text}", clause_num_str)

    print(f"  - 고급 하이브리드 청킹으로 {len(final_chunks)}개 청크 생성 완료.")
    return final_chunks

# --- Ollama (gemma3)를 이용한 평가 데이터셋 생성 ---
def generate_evaluation_set(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    처리된 청크를 기반으로 Ollama (gemma3)를 사용하여 질문-정답 쌍을 생성합니다.
    """
    print(f"\n--- Ollama ({OLLAMA_MODEL})로 Ragas 평가 데이터셋 생성 시작 ---")
    
    eval_data = []
    chunks_to_process = chunks[:MAX_CHUNKS_FOR_EVAL]
    
    for i, chunk in enumerate(chunks_to_process):
        print(f"  - 처리 중인 청크 {i+1}/{len(chunks_to_process)}...")
        prompt = f"""
        당신은 한국 법률 전문가입니다. RAG 시스템의 성능을 평가하기 위한 데이터셋을 생성하는 임무를 받았습니다.
        아래에 제공된 법률 조항 텍스트를 기반으로, 사실에 근거한 '질문'과 그에 대한 완벽한 '정답' 쌍을 1개 생성해주세요.

        [규칙]
        1. 질문은 반드시 주어진 텍스트 내용만으로 답변할 수 있어야 합니다.
        2. 정답은 질문에 대한 완전하고 상세한 답변이어야 하며, 텍스트의 내용을 충실히 반영해야 합니다.
        3. 외부 지식을 사용하거나 내용을 추측해서는 안 됩니다.

        [법률 조항 텍스트]
        ---
        {chunk['text']}
        ---

        [출력 형식]
        질문: [여기에 생성한 질문을 입력]
        정답: [여기에 생성한 정답을 입력]
        """
        
        try:
            # Ollama API 호출
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            # 응답 텍스트 파싱
            qa_text = response['message']['content']
            question_match = re.search(r"질문:\s*(.*)", qa_text)
            answer_match = re.search(r"정답:\s*(.*)", qa_text, re.DOTALL)
            
            if question_match and answer_match:
                question = question_match.group(1).strip()
                ground_truth = answer_match.group(1).strip()
                
                eval_data.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "source_chunk_id": chunk['chunk_id'],
                    "source_chunk_text": chunk['text']
                })
            else:
                print(f"[경고] 청크 {chunk['chunk_id']}에 대한 Q&A 파싱 실패.")
                
        except Exception as e:
            print(f"[오류] Ollama({OLLAMA_MODEL}) 호출 중 문제 발생: {e}")
            print("Ollama가 실행 중인지, 모델({OLLAMA_MODEL})이 설치되었는지 확인해주세요.")
            continue
            
    return eval_data

# --- 메인 실행 로직 ---
def run_pipeline():
    if not TOKENIZER:
        print("\n실행 중단: 토크나이저를 불러올 수 없습니다.")
        return

    all_chunks = []
    
    print("\n--- 1&2단계: 문서 파싱 및 하이브리드 청킹 시작 ---")
    doc_files = sorted(list(pathlib.Path(ROOT_DIR).rglob('*.docx')))
    if not doc_files:
        print(f"[경고] '{ROOT_DIR}' 디렉토리에서 .docx 파일을 찾을 수 없습니다.")
        return

    for file_path in doc_files:
        print(f"\n파일 처리 중: {file_path}")
        metadata, content = parse_law_document(str(file_path))
        if content:
            try:
                relative_path = file_path.relative_to(ROOT_DIR)
                metadata['hierarchy'] = list(relative_path.parts[:-1])
            except ValueError:
                metadata['hierarchy'] = []
            
            chunks = advanced_hybrid_chunking(metadata, content)
            all_chunks.extend(chunks)

    print(f"\n--- 총 {len(all_chunks)}개 청크 생성 완료 ---")

    if all_chunks:
        # 평가 데이터셋 생성
        evaluation_dataset = generate_evaluation_set(all_chunks)
        
        if evaluation_dataset:
            # 결과 저장
            os.makedirs(EVAL_DATA_DIR, exist_ok=True)
            output_path = os.path.join(EVAL_DATA_DIR, f"ragas_eval_data_ollama_gemma3_{MAX_CHUNKS_FOR_EVAL}_test1.json")
            
            # json 라이브러리를 직접 사용하여 표준 JSON 배열로 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_dataset, f, ensure_ascii=False, indent=4)   
            print(f"\n--- 성공! Ragas 평가 데이터셋을 '{output_path}'에 저장했습니다. ---")
            print(f"총 {len(evaluation_dataset)}개의 질문-정답 쌍이 생성되었습니다.")
        else:
            print("\n--- 생성된 평가 데이터가 없어 파일을 저장하지 않았습니다. ---")

if __name__ == "__main__":
    run_pipeline()


