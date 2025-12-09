# MCP 기반 git-filter-repo Wrapper  
## 1. 목적  
git-filter-repo 기능을 MCP(Multi-Command Plugin)로 Wrapping하여 Git 기록 재작성 작업을 자동화하고, AI 기반 커밋 메시지 변환 기능까지 포함하는 경량 도구 제공.

## 2. 요구사항 요약  
### 2.1 기능 요구사항  
- 커밋 메시지 재작성  
- Author/Committer 정보 변경  
- 파일 제거/이동  
- 경로 필터링  
- 대용량 파일 제거  
- 브랜치 일괄 처리  
- Merge 구조 flatten(optional)  
- AI 기반 commit message regenerate(옵션)

### 2.2 비기능 요구사항  
- Cross-platform(Windows/Linux)  
- python(uv) 기반 단일 배포  
- 성능: git-filter-repo 동일 수준  
- 로그 및 dry-run 지원  

## 3. 시스템 구성  
- **MCP Server**  
  - REST/JSON-RPC 인터페이스  
  - Git 작업 Queue 처리  
- **git-filter-repo Adapter**  
  - 파라미터 변환  
  - 결과 리턴  
- **AI Commit Message Engine(optional)**  
  - 로컬 Ollama 연동  
  - 메시지 변환 정책 포함

## 4. 간단 아키텍처  
```
Client → MCP Server → Adapter → git-filter-repo → Git Repo
                               ↳ (optional) AI Engine
```

## 5. 핵심 파라미터  
- target_branch  
- rewrite_mode(rewrite, flatten, cleanup)  
- author_map  
- file_filter  
- commit_message_policy(ai|manual)  

## 6. 제약  
- remote force-push 필요  
- merge 구조 깨짐 가능성  
- Git LFS 기록 rewrite 별도 처리 필요  

## 7. 예시 워크플로우  
1. MCP 서버 호출  
2. 히스토리 분석(dry run)  
3. rewrite 실행  
4. 결과 검증  
5. push --force-with-lease 수행  

