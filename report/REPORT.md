# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Anh Quân
**Nhóm:** E402 - Group 6
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai vector gần cùng hướng, tức là hai câu có biểu diễn ngữ nghĩa tương đồng. Giá trị càng gần 1 thì độ tương đồng càng cao.

**Ví dụ HIGH similarity:**
- Sentence A: `Python is a programming language for building software.`
- Sentence B: `Python is widely used to develop software applications.`
- Tại sao tương đồng: Cùng nói về Python và mục đích dùng để phát triển phần mềm.

**Ví dụ LOW similarity:**
- Sentence A: `Vector databases are used for similarity search.`
- Sentence B: `I like cooking pasta with tomato sauce.`
- Tại sao khác: Hai câu thuộc hai chủ đề khác hẳn nhau.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity tập trung vào hướng của vector (ngữ nghĩa), ít bị ảnh hưởng bởi độ lớn vector. Vì vậy nó ổn định hơn khi so sánh embeddings văn bản.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*  
> `step = chunk_size - overlap = 500 - 50 = 450`  
> `chunks = ceil((10000 - 500)/450) + 1 = ceil(9500/450) + 1 = 22 + 1 = 23`
> *Đáp án:* **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap = 100 thì `step = 400`, số chunk tăng thành `ceil(9500/400)+1 = 25`. Overlap lớn hơn giúp giữ ngữ cảnh liên tục giữa các chunk, đổi lại tốn thêm lưu trữ/tính toán.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** [ví dụ: Customer support FAQ, Vietnamese law, cooking recipes, ...]

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:*

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| | | | |
| | | | |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| | FixedSizeChunker (`fixed_size`) | | | |
| | SentenceChunker (`by_sentences`) | | | |
| | RecursiveChunker (`recursive`) | | | |

### Strategy Của Tôi

**Loại:** [FixedSizeChunker / SentenceChunker / RecursiveChunker / custom strategy]

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?*

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?*

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| | best baseline | | | |
| | **của tôi** | | | |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | | | | |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex `(?<=[.!?])(?:\s+|\n+)` để tách câu theo dấu kết câu và khoảng trắng/xuống dòng. Sau đó trim và gom `max_sentences_per_chunk` câu thành một chunk. Edge case text rỗng trả về `[]`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán đệ quy thử tách theo danh sách separator ưu tiên. Base case là độ dài đoạn <= `chunk_size`; nếu hết separator thì cắt cứng theo block `chunk_size`. Khi ghép part mới vượt ngưỡng thì flush chunk hiện tại và đệ quy sâu hơn.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` tạo record chuẩn hóa (`id`, `doc_id`, `content`, `metadata`, `embedding`) rồi lưu vào `_store`; nếu Chroma có sẵn thì đồng bộ thêm vào collection. `search` embed query một lần, tính dot product với từng embedding để ra `score`, sắp xếp giảm dần và trả về `top_k`.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` ưu tiên lọc theo metadata trước rồi mới chạy similarity search trên tập đã lọc. `delete_document` xóa toàn bộ record có `doc_id` tương ứng trong in-memory store, đồng thời thử xóa bên Chroma nếu đang bật.

### KnowledgeBaseAgent

**`answer`** — approach:
> Agent gọi `store.search(question, top_k)` để lấy context chunks, nối lại thành phần `Context` trong prompt. Prompt yêu cầu chỉ trả lời theo context và nói "don't know" nếu thiếu thông tin. Sau đó gọi `llm_fn(prompt)` để sinh câu trả lời.

### Test Results

```text
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.3, pluggy-1.6.0 -- /home/nauqhna/VIN_AITHUCCHIEN/Day-07-Lab-Data-Foundations/venv/bin/python3
cachedir: .pytest_cache
rootdir: /home/nauqhna/VIN_AITHUCCHIEN/Day-07-Lab-Data-Foundations
collected 42 items                                                             

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.04s ==============================

```

**Số tests pass:** **42 / 42**

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | | | high / low | | |
| 2 | | | high / low | | |
| 3 | | | high / low | | |
| 4 | | | high / low | | |
| 5 | | | high / low | | |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:*

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu queries trả về chunk relevant trong top-3?** __ / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Cách thiết kế metadata nhất quán từ đầu (`doc_id`, `lang`, `topic`) giúp search filter ổn định và dễ debug hơn nhiều. Ngoài ra việc thống nhất schema sớm giúp tránh sửa dữ liệu hàng loạt về sau.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Một số nhóm tách benchmark query thành câu hỏi factual ngắn và câu hỏi suy luận dài để đánh giá retrieval rõ hơn. Cách này làm lộ khác biệt chất lượng chunking/search rất nhanh.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ tăng số tài liệu thật và cân bằng đa dạng ngôn ngữ/chủ đề hơn thay vì dựa nhiều vào mock data. Tôi cũng sẽ lưu thêm metadata về source và thời gian để hỗ trợ lọc/trace tốt hơn.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
