# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Mạnh Quyền  
**Nhóm:** Nhóm Quyền - An - Dũng  
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai đoạn văn bản có vector biểu diễn gần nhau, tức là chúng mang nội dung hoặc ngữ cảnh khá giống nhau. Giá trị càng gần `1` thì mức độ tương đồng ngữ nghĩa càng cao.

**Ví dụ HIGH similarity:**
- Sentence A: Python là một ngôn ngữ lập trình phổ biến cho phát triển phần mềm.
- Sentence B: Python được dùng rộng rãi trong nhiều dự án lập trình và tự động hóa.
- Tại sao tương đồng: Cả hai câu đều nói về Python và mục đích sử dụng của nó trong lập trình, nên ý nghĩa gần nhau.

**Ví dụ LOW similarity:**
- Sentence A: Vector store dùng để lưu embedding phục vụ similarity search.
- Sentence B: Hôm nay trời mưa lớn và đường phố bị ngập nước.
- Tại sao khác: Hai câu nói về hai chủ đề hoàn toàn khác nhau, một bên là kỹ thuật AI còn một bên là thời tiết.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity tập trung vào hướng của vector thay vì độ lớn tuyệt đối, nên phù hợp hơn khi so sánh ý nghĩa giữa các embedding. Với text embeddings, điều quan trọng thường là mức độ cùng ngữ cảnh chứ không phải độ dài vector.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, `chunk_size=500`, `overlap=50`. Bao nhiêu chunks?**
> `num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23 chunks`

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào?**
> `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25 chunks`. Overlap lớn hơn giữ được ngữ cảnh ở ranh giới chunk tốt hơn, giảm rủi ro mất ý khi câu hoặc đoạn bị cắt giữa chừng.

---

## 2. Document Selection — Nhóm (10 điểm)

**Chosen domain:** `Python programming basics`

**Tại sao nhóm chọn domain này?**
> Nhóm chọn bộ tài liệu Python vì toàn bộ file trong `main_data/` đều cùng ngữ cảnh về Python cơ bản, dễ tạo benchmark query có thể verify rõ. Tài liệu cũng có cấu trúc theo chủ đề và section, phù hợp để so sánh nhiều cách chunking và metadata filtering.

### Shared document set (Tài liệu chung của nhóm)

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|---|---|---:|---|
| 1 | `python_intro_and_variables.txt` | Python Tutorial - Introduction | 16,468 | `topic=intro_variables`, `source=python_docs`, `level=basic` |
| 2 | `python_conditionals_loops_functions.txt` | Python Tutorial - Control Flow Tools | 35,177 | `topic=control_flow`, `source=python_docs`, `level=basic` |
| 3 | `python_dictionaries_sets_list_tuples.txt` | Python Tutorial - Data Structures | 22,041 | `topic=collections`, `source=python_docs`, `level=basic` |
| 4 | `python_input_output.txt` | Python Tutorial - Input and Output | 18,141 | `topic=input_output`, `source=python_docs`, `level=basic` |
| 5 | `python_error_exception.txt` | Python Tutorial - Errors and Exceptions | 21,243 | `topic=exceptions`, `source=python_docs`, `level=basic` |
| 6 | `python_module.txt` | Python Tutorial - Modules | 23,266 | `topic=modules`, `source=python_docs`, `level=basic` |

### Metadata schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|---|---|---|---|
| `topic` | string | `control_flow`, `modules` | Dùng để pre-filter query theo chủ đề, tăng độ chính xác khi câu hỏi đã biết domain con. |
| `source` | string | `python_docs` | Giúp trace chunk về file gốc, thuận tiện khi debug và viết report. |
| `section` | string | `introduction`, `exceptions` | Giúp biết chunk đến từ section nào, hỗ trợ phân tích chunk coherence. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu mẫu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|---|---|---:|---:|---|
| `python_intro_and_variables.txt` | FixedSizeChunker (`fixed_size`) | 37 | 493.7 | Khá |
| `python_intro_and_variables.txt` | SentenceChunker (`by_sentences`) | 48 | 341.6 | Tốt |
| `python_intro_and_variables.txt` | RecursiveChunker (`recursive`) | 48 | 341.1 | Trung bình |
| `python_conditionals_loops_functions.txt` | FixedSizeChunker (`fixed_size`) | 79 | 494.6 | Khá |
| `python_conditionals_loops_functions.txt` | SentenceChunker (`by_sentences`) | 70 | 500.6 | Tốt |
| `python_conditionals_loops_functions.txt` | RecursiveChunker (`recursive`) | 98 | 354.8 | Trung bình |
| `python_module.txt` | FixedSizeChunker (`fixed_size`) | 52 | 496.5 | Khá |
| `python_module.txt` | SentenceChunker (`by_sentences`) | 47 | 493.0 | Tốt |
| `python_module.txt` | RecursiveChunker (`recursive`) | 68 | 335.6 | Trung bình |

### Strategy Của Tôi (Quyền)

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**
> `RecursiveChunker` thử tách văn bản theo thứ tự separator ưu tiên: `\n\n`, `\n`, `. `, space, rồi cuối cùng mới cắt cứng theo độ dài. Cách này giữ được cấu trúc đoạn và header tốt hơn fixed-size khi tài liệu có nhiều section rõ ràng.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Bộ tài liệu Python docs có nhiều đoạn giải thích, heading, ví dụ và code block. `RecursiveChunker` tận dụng được cấu trúc đó nên chunk thường bám sát ý hơn so với cắt theo ký tự thuần túy.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|---|---|---:|---:|---|
| `python_module.txt` | best baseline = SentenceChunker | 47 | 492.1 | Tốt, cân bằng |
| `python_module.txt` | **của tôi = RecursiveChunker** | 68 | 335.6 | Tốt cho tài liệu nhiều section |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Chunks | Relevant@top1 | Avg cosine score | Điểm mạnh | Điểm yếu |
|---|---|---:|---:|---:|---|---|
| An | FixedSizeChunker | 306 | 5/5 | 0.7052 | Đơn giản, chunk count ổn định | Dễ cắt giữa câu khi không có overlap lớn |
| **Quyền (Tôi)** | **SentenceChunker** | 277 | 5/5 | 0.7237 | Giữ câu trọn ý, chunk dễ đọc | Chunk size không đều, có thể vượt limit |
| Dũng | RecursiveChunker | 389 | 5/5 | 0.7726 | Tôn trọng paragraph/section boundary | Chunk count cao nhất, nhiều chunk nhỏ hơn |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Với bộ Python tutorial notes này, `RecursiveChunker` là lựa chọn phù hợp nhất nếu ưu tiên giữ boundary theo heading và paragraph. Tuy nhiên, `SentenceChunker` cũng rất cạnh tranh nhờ việc giữ trọn vẹn ý nghĩa của từng câu đơn lẻ.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`**:
> Tôi dùng regex để tách câu theo dấu chấm, chấm hỏi, chấm than. Sau đó gom lại theo `max_sentences_per_chunk` và `strip()` khoảng trắng thừa để chunk sạch và dễ đọc hơn.

**`RecursiveChunker.chunk` / `_split`**:
> Triển khai đệ quy: thử tách bằng `\n\n`, `\n`, `. `, rồi đến khoảng trắng. Nếu đoạn vẫn quá dài sau khi tách, hàm tiếp tục gọi `_split()` với separator ở mức tiếp theo cho đến khi đạt `chunk_size`.

### EmbeddingStore

**`add_documents` + `search`**:
> Lưu trữ in-memory dưới dạng danh sách các record (`id`, `content`, `metadata`, `embedding`). Khi search, tôi tính dot product giữa query embedding và data embedding, sau đó sort giảm dần để lấy top-k.

**`search_with_filter` + `delete_document`**:
> Thực hiện filter metadata trước để thu hẹp không gian tìm kiếm, sau đó mới tính similarity. Hàm delete sẽ xóa mọi record có `doc_id` tương ứng trong metadata.

### KnowledgeBaseAgent

**`answer`**:
> Agent thực hiện retrieve top-k chunks liên quan, sau đó đưa vào prompt dưới dạng Context. Prompt yêu cầu LLM chỉ sử dụng thông tin trong Context để trả lời câu hỏi của người dùng, đảm bảo tính trung thực (groundedness).

### Test Results

```text
======================================= test session starts ========================================
platform win32 -- Python 3.12.7, pytest-8.3.5, pluggy-1.5.0
rootdir: D:\AI_thucchien\Day07\2A202600481-NguyenManhQuyen-Day07
collected 42 items

tests/test_solution.py .......................................... [100%]

======================================== 42 passed in 0.08s ========================================
```


## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Prediction | Actual Score | Correct? |
|---:|---|---|---|---:|---|
| 1 | Python is a popular programming language. | Python is widely used in software development. | high | 0.0188 | No |
| 2 | I like machine learning. | It is raining heavily today. | low | -0.1396 | Yes |
| 3 | Vector stores save embeddings. | Vector databases help with similarity search. | high | 0.0263 | Yes |
| 4 | The cat is sleeping on the chair. | Object-oriented programming uses classes and objects. | low | -0.0071 | Yes |
| 5 | Chunking splits a document into smaller parts. | Splitting text into chunks improves retrieval. | high | -0.0017 | No |

### Kết quả nào bất ngờ nhất? Điều này nói gì về embeddings?

> Pair 5 làm tôi bất ngờ nhất vì ngữ nghĩa rất gần nhau nhưng điểm số lại âm. Điều này cho thấy chất lượng similarity phụ thuộc cực lớn vào embedding backend. Với `_mock_embed` trong lab, điểm số chỉ mang tính chất kiểm tra luồng logic (pipeline validation) chứ chưa phản ánh đúng ngữ nghĩa thực tế như các mô hình OpenAI hay Local Embedder.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers (Nhóm thống nhất)

| # | Query | Gold Answer | Chunk dự kiến |
|---:|---|---|---|
| 1 | What does `range(5)` generate in Python? | `0, 1, 2, 3, 4` (endpoint is not included). | `python_conditionals_loops_functions.txt` |
| 2 | How can a list be used as a stack? | Use `append()` to push and `pop()` without index to remove last item (LIFO). | `python_dictionaries_sets_list_tuples.txt` |
| 3 | Difference between `str()` and `repr()`? | `str()` is human-readable, `repr()` is for interpreter/debugging. | `python_input_output.txt` |
| 4 | Exception matches except clause? | The except clause executes, then continues after the `try/except` block. | `python_error_exception.txt` |
| 5 | What does `import fibo` do? | Binds module name `fibo` in current namespace; access via `fibo.fib()`. | `python_module.txt` |

### Kết Quả Của Tôi (Quyền)

| # | Query | Top-1 Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---:|---|---|---:|---|---|
| 1 | `range(5)` generate? | `range()` example: `0 1 2 3 4` | 0.6904 | Yes | `0, 1, 2, 3, 4` (5 excluded) |
| 2 | List as stack? | `stack.append()` / `stack.pop()` | 0.7245 | Yes | Use `append`/`pop` for LIFO stack |
| 3 | `str()` vs `repr()`? | `str()` human vs `repr()` debug | 0.8614 | Yes | Human-readable vs interpreter-readable |
| 4 | Exception match? | `try/except` matching handler | 0.7834 | Yes | Matching clause runs, then continues |
| 5 | Import `fibo`? | `import fibo` / `fibo.fib()` | 0.8033 | Yes | Binds name, access via `fibo.method` |

**Bao nhiêu queries trả về chunk relevant trong top-3?**  
> `5 / 5`

---

## 7. What I Learned (5 điểm — Demo)

### Điều hay nhất tôi học được từ thành viên khác trong nhóm:
> Việc gán metadata chi tiết (như `topic`, `section`) cực kỳ quan trọng. Nó cho phép filter dữ liệu trước khi search, giúp tăng độ chính xác và giảm nhiễu cho hệ thống RAG.

### Điều hay nhất tôi học được từ nhóm khác (qua demo):
> Các chiến lược chunking khác nhau (như Semantic Chunking) có thể mang lại kết quả vượt trội so với các cách cắt text truyền thống khi xử lý các tài liệu có cấu trúc phức tạp.

### Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?
> Tôi sẽ thử nghiệm thêm việc chia nhỏ metadata đến mức heading (`H1`, `H2`) và thử nghiệm tính toán trọng số cho các trường metadata khác nhau để tối ưu hóa kết quả retrieval.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|---|---|---:|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** |  | **65 / 100** |
