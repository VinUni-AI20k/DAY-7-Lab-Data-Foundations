# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Mạnh Quyền
**Nhóm:** [Tên nhóm]
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai đoạn văn bản có hướng biểu diễn vector gần nhau, tức là chúng mang nội dung hoặc ngữ cảnh khá giống nhau. Giá trị càng gần `1` thì mức độ tương đồng ngữ nghĩa càng cao.

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

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* `num_chunks = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11)`
> *Đáp án:* `23 chunks`

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap tăng lên `100`, công thức trở thành `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25`, nên số chunk tăng lên. Overlap lớn hơn giúp giữ được ngữ cảnh giữa hai chunk liền kề, giảm nguy cơ mất ý khi câu hoặc đoạn bị cắt ở ranh giới chunk.

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
> Tôi dùng regex `(?<=[.!?])\s+|\.\n` để tách câu theo dấu chấm, chấm hỏi, chấm than và trường hợp chấm xuống dòng. Sau khi tách, tôi `strip()` từng câu và loại bỏ phần rỗng để tránh tạo ra chunk lỗi khi văn bản có khoảng trắng thừa. Cuối cùng, các câu được gom lại theo `max_sentences_per_chunk`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Tôi triển khai theo hướng đệ quy: thử tách văn bản bằng separator ưu tiên cao trước như `\n\n`, `\n`, `. `, rồi mới đến khoảng trắng và cuối cùng là cắt cứng theo độ dài. Base case là khi chuỗi rỗng hoặc khi đoạn hiện tại đã ngắn hơn hoặc bằng `chunk_size`. Nếu một đoạn vẫn quá dài sau khi tách, hàm tiếp tục gọi `_split()` với separator ở mức tiếp theo.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Tôi chuẩn hóa mỗi tài liệu thành một record gồm `id`, `content`, `metadata` và `embedding`, rồi lưu vào danh sách in-memory `_store`. Khi search, tôi embed câu query bằng cùng embedding function và tính điểm bằng dot product giữa query embedding và embedding của từng record. Sau đó tôi sắp xếp giảm dần theo score và lấy `top_k`.

**`search_with_filter` + `delete_document`** — approach:
> Tôi filter trước rồi mới search để chỉ tính similarity trên tập record phù hợp với metadata yêu cầu, đúng với bản chất của pre-filter retrieval. Với `delete_document`, tôi xóa toàn bộ record có `metadata['doc_id']` trùng với document cần xóa và trả về `True/False` tùy việc có xóa được gì hay không.

### KnowledgeBaseAgent

**`answer`** — approach:
> Tôi cho agent retrieve `top_k` chunk liên quan từ store trước, rồi ghép các chunk đó thành phần `Context` trong prompt. Prompt có cấu trúc đơn giản: vai trò trợ lý, phần context đã retrieve, câu hỏi của người dùng và phần `Answer:` để LLM điền tiếp. Cách này giúp câu trả lời có grounding rõ ràng từ dữ liệu đã lấy ra.

### Test Results

```
======================================= test session starts ========================================
platform win32 -- Python 3.12.7, pytest-8.3.5, pluggy-1.5.0 -- C:\Users\mnquyen26\anaconda3\python.exe
cachedir: .pytest_cache
rootdir: D:\AI_thucchien\Day07\2A202600481-NguyenManhQuyen-Day07
plugins: anyio-4.10.0, hydra-core-1.3.2
collected 42 items

tests/test_solution.py ..........................................

======================================== 42 passed in 0.08s ========================================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Python la ngon ngu lap trinh pho bien. | Python duoc dung rong rai trong phat trien phan mem. | high | 0.0188 | Không |
| 2 | Toi thich hoc machine learning. | Hom nay troi mua rat to. | low | -0.1396 | Có |
| 3 | Vector store dung de luu embedding. | Co so du lieu vector giup tim kiem tuong dong. | high | 0.0263 | Có |
| 4 | Con meo dang ngu tren ghe. | Lap trinh huong doi tuong dung class va object. | low | -0.0071 | Có |
| 5 | Chunking giup chia tai lieu thanh doan nho. | Chia van ban thanh cac chunk giup retrieval hieu qua hon. | high | -0.0017 | Không |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Cặp số 5 là kết quả làm tôi bất ngờ nhất vì về mặt ngữ nghĩa hai câu khá gần nhau nhưng điểm lại hơi âm. Điều này cho thấy kết quả similarity phụ thuộc rất nhiều vào chất lượng embedding backend; với `_mock_embed` dùng trong bài lab, điểm số chủ yếu phục vụ kiểm thử tính đúng của pipeline hơn là đo ngữ nghĩa thật. Nếu dùng local embedder hoặc OpenAI embedder, các cặp tương đồng về nghĩa có thể sẽ tách biệt rõ hơn.

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
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **50 / 100 + phần nhóm** |
