# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Hồ Trần Đình Nguyên - MSSV: 2A202600080
**Nhóm:** 70 (Nguyên - Quang - Toàn)  
**Ngày:** 2026-04-10

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**  
High cosine similarity nghĩa là hai đoạn văn bản có hướng biểu diễn gần nhau trong không gian embedding, tức là nội dung hoặc ý nghĩa của chúng tương đối giống nhau. Trong bài toán retrieval, điều này thường cho thấy một chunk có mức liên quan cao với câu hỏi của người dùng.

**Ví dụ HIGH similarity:**
- Sentence A: Python is used for data analysis and machine learning.
- Sentence B: Python is popular for analyzing data and training ML models.
- Tại sao tương đồng: Cả hai câu đều nói về việc dùng Python cho phân tích dữ liệu và học máy, chỉ khác cách diễn đạt.

**Ví dụ LOW similarity:**
- Sentence A: Vector stores help retrieve similar chunks from embeddings.
- Sentence B: My bicycle needs a new rear tire this weekend.
- Tại sao khác: Hai câu thuộc hai chủ đề hoàn toàn khác nhau, một câu về hệ thống AI còn câu kia về xe đạp nên mức tương đồng ngữ nghĩa rất thấp.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**  
Cosine similarity tập trung vào hướng của vector thay vì độ lớn tuyệt đối, nên phù hợp hơn với embedding văn bản, nơi ý nghĩa thường nằm ở hướng biểu diễn. Euclidean distance dễ bị ảnh hưởng bởi độ lớn vector, trong khi điều quan trọng hơn trong retrieval là mức độ giống nhau về ngữ nghĩa.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**  
Trình bày phép tính:  
`num_chunks =(doc_length - chunk_size) / (chunk_size - overlap) + 1`  
`= (10000 - 500) / (500 - 50) + 1`  
`= 9500 / 450 + 1`  
`= 21.11 + 1`  
`= 22 + 1 = 23`  
**Đáp án:** `23 chunks`

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**  
Khi overlap tăng lên 100 thì số chunks trở thành `(10000 - 500) / (500 - 100) + 1 = 9500 / 400 + 1 = 24 + 1 = 25`. Overlap lớn hơn giúp giữ lại ngữ cảnh giữa hai chunk liền kề, nhờ đó giảm nguy cơ một ý quan trọng bị cắt đúng ở ranh giới chunk.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** GitHub Actions documentation

**Tại sao nhóm chọn domain này?**  
Nhóm chọn domain GitHub Actions vì đây là tài liệu kỹ thuật có cấu trúc rõ ràng, nhiều khái niệm liên quan nhau như workflows, variables, contexts, caching và artifacts, nên rất phù hợp để đánh giá chất lượng retrieval. Ngoài ra, nội dung tài liệu đủ đa dạng để so sánh các chunking strategy nhưng vẫn nằm trong cùng một domain, giúp phép so sánh công bằng và dễ phân tích hơn.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Contexts | GitHub Docs | 2658 | `source`, `domain`, `topic=contexts`, `category=expressions_contexts`, `language=en`, `doc_type=docs` |
| 2 | Dependency caching | GitHub Docs | 1593 | `source`, `domain`, `topic=dependency_caching`, `category=storage_caching`, `language=en`, `doc_type=docs` |
| 3 | Reusing workflow configurations | GitHub Docs | 8929 | `source`, `domain`, `topic=reusing_workflow_configurations`, `category=workflow_reuse`, `language=en`, `doc_type=docs` |
| 4 | Variables | GitHub Docs | 1498 | `source`, `domain`, `topic=variables`, `category=configuration`, `language=en`, `doc_type=docs` |
| 5 | Workflow artifacts | GitHub Docs | 2843 | `source`, `domain`, `topic=workflow_artifacts`, `category=storage_artifacts`, `language=en`, `doc_type=docs` |
| 6 | Workflows | GitHub Docs | 2788 | `source`, `domain`, `topic=workflows`, `category=workflow_basics`, `language=en`, `doc_type=docs` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `source` | string | `github_clean_workflows.md` | Cho biết chunk đến từ file nào để kiểm tra nguồn và debug retrieval |
| `domain` | string | `github_actions` | Giữ toàn bộ tập dữ liệu cùng một domain khi mở rộng thêm corpus |
| `topic` | string | `workflows` | Hữu ích nhất cho filter theo chủ đề, ví dụ `topic=variables` |
| `category` | string | `workflow_basics` | Cho phép nhóm các tài liệu gần nhau ở mức chi tiết hơn `topic` |
| `language` | string | `en` | Giúp phân biệt ngôn ngữ nếu sau này thêm tài liệu tiếng Việt hoặc ngôn ngữ khác |
| `doc_type` | string | `docs` | Giữ schema thống nhất khi bổ sung thêm FAQ, notes hoặc tutorial |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu đại diện với `chunk_size=500`:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `github_clean_workflows.md` | FixedSizeChunker (`fixed_size`) | 7 | 441.14 | Trung bình |
| `github_clean_workflows.md` | SentenceChunker (`by_sentences`) | 5 | 554.60 | Cao |
| `github_clean_workflows.md` | RecursiveChunker (`recursive`) | 6 | 462.83 | Cao |
| `github_clean_contexts.md` | FixedSizeChunker (`fixed_size`) | 6 | 484.67 | Trung bình |
| `github_clean_contexts.md` | SentenceChunker (`by_sentences`) | 4 | 662.00 | Khá cao nhưng chunk dài |
| `github_clean_contexts.md` | RecursiveChunker (`recursive`) | 8 | 330.38 | Cao |
| `github_clean_reusing_workflow_configurations.md` | FixedSizeChunker (`fixed_size`) | 20 | 493.95 | Trung bình |
| `github_clean_reusing_workflow_configurations.md` | SentenceChunker (`by_sentences`) | 13 | 684.31 | Khá cao nhưng đôi lúc gom quá dài |
| `github_clean_reusing_workflow_configurations.md` | RecursiveChunker (`recursive`) | 26 | 341.65 | Cao nhất |

### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**  
`RecursiveChunker` chia văn bản theo thứ tự ưu tiên của các separator: đoạn trống `\n\n`, xuống dòng `\n`, dấu câu `. `, khoảng trắng và cuối cùng mới cắt cứng theo số ký tự nếu vẫn còn quá dài. Với mỗi mức separator, thuật toán cố gắng ghép các phần nhỏ lại sao cho không vượt quá `chunk_size`. Nếu một phần vẫn còn dài hơn giới hạn thì nó tiếp tục đệ quy xuống separator tiếp theo. Nhờ vậy, chunk được tạo ra thường bám theo cấu trúc tự nhiên của tài liệu trước khi phải cắt cơ học.

**Tại sao tôi chọn strategy này cho domain nhóm?**  
Tài liệu GitHub Actions có cấu trúc heading, paragraph, bullet list và section khá rõ, nên recursive chunking tận dụng rất tốt các ranh giới tự nhiên đó. So với fixed-size, strategy này ít cắt giữa ý hơn; so với sentence chunking, nó linh hoạt hơn với các phần dài có nhiều dòng hoặc heading ngắn nhưng liên quan chặt chẽ với đoạn phía dưới.

**Code snippet (nếu custom):**
```python
# Không dùng custom strategy.
# Tôi dùng RecursiveChunker(chunk_size=500) có sẵn trong src/chunking.py.
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `github_clean_reusing_workflow_configurations.md` | best baseline: SentenceChunker | 13 | 684.31 | Top-3 vẫn đúng, chunk dễ đọc nhưng đôi lúc quá dài và chưa luôn rơi đúng đoạn so sánh trực tiếp |
| `github_clean_reusing_workflow_configurations.md` | **của tôi: RecursiveChunker** | 26 | 341.65 | Top-3 = 5/5, top-1 rất sát ở Query 3 và Query 5, chunk gọn hơn nhưng vẫn giữ đúng section cần dùng |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi (Nguyên) | RecursiveChunker | 10/10 | Cân bằng tốt giữa độ dài chunk và ngữ cảnh, đặc biệt mạnh ở tài liệu dài có cấu trúc rõ | Tạo nhiều chunk hơn nên tốn công lưu trữ hơn |
| Quang | SentenceChunker | 9/10 | Chunk dễ đọc, giữ câu trọn vẹn, ổn định với tài liệu giải thích | Một số chunk hơi dài, đôi khi gom nhiều câu nên chưa thật sắc ở truy vấn cần section rất cụ thể |
| Toàn | FixedSizeChunker | 8.5/10 | Dễ cài đặt, làm baseline tốt, retrieval vẫn đúng | Dễ cắt giữa câu hoặc giữa ý nên preview và grounding kém tự nhiên hơn |

**Strategy nào tốt nhất cho domain này? Tại sao?**  
Trong domain GitHub Actions docs, `RecursiveChunker` là strategy tốt nhất vì tài liệu có cấu trúc rõ theo section và subsection. Kết quả benchmark của nhóm cho thấy strategy này vừa giữ được context tốt, vừa retrieve đúng phần nội dung trực tiếp liên quan đến câu hỏi, đặc biệt ở các truy vấn cần so sánh hoặc cần lấy đúng section chi tiết.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:  
Tôi dùng regex `(?<=[.!?])\s+` để tách câu dựa trên dấu chấm, chấm hỏi hoặc chấm than theo sau bởi khoảng trắng. Sau khi tách, tôi `strip()` từng phần để loại bỏ khoảng trắng thừa và bỏ các câu rỗng. Cách này đơn giản nhưng đủ tốt cho bài lab vì tài liệu chủ yếu là prose kỹ thuật tương đối chuẩn câu.

**`RecursiveChunker.chunk` / `_split`** — approach:  
Thuật toán bắt đầu bằng cách kiểm tra base case: nếu text rỗng thì trả về `[]`, nếu độ dài không vượt `chunk_size` thì trả về một chunk duy nhất. Nếu còn dài, hàm `_split` thử chia theo từng separator ưu tiên cao trước, sau đó ghép lại các mảnh sao cho không vượt giới hạn. Chỉ khi không còn separator phù hợp hoặc đến separator rỗng thì mới cắt cứng theo số ký tự.

### EmbeddingStore

**`add_documents` + `search`** — approach:  
Tôi thiết kế `EmbeddingStore` theo hướng ưu tiên dùng ChromaDB nếu có, còn không thì fallback sang in-memory store để bài lab luôn chạy được. Khi thêm tài liệu, mỗi `Document` được chuyển thành một record gồm `id`, `content`, `metadata` và `embedding`. Khi search, hệ thống tạo embedding cho query rồi xếp hạng các record theo độ tương đồng để lấy `top_k`.

**`search_with_filter` + `delete_document`** — approach:  
Với `search_with_filter`, tôi lọc record theo `metadata_filter` trước rồi mới tính similarity để giảm nhiễu và tăng precision. Nếu dùng ChromaDB thì filter được đẩy xuống `where` trong query; nếu dùng in-memory thì lọc bằng cách so khớp key-value trong metadata. Với `delete_document`, tôi xóa tất cả chunk có `doc_id` tương ứng, cả ở ChromaDB lẫn in-memory store.

### KnowledgeBaseAgent

**`answer`** — approach:  
Hàm `answer` trước hết retrieve `top_k` chunk liên quan nhất từ vector store. Sau đó tôi inject toàn bộ các chunk này vào prompt dưới dạng danh sách `[1]`, `[2]`, `[3]` để LLM thấy rõ từng nguồn context. Prompt cũng yêu cầu nếu context không đủ thì phải nói rõ, giúp agent bám sát retrieved knowledge thay vì tự suy diễn.

### Test Results

```text
(venv) > pytest tests/ -q
..........................................
42 passed in 0.05s
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Python is a programming language. | Python is used to write software. | high | 0.8640 | Đúng |
| 2 | Vector databases store embeddings. | Similarity search retrieves related chunks. | high | 0.3976 | Tương đối |
| 3 | The sky is blue today. | I forgot to charge my phone. | low | 0.0285 | Đúng |
| 4 | Customer support handles billing issues. | The team resolves payment and account problems. | high | 0.6123 | Đúng |
| 5 | Neural networks learn patterns from data. | A bicycle has two wheels and pedals. | low | 0.0259 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**  
Kết quả đáng chú ý nhất là pair 2, vì hai câu có liên quan về retrieval pipeline nhưng score chỉ ở mức trung bình `0.3976`, không cao như pair 1 hoặc pair 4. Điều này cho thấy local embeddings biểu diễn nghĩa theo mức độ gần nhau về ngữ cảnh và cách diễn đạt, chứ không phải cứ cùng domain là sẽ có similarity rất cao. So với `_mock_embed`, kết quả từ `LocalEmbedder` hợp lý hơn nhiều và phản ánh ngữ nghĩa thực tế tốt hơn.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What is a GitHub Actions workflow, and where is it defined? | A workflow is a configurable automated process that runs one or more jobs, and it is defined by a YAML file stored in the `.github/workflows` directory of a repository. |
| 2 | What is the main difference between dependency caching and workflow artifacts? | Dependency caching is used to reuse files that do not change often between jobs or workflow runs, such as downloaded dependencies, while artifacts are used to save files produced by a job after a workflow run ends, such as build logs or binaries. |
| 3 | In GitHub Actions, when are contexts more useful than default environment variables? | Contexts are useful when information is needed before a job is routed to a runner or when expressions and conditional logic must be evaluated, because default environment variables only exist on the runner that is executing the job. |
| 4 | What are the two ways to define custom variables in GitHub Actions workflows? | You can define a variable for a single workflow by using the `env` key in the workflow file, or define a configuration variable across multiple workflows at the organization, repository, or environment level. |
| 5 | When should reusable workflows be preferred over composite actions? | Reusable workflows should be preferred when you need multiple jobs, separate logging per job and step, or to specify runners at the job level. Composite actions are better for bundling multiple steps into a single reusable step inside one job. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | What is a GitHub Actions workflow, and where is it defined? | Chunk đầu của `github_clean_workflows.md`, định nghĩa workflow là automated process và nêu vị trí `.github/workflows` | 0.8376 | Có | Agent trả lời đúng workflow là quy trình tự động có một hoặc nhiều jobs và nằm trong thư mục `.github/workflows` |
| 2 | What is the main difference between dependency caching and workflow artifacts? | Chunk đầu của `github_clean_dependency_caching.md`, giải thích caching để tái sử dụng file; top-2 đồng thời lấy đúng chunk về artifacts | 0.7626 | Có | Agent tóm tắt đúng rằng caching dùng cho file ít thay đổi còn artifacts dùng để lưu output sinh ra sau workflow |
| 3 | In GitHub Actions, when are contexts more useful than default environment variables? | Chunk 3 của `github_clean_contexts.md`, đúng section “Determining when to use contexts” | 0.8401 | Có | Agent trả lời đúng rằng contexts hữu ích khi cần evaluate logic trước khi job chạy trên runner |
| 4 | What are the two ways to define custom variables in GitHub Actions workflows? | Chunk 1 của `github_clean_variables.md`, đúng tài liệu `variables`; query này có dùng filter `topic=variables` | 0.8029 | Có | Agent trả lời đúng hai cách: dùng `env` cho workflow đơn lẻ hoặc configuration variable ở mức org/repo/environment |
| 5 | When should reusable workflows be preferred over composite actions? | Chunk 9 của `github_clean_reusing_workflow_configurations.md`, chứa đoạn so sánh trực tiếp reusable workflows và composite actions | 0.8545 | Có | Agent trả lời đúng rằng reusable workflows phù hợp khi cần nhiều jobs, logging riêng và chọn runner theo job |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**  
Tôi học được rằng một strategy đơn giản như `SentenceChunker` hoặc `FixedSizeChunker` vẫn có thể cho kết quả rất tốt nếu bộ dữ liệu được làm sạch kỹ và benchmark queries được thiết kế sát tài liệu. Ngoài ra, việc gắn metadata rõ ràng như `topic` và `category` giúp nhóm so sánh retrieval công bằng hơn và cũng dễ debug hơn khi kết quả chưa đúng ý.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**  
Qua phần demo, tôi thấy nhiều nhóm chú ý rất nhiều đến cách chọn query kiểm thử thay vì chỉ tập trung vào code. Điều này giúp tôi nhận ra rằng một pipeline retrieval tốt không chỉ phụ thuộc vào chunking hay embedding mà còn phụ thuộc mạnh vào chất lượng bộ benchmark và cách đặt câu hỏi.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**  
Nếu làm lại, tôi sẽ mở rộng corpus thêm vài tài liệu GitHub Actions có liên quan mạnh đến troubleshooting hoặc secrets để tạo ra các query khó hơn và dễ nhiễu hơn. Tôi cũng sẽ chuẩn hóa thêm metadata như `section_title` hoặc `source_url` để việc filter và phân tích lỗi retrieval chi tiết hơn.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **89 / 100** |
 