# Task 3: RAG with Unsloth Dynamic 4-bit Quantization

A Retrieval-Augmented Generation (RAG) pipeline that answers questions from your own documents using an **Unsloth dynamic 4-bit quantized LLM**. It runs end to end on the free Google Colab **Tesla T4** GPU and peaks at only **2.58 GB of VRAM**.

> Internship project for ArchTechnologies  
> Author: **[Your Full Name]** | Domain: **[Your Internship Domain]** | Email: **[your.email@example.com]**

---

## What this project does

1. Loads `unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit`, a **dynamic** 4-bit model (note the `unsloth-bnb-4bit` suffix).
2. Loads domain documents (TXT / PDF) and splits them into overlapping chunks.
3. Embeds the chunks with `BAAI/bge-small-en-v1.5` and stores them in a **FAISS** index.
4. For each question, retrieves the top 4 most similar chunks with a **LangChain** retriever.
5. Builds a prompt that tells the model to answer **only from the retrieved context** and generates a grounded answer.
6. If the answer is not in the documents, the model replies: *"I could not find this in the provided documents."*

### Why dynamic 4-bit?

Standard 4-bit quantization (bitsandbytes NF4) quantizes every linear layer. Unsloth's dynamic quantization measures the quantization error of each layer and keeps the most sensitive layers/parameters in higher precision (16-bit), while the rest stay in 4-bit. The result is better accuracy than plain 4-bit for only a small increase in VRAM.

## Architecture

```
Documents (TXT/PDF)
      |
Text splitter (500 chars, 80 overlap)
      |
BGE-small embeddings --> FAISS index
                              |
User question --> Retriever (top-k = 4) --> Prompt (context + question)
                                                  |
                              Llama-3.2-3B, Unsloth dynamic 4-bit
                                                  |
                                           Grounded answer
```

## Tech stack

| Component | Choice |
|---|---|
| LLM | `unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit` |
| Inference engine | Unsloth `FastLanguageModel` |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Vector store | FAISS (`faiss-cpu`) |
| RAG framework | LangChain (retriever + LCEL chain) |
| Splitter | `RecursiveCharacterTextSplitter` |
| Loaders | `PyPDFLoader`, plain text loader |
| Hardware | Google Colab, Tesla T4 (14.56 GB VRAM) |

## How to run

1. Open `Task3_RAG_Unsloth_Dynamic4bit.ipynb` in Google Colab.
2. Select `Runtime > Change runtime type > T4 GPU`.
3. Run the cells from top to bottom.
   - Section 1 installs the dependencies (takes a few minutes).
   - If a package error appears after installing, restart the runtime and run again from Section 2.
4. The notebook creates three sample documents about LLM quantization in `/content/docs`.
5. **Use your own documents:** run the last cell, upload PDF/TXT files, and ask questions with `ask("your question")`.

## Notebook structure

| Section | Content |
|---|---|
| 1 | Install dependencies |
| 2 | GPU check and the `vram()` memory helper |
| 3 | Load the dynamic 4-bit model |
| 4 | Create the sample document collection |
| 5 | Load documents and split into chunks |
| 6 | Embeddings and FAISS index |
| 7 | Build the RAG chain (LangChain LCEL + Unsloth LLM) |
| 8 | Demo: grounded answers |
| 8b | Out-of-scope question and with/without-RAG comparison |
| 9 | Memory report |
| 10 | Optional: your own documents |

## Key settings

| Setting | Value | Reason |
|---|---|---|
| `max_seq_length` | 4096 | Fits 4 chunks, the question and the answer |
| Chunk size / overlap | 500 / 80 characters | Precise matches, facts are not cut at chunk borders |
| Top-k | 4 | Enough context, small prompt |
| Temperature | 0.2 | Stable, factual answers |
| `max_new_tokens` | 300 | Bounds generation and KV-cache memory |

## Results

### Answer quality (sample documents)

| Question | Result |
|---|---|
| What does dynamic 4-bit do differently from standard 4-bit? | Correct: it avoids quantizing the layers that cause the largest errors |
| How much extra VRAM does dynamic quantization need? | Correct: about 10 percent over standard 4-bit |
| Memory for a 7B model, 4-bit vs 16-bit? | Correct: about 3.5 to 4 GB vs about 14 GB |
| Why does RAG reduce hallucination? | Correct (brief) |
| What is NF4? | Correct (brief) |
| Who won the FIFA World Cup in 2018? (out of scope) | Correctly refused |

**With vs without RAG:** asked which suffix marks Unsloth dynamic models, the plain model guessed wrongly (a "U" suffix). With RAG it answered `-unsloth-bnb-4bit` correctly.

### GPU memory

| Stage | Allocated | Peak |
|---|---|---|
| Baseline | 0.01 GB | 0.01 GB |
| After loading the dynamic 4-bit LLM | 2.24 GB | 2.25 GB |
| After the RAG demo queries | 2.51 GB | 2.58 GB |

Peak usage was **2.58 GB of 14.6 GB (17.7 percent of the T4)**. `nvidia-smi` showed 2,745 MiB used out of 15,360 MiB.

### Own-document test

A 26-page lecture PDF was added and re-indexed into 41 chunks. Specific questions work well. See the limitations below for broad summaries.

## Memory-saving techniques used

- Dynamic 4-bit weights (3B model in about 2.2 GB instead of roughly 6 to 7 GB in fp16)
- A small embedding model (about 130 MB)
- Small, bounded context (500-character chunks, k = 4)
- `max_new_tokens=300` and `torch.inference_mode()`
- `gc.collect()` and `torch.cuda.empty_cache()` after the run
- `faiss-cpu` keeps the vector index in system RAM, not VRAM

## Known limitations

- **Broad summaries of long PDFs are weak.** A top-4 retriever with small chunks only sees fragments of a long slide deck. For summaries use a larger `k`, larger chunks, or a map-reduce summarisation step.
- **Mixed index.** After uploading a PDF, the sample files stay in `/content/docs` and can appear in the retrieved context. Delete them or build a separate index per document set.
- **FAISS scores are distances.** Lower means a better match.
- **Harmless warnings.** `max_new_tokens` and `max_length` both set (the former takes precedence), and a LangChain community deprecation notice.
- **Sampling.** `do_sample=True`, so answer wording can vary slightly between runs.

## Possible improvements

- Add a re-ranker and hybrid (keyword plus vector) search
- Try a larger dynamic 4-bit model, for example Llama-3.1-8B or Qwen2.5-7B
- Evaluate answers automatically (faithfulness, answer relevance)

## Files

| File | Description |
|---|---|
| `Task3_RAG_Unsloth_Dynamic4bit.ipynb` | The Google Colab notebook |
| `README_Task3.md` | This file |

## Links

- Colab / GitHub: **[paste link]**
- LinkedIn post: **[paste link]**
