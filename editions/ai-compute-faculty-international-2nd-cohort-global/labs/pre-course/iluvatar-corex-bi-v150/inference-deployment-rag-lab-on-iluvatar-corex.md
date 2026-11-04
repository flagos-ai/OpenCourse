# Inference Deployment Lab on Iluvatar CoreX BI-V150

**Course relationship:** pre-course lab for the *AI Large Language Model General Course*.
**Platform:** `iluvatar-corex-bi-v150`.

This lab deploys Qwen3-4B with vLLM, the FlagOS vLLM plugin, and FlagGems, calls the OpenAI-compatible service, builds a multi-turn chat client, and then optionally layers on a local RAG knowledge base and an agentic-search retrieval loop.

The course environment has the required packages preinstalled. Run the install section below only when preparing a new environment.

## Install FlagGems and vllm-plugin-FL

Install the FlagOS components FlagGems and vllm-plugin-FL:

```bash
cd /root
pip install -U scikit-build-core==0.11 pybind11 ninja cmake
git clone https://gitcode.com/flagos-ai/FlagGems.git
cd FlagGems
git checkout 6f2585dc9c48d440d856ad75f4aedee66fac365a
pip3 install --no-build-isolation -e .

cd /root
git clone https://gitcode.com/flagos-ai/vllm-plugin-FL.git
cd vllm-plugin-FL
git checkout f11a0f4707aecae245ec81289329b208ede5b06d
pip3 install --no-build-isolation -e . --no-deps
```

## Start the service

The first start takes about 15 minutes; a second start is under 2 minutes.

```bash
bash run_serve.sh
```

The Qwen3-4B weights must be downloaded in advance to `/flagos/lab/models/Qwen3-4B`:

```bash
# Download the Qwen3-4B model (takes about 5 minutes)
# pip install -U modelscope
# modelscope download \
#   --model Qwen/Qwen3-4B \
#   --local_dir /flagos/lab/models/Qwen3-4B

# Deploy the service
export VLLM_ENGINE_ITERATION_TIMEOUT_S=36000
export VLLM_RPC_TIMEOUT=36000000
vllm serve /flagos/lab/models/Qwen3-4B --served-model-name qwen --enforce-eager --max-model-len 4096 --gpu-memory-utilization 0.48
```

Parameters:

- `--max-model-len 4096`: the maximum context length supported by the model (in tokens).
- `--gpu-memory-utilization 0.48`: vLLM uses at most 48% of the total GPU memory.

## Call the service

The first run starts producing results after about 8 minutes; the second run starts producing results within seconds.

Create a `client.py` file in the terminal:

```bash
cat > client.py << 'EOF'
from openai import OpenAI

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key="EMPTY",
    base_url=openai_api_base,
)

response = client.chat.completions.create(
    model="qwen",
    messages=[
        {"role": "user", "content": "Give me a short introduction to large language models."},
    ],
    max_tokens=1024,
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    stream=True,
)

for chunk in response:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
EOF
```

Then run:

```bash
python3 client.py
```

Expected result:

```text
Large language models (LLMs) are advanced artificial intelligence systems designed to understand and generate human-like text. They are trained on vast amounts of text data from the internet, books, articles, and other sources, allowing them to recognize patterns, context, and meaning in language. LLMs can perform a wide range of tasks, such as answering questions, writing essays, coding, creating stories, and even engaging in conversations. These models are built using deep learning techniques, particularly transformer architectures, which enable them to process and generate text efficiently and effectively. Their ability to handle complex language tasks makes them powerful tools in fields like natural language processing, customer service, education, and more.
```

## Chat conversation

Create a `chat.py` file in the terminal:

```bash
cat > chat.py << 'EOF'
import openai
from openai import OpenAI
import socket
import sys

# 1. Auto-detect the gateway IP (works around Docker networking issues)
def get_gateway_ip():
    try:
        with open('/proc/net/route', 'r') as f:
            for line in f:
                parts = line.split()
                if parts[1] == '00000000': # default route
                    gateway_hex = parts[2]
                    # Convert hex to IP (note the byte-order reversal)
                    ip = '.'.join(str(int(gateway_hex[i:i+2], 16)) for i in (6, 4, 2, 0))
                    return ip
    except Exception:
        pass
    return "172.17.0.1" #  fallback

gateway_ip = get_gateway_ip()
# Key change: the service is confirmed to run inside this container, so use localhost directly
BASE_URL = f"http://localhost:8000/v1"
MODEL_NAME = "qwen"  # must match vllm serve --served-model-name

print(f"🔗 Connecting:: vLLM server: {BASE_URL}")
print(f"🤖 Model name: {MODEL_NAME}")
print("=" * 50)

# 2. Initialize the client (new SDK style)
client = OpenAI(
    api_key="EMPTY",
    base_url=BASE_URL,
)

# 3. Initialize the conversation history
messages = [
    {"role": "system", "content": "You are a helpful, friendly, and knowledgeable assistant powered by Qwen3-4B."}
]

try:
    # Test connectivity first
    client.models.list()
    print("✅ Connected! Start chatting (type 'quit' to exit)\n")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("💡 Please check:")
    print(f"   1. Whether the server is started and listening on --host 0.0.0.0")
    print(f"   2. Whether the server IP is {gateway_ip}")
    sys.exit(1)

# 4. Multi-turn conversation loop
while True:
    try:
        user_input = input("👤 You: ").strip()
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        # Append the user message
        messages.append({"role": "user", "content": user_input})

        # Call the API (streaming output for a better experience)
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            top_p=0.8,
            stream=True, # enable streaming
        )

        print("🤖 AI: ", end="", flush=True)
        full_response = ""

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content

        print("\n") # newline
        # Append the AI reply to the history
        messages.append({"role": "assistant", "content": full_response})

    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted, exiting")
        break
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("💡 The server may have disconnected; check the server logs.")
        break
EOF
```

Run it:

```bash
python3 chat.py
```

## Optional: set up RAG

Querying specialized knowledge such as "FlagOS Overview" against the base model gives unsatisfactory answers. Configure a local RAG system to ground the answers in documentation.

Create a download directory inside the container:

```bash
mkdir -p /tmp/my_embedding_model
cd /tmp/my_embedding_model
```

Use `wget` to download all necessary embedding-model files from a domestic mirror (`hf-mirror.com`):

```bash
BASE_URL="https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
echo "🚀 Downloading model files..."
# Download core files
wget -c "$BASE_URL/modules.json"
wget -c "$BASE_URL/config_sentence_transformers.json"
wget -c "$BASE_URL/config.json"
wget -c "$BASE_URL/tokenizer_config.json"
wget -c "$BASE_URL/special_tokens_map.json"
wget -c "$BASE_URL/vocab.txt"
wget -c "$BASE_URL/tokenizer.json"
# Download the model weight files (largest files, may take a while)
# Note: all-MiniLM-L6-v2 usually uses pytorch_model.bin or model.safetensors
# Try pytorch_model.bin first; fall back to safetensors if it is missing
if ! wget -c "$BASE_URL/pytorch_model.bin"; then
    echo "pytorch_model.bin not found, trying model.safetensors..."
    wget -c "$BASE_URL/model.safetensors"
fi
# Download the 1_Pooling config (if any)
mkdir -p 1_Pooling
cd 1_Pooling
wget -c "$BASE_URL/1_Pooling/config.json" || echo "No 1_Pooling config, skipping"
cd ..
echo "✅ Download complete! File list:"
ls -lh
```

Add the knowledge-base documents:

```bash
git clone https://gitcode.com/flagos-ai/docs
```

Install `sentence_transformers`:

```bash
pip install sentence_transformers
```

Replace `chat.py` with the RAG-enabled version:

```python
import os
import sys
from typing import List, Any

try:
    from openai import OpenAI
except ImportError:
    print("❌ openai library is missing, run: pip install openai")
    sys.exit(1)

# ==========================================
# 1. Configuration
# ==========================================

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "qwen")
DEFAULT_LOCAL_PATH = "/tmp/my_embedding_model"
LOCAL_EMBEDDING_PATH = os.environ.get("LOCAL_EMBEDDING_PATH", DEFAULT_LOCAL_PATH)

# 👇 [IMPORTANT] Confirm your knowledge base root directory here
# Point it to the outermost directory that contains docs; the program recursively scans subdirectories
# If your files live under /root/docs/docs/docs/..., set "/root/docs" here
KNOWLEDGE_BASE_DIR = os.environ.get("KNOWLEDGE_BASE_DIR", "/root/docs")

print(f"🔗 Connecting to vLLM: {BASE_URL}")
print(f"📝 Using model name: {MODEL_NAME}")
print(f"📂 Knowledge base root: {KNOWLEDGE_BASE_DIR}")

# ==========================================
# 2. Initialize the embedding model
# ==========================================

print(f"\n📚 Loading the embedding model...")
try:
    from sentence_transformers import SentenceTransformer

    if LOCAL_EMBEDDING_PATH and os.path.exists(LOCAL_EMBEDDING_PATH):
        files = os.listdir(LOCAL_EMBEDDING_PATH)
        has_weights = any(f.endswith('.bin') or f.endswith('.safetensors') for f in files)
        if not has_weights:
            raise FileNotFoundError("No model weight files found (.bin or .safetensors)")
        print(f"   ✅ Loading local model: {LOCAL_EMBEDDING_PATH}")
        embed_model = SentenceTransformer(LOCAL_EMBEDDING_PATH, trust_remote_code=True)
    else:
        print("   ⚠️ Local model not found, trying online download (all-MiniLM-L6-v2)...")
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", trust_remote_code=True)
    print("✅ Embedding model ready!\n")

except Exception as e:
    print(f"\n❌ Failed to load the embedding model: {e}")
    sys.exit(1)

# ==========================================
# 3. RAG logic (optimized to recursive scanning)
# ==========================================

class SimpleVectorStore:
    def __init__(self, model):
        self.model = model
        self.documents: List[str] = []
        self.vectors: List[Any] = []

    def add_documents(self, docs: List[str]):
        if not docs:
            return
        print(f"   📝 Encoding and indexing {len(docs)} text chunks...")
        vectors = self.model.encode(docs, convert_to_numpy=True, show_progress_bar=False)
        self.documents.extend(docs)
        self.vectors.extend(vectors)
        print(f"   ✅ Indexing complete. Total chunks in store: {len(self.documents)}\n")

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve the most relevant chunks"""
        if not self.documents:
            return []

        query_vector = self.model.encode([query], convert_to_numpy=True)[0]

        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        corpus_vectors = np.array(self.vectors)
        similarities = cosine_similarity([query_vector], corpus_vectors)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        # [Debug] Print the retrieved chunks and scores
        print(f"\n   🔍 [Retrieval details] Top {len(top_indices)} results:")
        results = []
        for i, idx in enumerate(top_indices):
            score = similarities[idx]
            doc = self.documents[idx]
            preview = doc[:80].replace('\n', ' ') + "..." if len(doc) > 80 else doc.replace('\n', ' ')
            print(f"      [{i+1}] Score: {score:.4f} | Content: {preview}")

            # Keep results with a score above 0.05, or at least the top few of top_k
            if score > 0.05:
                results.append(doc)

        return results

vector_store = SimpleVectorStore(embed_model)

def load_knowledge_base():
    """
    Recursively scan all .md and .txt files under the directory
    """
    docs = []
    abs_base_dir = os.path.abspath(KNOWLEDGE_BASE_DIR)

    if not os.path.exists(abs_base_dir):
        print(f"   ❌ Error: directory does not exist -> {abs_base_dir}")
        print("   💡 Tip: check the KNOWLEDGE_BASE_DIR setting or create the directory.")
        return docs

    print(f"   📂 Starting recursive scan: {abs_base_dir}")
    files_count = 0
    total_chunks = 0

    # ✅ Core change: use os.walk to recursively traverse all subdirectories
    for root, dirs, files in os.walk(abs_base_dir):
        # Optional: skip hidden folders
        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for filename in files:
            if filename.endswith(".txt") or filename.endswith(".md"):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()

                        # Smart chunking logic
                        paragraphs = content.split('\n\n')
                        chunks = []
                        for p in paragraphs:
                            p = p.strip()
                            if not p:
                                continue
                            # Split long paragraphs further by sentence-ending periods
                            if len(p) > 300:
                                sub_chunks = [s.strip() for s in p.replace('\n', ' ').split('.') if len(s.strip()) > 20]
                                chunks.extend(sub_chunks)
                            else:
                                # Short paragraphs become chunks directly
                                clean_p = p.replace('\n', ' ')
                                if len(clean_p) > 10:
                                    chunks.append(clean_p)

                        if chunks:
                            rel_path = os.path.relpath(filepath, abs_base_dir)
                            print(f"      ✅ Read: {rel_path} (split into {len(chunks)} chunks)")
                            docs.extend(chunks)
                            total_chunks += len(chunks)
                            files_count += 1

                except Exception as e:
                    print(f"   ⚠️ Failed to read {filepath}: {e}")

    if files_count == 0:
        print("   ⚠️ Warning: no .txt or .md files found in the directory or its subdirectories!")
        print("   💡 Check that the files exist and the KNOWLEDGE_BASE_DIR path is correct.")
    else:
        print(f"   🎉 Scan complete! Found {files_count} files, generated {total_chunks} text chunks.\n")

    return docs

# Load the knowledge base
kb_docs = load_knowledge_base()
if kb_docs:
    vector_store.add_documents(kb_docs)
else:
    print("   ⚠️ The knowledge base is empty, so RAG cannot provide context reference.\n")

# ==========================================
# 4. Conversation loop
# ==========================================

client = OpenAI(api_key="EMPTY", base_url=BASE_URL)

print("="*60)
print(f"🤖 RAG bot ready (model: {MODEL_NAME})")
print("="*60)
print("Type 'quit' or 'exit' to exit.\n")

while True:
    try:
        user_input = input("👤 You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        # Retrieve relevant context
        context_chunks = vector_store.search(user_input, top_k=5)

        # Build the prompt
        if context_chunks:
            # Take the top 3 most relevant chunks as context
            final_context = "\n".join([f"- {chunk}" for chunk in context_chunks[:3]])
            system_prompt = f"""You are an intelligent assistant. Answer the user's question strictly based on the following [Reference Information].
If the answer is not in the reference information, say "cannot answer based on the available materials" directly; do not make up facts.

[Reference Information]:
{final_context}
"""
        else:
            system_prompt = "You are an intelligent assistant. No related reference materials were found, so answer based on general knowledge, but note that specific document details may be missing."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        print(f"🤖 AI: ", end="", flush=True)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            stream=True,
            extra_body={"top_k": 20}
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n")

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
```

With RAG enabled, the assistant answers specialized questions ("FlagOS Overview") from the local knowledge base instead of hallucinating.

## Optional: set up RAG + agentic search

Upgrade `chat.py` so a retrieval agent analyzes each question, rewrites queries, runs multiple retrieval rounds, and verifies whether the retrieved context is sufficient before answering:

```python
import os
import sys
from typing import List, Any, Dict

try:
    from openai import OpenAI
except ImportError:
    print("❌ openai library is missing, run: pip install openai")
    sys.exit(1)

# ==========================================
# 1. Configuration
# ==========================================
BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "qwen")
DEFAULT_LOCAL_PATH = "/tmp/my_embedding_model"
LOCAL_EMBEDDING_PATH = os.environ.get("LOCAL_EMBEDDING_PATH", DEFAULT_LOCAL_PATH)
KNOWLEDGE_BASE_DIR = os.environ.get("KNOWLEDGE_BASE_DIR", "/root/docs")

print(f"🔗 Connecting to vLLM: {BASE_URL}")
print(f"📝 Using model name: {MODEL_NAME}")
print(f"📂 Knowledge base root: {KNOWLEDGE_BASE_DIR}")

# ==========================================
# 2. Initialize the embedding model
# ==========================================
print(f"\n📚 Loading the embedding model...")
try:
    from sentence_transformers import SentenceTransformer

    if LOCAL_EMBEDDING_PATH and os.path.exists(LOCAL_EMBEDDING_PATH):
        files = os.listdir(LOCAL_EMBEDDING_PATH)
        has_weights = any(f.endswith('.bin') or f.endswith('.safetensors') for f in files)
        if not has_weights:
            raise FileNotFoundError("No model weight files found (.bin or .safetensors)")
        print(f" ✅ Loading local model: {LOCAL_EMBEDDING_PATH}")
        embed_model = SentenceTransformer(LOCAL_EMBEDDING_PATH, trust_remote_code=True)
    else:
        print(" ⚠️ Local model not found, trying online download (all-MiniLM-L6-v2)...")
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", trust_remote_code=True)
    print("✅ Embedding model ready!\n")
except Exception as e:
    print(f"\n❌ Failed to load the embedding model: {e}")
    sys.exit(1)

# ==========================================
# 3. Vector store module (reuses the original logic)
# ==========================================
class SimpleVectorStore:
    def __init__(self, model):
        self.model = model
        self.documents: List[str] = []
        self.vectors: List[Any] = []

    def add_documents(self, docs: List[str]):
        if not docs:
            return
        print(f" 📝 Encoding and indexing {len(docs)} text chunks...")
        vectors = self.model.encode(docs, convert_to_numpy=True, show_progress_bar=False)
        self.documents.extend(docs)
        self.vectors.extend(vectors)
        print(f" ✅ Indexing complete. Total chunks in store: {len(self.documents)}\n")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the most relevant chunks (returns dicts with scores, so the Agent can judge)"""
        if not self.documents:
            return []
        query_vector = self.model.encode([query], convert_to_numpy=True)[0]
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        corpus_vectors = np.array(self.vectors)
        similarities = cosine_similarity([query_vector], corpus_vectors)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        # Return scored results so the Agent can analyze them
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            doc = self.documents[idx]
            results.append({
                "content": doc,
                "similarity_score": score,
                "preview": doc[:80].replace('\n', ' ') + "..." if len(doc) > 80 else doc.replace('\n', ' ')
            })
        return results

vector_store = SimpleVectorStore(embed_model)

# ==========================================
# 4. Knowledge base loading (reuses the original logic)
# ==========================================
def load_knowledge_base():
    docs = []
    abs_base_dir = os.path.abspath(KNOWLEDGE_BASE_DIR)
    if not os.path.exists(abs_base_dir):
        print(f" ❌ Error: directory does not exist -> {abs_base_dir}")
        print(" 💡 Tip: check the KNOWLEDGE_BASE_DIR setting or create the directory.")
        return docs

    print(f" 📂 Starting recursive scan: {abs_base_dir}")
    files_count = 0
    total_chunks = 0
    for root, dirs, files in os.walk(abs_base_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for filename in files:
            if filename.endswith(".txt") or filename.endswith(".md"):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        paragraphs = content.split('\n\n')
                        chunks = []
                        for p in paragraphs:
                            p = p.strip()
                            if not p:
                                continue
                            if len(p) > 300:
                                sub_chunks = [s.strip() for s in p.replace('\n', ' ').split('.') if len(s.strip()) > 20]
                                chunks.extend(sub_chunks)
                            else:
                                clean_p = p.replace('\n', ' ')
                                if len(clean_p) > 10:
                                    chunks.append(clean_p)
                        if chunks:
                            rel_path = os.path.relpath(filepath, abs_base_dir)
                            print(f" ✅ Read: {rel_path} (split into {len(chunks)} chunks)")
                            docs.extend(chunks)
                            total_chunks += len(chunks)
                            files_count += 1
                except Exception as e:
                    print(f" ⚠️ Failed to read {filepath}: {e}")

    if files_count == 0:
        print(" ⚠️ Warning: no .txt or .md files found in the directory or its subdirectories!")
        print(" 💡 Check that the files exist and the KNOWLEDGE_BASE_DIR path is correct.")
    else:
        print(f" 🎉 Scan complete! Found {files_count} files, generated {total_chunks} text chunks.\n")
    return docs

# Load the knowledge base
kb_docs = load_knowledge_base()
if kb_docs:
    vector_store.add_documents(kb_docs)
else:
    print(" ⚠️ The knowledge base is empty, so RAG cannot provide context reference.\n")

# ==========================================
# 5. Agentic Search core module (new)
# ==========================================
client = OpenAI(api_key="EMPTY", base_url=BASE_URL)

class RetrievalAgent:
    """Retrieval agent: autonomously analyzes questions, adjusts retrieval strategy, and performs multi-round retrieval"""
    def __init__(self, vector_store, llm_client, model_name):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.model_name = model_name
        self.max_retries = 2  # max retrieval retries
        self.min_score_threshold = 0.1 # minimum similarity threshold

    def analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze the user's question and output a retrieval strategy"""
        prompt = f"""
You are a retrieval strategy agent. Analyze the user's question and output a retrieval strategy in JSON format. Field descriptions:
1. is_complex: boolean, whether it is a complex question (needs decomposition/multi-round retrieval)
2. core_keywords: list, core retrieval keywords (3-5)
3. rewritten_queries: list, rewritten retrieval questions (2-3, more precise)
4. top_k: integer, suggested number of results (3-8)

User question: {question}

Requirements: your reply must be one and only one valid JSON object. Do not include any Markdown formatting (such as ```json), no preamble or closing remarks. Start with {{ and end with }}.
"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512
            )
            import json
            import re
            content = response.choices[0].message.content.strip()
            # Extract the first { ... } code block using regex
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                raise ValueError("No valid JSON object found in the model output")
        except Exception as e:
            # Fall back to the default strategy if parsing fails
            print(f" ⚠️ Question analysis failed, using the default strategy: {e}")
            return {
                "is_complex": False,
                "core_keywords": [question],
                "rewritten_queries": [question],
                "top_k": 5
            }

    def check_retrieval_quality(self, results: List[Dict], question: str) -> Dict[str, Any]:
        """Check retrieval result quality and decide whether supplementary retrieval is needed"""
        results_str = "\n".join([
            f"[{i+1}] similarity: {r['similarity_score']:.4f} | content: {r['preview']}"
            for i, r in enumerate(results)
        ])
        prompt = f"""
You are a retrieval result verification agent. Analyze whether the following retrieval results are sufficient to answer the user's question and output the result in JSON format:
1. is_sufficient: boolean, whether the results are sufficient (similarity > 0.1 and content is relevant)
2. need_supplement: boolean, whether supplementary retrieval is needed
3. supplement_query: string, the supplementary retrieval question (empty if not needed)

User question: {question}
Retrieval results:
{results_str}

Requirements: your reply must be one and only one valid JSON object. Do not include any Markdown formatting (such as ```json), no preamble or closing remarks. Start with {{ and end with }}.
"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512
            )
            import json
            import re
            content = response.choices[0].message.content.strip()
            # Extract the first { ... } code block using regex
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                raise ValueError("No valid JSON object found in the model output")
        except Exception as e:
            print(f" ⚠️ Result verification failed: {e}")
            return {
                "is_sufficient": len(results) > 0 and results[0]["similarity_score"] > self.min_score_threshold,
                "need_supplement": False,
                "supplement_query": ""
            }

    def agentic_search(self, question: str) -> List[str]:
        """Agentic Search main flow: multi-round retrieval + strategy adjustment"""
        print(f"\n🤖 The agent starts analyzing the retrieval strategy...")
        # Step 1: analyze the question and determine the initial retrieval strategy
        strategy = self.analyze_question(question)
        print(f" 📊 Question analysis: {'complex question' if strategy['is_complex'] else 'simple question'}")
        print(f" 🔑 Core keywords: {strategy['core_keywords']}")

        all_results = []
        retry_count = 0

        # Step 2: initial retrieval (using rewritten queries)
        for idx, query in enumerate(strategy["rewritten_queries"]):
            print(f" 🔍 Round {idx+1} retrieval (query: {query})...")
            results = self.vector_store.search(query, top_k=strategy["top_k"])
            all_results.extend(results)
            if len(all_results) >= strategy["top_k"]:
                break

        # Deduplicate and sort by similarity
        unique_results = self._deduplicate_results(all_results)
        unique_results = sorted(unique_results, key=lambda x: x["similarity_score"], reverse=True)

        # Step 3: verify results, supplement retrieval if needed
        while retry_count < self.max_retries:
            check_result = self.check_retrieval_quality(unique_results, question)
            if check_result["is_sufficient"]:
                break
            if not check_result["need_supplement"]:
                break

            # Supplementary retrieval
            retry_count += 1
            supplement_query = check_result["supplement_query"] or question
            print(f" 🔍 Supplementary retrieval (attempt {retry_count}): {supplement_query}")
            supplement_results = self.vector_store.search(supplement_query, top_k=strategy["top_k"])
            all_results.extend(supplement_results)
            # Re-deduplicate and re-sort
            unique_results = self._deduplicate_results(all_results)
            unique_results = sorted(unique_results, key=lambda x: x["similarity_score"], reverse=True)

        # Step 4: filter low-quality results and output the final context
        final_results = [
            r["content"] for r in unique_results if r["similarity_score"] > self.min_score_threshold
        ]

        # Print retrieval details
        print(f"\n 🔍 [Agentic Search final results] Top {len(final_results)} relevant chunks:")
        for i, r in enumerate(unique_results[:5]):
            print(f" [{i+1}] Score: {r['similarity_score']:.4f} | Content: {r['preview']}")

        return final_results[:3] # take the top 3 most relevant

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Deduplicate retrieval results (based on content similarity)"""
        seen = set()
        unique = []
        for r in results:
            # Deduplicate using a content hash (simple version)
            content_hash = hash(r["content"][:100])
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(r)
        return unique

# Initialize the retrieval agent
retrieval_agent = RetrievalAgent(vector_store, client, MODEL_NAME)

# ==========================================
# 6. Conversation loop (upgraded to Agentic Search)
# ==========================================
print("="*60)
print(f"🤖 Agentic RAG bot ready (model: {MODEL_NAME})")
print("="*60)
print("Type 'quit' or 'exit' to exit.\n")

while True:
    try:
        user_input = input("👤 You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        # Core upgrade: use Agentic Search instead of the original single-shot retrieval
        context_chunks = retrieval_agent.agentic_search(user_input)

        # Build the prompt (logic unchanged)
        if context_chunks:
            final_context = "\n".join([f"- {chunk}" for chunk in context_chunks[:3]])
            system_prompt = f"""You are an intelligent assistant. Answer the user's question strictly based on the following [Reference Information].
If the answer is not in the reference information, say "cannot answer based on the available materials" directly; do not make up facts.

[Reference Information]:
{final_context}
"""
        else:
            system_prompt = "You are an intelligent assistant. No related reference materials were found, so answer based on general knowledge, but note that specific document details may be missing."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        print(f"\n🤖 AI: ", end="", flush=True)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            stream=True,
            extra_body={"top_k": 20}
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n")

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
```

## Completion criteria

The service stays healthy while `client.py` streams a coherent completion and `chat.py` holds a multi-turn conversation. With the optional stages, the RAG bot grounds specialized answers in the local knowledge base, and the agentic-search bot logs its retrieval strategy and relevance scores before answering. Record the model, endpoint, platform, and any runtime error in your lab notes.
