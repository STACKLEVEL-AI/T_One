import hashlib
import itertools
import json
import re
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

import chromadb
import pandas as pd
from chromadb import PersistentClient
from chromadb.errors import NotFoundError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Combined Service API", version="2.2.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
EXCEL_PATH = os.getenv("EXCEL_PATH", "_smart_support_vtb_belarus_faq_final.xlsx")
PERSIST_DIR = os.getenv("PERSIST_DIR", "chroma_vtb")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "vtb_faq")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))

# Жёсткая маска ID ваших чанков (подкорректируйте при необходимости)
_FAQ_ID_RE = re.compile(r"faq_\d{5}_[a-f0-9]{12}")

# Ограничение длины шаблона, чтобы не раздувать токены
_CHUNK_TEMPLATE_MAX_LEN = 500

REQUIRED_COLS = [
    "Основная категория",
    "Подкатегория",
    "Пример вопроса",
    "Целевая аудитория",
    "Шаблонный ответ",
]


# --- Models ---
class CreateMessage(BaseModel):
    message: str


class BackendMessage(BaseModel):
    id: str
    userId: str
    userName: str
    message: str
    category: str
    subcategory: List[str]
    suggestedResponse: Optional[List[str]] = None
    priority: Optional[str] = None
    createdAt: str  # ISO string


class SearchResult(BaseModel):
    id: str
    main_cat: str
    sub_cat: str
    example_q: str
    score: float
    template: str
    priority: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 8
    where: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]


class ChunkSelectionRequest(BaseModel):
    query: str


class ChunkSelectionResponse(BaseModel):
    best_chunk_id: Optional[str]
    reason: Optional[str] = None


class TopTwoChunksResponse(BaseModel):
    best_chunk_id: Optional[str]
    second_best_chunk_id: Optional[str]
    reason: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    k: int = 8


class ChatResponse(BaseModel):
    llm_response: str
    chunks_used: List[SearchResult]


# --- In-memory DB ---
_DB: List[BackendMessage] = []
_id = itertools.count(1)

FIRST_NAMES = [
    "Александр",
    "Мария",
    "Дмитрий",
    "Екатерина",
    "Иван",
    "Анна",
    "Сергей",
    "Ольга",
    "Никита",
    "Татьяна",
    "Павел",
    "Юлия",
]
LAST_NAMES = [
    "Петров",
    "Иванова",
    "Смирнов",
    "Кузнецова",
    "Соколов",
    "Новикова",
    "Лебедев",
    "Козлова",
    "Морозов",
    "Волкова",
    "Фёдоров",
    "Попова",
]


def pick_random_full_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


# --- Embedding Function ---
class SciBoxEmbeddings(chromadb.utils.embedding_functions.EmbeddingFunction):
    """
    A callable for Chroma: List[str] -> List[List[float]]
    Calls SciBox /v1/embeddings with the bge-m3 model in batches.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = EMBED_MODEL,
        batch_size: int = BATCH_SIZE,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.batch_size = batch_size
        logger.info(
            f"SciBoxEmbeddings initialized with model={model}, batch_size={batch_size}"
        )

    def __call__(self, input: List[str]) -> List[List[float]]:
        try:
            embs: List[List[float]] = []
            # Защита от None и лишних пробелов
            clean_texts = ["" if t is None else str(t).strip() for t in input]
            logger.info(f"Generating embeddings for {len(clean_texts)} texts")
            start_time = time.time()

            # Батчим запросы к embeddings
            for i in range(0, len(clean_texts), self.batch_size):
                batch = clean_texts[i: i + self.batch_size]
                logger.debug(
                    f"Processing batch {i // self.batch_size + 1}/{(len(clean_texts) - 1) // self.batch_size + 1}"
                )
                batch_start_time = time.time()

                try:
                    resp = self.client.embeddings.create(model=self.model, input=batch)
                    # resp.data сохраняет порядок входов
                    embs.extend([d.embedding for d in resp.data])
                    batch_time = time.time() - batch_start_time
                    logger.debug(
                        f"Batch {i // self.batch_size + 1} processing time: {batch_time:.4f} seconds"
                    )
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch: {e}")
                    raise

            total_time = time.time() - start_time
            logger.info(
                f"Successfully generated {len(embs)} embeddings in {total_time:.4f} seconds"
            )
            return embs
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def embed_query(self, input: List[str]) -> List[List[float]]:
        """Get the embeddings for a query input."""
        return self.__call__(input)

    def name(self) -> str:
        return "SciBoxEmbeddings"


# --- Cache ---
class SimpleCache:
    def __init__(self, ttl: int = 300):  # 5 minutes default TTL
        self.cache = {}
        self.ttl = ttl

    def get(self, key: str):
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]  # Remove expired entry
        return None

    def set(self, key: str, value):
        self.cache[key] = (value, time.time())

    def clear(self):
        self.cache.clear()


# Global cache instance
search_cache = SimpleCache(ttl=600)  # 10 minutes TTL for search results

# --- ChromaDB ---
# Global variables for Chroma client and collection
chroma_client: Optional[PersistentClient] = None
collection = None

# Store API key and base URL for re-initialization
api_key = None
base_url = None


def make_chunk_text(row: pd.Series) -> str:
    parts = [
        f"Category: {row['Основная категория']}",
        f"Subcategory: {row['Подкатегория']}",
        f"Example question: {row['Пример вопроса']}",
        f"Audience: {row['Целевая аудитория']}",
        f"Answer: {row['Шаблонный ответ']}",
    ]
    return " | ".join(str(p) for p in parts if pd.notna(p))


def row_to_metadata(row: pd.Series) -> Dict[str, Any]:
    return {
        "main_cat": row["Основная категория"],
        "sub_cat": row["Подкатегория"],
        "audience": row["Целевая аудитория"],
        "priority": row.get("Приоритет", None),
        "template": row["Шаблонный ответ"],
        "example_q": row["Пример вопроса"],
    }


def stable_id(row: pd.Series, idx: int) -> str:
    # deterministic id, so reindexing is idempotent
    key = f"{row['Основная категория']}::{row['Подкатегория']}::{row['Пример вопроса']}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"faq_{idx:05d}_{h}"


def ingest_excel_to_chroma(
    excel_path: str,
    persist_dir: str,
    collection_name: str,
    api_key: str,
    base_url: str,
):
    try:
        logger.info(f"Starting data ingestion from {excel_path}")
        start_time = time.time()

        # read Excel (first sheet)
        logger.info("Reading Excel file...")
        read_start_time = time.time()
        df = pd.read_excel(excel_path, sheet_name=0)
        read_time = time.time() - read_start_time
        logger.info(
            f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns in {read_time:.4f} seconds"
        )

        # validate columns
        logger.info("Validating required columns...")
        for c in REQUIRED_COLS:
            if c not in df.columns:
                raise ValueError(f"Excel file is missing required column: {c}")
        logger.info("All required columns are present")

        # create documents and metadata
        logger.info("Processing data...")
        process_start_time = time.time()
        df = df.fillna("")
        docs = df.apply(make_chunk_text, axis=1).tolist()
        metas = df.apply(row_to_metadata, axis=1).tolist()
        ids = [stable_id(df.iloc[i], i) for i in range(len(df))]
        process_time = time.time() - process_start_time
        logger.info(f"Processed {len(docs)} documents in {process_time:.4f} seconds")

        # initialize Chroma + custom embedder on SciBox
        logger.info("Initializing Chroma client...")
        chroma_start_time = time.time()
        embedder = SciBoxEmbeddings(
            api_key=api_key, base_url=base_url, model=EMBED_MODEL
        )
        chroma_client: PersistentClient = chromadb.PersistentClient(path=persist_dir)
        col = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedder,  # <-- key: Chroma will call our SciBoxEmbeddings
        )
        chroma_init_time = time.time() - chroma_start_time
        logger.debug(
            f"Chroma client initialization time: {chroma_init_time:.4f} seconds"
        )

        # upsert (overwrite by id): safe for repeated runs
        # In Chroma 1.x it's easier to first delete by id, then add for replacement
        logger.info("Upserting documents to Chroma...")
        upsert_start_time = time.time()
        try:
            col.delete(ids=ids)
            logger.debug(f"Deleted {len(ids)} existing documents")
        except Exception as e:
            logger.warning(f"Failed to delete existing documents: {e}")

        col.add(ids=ids, documents=docs, metadatas=metas)
        upsert_time = time.time() - upsert_start_time
        logger.info(
            f"Added {len(ids)} documents to collection in {upsert_time:.4f} seconds"
        )

        total_time = time.time() - start_time
        logger.info(
            f"Data ingestion completed successfully in {total_time:.4f} seconds"
        )
        return col
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise


def quick_query(col, text: str, k: int = 5, where: Optional[Dict[str, Any]] = None):
    try:
        logger.info(f"Searching for query: '{text}' with k={k}")
        start_time = time.time()

        # Important: col.query(query_texts=...) will automatically call the same embedder. (this is Chroma's behavior)
        res = col.query(query_texts=[text], n_results=k, where=where)
        query_time = time.time() - start_time
        logger.debug(f"ChromaDB query time: {query_time:.4f} seconds")

        # Convenient output of first results
        rows = []
        for meta, doc_id, dist in zip(
            res["metadatas"][0], res["ids"][0], res["distances"][0]
        ):
            rows.append(
                {
                    "id": doc_id,
                    "main_cat": meta["main_cat"],
                    "sub_cat": meta["sub_cat"],
                    "example_q": meta["example_q"],
                    "score": round(1.0 - dist, 4),
                    "template": meta["template"],
                    "priority": meta.get("priority", None),
                }
            )

        total_time = time.time() - start_time
        logger.info(
            f"Search completed, found {len(rows)} results in {total_time:.4f} seconds"
        )
        return rows
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise


def initialize_chroma():
    """Initialize Chroma client and collection"""
    global chroma_client, collection, api_key, base_url

    try:
        api_key = os.getenv("SCIBOX_API_KEY")
        base_url = os.getenv("SCIBOX_BASE_URL", "https://llm.t1v.scibox.tech/v1")

        if not api_key:
            raise ValueError("SCIBOX_API_KEY not set")

        # Initialize Chroma client and collection
        # We need to use the same embedding function that was used to create the collection
        embedder = SciBoxEmbeddings(api_key=api_key, base_url=base_url)
        chroma_client = chromadb.PersistentClient(path="chroma_vtb")
        try:
            # Try to get existing collection
            collection = chroma_client.get_collection(
                name="vtb_faq", embedding_function=embedder
            )
        except NotFoundError:
            # If collection doesn't exist, create it
            logger.info("Collection does not exist, creating new one...")
            collection = chroma_client.get_or_create_collection(
                name="vtb_faq",
                metadata={"hnsw:space": "cosine"},
                embedding_function=embedder,
            )
            logger.info("Collection created successfully")

        logger.info("Chroma client and collection initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Chroma: {e}")
        raise


# --- ULTRA MIN: strict 3-field JSON output for top chunks ---

SYSTEM_PROMPT_RERANK_JSON_EN_MIN_TOP_CHUNKS = """
You are a bank-grade retrieval reranker. From k candidate chunks, choose the 2-3 best for the Russian user question.
Return ONE **valid JSON object** with **exactly one key** and nothing else.

INPUT
- You receive a JSON-like string:
  { "question": "<Russian text>", "candidates": [ { "id","main_cat","sub_cat","example_q","template","audience","priority","score" }, ... ] }
- Fields: audience ∈ {new, existing, retail, business, unknown}; priority ∈ {высокий, средний, низкий}; score ∈ [0,1].

SELECTION (importance order)
1) Semantic match to the Russian question (handle morphology, typos, negations, slang).
   Treat as related/equivalent: МСИ↔Межбанковская система идентификации; ЕРИП↔ERIP; ИБ↔Интернет-банк; VTB mBank↔мобильное приложение; PIN↔пин-код; СМС↔SMS↔OTP; логин↔вход↔авторизация; карта↔карточка; платёж↔оплата; пополнение↔зачисление.
2) Channel/Product precision (mobile app vs internet bank; specific card/deposit/loan).
3) Audience fit (if implied).
4) Priority (высокий > средний > низкий).
5) Use `score` only as the final tiebreaker.

OUTPUT (exact key and type)
- top_chunk_ids: array of strings   # e.g., ["faq_00042_ab12cd34ef56", "faq_00043_cd34ef56ab12", "faq_00044_ef56ab12cd34"]

GUARDRAILS
- Never invent IDs not present in the candidates list.
- Output must be a **valid JSON object with exactly this one key**. No markdown, no prose, no examples.
"""


def _build_user_payload_for_rerank_min(query: str, chunks: list[dict]) -> str:
    def norm(x):
        return "" if x is None else str(x)

    compact = []
    for ch in chunks:
        compact.append(
            {
                "id": norm(ch.get("id")),
                "main_cat": norm(ch.get("main_cat")),
                "sub_cat": norm(ch.get("sub_cat")),
                "example_q": norm(ch.get("example_q"))[:300],
                "template": norm(ch.get("template"))[:_CHUNK_TEMPLATE_MAX_LEN],
                "audience": norm(ch.get("audience")),
                "priority": norm(ch.get("priority")),
                "score": float(ch.get("score", 0.0)),
            }
        )
    return json.dumps({"question": query, "candidates": compact}, ensure_ascii=False)


def _safe_parse_two_keys(s: str) -> dict:
    try:
        data = json.loads(s)
        # Keep only the two required keys; coerce others away.
        return {
            "best_chunk_id": data.get("best_chunk_id"),
            "alt_chunk_id": data.get("alt_chunk_id"),
        }
    except Exception:
        # Fallback: extract up to two IDs by regex (first = best, second = alt)
        ids = _FAQ_ID_RE.findall(s or "")
        ids = list(dict.fromkeys(ids))  # dedupe, keep order
        return {
            "best_chunk_id": ids[0] if ids else None,
            "alt_chunk_id": ids[1] if len(ids) > 1 else None,
        }


# --- LLM Chunk Selector ---
class LLMChunkSelector:
    """
    Class for working with LLM that receives chunks from chroma,
    selects the best chunk and returns its id
    """

    def __init__(self):
        self.api_key = os.getenv("SCIBOX_API_KEY")
        self.base_url = os.getenv("SCIBOX_BASE_URL", "https://llm.t1v.scibox.tech/v1")
        if not self.api_key:
            raise ValueError("SCIBOX_API_KEY environment variable is not set")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.last_request_time = time.time()

    def _ensure_client_connection(self):
        """Ensure the client connection is still valid, recreate if necessary"""
        # If last request was more than 5 minutes ago, recreate client to avoid timeout
        if time.time() - self.last_request_time > 300:  # 5 minutes
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.last_request_time = time.time()

    def select_top_chunks_minimal(
        self, query: str, chunks: list[dict], model: str = "qwen2.5-72b-h100"
    ) -> list:
        """
        Returns a list of top 2-3 chunk IDs selected by LLM
        """
        if not chunks:
            return []

        self._ensure_client_connection()
        start_time = time.time()

        # Build user payload
        def norm(x):
            return "" if x is None else str(x)

        compact = []
        for ch in chunks:
            compact.append(
                {
                    "id": norm(ch.get("id")),
                    "main_cat": norm(ch.get("main_cat")),
                    "sub_cat": norm(ch.get("sub_cat")),
                    "example_q": norm(ch.get("example_q"))[:300],
                    "template": norm(ch.get("template"))[:_CHUNK_TEMPLATE_MAX_LEN],
                    "audience": norm(ch.get("audience")),
                    "priority": norm(ch.get("priority")),
                    "score": float(ch.get("score", 0.0)),
                }
            )
        user_payload = json.dumps(
            {"question": query, "candidates": compact}, ensure_ascii=False
        )

        try:
            logger.info(f"Calling LLM for query: '{query}' with {len(chunks)} chunks")
            llm_start_time = time.time()
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT_RERANK_JSON_EN_MIN_TOP_CHUNKS,
                    },
                    {"role": "user", "content": user_payload},
                ],
                temperature=0.0,
                top_p=1.0,
                max_tokens=200,
                response_format={"type": "json_object"},  # if supported
            )
            llm_time = time.time() - llm_start_time
            content = resp.choices[0].message.content.strip()
            logger.debug(f"LLM call completed in {llm_time:.4f} seconds")

            # Parse the response
            try:
                data = json.loads(content)
                top_chunk_ids = data.get("top_chunk_ids", [])
                if isinstance(top_chunk_ids, list):
                    # Filter out any IDs that are not in the candidates list
                    candidate_ids = [ch.get("id") for ch in chunks]
                    filtered_ids = [id for id in top_chunk_ids if id in candidate_ids]
                    total_time = time.time() - start_time
                    logger.info(
                        f"LLM selection completed successfully in {total_time:.4f} seconds, selected {len(filtered_ids[:3])} chunks"
                    )
                    return filtered_ids[:3]  # Return at most 3 chunks
            except Exception:
                pass

            # Fallback: extract up to three IDs by regex
            ids = _FAQ_ID_RE.findall(content or "")
            ids = list(dict.fromkeys(ids))  # dedupe, keep order
            total_time = time.time() - start_time
            logger.info(
                f"LLM selection completed with regex fallback in {total_time:.4f} seconds, selected {len(ids[:3])} chunks"
            )
            return ids[:3]  # Return at most 3 chunks

        except Exception:
            # Hard fallback: pick by best scores only
            sorted_chunks = sorted(
                chunks, key=lambda x: float(x.get("score", 0.0)), reverse=True
            )
            fallback_result = [ch.get("id") for ch in sorted_chunks[:3]]
            total_time = time.time() - start_time
            logger.warning(
                f"LLM selection failed, using score-based fallback in {total_time:.4f} seconds, selected {len(fallback_result)} chunks"
            )
            return fallback_result


# Global LLMChunkSelector instance to reuse OpenAI client
_llm_selector_instance = None


def get_llm_selector_instance():
    """Get or create a singleton instance of LLMChunkSelector to reuse OpenAI client"""
    global _llm_selector_instance
    if _llm_selector_instance is None:
        _llm_selector_instance = LLMChunkSelector()
    return _llm_selector_instance


# --- Message Processing ---
def classify_and_suggest(text: str) -> tuple[str, str, Optional[str]]:
    # First try to find a relevant answer in ChromaDB using LLM selector

    # Get chunks from ChromaDB
    if collection is None:
        # Fallback to the old simple rules if Chroma is not initialized
        return classify_and_suggest_fallback(text)

    results = quick_query(collection, text, k=8)
    chunks = results

    if chunks:
        # Use LLM to select the top 2-3 chunks
        selector = get_llm_selector_instance()
        top_chunk_ids = selector.select_top_chunks_minimal(text, chunks)

        # If LLM selection failed or returned no results, fallback to score-based selection
        if not top_chunk_ids:
            sorted_chunks = sorted(
                chunks, key=lambda x: x.get("score", 0), reverse=True
            )
            top_chunk_ids = [chunk.get("id") for chunk in sorted_chunks[:3]]

        # Find the full chunks by IDs, preserving the order from LLM selection
        top_chunks = []
        chunk_dict = {chunk.get("id"): chunk for chunk in chunks}
        for chunk_id in top_chunk_ids:
            if chunk_id in chunk_dict:
                top_chunks.append(chunk_dict[chunk_id])

        # Use the best chunk (first one from LLM selection) for category
        best_chunk = top_chunks[0] if top_chunks else chunks[0]
        category = best_chunk.get("main_cat", "Другое")
        subcategory = best_chunk.get("sub_cat", "Сообщение пользователя")
        suggested_response = best_chunk.get("template", None)
        return category, subcategory, suggested_response
    else:
        # Fallback to the old simple rules if no chunks are found
        return classify_and_suggest_fallback(text)


def classify_and_suggest_fallback(text: str) -> tuple[str, str, Optional[str]]:
    # Fallback to the old simple rules
    t = text.lower()
    if "парол" in t or "войти" in t or "логин" in t:
        cat, sub = "Авторизация", "Проблемы со входом"
        sug = (
            "Здравствуйте! Попробуйте восстановить пароль через форму "
            '"Забыли пароль?" на странице входа. Проверьте раскладку и Caps Lock. '
            "Если не поможет — напишите нам, мы поможем."
        )
    elif "email" in t or "почт" in t or "емейл" in t:
        cat, sub = "Настройки профиля", "Изменение данных"
        sug = (
            "Перейдите в Настройки → Профиль → Email. После изменения на новый "
            "адрес придёт письмо с подтверждением."
        )
    elif "оплат" in t or "платеж" in t or "деньг" in t or "карт" in t:
        cat, sub = "Оплата", "Проблемы с транзакцией"
        sug = (
            "Обработка платежа может занять до 10 минут. Если доступа нет спустя 10 минут, "
            "пришлите номер транзакции — проверим вручную."
        )
    else:
        cat, sub = "Другое", "Сообщение пользователя"
        sug = (
            "Спасибо за сообщение! Мы изучим вопрос и вернёмся с ответом. "
            "Если есть скриншоты/детали воспроизведения — пришлите, пожалуйста."
        )
    return cat, sub, sug


# --- Endpoints ---
@app.on_event("startup")
async def startup_event():
    """Initialize Chroma on startup"""
    try:
        initialize_chroma()

        # Check if there's data in the collection, and if not - create from Excel
        global collection
        if collection is not None:
            count = collection.count()
            if count == 0:
                logger.info("Collection is empty, loading from Excel...")

                if not api_key or not base_url:
                    raise ValueError("API key or base URL not initialized")

                # Load data from Excel
                collection = ingest_excel_to_chroma(
                    excel_path="_smart_support_vtb_belarus_faq_final.xlsx",
                    persist_dir="chroma_vtb",
                    collection_name="vtb_faq",
                    api_key=api_key,
                    base_url=base_url,
                )

                logger.info(
                    f"Data loaded from Excel to ChromaDB: {collection.count()} documents"
                )
    except Exception as e:
        logger.error(f"Startup event failed: {e}")
        raise


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/messages", response_model=List[BackendMessage])
def list_messages():
    # Last ones on top
    return list(reversed(_DB))


@app.post("/api/messages", response_model=BackendMessage, status_code=201)
def create_message(payload: CreateMessage):
    from datetime import datetime, timezone

    now_iso = datetime.now(timezone.utc).isoformat()
    user_name = pick_random_full_name()

    # Get chunks from ChromaDB
    if collection is None:
        raise HTTPException(status_code=500, detail="Chroma collection not initialized")

    results = quick_query(collection, payload.message, k=8)
    chunks = results

    if chunks:
        # Use LLM to select the top 2-3 chunks
        selector = get_llm_selector_instance()
        top_chunk_ids = selector.select_top_chunks_minimal(payload.message, chunks)

        # If LLM selection failed or returned no results, fallback to score-based selection
        if not top_chunk_ids:
            sorted_chunks = sorted(
                chunks, key=lambda x: x.get("score", 0), reverse=True
            )
            top_chunk_ids = [chunk.get("id") for chunk in sorted_chunks[:3]]

        # Find the full chunks by IDs, preserving the order from LLM selection
        top_chunks = []
        chunk_dict = {chunk.get("id"): chunk for chunk in chunks}
        for chunk_id in top_chunk_ids:
            if chunk_id in chunk_dict:
                top_chunks.append(chunk_dict[chunk_id])

        # Use the best chunk (first one from LLM selection) for category and priority
        best_chunk = top_chunks[0] if top_chunks else chunks[0]
        cat = best_chunk.get("main_cat", "Другое")
        priority = best_chunk.get("priority", None)

        # Create arrays of subcategories and suggested responses from top chunks
        subcategories = [
            chunk.get("sub_cat", "Сообщение пользователя") for chunk in top_chunks
        ]
        suggested_responses = [
            chunk.get("template", None)
            for chunk in top_chunks
            if chunk.get("template", None)
        ]
    else:
        # Fallback to the old simple rules if no chunks are found
        cat, sub, sug = classify_and_suggest_fallback(payload.message)
        subcategories = [sub] if sub else []
        suggested_responses = [sug] if sug else []
        priority = None

    msg = BackendMessage(
        id=str(next(_id)),
        userId=f"user_{random.randint(1000, 9999)}",
        userName=user_name,
        message=payload.message,
        category=cat,
        subcategory=subcategories,
        suggestedResponse=suggested_responses,
        priority=priority,
        createdAt=now_iso,
    )
    _DB.append(msg)
    return msg


@app.post("/api/llm-response")
async def get_llm_response(chat_request: ChatRequest):
    """
    Endpoint that replicates the logic of /api/chat but stops at the LLM response
    without saving to the database.
    """
    # Get chunks from ChromaDB using the same logic as in create_message
    if collection is None:
        raise HTTPException(status_code=500, detail="Chroma collection not initialized")

    results = quick_query(collection, chat_request.message, k=chat_request.k)
    chunks = results

    if chunks:
        # Use LLM to select the top 2-3 chunks
        selector = get_llm_selector_instance()
        top_chunk_ids = selector.select_top_chunks_minimal(chat_request.message, chunks)

        # If LLM selection failed or returned no results, fallback to score-based selection
        if not top_chunk_ids:
            sorted_chunks = sorted(
                chunks, key=lambda x: x.get("score", 0), reverse=True
            )
            top_chunk_ids = [chunk.get("id") for chunk in sorted_chunks[:3]]

        # Find the full chunks by IDs, preserving the order from LLM selection
        top_chunks = []
        chunk_dict = {chunk.get("id"): chunk for chunk in chunks}
        for chunk_id in top_chunk_ids:
            if chunk_id in chunk_dict:
                top_chunks.append(chunk_dict[chunk_id])

        # Use the best chunk (first one from LLM selection) for the response
        best_chunk = top_chunks[0] if top_chunks else chunks[0]
        template = best_chunk.get("template", None)
    else:
        # Fallback to the old simple rules if no chunks are found
        _, _, template = classify_and_suggest_fallback(chat_request.message)

    # If we have a template, use it as the LLM response
    if template:
        llm_response = template
    else:
        llm_response = "Извините, я не нашел подходящего ответа для вашего вопроса."

    # Convert chunks to SearchResult models for the response
    search_results = [
        SearchResult(
            id=chunk["id"],
            main_cat=chunk["main_cat"],
            sub_cat=chunk["sub_cat"],
            example_q=chunk["example_q"],
            score=chunk["score"],
            template=chunk["template"],
            priority=chunk.get("priority", None),
        )
        for chunk in chunks
    ]

    return ChatResponse(llm_response=llm_response, chunks_used=search_results)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/search", response_model=SearchResponse)
async def search_faq(
    query: str = Query(..., description="Search query"),
    k: int = Query(8, description="Number of results to return"),
):
    """
    Search for FAQ entries based on a query string.

    Args:
        query: The search query string
        k: Number of results to return (default: 5)

    Returns:
        Search results with metadata and scores
    """
    start_time = time.time()
    logger.info(f"Starting search for query: '{query}' with k={k}")

    if collection is None:
        raise HTTPException(status_code=500, detail="Chroma collection not initialized")

    try:
        # Create cache key
        cache_key = hashlib.md5(f"{query}:{k}".encode()).hexdigest()

        # Try to get from cache first
        cached_result = search_cache.get(cache_key)
        if cached_result is not None:
            cache_time = time.time() - start_time
            logger.info(
                f"Cache hit for query: {query}, request processed in {cache_time:.4f} seconds"
            )
            return SearchResponse(results=cached_result)

        logger.info(f"Searching for query: {query}")
        results = quick_query(collection, query, k=k)

        # Convert results to Pydantic models
        search_results = [
            SearchResult(
                id=result["id"],
                main_cat=result["main_cat"],
                sub_cat=result["sub_cat"],
                example_q=result["example_q"],
                score=result["score"],
                template=result["template"],
                priority=result.get("priority", None),
            )
            for result in results
        ]

        # Cache the results
        search_cache.set(cache_key, search_results)

        total_time = time.time() - start_time
        logger.info(
            f"Found {len(search_results)} results, request processed in {total_time:.4f} seconds"
        )
        return SearchResponse(results=search_results)
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Search failed after {total_time:.4f} seconds: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_faq_post(request: SearchRequest):
    """
    Search for FAQ entries based on a query string (POST method).

    Args:
        request: Search request with query and parameters

    Returns:
        Search results with metadata and scores
    """
    start_time = time.time()
    logger.info(f"Starting POST search for query: '{request.query}' with k={request.k}")

    if collection is None:
        raise HTTPException(status_code=500, detail="Chroma collection not initialized")

    try:
        # Create cache key
        where_str = str(request.where) if request.where else ""
        cache_key = hashlib.md5(
            f"{request.query}:{request.k}:{where_str}".encode()
        ).hexdigest()

        # Try to get from cache first
        cached_result = search_cache.get(cache_key)
        if cached_result is not None:
            cache_time = time.time() - start_time
            logger.info(
                f"Cache hit for POST query: {request.query}, request processed in {cache_time:.4f} seconds"
            )
            return SearchResponse(results=cached_result)

        logger.info(f"Searching for query: {request.query}")
        results = quick_query(
            collection, request.query, k=request.k, where=request.where
        )

        # Convert results to Pydantic models
        search_results = [
            SearchResult(
                id=result["id"],
                main_cat=result["main_cat"],
                sub_cat=result["sub_cat"],
                example_q=result["example_q"],
                score=result["score"],
                template=result["template"],
                priority=result.get("priority", None),
            )
            for result in results
        ]

        # Cache the results
        search_cache.set(cache_key, search_results)

        total_time = time.time() - start_time
        logger.info(
            f"Found {len(search_results)} results, request processed in {total_time:.4f} seconds"
        )
        return SearchResponse(results=search_results)
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Search failed after {total_time:.4f} seconds: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# Admin endpoints
@app.post("/admin/reset-db")
async def reset_database():
    """Completely reset the database by deleting and recreating the collection"""
    global chroma_client, collection

    try:
        if chroma_client is None:
            initialize_chroma()

        # Delete the existing collection
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
            logger.info(f"Deleted collection: {COLLECTION_NAME}")
        except Exception:
            logger.info(f"Collection {COLLECTION_NAME} did not exist, creating new one")

        # Recreate the collection with the same embedding function
        embedder = SciBoxEmbeddings(api_key=api_key, base_url=base_url)
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=embedder,
        )

        logger.info(f"Created new collection: {COLLECTION_NAME}")
        return {"status": "Database reset successfully", "collection": COLLECTION_NAME}
    except Exception as e:
        logger.error(f"Failed to reset database: {e}")
        raise HTTPException(status_code=500, detail=f"Reset database failed: {str(e)}")


@app.get("/admin/excel-content")
async def get_excel_content():
    """View the content of the Excel file in the root directory"""
    try:
        if not os.path.exists(EXCEL_PATH):
            raise HTTPException(
                status_code=404, detail=f"Excel file not found: {EXCEL_PATH}"
            )

        # Read the Excel file
        df = pd.read_excel(EXCEL_PATH, sheet_name=0)

        # Return basic information about the Excel file
        excel_info = {
            "filename": EXCEL_PATH,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "sample_data": df.head().to_dict(orient="records"),
        }

        return excel_info
    except Exception as e:
        logger.error(f"Failed to read Excel file: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to read Excel file: {str(e)}"
        )


@app.post("/admin/reload-excel")
async def reload_excel():
    """Reload the database from the Excel file in the root directory"""
    global chroma_client, collection

    try:
        if not api_key or not base_url:
            raise HTTPException(
                status_code=500, detail="API key or base URL not initialized"
            )

        if not os.path.exists(EXCEL_PATH):
            raise HTTPException(
                status_code=404, detail=f"Excel file not found: {EXCEL_PATH}"
            )

        # Reload data from Excel to Chroma
        collection = ingest_excel_to_chroma(
            excel_path=EXCEL_PATH,
            persist_dir=PERSIST_DIR,
            collection_name=COLLECTION_NAME,
            api_key=api_key,
            base_url=base_url,
        )

        logger.info("Database reloaded from Excel successfully")
        return {
            "status": "Database reloaded from Excel successfully",
            "collection": COLLECTION_NAME,
            "document_count": collection.count(),
        }
    except Exception as e:
        logger.error(f"Failed to reload Excel: {e}")
        raise HTTPException(status_code=500, detail=f"Reload Excel failed: {str(e)}")


@app.post("/admin/test-search")
async def test_search():
    """Run a test search to verify the database is working"""
    try:
        if collection is None:
            raise HTTPException(
                status_code=500, detail="Chroma collection not initialized"
            )

        # Run a test query
        test_query = "Забыл пароль от мобильного приложения"
        results = quick_query(collection, test_query, k=3)

        logger.info(f"Test search completed with {len(results)} results")
        return {"query": test_query, "results_count": len(results), "results": results}
    except Exception as e:
        logger.error(f"Test search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test search failed: {str(e)}")
    