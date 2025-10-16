# Инструкция по использованию моделей LLM-сервиса SciBox

В этой инструкции показано, как получить список моделей и выполнить запросы к ним с помощью `curl`. Вместо реального токена используйте переменную окружения или подставляйте свой токен в заголовке `Authorization`.


## 0. Swagger и базовые URL

- Swagger: по домену `https://llm.t1v.scibox.tech/` или по IP `http://45.145.191.148:4000`.
- Endpoint списка моделей (пример по IP): `http://45.145.191.148:4000/v1/models`.

Вы можете использовать либо домен без порта, либо IP с портом 4000. В командах ниже оставлен IP с портом, но доменное имя также будет работать без указания порта.


## 1. Получение списка моделей

```bash
curl -H "Authorization: Bearer <YOUR_TOKEN>" \
     https://llm.t1v.scibox.tech/v1/models
```

Пример ответа:

```json
{
  "data": [
    {"id": "Qwen2.5-72B-Instruct-AWQ", ...},
    {"id": "bge-m3", ...}
  ],
  "object": "list"
}
```

---

## 2. Запрос к чат-моделям

**Что это за модели:**

- Используются инструкции‑ориентированные текстовые модели (например, `Qwen2.5-72B-Instruct-AWQ`). Это генеративные LLM, оптимизированные для следования инструкциям и диалогового взаимодействия.
- Подходят для: ответа на вопросы, суммаризации, переписывания текста, рассуждений, пошаговых инструкций, код‑ассистирования.
- Доступный endpoint: **`/v1/chat/completions`** (OpenAI‑совместимый). Поддерживается потоковая выдача через параметр `stream: true`.
- Ключевые параметры: `messages` (диалог), `temperature`, `top_p`, `max_tokens`, а также (если доступны) `frequency_penalty`/`presence_penalty`.

### 2.1 Qwen2.5-72B-Instruct-AWQ

```bash
curl -X POST \
     -H "Authorization: Bearer <YOUR_TOKEN>" \
     -H "Content-Type: application/json" \
     https://llm.t1v.scibox.tech/v1/chat/completions \
     -d '{
           "model": "Qwen2.5-72B-Instruct-AWQ",
           "messages": [
             {"role":"system","content":"Ты дружелюбный помощник"},
             {"role":"user","content":"Расскажи анекдот"}
           ],
           "temperature": 0.7,
           "top_p": 0.9,
           "max_tokens": 256
         }'
```


### 2.2 Потоковый ответ (stream)

```bash
curl -N -X POST \
     -H "Authorization: Bearer <YOUR_TOKEN>" \
     -H "Content-Type: application/json" \
     https://llm.t1v.scibox.tech/v1/chat/completions \
     -d '{
           "model": "Qwen2.5-72B-Instruct-AWQ",
           "messages": [{"role":"user","content":"Сделай краткое резюме книги Война и мир"}],
           "stream": true,
           "max_tokens": 400
         }'
```

Ответ приходит чанками (SSE). Каждая строка начинается с `data:` и содержит частичный фрагмент. Поток завершается сообщением `data: [DONE]`.


---

## 3. Запрос к эмбеддинг‑модели `bge-m3`

**Что это за модель:**

- `bge-m3` — многоязычная эмбеддинг‑модель (векторизации текста) от BAAI из семейства BGE. Она возвращает числовые вектора для текста и не генерирует ответы в стиле чат‑моделей.
- Подходит для задач: поиска и ранжирования (retrieval), классификации, кластеризации, дедупликации.
- В этом API для неё доступен endpoint: **`/v1/embeddings`**.

### Пример: эмбеддинг одной строки

```bash
curl -X POST \
     -H "Authorization: Bearer <YOUR_TOKEN>" \
     -H "Content-Type: application/json" \
     https://llm.t1v.scibox.tech/v1/embeddings \
     -d '{
           "model": "bge-m3",
           "input": "Напиши короткое стихотворение про осень"
         }'
```

### Пример: батч из нескольких текстов

```bash
curl -X POST \
     -H "Authorization: Bearer <YOUR_TOKEN>" \
     -H "Content-Type: application/json" \
     https://llm.t1v.scibox.tech/v1/embeddings \
     -d '{
           "model": "bge-m3",
           "input": [
             "Что такое квантовая запутанность?",
             "Квантовая запутанность — это корреляция состояний частиц"
           ]
         }'
```

Ответ содержит массив эмбеддингов. Для поиска схожести вычисляйте метрику (например, косинусное сходство) на клиентской стороне.

---

## 4. Советы по использованию

- Всегда указывайте заголовок `Authorization: Bearer <YOUR_TOKEN>`.
- Заголовок `Content-Type: application/json` обязателен при POST-запросах.
- В случае ошибки InternalServerError или 429 попробуйте повторить запрос через несколько секунд или переключиться на другую модель.
- Параметры (`max_tokens`, `temperature` и др.) можно настроить по своему усмотрению. Рекомендуем посмотреть раздел «Best practices по параметрам» в репозиториях моделей.


---

## 5. Примеры с OpenAI Python client

Ниже показано, как использовать совместимый клиент `openai` (ветка `openai>=1.0.0`) с вашим API.

Установка:

```bash
pip install --upgrade openai
```

> Базовый URL: можно домен без порта по HTTPS (`https://llm.t1v.scibox.tech/v1`) или IP с портом (`http://45.145.191.148:4000/v1`).

### 5.1 Chat Completions (non‑stream)

```python
from openai import OpenAI

API_KEY = "<YOUR_TOKEN>"
# Вариант с доменом без порта (HTTPS):
BASE_URL = "https://llm.t1v.scibox.tech/v1"
# Альтернатива с IP:порт
# BASE_URL = "http://45.145.191.148:4000/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

resp = client.chat.completions.create(
    model="Qwen2.5-72B-Instruct-AWQ",
    messages=[
        {"role": "system", "content": "Ты дружелюбный помощник"},
        {"role": "user", "content": "Расскажи анекдот"},
    ],
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
)

print(resp.choices[0].message.content)
```

### 5.2 Chat Completions (stream)

```python
from openai import OpenAI

client = OpenAI(api_key="<YOUR_TOKEN>", base_url="https://llm.t1v.scibox.tech/v1")

with client.chat.completions.stream(
    model="Qwen2.5-72B-Instruct-AWQ",
    messages=[{"role": "user", "content": "Сделай краткое резюме книги Война и мир"}],
    max_tokens=400,
) as stream:
    for event in stream:
        if event.type == "message.delta" and event.delta.get("content"):
            print(event.delta["content"], end="", flush=True)
        elif event.type == "message.completed":
            print()  # newline
```

### 5.3 Embeddings

```python
from openai import OpenAI

client = OpenAI(api_key="<YOUR_TOKEN>", base_url="https://llm.t1v.scibox.tech/v1")

emb = client.embeddings.create(
    model="bge-m3",
    input=[
        "Что такое квантовая запутанность?",
        "Квантовая запутанность — это корреляция состояний частиц",
    ],
)

print(len(emb.data), len(emb.data[0].embedding))
```
