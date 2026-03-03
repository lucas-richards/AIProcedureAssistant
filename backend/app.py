import json
import os
import re
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.response_synthesizers import ResponseMode

NO_INFO_TEXT = "I don’t have that information in the procedures."

# Your System Prompt
SYSTEM_PROMPT_STR = (
    "You are an internal procedures assistant.\n"
    "Rules:\n"
    "- You MUST answer using only the provided procedures context.\n"
    "- Never use outside knowledge, assumptions, or guesses.\n"
    "- If the context does not explicitly contain the answer, say exactly: "
    "'I don’t have that information in the procedures.'\n"
    "- Answer in precise, step-by-step instructions tailored to the user question.\n"
    "- Start with a direct answer in 1-2 sentences, then list only the required steps.\n"
    "- Include specific values only when they appear in the provided context.\n"
    "- Do not add extra information that is not explicitly in the context.\n"
    "- Do not use generic filler like 'I'd be happy to help'.\n"
    "- Do not include source labels, file paths, page numbers, or quotes unless the user asks for them.\n"
    "Context: {context_str}\n"
    "Question: {query_str}\n"
)
SYSTEM_PROMPT = PromptTemplate(SYSTEM_PROMPT_STR)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

query_engine = None
retriever = None
chat_sessions = {}
procedure_catalog = []

app.mount("/static", StaticFiles(directory="static"), name="static")


def _extract_chapter(file_name: str, text: str) -> str:
    patterns = [
        r"\bchapter\s*(\d+)\b",
        r"\bch[\s_\-]*(\d+)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, file_name, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    for pattern in patterns:
        match = re.search(pattern, text[:1500], flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return "N/A"


def _clean_quote(text: str, max_len: int = 450) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[:max_len].rstrip() + "..."


def _tokenize_text(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def _question_keywords(question: str):
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from", "get",
        "has", "have", "how", "i", "if", "in", "is", "it", "my", "of", "on", "or", "the",
        "to", "what", "when", "where", "which", "who", "why", "with", "you", "your",
    }
    return {
        token
        for token in _tokenize_text(question)
        if len(token) >= 3 and token not in stopwords
    }


def _quote_matches_question(quote: str, question_terms: set[str]):
    if not question_terms:
        return True
    quote_terms = set(_tokenize_text(quote))
    return bool(question_terms.intersection(quote_terms))


def _extract_sources(response, question: str, max_sources: int = 2):
    sources_out = []
    seen_quotes = set()
    question_terms = _question_keywords(question)

    source_nodes = getattr(response, "source_nodes", []) or []

    for source in source_nodes:
        node = getattr(source, "node", None)
        if node is None:
            continue

        metadata = getattr(node, "metadata", {}) or {}
        file_name = str(metadata.get("file_name") or metadata.get("filename") or "")

        try:
            node_text = str(node.get_content(metadata_mode="none") or "")
        except Exception:
            node_text = ""

        quote = _clean_quote(node_text)
        if not quote:
            continue

        if not _quote_matches_question(quote, question_terms):
            continue

        if quote in seen_quotes:
            continue
        seen_quotes.add(quote)

        page_candidate = metadata.get("page_label") or metadata.get("page")
        page = str(page_candidate) if page_candidate not in (None, "") else "N/A"
        chapter = _extract_chapter(file_name, node_text)

        sources_out.append(
            {
                "chapter": chapter,
                "page": page,
                "quote": quote,
            }
        )

        if len(sources_out) >= max_sources:
            break

    return sources_out


def _format_sources_section(sources):
    if not sources:
        return "Exact words of the procedure:\n- No matching procedure text found."

    quote_lines = ["Exact words of the procedure:"]
    for source in sources:
        quote_lines.append(f"- \"{source['quote']}\"")

    return "\n".join(quote_lines)


def _is_speculative_answer(text: str):
    lowered = text.lower()
    literal_patterns = [
        "i'd be happy",
        "i’d be happy",
        "based on general knowledge",
        "i am using outside knowledge",
        "i'm using outside knowledge",
        "i’m using outside knowledge",
    ]
    if any(pattern in lowered for pattern in literal_patterns):
        return True

    regex_patterns = [
        r"\bassum(?:e|ing|ption)?\b",
        r"\bcould be\b",
        r"\baccording to industry\b",
        r"\bgenerally speaking\b",
    ]
    return any(re.search(pattern, lowered) for pattern in regex_patterns)


def _is_no_info_variant(text: str):
    lowered = text.lower().strip()
    variants = [
        "i don't have that information in the procedures",
        "i don’t have that information in the procedures",
        "i don't have information in the procedures",
        "i don’t have information in the procedures",
        "no information in the procedures",
    ]
    return any(variant in lowered for variant in variants)


def _wants_source_text(question: str):
    lowered = question.lower()
    patterns = [
        r"\bexact words\b",
        r"\bquote\b",
        r"\bquoted\b",
        r"\bcite\b",
        r"\bcitation\b",
        r"\bsource\b",
        r"\bwhere found\b",
        r"\bpage\b",
        r"\bchapter\b",
        r"\bshow (the )?text\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def _collect_context_texts(response, question: str | None = None):
    texts = []
    seen = set()

    source_nodes = getattr(response, "source_nodes", []) or []
    for source in source_nodes:
        node = getattr(source, "node", None)
        if node is None:
            continue
        try:
            node_text = str(node.get_content(metadata_mode="none") or "")
        except Exception:
            node_text = ""
        compact = " ".join(node_text.split())
        if compact and compact not in seen:
            texts.append(compact)
            seen.add(compact)

    if retriever is not None and question:
        try:
            hits = retriever.retrieve(question)
        except Exception:
            hits = []

        for hit in hits:
            node = getattr(hit, "node", None)
            if node is None:
                continue
            try:
                node_text = str(node.get_content(metadata_mode="none") or "")
            except Exception:
                node_text = ""
            compact = " ".join(node_text.split())
            if compact and compact not in seen:
                texts.append(compact)
                seen.add(compact)

    return texts


def _extract_relevant_sentences(question: str, texts, max_sentences: int = 5):
    question_terms = _question_keywords(question)
    candidates = []
    seen = set()

    for text in texts:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            cleaned = " ".join(sentence.split()).strip("-• \t")
            if len(cleaned) < 24:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue

            sentence_terms = set(_tokenize_text(cleaned))
            overlap = len(question_terms.intersection(sentence_terms))
            if question_terms and overlap == 0:
                continue

            number_bonus = 1 if re.search(r"\d", cleaned) else 0
            score = overlap + number_bonus
            candidates.append((score, cleaned))
            seen.add(lowered)

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in candidates[:max_sentences]]


def _build_generic_fallback_answer(question: str, response):
    texts = _collect_context_texts(response, question)
    if not texts:
        return None

    relevant = _extract_relevant_sentences(question, texts)
    if not relevant:
        return None

    direct = _clean_quote(relevant[0], max_len=220)
    steps = relevant[1:4]

    lines = [
        f"From the procedures: {direct}",
    ]
    if steps:
        lines.extend(["", "Steps:"])
        for step in steps:
            lines.append(f"- {_clean_quote(step, max_len=220)}")

    return "\n".join(lines)


def _is_upc_creation_question(text: str):
    lowered = text.lower()
    mentions_upc = bool(re.search(r"\bupc\b|\bupc-a\b", lowered))
    asks_how = bool(re.search(r"\b(create|make|generate|build|setup|set up|how to|how do i)\b", lowered))
    return mentions_upc and asks_how


def _collect_upc_context_texts(response):
    texts = []
    seen = set()

    source_nodes = getattr(response, "source_nodes", []) or []
    for source in source_nodes:
        node = getattr(source, "node", None)
        if node is None:
            continue
        try:
            node_text = str(node.get_content(metadata_mode="none") or "")
        except Exception:
            node_text = ""
        compact = " ".join(node_text.split())
        if compact and compact not in seen:
            texts.append(compact)
            seen.add(compact)

    if retriever is not None:
        try:
            hits = retriever.retrieve(
                "Appendix C UPC-A code generation formatting company code product code final digit check sum"
            )
        except Exception:
            hits = []

        for hit in hits:
            node = getattr(hit, "node", None)
            if node is None:
                continue
            try:
                node_text = str(node.get_content(metadata_mode="none") or "")
            except Exception:
                node_text = ""
            compact = " ".join(node_text.split())
            if compact and compact not in seen:
                texts.append(compact)
                seen.add(compact)

    return texts


def _build_upc_fallback_answer(response):
    texts = _collect_upc_context_texts(response)
    if not texts:
        return None

    combined = "\n".join(texts)
    lowered = combined.lower()

    has_11_digit = bool(re.search(r"11\s*digit\s*upc-?a", lowered))
    has_company_pos = bool(re.search(r"position\s*1\s*[-–]\s*6", lowered))
    has_product_pos = bool(re.search(r"position\s*7\s*[-–]\s*11", lowered))
    has_pick_product = "pick a new product code" in lowered or "new product code" in lowered
    has_unique_warning = "already in use" in lowered or "be careful not to pick" in lowered
    has_check_sum = bool(re.search(r"final\s+digit.*check\s*sum|check\s*sum.*final\s+digit", lowered))
    has_pkg_identifier_note = "package identifier" in lowered and "check sum" in lowered

    prefix_example_match = re.search(r"[\"“](\d{6})[\"”]\s+to\s+show\s+westridge\s+upc\s+codes", combined, flags=re.IGNORECASE)
    prefix_example = prefix_example_match.group(1) if prefix_example_match else None

    fact_count = sum(
        [
            has_11_digit,
            has_company_pos,
            has_product_pos,
            has_pick_product,
            has_check_sum,
        ]
    )
    if fact_count < 2:
        return None

    lines = [
        "To create a UPC from the procedure: build the code from your company code and product code, then calculate the final check-sum digit.",
        "",
        "Step 1 — Use the company code portion of the UPC.",
    ]

    if has_company_pos:
        lines.append("- Company code is in positions 1–6.")
    if prefix_example:
        lines.append(f"- Example in DS1000: {prefix_example}.")

    lines.extend(
        [
            "",
            "Step 2 — Assign a new product code.",
        ]
    )
    if has_product_pos:
        lines.append("- Product code is in positions 7–11.")
    if has_pick_product or has_unique_warning:
        lines.append("- Pick a code that is not already in use.")

    lines.extend(
        [
            "",
            "Step 3 — Calculate the last digit.",
        ]
    )
    if has_check_sum:
        lines.append("- The final UPC digit is the check sum.")
    if has_11_digit:
        lines.append("- Appendix C describes an 11-digit UPC-A code plus the final check-sum digit.")
    if has_pkg_identifier_note:
        lines.append("- If package identifier is 0, use the product UPC check-sum rule noted in Appendix D.")

    return "\n".join(lines)


def _extract_min_fill_question_oz(question: str):
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:fl\s*)?oz\b", question.lower())
    if not match:
        return None
    return match.group(1)


def _extract_min_fill_from_response(question: str, response):
    oz_value = _extract_min_fill_question_oz(question)
    if not oz_value:
        return None

    source_nodes = getattr(response, "source_nodes", []) or []
    for source in source_nodes:
        node = getattr(source, "node", None)
        if node is None:
            continue

        metadata = getattr(node, "metadata", {}) or {}
        file_name = str(metadata.get("file_name") or metadata.get("filename") or "")
        page_candidate = metadata.get("page_label") or metadata.get("page")
        page = str(page_candidate) if page_candidate not in (None, "") else "N/A"

        try:
            node_text = str(node.get_content(metadata_mode="none") or "")
        except Exception:
            node_text = ""

        compact = " ".join(node_text.split())
        row_pattern = re.compile(
            rf"(?:^|\s){re.escape(oz_value)}\s+(\d+(?:\.\d+)?)\s+(\d+)\s+(\d+)(?:\s|$)",
            flags=re.IGNORECASE,
        )
        row_match = row_pattern.search(compact)
        if not row_match:
            continue

        return {
            "oz": oz_value,
            "oz_to_ml": row_match.group(1),
            "label_ml": row_match.group(2),
            "min_fill_ml": row_match.group(3),
            "file_name": file_name or "N/A",
            "page": page,
        }

    return None


def _format_min_fill_answer(match_data):
    return (
        "Based on Table 3-3: Quantity Labeling Conversion and Minimum Fill Requirements "
        f"in {match_data['file_name']}:\n\n"
        f"- Label (fl oz): {match_data['oz']}\n"
        f"- Label (ml): {match_data['label_ml']} ml\n"
        f"- Minimum Fill (ml): {match_data['min_fill_ml']} ml\n\n"
        "📌 Answer:\n"
        f"If your label says {match_data['oz']} oz, the minimum production fill must be "
        f"{match_data['min_fill_ml']} ml."
    )


def _get_or_create_session(request: Request):
    session_id = request.cookies.get("session_id")
    is_new = False

    if not session_id or session_id not in chat_sessions:
        session_id = str(uuid.uuid4())
        chat_sessions[session_id] = {
            "name": None,
            "procedure_turns_left": 0,
            "procedure_history": [],
            "pending_scope_query": None,
            "pending_scope_candidates": [],
            "active_scope_files": [],
        }
        is_new = True

    return session_id, chat_sessions[session_id], is_new


def _json_chat_response(text: str, session_id: str, set_cookie: bool):
    response = JSONResponse({"text": text})
    if set_cookie:
        response.set_cookie(
            key="session_id",
            value=session_id,
            max_age=60 * 60 * 24 * 30,
            httponly=True,
            samesite="lax",
        )
    return response


def _extract_name_from_intro(text: str):
    match = re.search(
        r"\b(?:my name is|i am|i'm)\s+([A-Za-z][A-Za-z\-\']{0,30})\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    raw_name = match.group(1).strip()
    if raw_name.lower() in {"trying", "working", "looking", "asking", "wondering"}:
        return None
    return raw_name[:1].upper() + raw_name[1:].lower()


def _is_pure_name_intro(text: str):
    compact = re.sub(r"\s+", " ", text.strip())
    if not compact:
        return False
    if len(compact.split()) > 6:
        return False
    return bool(re.match(r"^(my name is|i am|i'm)\s+[A-Za-z][A-Za-z\-\']{0,30}[.!?]?$", compact, flags=re.IGNORECASE))


def _is_greeting(text: str):
    return bool(re.match(r"^\s*(hi|hello|hey|good morning|good afternoon|good evening)\b", text, flags=re.IGNORECASE))


def _is_name_memory_question(text: str):
    return bool(
        re.search(
            r"\b(what(?:'s| is) my name|do you remember my name|who am i)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _is_procedure_question(text: str):
    lower_text = text.lower()
    procedure_keywords = [
        "procedure",
        "procedures",
        "policy",
        "policies",
        "sop",
        "step",
        "steps",
        "work instruction",
        "how do i",
        "how to",
        "what is the process",
        "document",
        "requirement",
        "qa",
        "upc",
        "barcode",
        "scc-14",
        "scc14",
        "private label",
    ]
    return any(keyword in lower_text for keyword in procedure_keywords)


def _is_small_talk(text: str):
    lowered = text.lower().strip()
    small_talk_patterns = [
        r"\bhow are you\b",
        r"\bhow's it going\b",
        r"\bthank you\b",
        r"\bthanks\b",
        r"\bgood job\b",
        r"\bnice\b",
        r"\bawesome\b",
        r"\bgood morning\b",
        r"\bgood afternoon\b",
        r"\bgood evening\b",
    ]
    return any(re.search(pattern, lowered) for pattern in small_talk_patterns)


def _looks_like_procedure_follow_up(text: str):
    lowered = text.lower().strip()
    follow_up_patterns = [
        r"\bwhat about\b",
        r"\bhow about\b",
        r"\bcan you clarify\b",
        r"\bcan you explain\b",
        r"\bfor that\b",
        r"\bfor this\b",
        r"\bthat step\b",
        r"\bthose steps\b",
        r"\bnext step\b",
        r"\bwhich one\b",
    ]
    if any(re.search(pattern, lowered) for pattern in follow_up_patterns):
        return True

    short_follow_up = len(lowered.split()) <= 7 and bool(re.search(r"\b(it|that|this|those|them)\b", lowered))
    return short_follow_up


def _is_procedure_inventory_question(text: str):
    lowered = text.lower()
    patterns = [
        "what procedures do you have access to",
        "which procedures do you have access to",
        "what procedures can you access",
        "which procedures can you access",
        "list procedures",
        "show procedures",
        "what documents do you have",
        "which documents do you have",
    ]
    return any(pattern in lowered for pattern in patterns)


def _append_procedure_history(session_data, user_text: str, max_items: int = 6):
    history = session_data.get("procedure_history", [])
    history.append(user_text.strip())
    session_data["procedure_history"] = history[-max_items:]


def _build_procedure_query(session_data, current_question: str):
    history = session_data.get("procedure_history", [])
    if not history:
        return current_question

    history_lines = "\n".join(f"- {item}" for item in history[-4:])
    return (
        "Conversation context (for resolving follow-ups):\n"
        f"{history_lines}\n"
        f"Current user message: {current_question}\n"
        "Answer strictly from the procedures context only."
    )


def _derive_title_from_text(text: str, fallback_name: str):
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip(" -•\t")
        if not line:
            continue
        if len(line) < 4 or len(line) > 120:
            continue
        if re.match(r"^(page\s+\d+|\d+)$", line, flags=re.IGNORECASE):
            continue
        if len(re.findall(r"[A-Za-z]", line)) < 3:
            continue
        return line
    return os.path.splitext(fallback_name)[0]


def _build_procedure_catalog(documents):
    by_file = {}
    for doc in documents:
        metadata = getattr(doc, "metadata", {}) or {}
        file_name = str(metadata.get("file_name") or metadata.get("filename") or "").strip()
        if not file_name:
            continue
        text = str(getattr(doc, "text", "") or "")
        if file_name not in by_file:
            by_file[file_name] = text

    catalog = []
    for file_name in sorted(by_file.keys()):
        title = _derive_title_from_text(by_file[file_name], file_name)
        tokens = set(re.findall(r"[a-z0-9]+", f"{title} {file_name}".lower()))
        catalog.append(
            {
                "file_name": file_name,
                "title": title,
                "tokens": tokens,
            }
        )
    return catalog


def _format_procedure_catalog(catalog):
    if not catalog:
        return "I currently do not see any procedures in the procedures folder."

    lines = ["I have access to these procedures:"]
    for idx, item in enumerate(catalog, start=1):
        lines.append(f"{idx}. {item['title']} ({item['file_name']})")
    lines.append("If you want, ask a question and I will suggest which procedure(s) to search first.")
    return "\n".join(lines)


def _keyword_candidate_scores(question: str, catalog):
    question_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    scores = {}
    for item in catalog:
        overlap = len(question_tokens.intersection(item["tokens"]))
        if overlap > 0:
            scores[item["file_name"]] = float(overlap)
    return scores


def _semantic_candidate_scores(question: str):
    if retriever is None:
        return {}

    scores = {}
    try:
        nodes = retriever.retrieve(question)
    except Exception:
        return {}

    for hit in nodes:
        node = getattr(hit, "node", None)
        if node is None:
            continue
        metadata = getattr(node, "metadata", {}) or {}
        file_name = str(metadata.get("file_name") or metadata.get("filename") or "").strip()
        if not file_name:
            continue
        score = float(getattr(hit, "score", 0.0) or 0.0)
        scores[file_name] = max(scores.get(file_name, 0.0), score)
    return scores


def _hybrid_candidates(question: str, catalog, max_items: int = 5):
    if not catalog:
        return []

    keyword_scores = _keyword_candidate_scores(question, catalog)
    semantic_scores = _semantic_candidate_scores(question)

    candidates = []
    for item in catalog:
        file_name = item["file_name"]
        keyword_score = keyword_scores.get(file_name, 0.0)
        semantic_score = semantic_scores.get(file_name, 0.0)
        combined = keyword_score * 1.0 + semantic_score * 3.0
        if combined <= 0:
            continue
        candidates.append(
            {
                "file_name": file_name,
                "title": item["title"],
                "score": combined,
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    trimmed = candidates[:max_items]
    for idx, item in enumerate(trimmed, start=1):
        item["id"] = idx
    return trimmed


def _format_candidate_prompt(question: str, candidates):
    if not candidates:
        return (
            "I can search all procedures for this question.\n"
            f"Question: {question}\n\n"
            "Should I search all procedures? Reply with: all"
        )

    lines = [
        "I found your answer could be in these procedures:",
    ]
    for item in candidates:
        lines.append(f"{item['id']}. {item['title']} ({item['file_name']})")
    lines.append("")
    lines.append("Should I search all of them, or any in particular?")
    lines.append("Reply with: all, or one/more numbers (example: 1 or 1,2).")
    return "\n".join(lines)


def _format_optional_scope_hint(candidates, max_items: int = 3):
    if not candidates:
        return ""

    lines = [
        "",
        "Optional: narrow the next answer to specific procedure(s) by replying with numbers.",
    ]
    for item in candidates[:max_items]:
        lines.append(f"- {item['id']}. {item['title']} ({item['file_name']})")
    lines.append("Example reply: 1 or 1,2")
    return "\n".join(lines)


def _parse_scope_choice(user_text: str, candidates):
    lowered = user_text.lower().strip()
    if not lowered:
        return None

    if re.search(r"\b(all|search all|everything|all of them)\b", lowered):
        return {"mode": "all", "selected_files": []}

    by_id = {item["id"]: item for item in candidates}
    selected_files = []

    for token in re.findall(r"\d+", lowered):
        idx = int(token)
        if idx in by_id:
            selected_files.append(by_id[idx]["file_name"])

    if not selected_files:
        for item in candidates:
            if item["file_name"].lower() in lowered or item["title"].lower() in lowered:
                selected_files.append(item["file_name"])

    selected_files = list(dict.fromkeys(selected_files))
    if selected_files:
        return {"mode": "selected", "selected_files": selected_files}

    return None


def _build_general_chat_reply(question: str, name: str | None):
    lower_q = question.lower()
    if "what can you do" in lower_q or "your ability" in lower_q or "what do you do" in lower_q:
        if name:
            return (
                f"I can help with procedures, {name}. "
                "I can list the procedures I can access, suggest which ones match your question, and answer using only those procedures."
            )
        return (
            "I can help with procedures. "
            "I can list the procedures I can access, suggest which ones match your question, and answer using only those procedures."
        )

    if name:
        return f"I can chat briefly, {name}, and I’m best at procedure questions. You can ask: what procedures do you have access to?"
    return "I can chat briefly, and I’m best at procedure questions. You can ask: what procedures do you have access to?"

@app.on_event("startup")
def startup():
    global query_engine, retriever, procedure_catalog
    Settings.llm = Ollama(
            model="llama3.1:latest",
        base_url="http://127.0.0.1:11434",
        request_timeout=300.0,
        additional_kwargs={
            "num_ctx": 4096,
            "temperature": 0.0,
        }
    )

    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://127.0.0.1:11434",
    )

    procedures_path = os.path.abspath("../procedures")
    documents = SimpleDirectoryReader(procedures_path).load_data()
    procedure_catalog = _build_procedure_catalog(documents)

    print(f"📚 Procedures path: {procedures_path}")
    print(f"📄 Loaded documents: {len(documents)}")
    if not documents:
        query_engine = None
        retriever = None
        print("⚠️ No procedures loaded. Add files to the procedures folder and restart.")
        return

    index = VectorStoreIndex.from_documents(documents)
    retriever = index.as_retriever(similarity_top_k=8)

    query_engine = index.as_query_engine(
        response_mode=ResponseMode.COMPACT,
        similarity_top_k=6,
        streaming=True,
    )
    # Apply your system prompt
    query_engine.update_prompts({"response_synthesizer:text_qa_template": SYSTEM_PROMPT})
    print(
        f"✅ RAG index loaded ({len(procedure_catalog)} procedures) | "
        "retriever_top_k=8 | answer_top_k=6 | num_ctx=4096"
    )

@app.get("/chat")
def chat_page():
    return FileResponse("static/chat.html")

@app.get("/")
def root():
    return RedirectResponse(url="/chat")
    
@app.post("/chat")
async def chat(request: Request, payload: dict):
    try:
        session_id, session_data, set_cookie = _get_or_create_session(request)

        question = ""
        messages = payload.get("messages")
        if isinstance(messages, list) and messages:
            last_message = messages[-1]
            if isinstance(last_message, dict):
                question = str(last_message.get("text", "")).strip()

        if not question:
            question = str(payload.get("text", "")).strip()

        if not question:
            message = payload.get("message")
            if isinstance(message, dict):
                question = str(message.get("text", "")).strip()

        if not question:
            return _json_chat_response("Please enter a message.", session_id, set_cookie)

        if _is_procedure_inventory_question(question):
            return _json_chat_response(_format_procedure_catalog(procedure_catalog), session_id, set_cookie)

        pending_query = session_data.get("pending_scope_query")
        pending_candidates = session_data.get("pending_scope_candidates") or []
        if pending_query and pending_candidates:
            scope_choice = _parse_scope_choice(question, pending_candidates)
            if scope_choice is not None:
                session_data["active_scope_files"] = scope_choice.get("selected_files", [])
                question = pending_query
            session_data["pending_scope_query"] = None
            session_data["pending_scope_candidates"] = []

        introduced_name = _extract_name_from_intro(question)
        if introduced_name and _is_pure_name_intro(question):
            session_data["name"] = introduced_name
            return _json_chat_response(
                f"Nice to meet you, {introduced_name}. I will remember your name.",
                session_id,
                set_cookie,
            )

        if _is_name_memory_question(question):
            if session_data.get("name"):
                return _json_chat_response(
                    f"Your name is {session_data['name']}.",
                    session_id,
                    set_cookie,
                )
            return _json_chat_response(
                "I don't know your name yet. Tell me by saying 'my name is ...'.",
                session_id,
                set_cookie,
            )

        greeting_match = re.match(r"^\s*(hi|hello|hey|good morning|good afternoon|good evening)\b[\s,!?.-]*", question, flags=re.IGNORECASE)
        if greeting_match:
            remainder = question[greeting_match.end():].strip()
            if remainder:
                question = remainder
            else:
                if session_data.get("name"):
                    return _json_chat_response(
                        f"Hi {session_data['name']}! How can I help you today?",
                        session_id,
                        set_cookie,
                    )
                return _json_chat_response("Hi! How can I help you today?", session_id, set_cookie)

        if _is_greeting(question):
            if session_data.get("name"):
                return _json_chat_response(
                    f"Hi {session_data['name']}! How can I help you today?",
                    session_id,
                    set_cookie,
                )
            return _json_chat_response("Hi! How can I help you today?", session_id, set_cookie)

        is_explicit_procedure = _is_procedure_question(question)
        follow_up_window = int(session_data.get("procedure_turns_left", 0) or 0)
        is_procedure_follow_up = (
            (not is_explicit_procedure)
            and follow_up_window > 0
            and (not _is_small_talk(question))
            and _looks_like_procedure_follow_up(question)
        )

        if is_explicit_procedure:
            session_data["procedure_turns_left"] = 4
        elif is_procedure_follow_up:
            session_data["procedure_turns_left"] = max(follow_up_window - 1, 0)

        if not is_explicit_procedure and not is_procedure_follow_up:
            session_data["procedure_turns_left"] = 0
            return _json_chat_response(
                _build_general_chat_reply(question, session_data.get("name")),
                session_id,
                set_cookie,
            )

        if query_engine is None:
            return _json_chat_response(
                "Assistant is still starting. Please try again in a few seconds.",
                session_id,
                set_cookie,
            )

        active_scope_files = session_data.get("active_scope_files") or []
        scope_hint = ""

        _append_procedure_history(session_data, question)

        if active_scope_files:
            scoped_files_text = ", ".join(active_scope_files)
            scoped_question = (
                f"Use only these procedures: {scoped_files_text}. "
                f"If the answer is not in them, reply exactly: {NO_INFO_TEXT}\n"
                f"User question: {question}"
            )
            procedure_query = _build_procedure_query(session_data, scoped_question)
            session_data["active_scope_files"] = []
        else:
            procedure_query = _build_procedure_query(session_data, question)

        response = query_engine.query(procedure_query)
        answer_text = str(response).strip()
        sources = _extract_sources(response, question)

        min_fill_match = _extract_min_fill_from_response(question, response)
        if min_fill_match is not None:
            answer_text = _format_min_fill_answer(min_fill_match)
            table_quote = (
                f"Table 3-3 row: {min_fill_match['oz']} {min_fill_match['oz_to_ml']} "
                f"{min_fill_match['label_ml']} {min_fill_match['min_fill_ml']}"
            )
            sources = [
                {
                    "chapter": "N/A",
                    "page": min_fill_match["page"],
                    "quote": table_quote,
                }
            ]

        if answer_text == NO_INFO_TEXT or _is_no_info_variant(answer_text):
            generic_fallback_answer = _build_generic_fallback_answer(question, response)
            if generic_fallback_answer:
                answer_text = generic_fallback_answer

            if _is_upc_creation_question(question):
                upc_fallback_answer = _build_upc_fallback_answer(response)
                if upc_fallback_answer:
                    answer_text = upc_fallback_answer

            if answer_text == NO_INFO_TEXT or _is_no_info_variant(answer_text):
                sources = []
                answer_text = NO_INFO_TEXT
        elif _is_speculative_answer(answer_text):
            generic_fallback_answer = _build_generic_fallback_answer(question, response)
            if generic_fallback_answer:
                answer_text = generic_fallback_answer
            elif not sources:
                answer_text = NO_INFO_TEXT
                sources = []

        wants_sources = _wants_source_text(question)
        if wants_sources:
            formatted_sources = _format_sources_section(sources)
            final_text = f"Answer:\n{answer_text}\n\n{formatted_sources}{scope_hint}"
        else:
            final_text = f"Answer:\n{answer_text}{scope_hint}"
        return _json_chat_response(final_text, session_id, set_cookie)
    except Exception:
        return JSONResponse({"text": "Error, please try again."})


