from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from panda_matching.config import get_settings
from panda_matching.observability import SpanType, start_span, trace


class LLMRateLimitError(RuntimeError):
    pass


_LLM_COOLDOWN_UNTIL = 0.0


def llm_enabled() -> bool:
    settings = get_settings()
    return (
        settings.databricks_llm_enabled
        and bool(settings.databricks_host)
        and bool(settings.databricks_token)
        and bool(settings.databricks_llm_endpoint)
    )


def extract_json_object(text_value: str) -> dict[str, Any] | None:
    stripped = text_value.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    return parsed


@trace(name="databricks_llm_invoke", span_type=SpanType.CHAT_MODEL)
def invoke_databricks_llm(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    global _LLM_COOLDOWN_UNTIL
    settings = get_settings()
    host = settings.databricks_host
    token = settings.databricks_token
    endpoint = settings.databricks_llm_endpoint
    if not host or not token or not endpoint:
        raise RuntimeError("Databricks LLM is not configured")
    now = time.time()
    if now < _LLM_COOLDOWN_UNTIL:
        remaining = max(0.0, _LLM_COOLDOWN_UNTIL - now)
        raise LLMRateLimitError(
            f"Databricks LLM cooldown active after 429; retry in {remaining:.1f}s"
        )

    host = host.rstrip("/")
    payload = json.dumps(
        {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode("utf-8")
    invocation_urls = [
        f"{host}/serving-endpoints/{endpoint}/invocations",
        f"{host}/api/2.0/serving-endpoints/{endpoint}/invocations",
    ]
    body: str | None = None
    last_exc: Exception | None = None
    for url in invocation_urls:
        for attempt in range(settings.databricks_llm_max_retries + 1):
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with start_span(
                    "databricks_http_request",
                    span_type=SpanType.TOOL,
                    attributes={
                        "endpoint_url": url,
                        "retry_attempt": attempt,
                    },
                ) as span:
                    with urllib.request.urlopen(req, timeout=45) as resp:
                        if span is not None:
                            span.set_attribute("http_status_code", resp.status)
                        body = resp.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                if span is not None:
                    span.set_attribute("http_status_code", exc.code)
                last_exc = exc
                if exc.code == 404:
                    break
                if exc.code == 429:
                    _LLM_COOLDOWN_UNTIL = time.time() + settings.databricks_llm_cooldown_seconds
                    if span is not None:
                        span.set_attribute("rate_limited", True)
                        span.set_attribute(
                            "cooldown_seconds",
                            settings.databricks_llm_cooldown_seconds,
                        )
                    if attempt < settings.databricks_llm_max_retries:
                        sleep_seconds = (
                            settings.databricks_llm_retry_backoff_seconds * (attempt + 1)
                        )
                        time.sleep(sleep_seconds)
                        continue
                    raise LLMRateLimitError(f"Databricks LLM call failed: {exc}") from exc
                raise RuntimeError(f"Databricks LLM call failed: {exc}") from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Databricks LLM call failed: {exc}") from exc
        if body is not None:
            _LLM_COOLDOWN_UNTIL = 0.0
            break

    if body is None:
        raise RuntimeError(f"Databricks LLM call failed: {last_exc}")

    parsed = extract_json_object(body)
    if parsed is None:
        raise RuntimeError("Databricks LLM response was not valid JSON")

    choices = parsed.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    return content

    predictions = parsed.get("predictions")
    if isinstance(predictions, list) and predictions:
        first_pred = predictions[0]
        if isinstance(first_pred, str):
            return first_pred
        return json.dumps(first_pred)

    output = parsed.get("output")
    if isinstance(output, str):
        return output

    return body


def llm_plan_message(message: str, memory: dict[str, str]) -> dict[str, Any] | None:
    system_prompt = (
        "You are a planning layer for a panda matching assistant. "
        "Return strict JSON only. Choose either mode=tool or mode=respond. "
        "Available tools and args: "
        "top_matches(panda_name,k), explain_match(focal_id,candidate_id), "
        "compare_candidates(focal_panda_name,candidate_a_name,candidate_b_name), "
        "blockers(focal_ref), panda_profile(name), best_overall(), "
        "count_eligible(), count_pandas(), count_status(status), count_sex(sex), "
        "count_curated(), count_matches_for_panda(panda_name). "
        "Use mode=tool for factual/data requests. "
        "Use explain_match only when the user provides actual pair IDs such as 'explain 16 81'. "
        "Do not use explain_match for natural-language ranking questions about names. "
        "For questions like 'why is X first top match for Y', 'why is X only second for Y', "
        "or 'who are Y's top matches', use top_matches with the focal panda name. "
        "For direct profile or count questions, use panda_profile or the relevant count tool. "
        "Use mode=respond only for greetings/chitchat/help. "
        "JSON schema: "
        "{\"mode\":\"tool|respond\",\"tool\":\"...\",\"args\":{},\"response\":\"...\"}."
    )
    user_prompt = json.dumps({"message": message, "memory": memory}, ensure_ascii=True)
    response_text = invoke_databricks_llm(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        max_tokens=220,
        temperature=0.0,
    )
    return extract_json_object(response_text)


def _compose_task_instruction(user_message: str, tool: str) -> str:
    lower = user_message.lower()
    if tool == "top_matches":
        return (
            "For ranked match questions, answer with a numbered list. "
            "Name each candidate explicitly and give one short causal reason "
            "grounded in the tool output. "
            "If the user asks why a candidate is first or second, explicitly "
            "answer that why-question using ranking position, score differences, "
            "and listed strengths or weaknesses when available."
        )
    if tool == "compare_candidates":
        return (
            "For comparison questions, name the better candidate directly, then "
            "explain the tradeoff using only ranking or score evidence from the "
            "tool output."
        )
    if tool == "panda_profile" and "fun fact" in lower:
        return (
            "For fun fact questions, give exactly one short engaging fact taken "
            "directly from the profile data. Prefer facts from description_text, "
            "babies_had_count, personality_text, or location fields. "
            "Do not mention any facility, country, or history that is not present "
            "in the tool output. "
            "Do not end with a follow-up question."
        )
    if tool == "panda_profile":
        return (
            "For profile questions, answer the exact question directly in 1-3 "
            "sentences using only the profile fields present in the tool output. "
            "If a field is missing, say that you do not have that detail."
        )
    if tool == "blockers":
        return "For blockers, summarize the main blocker patterns in plain language."
    if tool.startswith("count_"):
        return "For count questions, answer in one sentence with the count first."
    return (
        "Answer directly and stay grounded in the provided tool output. "
        "Do not add facts that are not explicitly present."
    )


def llm_compose_answer(user_message: str, tool: str, tool_data: dict[str, Any]) -> str:
    task_instruction = _compose_task_instruction(user_message, tool)
    system_prompt = (
        "You are a friendly panda breeding assistant. "
        "Answer clearly using only provided tool output. "
        "Do not invent facts not present in tool output. "
        "Never return JSON or raw field dumps. "
        "If a fact is not present in the tool output, do not guess. "
        f"{task_instruction}"
    )
    user_prompt = json.dumps(
        {"user_message": user_message, "tool": tool, "tool_output": tool_data},
        ensure_ascii=True,
        default=str,
    )
    return invoke_databricks_llm(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        max_tokens=380,
        temperature=0.0,
    ).strip()
