from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from panda_matching.config import get_settings


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


def invoke_databricks_llm(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    settings = get_settings()
    host = settings.databricks_host
    token = settings.databricks_token
    endpoint = settings.databricks_llm_endpoint
    if not host or not token or not endpoint:
        raise RuntimeError("Databricks LLM is not configured")

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
            with urllib.request.urlopen(req, timeout=45) as resp:
                body = resp.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 404:
                continue
            raise RuntimeError(f"Databricks LLM call failed: {exc}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Databricks LLM call failed: {exc}") from exc

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
        "blockers(focal_ref), panda_profile(name), best_overall(), "
        "count_eligible(), count_pandas(), count_status(status), count_sex(sex), "
        "count_curated(), count_matches_for_panda(panda_name). "
        "Use mode=tool for factual/data requests. "
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


def llm_compose_answer(user_message: str, tool: str, tool_data: dict[str, Any]) -> str:
    system_prompt = (
        "You are a friendly panda breeding assistant. "
        "Answer clearly using only provided tool output. "
        "Do not invent facts not present in tool output. "
        "Never return JSON or raw field dumps. "
        "When the tool returns ranked matches, present a short ranked list with a concise reason "
        "for each candidate based on the score factors. "
        "When the tool returns profile, blockers, or counts, summarize them in plain language."
    )
    user_prompt = json.dumps(
        {"user_message": user_message, "tool": tool, "tool_output": tool_data},
        ensure_ascii=True,
        default=str,
    )
    return invoke_databricks_llm(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        max_tokens=380,
        temperature=0.3,
    ).strip()
