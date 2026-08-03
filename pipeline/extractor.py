import json
from dotenv import load_dotenv
from models.prompts import SYSTEM_PROMPT, EXTRACTION_PROMPT
from pipeline.schemas import SLAClause, ExtractionResult
from pipeline.llm import call_llm
from config import LLM_PROVIDER, LLM_MODEL, MAX_TOKENS_PER_CALL

load_dotenv()


def extract_sla(
    contract_id: str,
    file_path: str,
    context: str,
    provider: str | None = None,
    model: str | None = None,
    json_mode: bool = False,
    num_ctx: int | None = None,
    max_tokens: int | None = None,
) -> ExtractionResult:
    prompt = EXTRACTION_PROMPT.format(context=context)

    try:
        resp = call_llm(
            system=SYSTEM_PROMPT,
            user=prompt,
            provider=provider or LLM_PROVIDER,
            model=model or LLM_MODEL,
            max_tokens=max_tokens or MAX_TOKENS_PER_CALL,
            json_mode=json_mode,
            num_ctx=num_ctx,
        )

        data = _parse_json(resp.text)
        data = _coerce(data)
        sla = SLAClause(**data)

        populated = sum(1 for v in sla.model_dump().values() if v is not None)
        status = "success" if populated >= 2 else "partial"

        return ExtractionResult(
            contract_id=contract_id,
            file_path=file_path,
            status=status,
            sla=sla,
            raw_response=resp.text,
            tokens_used=resp.prompt_tokens + resp.completion_tokens,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            num_ctx=resp.num_ctx or num_ctx or 0,
        )

    except Exception as e:
        return ExtractionResult(
            contract_id=contract_id,
            file_path=file_path,
            status="failed",
            sla=SLAClause(),
            error=f"{type(e).__name__}: {e}",
        )


def _coerce(data: dict) -> dict:
    """
    Smaller local models drift from the schema in predictable ways: they echo the
    "EXACT QUOTE or null" placeholder, emit the string "null", wrap values in
    {"value": ...}, or return a list of candidate quotes. Normalise rather than
    fail — a parse failure and a placeholder answer are different bugs and the
    benchmark needs to tell them apart.
    """
    known = set(SLAClause.model_fields)
    out = {}
    for k, v in data.items():
        if k not in known:
            continue
        if isinstance(v, dict):
            v = v.get("value", next(iter(v.values()), None))
        if isinstance(v, list):
            v = next((x for x in v if x), None)
        if isinstance(v, str):
            s = v.strip()
            if (
                s.lower() in ("null", "none", "n/a", "na", "", "not specified",
                              "not found", "not mentioned", "not applicable")
                or s.upper().startswith("EXACT QUOTE")
            ):
                v = None
            else:
                v = s
        out[k] = v
    return out


def _parse_json(raw: str) -> dict:
    raw = (raw or "").strip()
    # Strip markdown fences if model adds them despite instructions
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Best-effort: extract first {...} block
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
        raise
