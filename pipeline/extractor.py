import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from models.prompts import SYSTEM_PROMPT, EXTRACTION_PROMPT
from pipeline.schemas import SLAClause, ExtractionResult
from config import DEEPSEEK_MODEL, DEEPSEEK_BASE_URL, MAX_TOKENS_PER_CALL

load_dotenv()


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=DEEPSEEK_BASE_URL,
    )


def extract_sla(contract_id: str, file_path: str, context: str) -> ExtractionResult:
    client = _get_client()

    prompt = EXTRACTION_PROMPT.format(context=context)

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            max_tokens=MAX_TOKENS_PER_CALL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        raw = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if response.usage else 0

        data = _parse_json(raw)
        sla = SLAClause(**data)

        populated = sum(1 for v in data.values() if v is not None)
        status = "success" if populated >= 2 else "partial"

        return ExtractionResult(
            contract_id=contract_id,
            file_path=file_path,
            status=status,
            sla=sla,
            raw_response=raw,
            tokens_used=tokens_used,
        )

    except Exception as e:
        return ExtractionResult(
            contract_id=contract_id,
            file_path=file_path,
            status="failed",
            sla=SLAClause(),
            error=str(e),
        )


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
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
