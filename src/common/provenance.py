"""数据源可信度与审计元数据（Data Provenance & Trustworthiness Metadata）。

设计原则（实事求是）：
1. 爬虫检查了 robots.txt 与请求频率，仅代表遵守了网络爬虫协议，并不代表完成版权/数据授权；
2. 公开百科网络抓取的数据未经过专业医师人工审核，绝不宣称“权威循证数据”或“全面合规”；
3. 为每个处理记录与检索片段附带透明的 Provenance 元数据，可追溯、防虚标。
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
import json


@dataclass
class SourceProvenance:
    source_platform: str  # 采集源平台（如"好大夫在线公开百科"、"寻医问药公开科普"）
    source_url: str  # 原始页面 URL
    crawl_timestamp: str  # 抓取时间（ISO-8601）
    robots_txt_status: str = "unknown"
    rate_limit_applied: bool | None = None
    medical_reviewed: bool = False  # 明确标识是否经过专业临床医师人工复核
    evidence_grade: str = "web_scraping_unreviewed"  # 证据等级：互联网公开网页抓取（未临床审核）
    disclaimer: str = "本数据源自互联网公开网页，仅反映网络科普内容，未经过临床专业人工双盲复核，严禁作为临床诊疗决断依据。"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def create_for_crawler(
        cls,
        url: str,
        platform: str = "unknown",
        robots_allowed: bool | None = None,
        rate_limit_applied: bool | None = None,
    ) -> "SourceProvenance":
        return cls(
            source_platform=platform,
            source_url=url,
            crawl_timestamp=datetime.now(timezone.utc).isoformat(),
            robots_txt_status=("allowed_by_robots_txt" if robots_allowed is True
                               else "disallowed" if robots_allowed is False else "unknown"),
            rate_limit_applied=rate_limit_applied,
            medical_reviewed=False,
            evidence_grade="web_scraping_unreviewed",
        )

    @classmethod
    def create_for_seed(cls, disease_name: str) -> "SourceProvenance":
        return cls(
            source_platform="demo_curated_seed",
            source_url=f"internal://demo_seed/{disease_name}",
            crawl_timestamp=datetime.now(timezone.utc).isoformat(),
            robots_txt_status="not_applicable_internal",
            rate_limit_applied=False,
            medical_reviewed=False,
            evidence_grade="internal_demo_knowledge",
            disclaimer="本数据为系统预置演示医学常识，未由国家临床执业医师进行同行复审，仅供工程测试与原型演示。",
        )


def decode_provenance(value) -> list[dict]:
    """Read stored provenance without inventing acquisition or review status."""
    if isinstance(value, SourceProvenance):
        value = value.to_dict()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return []
    if isinstance(value, dict):
        return [dict(value)] if value else []
    if isinstance(value, list):
        return [record for item in value for record in decode_provenance(item)]
    return []


def merge_provenance_records(*values) -> list[dict]:
    records, seen = [], set()
    for value in values:
        for item in decode_provenance(value):
            key = json.dumps(item, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                records.append(item)
    return records


def normalize_record_provenance(record: dict) -> dict:
    """Preserve recorded facts; absence means unknown, never checked/approved."""
    existing = record.get("provenance")
    existing = existing if isinstance(existing, dict) else {}
    return {
        "source_platform": existing.get("source_platform") or record.get("source_platform") or "unknown",
        "source_url": existing.get("source_url") or record.get("url") or "",
        "crawl_timestamp": existing.get("crawl_timestamp") or record.get("crawled_at") or "unknown",
        "robots_txt_status": existing.get("robots_txt_status") or "unknown",
        "rate_limit_applied": existing.get("rate_limit_applied"),
        "medical_reviewed": existing.get("medical_reviewed", False),
        "evidence_grade": existing.get("evidence_grade") or "unverified",
    }


def format_context_with_provenance(context: str, provenance=None) -> str:
    """在输入给 LLM 的知识片段末尾附加透明的来源与证据等级标注，杜绝无事实依据的假定。"""
    records = decode_provenance(provenance)
    if not records:
        tag = "[数据源: unknown | 未经过临床医学审核 | 来源记录缺失]"
    else:
        tag = "[数据源: " + json.dumps(records, ensure_ascii=False, sort_keys=True) + "]"

    return f"{context.strip()}  {tag}"
