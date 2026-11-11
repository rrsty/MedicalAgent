"""医疗网站爬虫：好大夫在线疾病百科（requests + BeautifulSoup，含 robots.txt 合规）。

功能：
- 预定义疾病列表（crawler/disease_list.py，不依赖网站列表页的 ~15 种）；
- robots.txt 合规检查（can_fetch）；
- 随机 User-Agent + 请求延时 3-5 秒；
- 异常重试（最多 3 次）+ 断点续爬（data/raw/crawled_urls.json）；
- 输出 data/raw/diseases.json。

用法：
    python crawler/medical_crawler.py                # 默认 --max 500（以预定义列表为准）
    python crawler/medical_crawler.py --max 50       # 小批量验证解析质量
    python crawler/medical_crawler.py --site xywy    # 补充源：寻医问药
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
from bs4 import BeautifulSoup

from crawler.disease_list import PREDEFINED_DISEASES, get_disease_url
from src.common.logger import setup_logger

logger = setup_logger("crawler.medical", "crawler.log")

SITES = {
    "haodf": {
        "base": "https://www.haodf.com",
        "list": "https://www.haodf.com/jibing/",
        "robots": "https://www.haodf.com/robots.txt",
    },
    "xywy": {
        "base": "https://jib.xywy.com",
        "list": "https://jib.xywy.com/html/list.html",
        "robots": "https://jib.xywy.com/robots.txt",
    },
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUTPUT_FILE = RAW_DIR / "diseases.json"
CHECKPOINT_FILE = RAW_DIR / "crawled_urls.json"


class MedicalCrawler:
    def __init__(self, site: str = "haodf", delay: tuple[float, float] = (3.0, 5.0)):
        self.site_id = site
        self.site = SITES[site]
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
        self._robots_rules: list[tuple[bool, str]] | None = None
        self.crawled: dict[str, str] = {}
        self._load_checkpoint()

    # ---------- robots.txt 合规（RFC 9309 / Google 规范：最长匹配优先，平局 Allow 胜）----------
    def _load_robots(self) -> None:
        text = self._get(self.site["robots"], max_retries=1)
        if text is None:
            raise RuntimeError("robots.txt unavailable; permission is unverified")
        self._robots_rules: list[tuple[bool, str]] = []
        agent_block = False
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip()
            if key == "user-agent":
                agent_block = value == "*"
            elif agent_block and key in ("allow", "disallow") and value:
                self._robots_rules.append((key == "allow", value))

    def _rule_length(self, pattern: str, path: str) -> int:
        """通配符前缀匹配；返回匹配长度（0=不匹配）。支持 * 与结尾 $。"""
        if pattern.endswith("$"):
            import fnmatch

            return len(pattern) - 1 if fnmatch.fnmatchcase(path, pattern[:-1]) else 0
        # 前缀 + 中间 * 通配
        import re

        regex = "^" + re.escape(pattern).replace(r"\*", ".*")
        return len(pattern) - pattern.count("*") if re.match(regex, path) else 0

    def can_fetch(self, url: str) -> bool:
        try:
            if self._robots_rules is None:
                self._load_robots()
            path = url.replace("://", ":", 1).split(":", 1)[1]  # 去 scheme+host
            allow_len, disallow_len = 0, 0
            for is_allow, pattern in self._robots_rules:
                matched = self._rule_length(pattern, path)
                if not matched:
                    continue
                if is_allow:
                    allow_len = max(allow_len, matched)
                else:
                    disallow_len = max(disallow_len, matched)
            allowed = allow_len >= disallow_len  # 平局 Allow 胜（Google 规范）
            if not allowed:
                logger.warning("robots.txt 禁止爬取，跳过: %s", url)
            return allowed
        except Exception as exc:  # noqa: BLE001
            logger.warning("robots.txt 检查失败（保守跳过）: %s | %s", url, exc)
            return False

    # ---------- 断点续爬 ----------
    def _load_checkpoint(self) -> None:
        if CHECKPOINT_FILE.exists():
            try:
                self.crawled = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
                logger.info("断点续爬：已有 %d 条记录", len(self.crawled))
            except Exception:  # noqa: BLE001
                self.crawled = {}

    def _save_checkpoint(self) -> None:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_FILE.write_text(
            json.dumps(self.crawled, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ---------- HTTP ----------
    def _sleep(self) -> None:
        time.sleep(random.uniform(*self.delay))

    def _get(self, url: str, max_retries: int = 3) -> str | None:
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.session.get(url, timeout=15)
                if resp.status_code == 200:
                    resp.encoding = resp.apparent_encoding or "utf-8"
                    return resp.text
                logger.warning("HTTP %s | %s（第 %d 次）", resp.status_code, url, attempt)
            except Exception as exc:  # noqa: BLE001
                logger.warning("请求异常: %s | %s（第 %d 次）", exc, url, attempt)
            time.sleep(2 * attempt)
        return None

    # ---------- 解析 ----------
    def _parse_detail(self, url: str, fallback_name: str) -> dict | None:
        """解析疾病详情页。字段缺失留空——由 LLM 抽取层从治疗文本补齐。"""
        html = self._get(url)
        if html is None:
            return None
        soup = BeautifulSoup(html, "lxml")

        name = ""
        title_tag = soup.find("h1")
        if title_tag:
            name = title_tag.get_text(strip=True)

        summary = ""
        summary_tag = (
            soup.select_one(".jib-summary") or soup.select_one(".lead")
            or soup.select_one(".jib-articl")
        )
        if summary_tag:
            summary = summary_tag.get_text("\n", strip=True)

        # 分节提取（兼容「概述/症状/治疗/预防」等标题块）
        sections: dict[str, str] = {}
        for header in soup.select("h2, h3"):
            title = header.get_text(strip=True)
            content_parts = []
            for sibling in header.next_siblings:
                if sibling.name in ("h2", "h3"):
                    break
                text = sibling.get_text("\n", strip=True) if sibling.name else ""
                if text:
                    content_parts.append(text)
            if title and content_parts:
                sections[title] = "\n".join(content_parts)[:3000]

        if not summary and sections:
            summary = next(iter(sections.values()), "")

        record = {
            "name": name or fallback_name,
            "url": url,
            "description": summary[:2000],
            "symptoms": _extract_list_after(sections, ["症状", "表现"]),
            "treatment": _join_sections(sections, ["治疗", "用药", "疗法"]),
            "department": "",
            "medications": _extract_list_after(sections, ["常用药物", "用药", "药品"]),
            "precautions": _join_sections(sections, ["预防", "护理", "注意"]),
            "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return record

    # ---------- 主流程 ----------
    def crawl(self, max_count: int = 500) -> list[dict]:
        targets = [
            (name, get_disease_url(self.site["base"], slug))
            for name, slug in PREDEFINED_DISEASES[:max_count]
        ]
        logger.info("目标疾病 %d 个（预定义列表 %d）", len(targets), len(PREDEFINED_DISEASES))

        records: list[dict] = []
        for i, (name, url) in enumerate(targets, 1):
            if url in self.crawled:
                logger.info("[%d/%d] 跳过已爬取: %s", i, len(targets), name)
                continue
            if not self.can_fetch(url):
                continue

            logger.info("[%d/%d] 爬取: %s | %s", i, len(targets), name, url)
            self._sleep()
            record = self._parse_detail(url, name)
            if record and record["description"]:
                from src.common.provenance import SourceProvenance

                record["provenance"] = SourceProvenance.create_for_crawler(
                    url, platform=self.site_id, robots_allowed=True,
                    rate_limit_applied=self.delay[0] > 0,
                ).to_dict()
                records.append(record)
                self.crawled[url] = record["name"]
            else:
                logger.warning("爬取失败或内容为空: %s", name)

            self._save_checkpoint()

        self._merge_and_save(records)
        return records

    def _merge_and_save(self, new_records: list[dict]) -> None:
        """断点续爬：与已有数据合并去重后写 diseases.json。"""
        existing: dict[str, dict] = {}
        if OUTPUT_FILE.exists():
            try:
                data = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
                existing = {d["name"]: d for d in data.get("diseases", [])}
            except Exception:  # noqa: BLE001
                existing = {}
        for record in new_records:
            existing[record["name"]] = record

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(
            json.dumps({"diseases": list(existing.values())}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("已保存 %d 条疾病数据 → %s", len(existing), OUTPUT_FILE)


def _extract_list_after(sections: dict[str, str], keywords: list[str]) -> list[str]:
    for title, content in sections.items():
        if any(kw in title for kw in keywords):
            return [line.strip() for line in content.splitlines() if line.strip()][:15]
    return []


def _join_sections(sections: dict[str, str], keywords: list[str]) -> str:
    parts = [content for title, content in sections.items() if any(kw in title for kw in keywords)]
    return "\n".join(parts)[:3000]


def main() -> None:
    parser = argparse.ArgumentParser(description="医疗疾病百科爬虫")
    parser.add_argument("--max", type=int, default=500, help="爬取疾病数量上限（默认 500）")
    parser.add_argument("--site", choices=list(SITES), default="haodf", help="数据源站点")
    args = parser.parse_args()

    crawler = MedicalCrawler(site=args.site)
    records = crawler.crawl(max_count=args.max)
    print("\n===== 爬取结果统计 =====")
    print(f"本次新增: {len(records)} 条")
    print(f"累计数据: {len(crawler.crawled)} 条 → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
