from __future__ import annotations

import random

DOMAIN_TOPICS = (
    ("finance", "予算策定", "Budget planning"),
    ("technology", "システム移行", "System migration"),
    ("healthcare", "医療連携", "Healthcare coordination"),
    ("manufacturing", "生産計画", "Production planning"),
    ("retail", "在庫管理", "Inventory management"),
    ("legal", "契約審査", "Contract review"),
    ("hr", "人材育成", "Talent development"),
)


def generate_business_dataset(seed: int, samples: int) -> list[dict[str, str]]:
    if samples < 1:
        raise ValueError("samples must be positive")

    generator = random.Random(seed)
    records: list[dict[str, str]] = []
    for index in range(samples):
        domain, japanese_topic, english_topic = generator.choice(DOMAIN_TOPICS)
        if index % 2 == 0:
            topic = japanese_topic
            language = "Japanese"
            content = f"{topic}に関する検討事項です。予算と進行状況を確認します。"
        else:
            topic = english_topic
            language = "English"
            content = f"{topic} covers budget, schedule, and resource allocation."
        records.append(
            {
                "id": f"{domain}_{index}",
                "language": language,
                "domain": domain,
                "topic": topic,
                "content": content,
                "source": f"synthetic-{domain}",
            }
        )
    return records
