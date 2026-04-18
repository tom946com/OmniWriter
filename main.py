import argparse
import json
import sys
from typing import Optional

from src.pipeline.orchestrator import WriterAgentOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OmniWriter CLI entrypoint",
    )
    parser.add_argument(
        "--topic",
        required=True,
        help="写作主题（必填）",
    )
    parser.add_argument(
        "--demand",
        default=None,
        help="写作要求（可选）",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="会话 ID（可选，不传则自动生成）",
    )
    return parser


def _build_output(result: dict) -> dict:
    return {
        "thread_id": result.get("thread_id"),
        "status": result.get("status"),
        "article_file_path": result.get("article_file_path"),
        "error": result.get("error"),
    }


def run(topic: str, demand: Optional[str] = None, thread_id: Optional[str] = None) -> dict:
    orchestrator = WriterAgentOrchestrator()
    return orchestrator.run_sync(topic=topic, demand=demand, thread_id=thread_id)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    topic = (args.topic or "").strip()
    if not topic:
        parser.error("--topic 不能为空")

    try:
        result = run(topic=topic, demand=args.demand, thread_id=args.thread_id)
    except Exception as exc:
        print(f"执行失败: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(_build_output(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
