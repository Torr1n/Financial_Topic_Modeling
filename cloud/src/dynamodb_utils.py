"""
DynamoDB utility classes for the Financial Topic Modeling pipeline.

Provides writers for both map and reduce phases:
    - MapPhaseDynamoDBWriter: Writes sentences to DynamoDB (by topic)
    - ReducePhaseDynamoDBWriter: Writes themes to DynamoDB (Phase 3)

Single-table design with composite keys enables hierarchical queries.
"""

import logging
from typing import Dict, List, Any

import boto3

from cloud.src.models import TranscriptSentence

logger = logging.getLogger(__name__)


class MapPhaseDynamoDBWriter:
    """
    Write sentences to DynamoDB during map phase.

    Each sentence is stored with:
        - PK: TOPIC#{firm_id}#{topic_id}
        - SK: SENTENCE#{sentence_id}
        - GSI1PK: FIRM#{firm_id}  (for firm-level queries)
        - GSI1SK: SENTENCE#{sentence_id}

    This allows querying:
        - All sentences for a topic: PK = "TOPIC#1001#0"
        - All sentences for a firm: GSI1PK = "FIRM#1001"
    """

    def __init__(self, table_name: str, region: str = "us-east-1"):
        """
        Initialize writer with DynamoDB table.

        Args:
            table_name: DynamoDB table name
            region: AWS region (default: us-east-1)
        """
        self.table_name = table_name
        self.dynamodb = boto3.resource("dynamodb", region_name=region)
        self.table = self.dynamodb.Table(table_name)

        logger.info(f"Initialized MapPhaseDynamoDBWriter for table: {table_name}")

    def write_firm_sentences(
        self,
        firm_result: Dict[str, Any],
        sentences: List[TranscriptSentence],
    ) -> int:
        """
        Write sentence records for all topics in a firm.

        Args:
            firm_result: FirmTopicOutput dict from FirmProcessor
            sentences: List of TranscriptSentence objects

        Returns:
            Number of items written
        """
        firm_id = firm_result["firm_id"]

        # Build lookup dict for sentences by ID
        sentence_map = {s.sentence_id: s for s in sentences}

        items_written = 0

        with self.table.batch_writer() as batch:
            for topic in firm_result["topics"]:
                topic_id = topic["topic_id"]

                for sentence_id in topic["sentence_ids"]:
                    sentence = sentence_map.get(sentence_id)
                    if sentence is None:
                        logger.warning(f"Sentence {sentence_id} not found in sentences list")
                        continue

                    item = {
                        "PK": f"TOPIC#{firm_id}#{topic_id}",
                        "SK": f"SENTENCE#{sentence_id}",
                        "text": sentence.text,
                        "position": sentence.position,
                        "speaker_type": sentence.speaker_type or "UNKNOWN",
                        "firm_id": firm_id,
                        "topic_id": topic_id,
                        "GSI1PK": f"FIRM#{firm_id}",
                        "GSI1SK": f"SENTENCE#{sentence_id}",
                    }

                    batch.put_item(Item=item)
                    items_written += 1

        logger.info(f"Wrote {items_written} sentence items for firm {firm_id}")
        return items_written


class ReducePhaseDynamoDBWriter:
    """
    Write themes to DynamoDB during reduce phase.

    Each theme creates:
        - Theme metadata: PK=THEME#{theme_id}, SK=METADATA
        - Topic links: PK=THEME#{theme_id}, SK=TOPIC#{firm_id}#{topic_id}

    Note: Sentences are already written by the map phase.
    """

    def __init__(self, table_name: str, region: str = "us-east-1"):
        """
        Initialize writer with DynamoDB table.

        Args:
            table_name: DynamoDB table name
            region: AWS region (default: us-east-1)
        """
        self.table_name = table_name
        self.dynamodb = boto3.resource("dynamodb", region_name=region)
        self.table = self.dynamodb.Table(table_name)

        logger.info(f"Initialized ReducePhaseDynamoDBWriter for table: {table_name}")

    def write_themes(self, themes: List[Dict[str, Any]]) -> int:
        """
        Write theme and topic records.

        Args:
            themes: List of ThemeOutput dicts

        Returns:
            Number of items written
        """
        items_written = 0

        with self.table.batch_writer() as batch:
            for theme in themes:
                theme_id = theme["theme_id"]

                # 1. Theme metadata record
                metadata_item = {
                    "PK": f"THEME#{theme_id}",
                    "SK": "METADATA",
                    "name": theme.get("name", ""),
                    "keywords": theme.get("keywords", [])[:20],  # Limit keywords
                    "n_firms": theme.get("n_firms", 0),
                    "n_topics": theme.get("n_topics", 0),
                    "metadata": theme.get("metadata", {}),
                }
                batch.put_item(Item=metadata_item)
                items_written += 1

                # 2. Topic records (links theme to topics)
                for topic in theme.get("topics", []):
                    topic_item = {
                        "PK": f"THEME#{theme_id}",
                        "SK": f"TOPIC#{topic['firm_id']}#{topic['topic_id']}",
                        "firm_id": topic["firm_id"],
                        "topic_id": topic["topic_id"],
                        "representation": topic.get("representation", ""),
                        "GSI1PK": f"FIRM#{topic['firm_id']}",
                        "GSI1SK": f"THEME#{theme_id}",
                    }
                    batch.put_item(Item=topic_item)
                    items_written += 1

        logger.info(f"Wrote {items_written} items for {len(themes)} themes")
        return items_written
