"""
FirmProcessor - Map phase logic for single firm topic modeling.

This module converts firm transcript data into the FirmTopicOutput schema,
which is serialized to JSON for S3 storage in the map phase.

Design:
    - Uses dependency injection for TopicModel (testable, swappable)
    - Produces FirmTopicOutput schema per approved plan
    - Tracks outlier sentences separately (topic_id = -1)
"""

import logging
from datetime import datetime
from typing import Dict, List, Any

from cloud.src.interfaces import TopicModel
from cloud.src.models import FirmTranscriptData, TopicModelResult

logger = logging.getLogger(__name__)


class FirmProcessor:
    """
    Process a single firm's transcripts into topics.

    This class orchestrates:
        1. Extract sentence texts from FirmTranscriptData
        2. Run topic model to get assignments
        3. Map sentence_ids to their assigned topics
        4. Build FirmTopicOutput dict for JSON serialization

    Args:
        topic_model: TopicModel implementation (BERTopicModel, etc.)
        config: Configuration dict (passed to output metadata)
    """

    def __init__(self, topic_model: TopicModel, config: Dict[str, Any]):
        """
        Initialize FirmProcessor with topic model and config.

        Args:
            topic_model: Injected TopicModel implementation
            config: Configuration dict for model parameters
        """
        self.model = topic_model
        self.config = config

    def process(self, firm_data: FirmTranscriptData) -> Dict[str, Any]:
        """
        Run topic modeling and return FirmTopicOutput.

        Args:
            firm_data: FirmTranscriptData from DataConnector

        Returns:
            Dict matching FirmTopicOutput schema (JSON-serializable):
            {
                "firm_id": str,
                "firm_name": str,
                "n_topics": int,
                "topics": [
                    {
                        "topic_id": int,
                        "representation": str,
                        "keywords": List[str],
                        "size": int,
                        "sentence_ids": List[str]
                    },
                    ...
                ],
                "outlier_sentence_ids": List[str],
                "metadata": {
                    "processing_timestamp": str,
                    "model_config": dict,
                    "n_sentences_processed": int
                }
            }
        """
        logger.info(f"Processing firm {firm_data.firm_id} ({firm_data.firm_name})")

        # Extract texts and IDs
        sentences = [s.text for s in firm_data.sentences]
        sentence_ids = [s.sentence_id for s in firm_data.sentences]

        logger.info(f"Running topic model on {len(sentences)} sentences")

        # Run topic model
        result = self.model.fit_transform(sentences)

        # Convert to output schema
        return self._to_output(firm_data, result, sentence_ids)

    def _to_output(
        self,
        firm_data: FirmTranscriptData,
        result: TopicModelResult,
        sentence_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Convert TopicModelResult to FirmTopicOutput schema.

        Args:
            firm_data: Original firm data
            result: TopicModelResult from model
            sentence_ids: List of sentence IDs in same order as model input

        Returns:
            FirmTopicOutput dict
        """
        # Build topic list (excluding outliers)
        topics = []
        for topic_id in range(result.n_topics):
            # Find sentences assigned to this topic
            mask = result.topic_assignments == topic_id
            topic_sentence_ids = [
                sid for sid, is_match in zip(sentence_ids, mask) if is_match
            ]

            topics.append({
                "topic_id": topic_id,
                "representation": result.topic_representations.get(topic_id, f"Topic {topic_id}"),
                "keywords": result.topic_keywords.get(topic_id, []),
                "size": int(mask.sum()),
                "sentence_ids": topic_sentence_ids,
            })

        # Collect outlier sentence IDs (topic_id = -1)
        outlier_mask = result.topic_assignments == -1
        outlier_sentence_ids = [
            sid for sid, is_outlier in zip(sentence_ids, outlier_mask) if is_outlier
        ]

        logger.info(
            f"Firm {firm_data.firm_id}: {result.n_topics} topics, "
            f"{len(outlier_sentence_ids)} outliers"
        )

        return {
            "firm_id": firm_data.firm_id,
            "firm_name": firm_data.firm_name,
            "n_topics": result.n_topics,
            "topics": topics,
            "outlier_sentence_ids": outlier_sentence_ids,
            "metadata": {
                "processing_timestamp": datetime.utcnow().isoformat() + "Z",
                "model_config": self.config,
                "n_sentences_processed": len(sentence_ids),
            },
        }
