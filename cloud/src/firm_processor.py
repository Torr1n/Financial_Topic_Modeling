"""
FirmProcessor - Map phase logic for single firm topic modeling.

This module converts firm transcript data into the FirmTopicOutput schema.

Design:
    - Uses dependency injection for TopicModel (testable, swappable)
    - Produces FirmTopicOutput schema per approved plan
    - Tracks outlier sentences separately (topic_id = -1)
    - Accepts optional pre-computed embeddings (Phase 2 - Pipeline Unification)
    - Returns topic_assignments for Postgres sentence→topic mapping
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

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

    def process(
        self,
        firm_data: FirmTranscriptData,
        embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Run topic modeling and return FirmTopicOutput with topic assignments.

        Args:
            firm_data: FirmTranscriptData from DataConnector
            embeddings: Optional pre-computed embeddings. If provided, passed to
                       model.fit_transform() to skip internal encoding. Shape:
                       (len(firm_data.sentences), embedding_dim). Use this when
                       the unified pipeline computes embeddings externally.

        Returns:
            Tuple of:
                - Dict matching FirmTopicOutput schema (JSON-serializable)
                - np.ndarray of topic_assignments (for Postgres sentence→topic mapping)

            FirmTopicOutput schema:
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

        # Extract cleaned texts and IDs (use cleaned_text for topic modeling)
        sentences = [s.cleaned_text for s in firm_data.sentences]
        sentence_ids = [s.sentence_id for s in firm_data.sentences]

        logger.info(f"Running topic model on {len(sentences)} sentences")

        # Run topic model with optional pre-computed embeddings
        result = self.model.fit_transform(sentences, embeddings=embeddings)

        # Convert to output schema
        output = self._to_output(firm_data, result, sentence_ids)

        # Return both output dict and topic_assignments for Postgres mapping
        return output, result.topic_assignments

    def _to_output(
        self,
        firm_data: FirmTranscriptData,
        result: TopicModelResult,
        sentence_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Convert TopicModelResult to FirmTopicOutput schema.

        Sentence IDs within each topic are sorted by their probability
        for that topic (highest first). This enables downstream use cases
        like selecting "top K most representative sentences" for a topic.

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
            topic_indices = [i for i, is_match in enumerate(mask) if is_match]

            # Sort by probability for this topic (highest first)
            # probabilities is (n_docs, n_topics) - get column for this topic
            if result.probabilities is not None and result.probabilities.shape[1] > topic_id:
                # Get probabilities for this topic and sort indices by descending prob
                topic_probs = [(idx, result.probabilities[idx, topic_id]) for idx in topic_indices]
                topic_probs.sort(key=lambda x: x[1], reverse=True)
                topic_sentence_ids = [sentence_ids[idx] for idx, _ in topic_probs]
            else:
                # Fallback: keep original order if probabilities not available
                topic_sentence_ids = [sentence_ids[idx] for idx in topic_indices]

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
                "processing_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "model_config": self.config,
                "n_sentences_processed": len(sentence_ids),
            },
        }
