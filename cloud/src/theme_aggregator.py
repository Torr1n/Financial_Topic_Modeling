"""
ThemeAggregator: Aggregate firm-level topics into cross-firm themes.

This module implements the reduce phase of the map-reduce pipeline using
Dual-BERTopic: topic representations (strings) from the map phase become
"documents" for re-embedding and clustering into themes.

The same TopicModel interface used for firm-level topic modeling is reused
for theme-level clustering - this is the elegance of Dual-BERTopic.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List
from collections import Counter
import logging

from cloud.src.interfaces import TopicModel

logger = logging.getLogger(__name__)


class ThemeAggregator:
    """
    Aggregate firm topics into cross-firm themes using Dual-BERTopic.

    Process:
    1. Collect topic representations (strings) from all firms
    2. Embed these representations as new documents
    3. Run BERTopic again to cluster topics into themes
    4. Validate themes (diversity, dominance filters)
    5. Return ThemeOutput dicts

    Args:
        topic_model: TopicModel implementation (e.g., BERTopicModel)
        config: Configuration dict with validation settings
    """

    def __init__(self, topic_model: TopicModel, config: Dict[str, Any]):
        self.model = topic_model
        self.config = config

        # Extract validation settings with defaults
        validation = config.get("validation", {})
        self.min_firms = validation.get("min_firms", 2)
        self.max_dominance = validation.get("max_firm_dominance", 0.4)

    def aggregate(self, firm_results: List[Dict]) -> List[Dict]:
        """
        Aggregate firm topics into cross-firm themes.

        Args:
            firm_results: List of FirmTopicOutput dicts from map phase

        Returns:
            List of ThemeOutput dicts, sorted by n_topics descending
        """
        # Handle empty input
        if not firm_results:
            logger.info("No firm results to aggregate")
            return []

        # 1. Extract topic representations as "documents" for theme modeling
        topic_docs, topic_metadata = self._extract_topic_documents(firm_results)

        if not topic_docs:
            logger.info("No topics to aggregate (all firms have n_topics=0)")
            return []

        logger.info(f"Aggregating {len(topic_docs)} topics from {len(firm_results)} firms")

        # 2. Run BERTopic on topic representations (RE-EMBEDS them)
        theme_result = self.model.fit_transform(topic_docs)

        # 3. Group topics by theme assignment
        raw_themes = self._group_into_themes(theme_result, topic_metadata)

        # 4. Apply validation filters
        validated_themes = self._validate_themes(raw_themes)

        # 5. Sort by n_topics descending and assign theme IDs
        sorted_themes = sorted(validated_themes, key=lambda t: -t["n_topics"])
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d")
        for i, theme in enumerate(sorted_themes):
            theme["theme_id"] = f"theme_{run_id}_{i:03d}"

        logger.info(f"Discovered {len(sorted_themes)} validated themes")
        return sorted_themes

    def _extract_topic_documents(
        self, firm_results: List[Dict]
    ) -> tuple[List[str], List[Dict]]:
        """
        Extract topic representations as "documents" for theme modeling.

        Args:
            firm_results: List of FirmTopicOutput dicts

        Returns:
            (topic_docs, topic_metadata) - Lists aligned by index
            topic_docs: List of representation strings
            topic_metadata: List of dicts with firm_id, topic_id, representation, size
        """
        topic_docs = []
        topic_metadata = []

        for firm_result in firm_results:
            firm_id = firm_result["firm_id"]

            # Skip firms with no topics
            if firm_result.get("n_topics", 0) == 0:
                logger.debug(f"Skipping firm {firm_id}: no topics")
                continue

            for topic in firm_result.get("topics", []):
                representation = topic.get("representation", "")
                if not representation:
                    logger.warning(f"Skipping topic {topic.get('topic_id')} from firm {firm_id}: empty representation")
                    continue

                topic_docs.append(representation)
                topic_metadata.append({
                    "firm_id": firm_id,
                    "topic_id": topic["topic_id"],
                    "representation": representation,
                    "size": topic.get("size", 0),
                })

        return topic_docs, topic_metadata

    def _group_into_themes(
        self, theme_result, topic_metadata: List[Dict]
    ) -> List[Dict]:
        """
        Group topics by their theme assignment from BERTopic.

        Args:
            theme_result: TopicModelResult from BERTopic
            topic_metadata: List of dicts with firm_id, topic_id, etc.

        Returns:
            List of raw theme dicts (before validation)
        """
        from collections import defaultdict

        # Group topics by theme assignment
        theme_topics = defaultdict(list)
        for idx, theme_id in enumerate(theme_result.topic_assignments):
            # Skip outliers (theme_id = -1)
            if theme_id == -1:
                logger.debug(f"Topic from firm {topic_metadata[idx]['firm_id']} assigned to outlier")
                continue

            theme_topics[int(theme_id)].append(topic_metadata[idx])

        # Build theme dicts
        themes = []
        for theme_id, topics in theme_topics.items():
            # Get theme representation and keywords from model result
            name = theme_result.topic_representations.get(theme_id, f"Theme {theme_id}")
            keywords = theme_result.topic_keywords.get(theme_id, [])

            # Count distinct firms
            distinct_firms = set(t["firm_id"] for t in topics)

            theme = {
                "theme_id": "",  # Will be assigned after sorting
                "name": name,
                "keywords": keywords,
                "n_firms": len(distinct_firms),
                "n_topics": len(topics),
                "topics": topics,
                "metadata": {
                    "processing_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "model_config": self.config,
                    "validation": {
                        "min_firms": self.min_firms,
                        "max_firm_dominance": self.max_dominance,
                    },
                },
            }
            themes.append(theme)

        return themes

    def _validate_themes(self, themes: List[Dict]) -> List[Dict]:
        """
        Apply validation filters to themes.

        Filters:
        1. min_firms: Theme must have topics from at least min_firms distinct firms
        2. max_dominance: No single firm can have more than max_dominance share of topics

        Args:
            themes: List of raw theme dicts

        Returns:
            List of validated theme dicts
        """
        validated = []

        for theme in themes:
            # Get firm distribution
            firm_ids = [t["firm_id"] for t in theme["topics"]]
            distinct_firms = set(firm_ids)
            firm_counts = Counter(firm_ids)

            # Tier 1: Minimum firms filter
            if len(distinct_firms) < self.min_firms:
                logger.debug(
                    f"Filtering theme '{theme['name']}': only {len(distinct_firms)} firms (min: {self.min_firms})"
                )
                continue

            # Tier 2: Maximum dominance filter
            max_count = max(firm_counts.values())
            max_share = max_count / len(firm_ids)
            if max_share > self.max_dominance:
                logger.debug(
                    f"Filtering theme '{theme['name']}': {max_share:.1%} dominance by one firm (max: {self.max_dominance:.0%})"
                )
                continue

            validated.append(theme)

        logger.info(f"Validated {len(validated)}/{len(themes)} themes")
        return validated
