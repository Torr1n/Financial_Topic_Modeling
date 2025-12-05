"""
UnifiedPipeline - Single-process pipeline for firm topic modeling and theme aggregation.

This class replaces the distributed AWS Batch map-reduce pattern with a single
GPU instance that:
    - Loads embedding model ONCE
    - Processes firms sequentially with per-firm checkpoints
    - Stores all results in PostgreSQL (no S3 intermediate)
    - Generates LLM summaries for topics and themes (Phase 3)

Phase 2+3 of the architecture pivot.
"""

import asyncio
import concurrent.futures
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from cloud.src.database.models import Base, Firm, Sentence, Topic
from cloud.src.database.repository import DatabaseRepository
from cloud.src.interfaces import DataConnector
from cloud.src.topic_models.bertopic_model import BERTopicModel
from cloud.src.firm_processor import FirmProcessor
from cloud.src.theme_aggregator import ThemeAggregator

logger = logging.getLogger(__name__)


class UnifiedPipeline:
    """
    Single-process pipeline for firm topic modeling and theme aggregation.

    Features:
    - Embedding model loaded ONCE at initialization
    - Per-firm checkpoints for spot instance resume
    - No S3 intermediate storage
    - Direct PostgreSQL writes with pgvector embeddings
    - LLM-generated topic summaries and theme descriptions (Phase 3)

    Usage:
        pipeline = UnifiedPipeline(database_url, config, device="cuda")
        pipeline.run(data_connector)
    """

    def __init__(
        self,
        database_url: str,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        """
        Initialize the unified pipeline.

        Args:
            database_url: PostgreSQL connection URL
            config: Configuration dict - see cloud/config/production.yaml for structure:
                   - embedding: {model, dimension, device}
                   - firm_topic_model: {umap, hdbscan, vectorizer, representation}
                   - theme_topic_model: {umap, hdbscan, vectorizer, representation}
                   - validation: {min_firms, max_firm_dominance}
                   - llm: {model, max_concurrent, timeout, max_retries}
            device: Device for embedding model ("cuda" or "cpu").
                   If None, uses config["embedding"]["device"] or defaults to "cpu".
        """
        logger.info("Initializing UnifiedPipeline")

        # Store config
        self.config = config

        # Extract embedding config (new structure)
        embedding_config = config.get("embedding", {})

        # Determine device: explicit param > embedding config > legacy config > default "cpu"
        if device is not None:
            self.device = device
        else:
            self.device = embedding_config.get("device", config.get("device", "cpu"))

        # Store embedding dimension for database operations
        self.embedding_dim = embedding_config.get("dimension", 768)

        # Database setup
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.SessionFactory = sessionmaker(bind=self.engine)

        # ML models - embedding model loaded ONCE
        embedding_model_name = embedding_config.get("model", config.get("embedding_model", "all-mpnet-base-v2"))
        logger.info(f"Loading embedding model: {embedding_model_name} (dim={self.embedding_dim}) on {self.device}")

        if self.device == "cpu":
            logger.warning(
                "⚠️  RUNNING ON CPU - For production/cloud use, set device='cuda' "
                "in config for 10x faster embeddings. CPU mode is for local testing only."
            )

        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)

        # Separate BERTopicModel instances with DIFFERENT configs for firm vs theme
        # This allows different hyperparameters for each stage
        firm_config = config.get("firm_topic_model", config)  # Fallback to legacy config
        theme_config = config.get("theme_topic_model", config)  # Fallback to legacy config

        # Add device to configs for BERTopicModel
        firm_config["device"] = self.device
        theme_config["device"] = self.device

        self.firm_topic_model = BERTopicModel(firm_config, embedding_model=self.embedding_model)
        self.theme_topic_model = BERTopicModel(theme_config, embedding_model=self.embedding_model)

        # Processors - pass validation config
        self.firm_processor = FirmProcessor(self.firm_topic_model, config)
        self.theme_aggregator = ThemeAggregator(self.theme_topic_model, config)

        # LLM client for topic/theme summarization (Phase 3)
        self._xai_client = self._init_llm_client()

        logger.info("UnifiedPipeline initialized")

    def _init_llm_client(self):
        """
        Initialize LLM client if API key is present.

        Returns:
            XAIClient instance or None if no API key
        """
        api_key = os.environ.get("XAI_API_KEY")
        if api_key:
            from cloud.src.llm import XAIClient
            logger.info("XAI_API_KEY found - LLM summaries enabled")
            return XAIClient(api_key=api_key, config=self.config.get("llm", {}))
        else:
            logger.info("XAI_API_KEY not found - using keyword fallbacks for summaries")
            return None

    def _run_async(self, coro):
        """
        Safely run async coroutine from sync context.

        Handles both cases:
        - No existing event loop: uses asyncio.run()
        - Existing event loop (notebook/async): runs in thread pool

        Args:
            coro: Async coroutine to execute

        Returns:
            Result of the coroutine
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - use asyncio.run()
            return asyncio.run(coro)
        else:
            # Loop exists (notebook/async context) - run in thread
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()

    def run(self, data_source: DataConnector) -> None:
        """
        Execute full pipeline: firm processing + theme aggregation.

        Args:
            data_source: DataConnector providing transcript data
        """
        logger.info("Starting unified pipeline")

        # Stage 1: Firm Processing
        self._process_firms(data_source)

        # Stage 2: Theme Aggregation
        self._aggregate_themes()

        # Finalize: Build vector indexes (HNSW)
        with self.SessionFactory() as session:
            repo = DatabaseRepository(session)
            logger.info("Building vector indexes...")
            repo.build_vector_indexes()
            session.commit()

        logger.info("Unified pipeline completed")

    def _process_firms(self, data_source: DataConnector) -> None:
        """
        Process all unprocessed firms with checkpoints.

        Uses context manager per firm for clean session management.
        """
        # Get unprocessed firm IDs
        unprocessed = self._get_unprocessed_firm_ids(data_source)
        total = len(unprocessed)

        logger.info(f"Processing {total} unprocessed firms")

        for i, firm_id in enumerate(unprocessed, 1):
            logger.info(f"[{i}/{total}] Processing firm {firm_id}")
            self._process_single_firm(firm_id, data_source)

        logger.info(f"Firm processing complete: {total} firms processed")

    def _process_single_firm(
        self,
        firm_id: str,
        data_source: DataConnector,
    ) -> None:
        """
        Process a single firm with checkpoint.

        Pattern per Codex review:
        1. Insert topics FIRST, flush to get IDs
        2. Insert sentences WITH topic_id already set
        3. Mark firm as processed and commit (checkpoint)

        Phase 3 addition: Generate LLM summaries for topics before writing.
        """
        # Use fresh session per firm (Codex review)
        with self.SessionFactory() as session:
            repo = DatabaseRepository(session)

            try:
                # Load firm data
                transcript_data = data_source.fetch_transcripts(
                    firm_ids=[firm_id],
                    start_date="1900-01-01",
                    end_date="2100-01-01",
                )
                firm_data = transcript_data.firms[firm_id]

                # Get or create firm record
                firm = repo.get_or_create_firm(
                    company_id=firm_id,
                    name=firm_data.firm_name,
                )
                session.flush()

                # Compute sentence embeddings from cleaned_text (model loaded once)
                texts = [s.cleaned_text for s in firm_data.sentences]
                logger.info(f"Computing embeddings for {len(texts)} sentences")
                sentence_embeddings = self.embedding_model.encode(texts)

                # Process with pre-computed embeddings
                output, topic_assignments = self.firm_processor.process(
                    firm_data,
                    embeddings=sentence_embeddings,
                )

                # Generate LLM summaries for topics (Phase 3)
                # Pass raw sentences grouped by topic for richer context
                self._generate_topic_summaries(
                    output=output,
                    firm_data=firm_data,
                    topic_assignments=topic_assignments,
                )

                # Write results to Postgres (topics first, then sentences)
                self._write_firm_results(
                    session=session,
                    repo=repo,
                    firm=firm,
                    firm_data=firm_data,
                    output=output,
                    topic_assignments=topic_assignments,
                    sentence_embeddings=sentence_embeddings,
                )

                # CHECKPOINT: Mark firm as processed and commit
                repo.mark_firm_processed(firm.id)
                session.commit()

                logger.info(f"Firm {firm_id} processed: {output['n_topics']} topics")

            except Exception as e:
                session.rollback()
                logger.error(f"Failed to process firm {firm_id}: {e}")
                raise

    def _generate_topic_summaries(
        self,
        output: Dict[str, Any],
        firm_data,
        topic_assignments: np.ndarray,
    ) -> None:
        """
        Generate LLM summaries for topics.

        Updates output["topics"] in-place with "summary" and "sentences" fields.
        Falls back to representation (keywords) if LLM unavailable or fails.

        Args:
            output: FirmProcessor output dict with "topics" list
            firm_data: FirmTranscriptData with raw sentences
            topic_assignments: Array mapping sentence index to topic ID
        """
        topics = output.get("topics", [])
        if not topics:
            return

        # Group raw sentences by topic ID
        topic_sentences = {}
        for i, sentence in enumerate(firm_data.sentences):
            topic_id = int(topic_assignments[i])
            if topic_id >= 0:  # Skip outliers (-1)
                if topic_id not in topic_sentences:
                    topic_sentences[topic_id] = []
                topic_sentences[topic_id].append(sentence.raw_text)

        # Add sentences to each topic dict
        for topic in topics:
            topic["sentences"] = topic_sentences.get(topic["topic_id"], [])

        if self._xai_client is None:
            # No LLM client - use keyword fallback
            for topic in topics:
                topic["summary"] = topic["representation"]
            return

        # Generate summaries via LLM (log first prompt for observability)
        logger.info(f"Generating LLM summaries for {len(topics)} topics")

        async def generate_all():
            return await self._xai_client.generate_batch_summaries(
                topics,
                log_first_prompt=True,  # Log one example prompt per firm
            )

        try:
            summaries = self._run_async(generate_all())

            # Apply summaries (fallback to representation if None)
            for i, topic in enumerate(topics):
                if summaries[i] is not None:
                    topic["summary"] = summaries[i]
                else:
                    topic["summary"] = topic["representation"]
                    logger.warning(f"LLM failed for topic {topic['topic_id']}, using fallback")

        except Exception as e:
            logger.error(f"LLM batch generation failed: {e}, using fallbacks")
            for topic in topics:
                topic["summary"] = topic["representation"]

    def _write_firm_results(
        self,
        session: Session,
        repo: DatabaseRepository,
        firm: Firm,
        firm_data,
        output: Dict[str, Any],
        topic_assignments: np.ndarray,
        sentence_embeddings: np.ndarray,
    ) -> None:
        """
        Write firm results to Postgres.

        Pattern per Codex review: Insert topics FIRST, flush to get IDs,
        then insert sentences WITH topic_id already set (no update needed).

        Phase 3: Topic embeddings now use summaries (not keywords).
        """
        # 1. Compute topic embeddings from summaries (Phase 3)
        # Use summary if available, fallback to representation
        topic_texts = []
        for t in output["topics"]:
            # Summary should be set by _generate_topic_summaries(), but fallback just in case
            text = t.get("summary") or t["representation"]
            topic_texts.append(text)

        if topic_texts:
            topic_embeddings = self.embedding_model.encode(topic_texts)
        else:
            topic_embeddings = np.array([])

        # 2. Insert topics FIRST, flush to get IDs
        topic_id_map = {}  # local_topic_id -> db_topic_id
        for i, topic in enumerate(output["topics"]):
            # Get summary (fallback to representation if not set)
            summary = topic.get("summary") or topic["representation"]

            db_topic = Topic(
                firm_id=firm.id,
                local_topic_id=topic["topic_id"],
                representation=topic["representation"],  # Keywords preserved
                summary=summary,  # LLM summary or fallback
                n_sentences=topic["size"],
                embedding=topic_embeddings[i].tolist() if len(topic_embeddings) > 0 else None,
            )
            session.add(db_topic)
            session.flush()  # Get ID
            topic_id_map[topic["topic_id"]] = db_topic.id

        # 3. Build sentence records WITH topic_id already set
        sentence_records = []
        for i, sentence in enumerate(firm_data.sentences):
            # Get topic assignment (-1 for outliers)
            local_topic_id = int(topic_assignments[i])
            db_topic_id = topic_id_map.get(local_topic_id)  # None for outliers

            sentence_records.append({
                "firm_id": firm.id,
                "raw_text": sentence.raw_text,  # Original for observability
                "cleaned_text": sentence.cleaned_text,  # Preprocessed for topic modeling
                "position": sentence.position,
                "speaker_type": sentence.speaker_type,
                "embedding": sentence_embeddings[i].tolist(),
                "topic_id": db_topic_id,  # Already set, no update needed
            })

        # 4. Bulk insert sentences
        repo.bulk_insert_sentences(sentence_records)

    def _get_unprocessed_firm_ids(self, data_source: DataConnector) -> List[str]:
        """
        Get firm IDs that haven't been processed yet.

        Enables resume from spot instance interruption.
        """
        with self.SessionFactory() as session:
            repo = DatabaseRepository(session)
            processed_ids = set(repo.get_processed_firm_ids())

        all_firm_ids = data_source.get_available_firm_ids()
        unprocessed = [f for f in all_firm_ids if f not in processed_ids]

        logger.info(f"Found {len(unprocessed)} unprocessed firms out of {len(all_firm_ids)} total")
        return unprocessed

    def _aggregate_themes(self) -> None:
        """
        Aggregate firm topics into cross-firm themes.

        Stage 2 of the pipeline: reads topics from Postgres,
        clusters them into themes, writes results back.

        Phase 3: Generates LLM descriptions for themes before writing.
        """
        logger.info("Starting theme aggregation")

        with self.SessionFactory() as session:
            repo = DatabaseRepository(session)

            # Get all topics from all firms
            topics = session.query(Topic).all()

            if not topics:
                logger.warning("No topics to aggregate")
                return

            logger.info(f"Aggregating {len(topics)} topics into themes")

            # Prepare topic data for aggregation (includes summaries for clustering)
            firm_topic_outputs = self._build_firm_topic_outputs(session, topics)

            # Run theme aggregation
            theme_results = self.theme_aggregator.aggregate(firm_topic_outputs)

            # Generate LLM descriptions for themes (Phase 3)
            self._generate_theme_descriptions(theme_results, firm_topic_outputs)

            # Write themes to Postgres
            self._write_themes(session, repo, theme_results, topics)

            session.commit()

        logger.info(f"Theme aggregation complete: {len(theme_results)} themes")

    def _generate_theme_descriptions(
        self,
        theme_results: List[Dict[str, Any]],
        firm_topic_outputs: List[Dict[str, Any]],
    ) -> None:
        """
        Generate LLM descriptions for themes.

        Updates theme_results in-place with "description" field.
        Uses topic summaries (not keywords) in the prompt.

        Args:
            theme_results: List of theme dicts from ThemeAggregator
            firm_topic_outputs: Topic outputs with summaries for lookup
        """
        if not theme_results:
            return

        if self._xai_client is None:
            # No LLM client - use name as fallback description
            for theme in theme_results:
                theme["description"] = theme.get("name", "")
            return

        # Build topic summary lookup
        topic_summary_lookup = {}
        for firm_output in firm_topic_outputs:
            for topic in firm_output["topics"]:
                key = (firm_output["firm_id"], topic["topic_id"])
                topic_summary_lookup[key] = topic.get("summary", topic["representation"])

        logger.info(f"Generating LLM descriptions for {len(theme_results)} themes")

        async def generate_all():
            tasks = []
            for i, theme in enumerate(theme_results):
                # Use theme keywords when available; fallback to name/representation
                keyword_list = theme.get("keywords", [])
                theme_keywords = ", ".join(keyword_list[:10]) if keyword_list else theme.get("name", "")

                # Collect topic summaries for this theme
                topic_summaries = []
                for topic_meta in theme.get("topics", []):
                    key = (topic_meta["firm_id"], topic_meta["topic_id"])
                    summary = topic_summary_lookup.get(key, "")
                    if summary:
                        topic_summaries.append(summary)

                tasks.append(
                    self._xai_client.generate_theme_description(
                        theme_keywords=theme_keywords,
                        topic_summaries=topic_summaries[:10],  # Limit for prompt size
                        log_prompt=(i == 0),  # Log first theme prompt for observability
                    )
                )
            return await asyncio.gather(*tasks, return_exceptions=True)

        try:
            descriptions = self._run_async(generate_all())

            # Apply descriptions (fallback to name if None or exception)
            for i, theme in enumerate(theme_results):
                desc = descriptions[i]
                if isinstance(desc, Exception) or desc is None:
                    theme["description"] = theme.get("name", "")
                    logger.warning(f"LLM failed for theme {i}, using fallback")
                else:
                    theme["description"] = desc

        except Exception as e:
            logger.error(f"LLM theme description generation failed: {e}, using fallbacks")
            for theme in theme_results:
                theme["description"] = theme.get("name", "")

    def _build_firm_topic_outputs(
        self,
        session: Session,
        topics: List[Topic],
    ) -> List[Dict[str, Any]]:
        """
        Build FirmTopicOutput-like dicts from database topics.

        This bridges the gap between Postgres storage and the
        ThemeAggregator which expects the original dict format.

        Phase 3: Includes summary field for theme aggregation.
        ThemeAggregator will cluster on summaries (not keywords).
        """
        # Group topics by firm
        from collections import defaultdict
        firm_topics = defaultdict(list)
        for topic in topics:
            firm_topics[topic.firm_id].append(topic)

        # Build output format
        outputs = []
        for firm_id, firm_topic_list in firm_topics.items():
            firm = session.get(Firm, firm_id)
            outputs.append({
                "firm_id": firm.company_id,
                "firm_name": firm.name,
                "n_topics": len(firm_topic_list),
                "topics": [
                    {
                        "topic_id": t.local_topic_id,
                        "representation": t.representation,
                        # Phase 3: Include summary for theme clustering
                        # Use summary when available, fallback to representation
                        "summary": t.summary or t.representation,
                        "keywords": [],  # Not stored in DB
                        "size": t.n_sentences,
                        "sentence_ids": [],  # Not needed for aggregation
                    }
                    for t in firm_topic_list
                ],
                "outlier_sentence_ids": [],
                "metadata": {},
            })

        return outputs

    def _write_themes(
        self,
        session: Session,
        repo: DatabaseRepository,
        theme_results: List[Dict[str, Any]],
        all_topics: List[Topic],
    ) -> None:
        """
        Write themes to Postgres and update topic.theme_id.

        Pattern per Codex review: Create theme, flush to get ID,
        then update associated topics' theme_id.

        Phase 3: Theme embeddings now use descriptions (not keywords).
        """
        from cloud.src.database.models import Theme

        # Build lookup for topics by (firm_company_id, local_topic_id)
        topic_lookup = {}
        for topic in all_topics:
            firm = session.get(Firm, topic.firm_id)
            key = (firm.company_id, topic.local_topic_id)
            topic_lookup[key] = topic

        for theme_data in theme_results:
            theme_keywords = theme_data.get("name", "") or theme_data.get("representation", "")

            # Phase 3: Use description for embedding (richer text from LLM)
            # Fall back to name if description not available
            description = theme_data.get("description") or theme_keywords
            theme_text = description if description else theme_keywords

            # Compute embedding from description (or fallback text)
            theme_embedding = self.embedding_model.encode([theme_text])[0] if theme_text else None

            # Create theme, flush to get ID
            theme = Theme(
                name=theme_data.get("name", f"Theme {theme_data.get('theme_id', 0)}"),
                description=description,  # Phase 3: Store LLM description
                n_topics=theme_data.get("n_topics", 0),
                n_firms=theme_data.get("n_firms", 0),
                embedding=theme_embedding.tolist() if theme_embedding is not None else None,
            )
            session.add(theme)
            session.flush()  # Get ID

            # Update topics with theme_id
            for topic_meta in theme_data.get("topics", []):
                key = (topic_meta["firm_id"], topic_meta["topic_id"])
                if key in topic_lookup:
                    topic_lookup[key].theme_id = theme.id
