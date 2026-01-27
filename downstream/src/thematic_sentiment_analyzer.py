"""
Thematic Sentiment Analyzer for Investment Strategy

This module implements sentiment analysis on thematic clusters from earnings calls,
using FinBERT-tone to compute sentiment scores for event study analysis.

Key Features:
- Hierarchical sentiment aggregation (sentence → firm → theme)
- Multiple aggregation strategies for research flexibility
- Batch processing with GPU support
- Event study output format with PERMNO identifiers

Author: Assistant
Date: 2025-01-31
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
from transformers import BertForSequenceClassification, BertTokenizer, pipeline
from tqdm import tqdm
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Container for sentiment analysis results."""
    text: str
    label: str  # 'positive', 'negative', or 'neutral'
    score: float  # confidence score from model


class ThematicSentimentAnalyzer:
    """
    Analyzes sentiment of thematic clusters using FinBERT-tone.
    
    This class processes the hierarchical output from thematic analysis
    and produces sentiment scores suitable for event study analysis.
    
    Attributes:
        model: FinBERT-tone model for sentiment classification
        tokenizer: BERT tokenizer for text preprocessing
        device: Computing device (CPU/GPU)
        batch_size: Number of sentences to process simultaneously
    """
    
    def __init__(
        self,
        model_name: str = 'yiyanghkust/finbert-tone',
        batch_size: int = 32,
        use_gpu: bool = True
    ):
        """
        Initialize the sentiment analyzer with FinBERT-tone.
        
        Args:
            model_name: HuggingFace model identifier
            batch_size: Batch size for processing
            use_gpu: Whether to use GPU if available
        """
        logger.info(f"Initializing ThematicSentimentAnalyzer with {model_name}")
        
        # Set device
        device_num = 0 if use_gpu and torch.cuda.is_available() else -1
        logger.info(f"Using device: {'GPU' if device_num == 0 else 'CPU'}")
        
        # Load the FinBERT model and tokenizer explicitly
        finbert = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Initialize the HuggingFace pipeline with the model and tokenizer
        self.nlp = pipeline(
            "sentiment-analysis",
            model=finbert,
            tokenizer=tokenizer,
            device=device_num
        )
        
        # Configuration
        self.batch_size = batch_size
        self.max_length = 512  # BERT maximum sequence length
        
        # Aggregation weights for different speaker roles
        self.speaker_weights = {
            "CEO": 1.5,
            "CFO": 1.5,
            "Chief Executive Officer": 1.5,
            "Chief Financial Officer": 1.5,
            "President": 1.3,
            "COO": 1.2,
            "Chief Operating Officer": 1.2,
            "default": 1.0
        }
        
        logger.info("Initialization complete")
    
    def analyze_themes(
        self,
        thematic_output: Dict[str, Any],
        aggregation_strategy: str = 'confidence_weighted',
        permno_mapping: Optional[Dict[str, int]] = None,
        output_csv: bool = True,
        csv_directory: str = 'sentiment_analysis_output'
    ) -> Dict[str, Any]:
        """
        Main entry point for analyzing thematic output.
        
        Processes the hierarchical theme structure and returns event study format
        organized by theme, with optional CSV export.
        
        Args:
            thematic_output: Output from thematic analysis pipeline
            aggregation_strategy: Method for aggregating sentence sentiments
            permno_mapping: Mapping from firm IDs to PERMNO (if None, uses placeholder)
            output_csv: Whether to export CSV files for each theme
            csv_directory: Directory to save CSV files
            
        Returns:
            Dictionary with:
                - 'by_theme': Dict mapping theme_id to list of event dicts
                - 'all_events': Flat list of all events across themes
                - 'csv_files': List of generated CSV file paths (if output_csv=True)
        """
        logger.info(f"Analyzing {len(thematic_output.get('themes', []))} themes")
        
        results_by_theme = {}
        all_events = []
        csv_files = []
        
        # Create output directory if needed
        if output_csv:
            from pathlib import Path
            csv_path = Path(csv_directory)
            csv_path.mkdir(parents=True, exist_ok=True)
        
        # Process each theme
        for theme in thematic_output.get('themes', []):
            theme_id = theme['theme_id']
            theme_name = theme['theme_name']
            theme_events = []
            
            logger.info(f"Processing theme: {theme_name} ({theme_id})")
            
            # Process each firm's contribution to the theme
            for firm_contribution in theme['firm_contributions']:
                firm_id = firm_contribution['firm_id']
                firm_name = firm_contribution['firm_name']

                # Extract PERMNO if available (directly from contribution)
                permno = firm_contribution.get('permno', None)

                # Extract earnings date if available
                earnings_date = firm_contribution.get('earnings_call_date', None)

                # Extract sentences
                sentences_data = firm_contribution['sentences']
                
                if not sentences_data:
                    logger.warning(f"No sentences for {firm_name} in {theme_name}")
                    continue
                
                # Compute sentiments for all sentences
                sentiment_results = self.compute_sentence_sentiments(sentences_data)
                
                # Aggregate to firm-theme level
                firm_theme_sentiment = self.aggregate_firm_theme_sentiment(
                    sentiment_results,
                    sentences_data,
                    strategy=aggregation_strategy
                )
                
                # Create event for event study
                event = self.format_for_event_study(
                    firm_id=firm_id,
                    firm_name=firm_name,
                    sentiment=firm_theme_sentiment,
                    theme_id=theme_id,
                    theme_name=theme_name,
                    sentences_data=sentences_data,
                    permno_mapping=permno_mapping,
                    earnings_date=earnings_date,
                    permno=permno  # Pass PERMNO directly from contribution
                )
                
                # Only add event if it has a valid PERMNO
                if event is not None:
                    theme_events.append(event)
                    all_events.append(event)
                else:
                    logger.warning(f"Skipping {firm_name} - no PERMNO mapping")
            
            # Store theme results
            results_by_theme[theme_id] = {
                'theme_name': theme_name,
                'events': theme_events,
                'n_firms': len(theme_events)
            }
            
            # Export to CSV if requested
            if output_csv and theme_events:
                csv_file = self._export_theme_to_csv(
                    theme_id=theme_id,
                    theme_name=theme_name,
                    events=theme_events,
                    csv_directory=csv_directory
                )
                csv_files.append(csv_file)
                logger.info(f"  Exported {len(theme_events)} events to {csv_file}")
        
        logger.info(f"Generated {len(all_events)} total events across {len(results_by_theme)} themes")
        
        return {
            'by_theme': results_by_theme,
            'all_events': all_events,
            'csv_files': csv_files if output_csv else []
        }
    
    def compute_sentence_sentiments(
        self,
        sentences_data: List[Dict[str, Any]]
    ) -> List[SentimentResult]:
        """
        Compute sentiment for a list of sentences using FinBERT.
        
        Args:
            sentences_data: List of sentence dictionaries with 'text' field
            
        Returns:
            List of SentimentResult objects
        """
        # Extract texts
        texts = [s['text'] for s in sentences_data]
        
        if not texts:
            return []
        
        results = []
        
        # Process in batches using the pipeline
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Use the HuggingFace pipeline to get predictions
            # This returns a list of dicts with 'label' and 'score'
            predictions = self.nlp(batch_texts)
            # Convert to SentimentResult objects
            for text, pred in zip(batch_texts, predictions):
                # FinBERT-tone labels are already in the correct format
                result = SentimentResult(
                    text=text,
                    label=pred['label'].lower(),  # 'positive', 'negative', or 'neutral'
                    score=pred['score']  # confidence score
                )
                results.append(result)
        
        return results
    
    
    def aggregate_firm_theme_sentiment(
        self,
        sentiment_results: List[SentimentResult],
        sentences_data: List[Dict[str, Any]],
        strategy: str = 'confidence_weighted'
    ) -> float:
        """
        Aggregate sentence-level sentiments to firm-theme level.
        
        Uses the formula: (positive_count - negative_count) / total_count
        
        Args:
            sentiment_results: List of sentiment results
            sentences_data: Original sentence data with metadata
            strategy: Aggregation strategy to use (kept for compatibility)
            
        Returns:
            Aggregated sentiment score: (positive - negative) / total
        """
        if not sentiment_results:
            return 0.0
        
        # Count sentences by sentiment label
        positive_count = sum(1 for r in sentiment_results if r.label == 'positive')
        negative_count = sum(1 for r in sentiment_results if r.label == 'negative')
        neutral_count = sum(1 for r in sentiment_results if r.label == 'neutral')
        total_count = len(sentiment_results)
        
        # Calculate sentiment score as (positive - negative) / total
        sentiment_score = (positive_count - negative_count) / total_count
        
        # Log the counts for transparency
        logger.debug(f"Sentiment counts: Positive={positive_count}, Negative={negative_count}, "
                    f"Neutral={neutral_count}, Total={total_count}, Score={sentiment_score:.4f}")
        
        return sentiment_score
    
    
    
    
    def format_for_event_study(
        self,
        firm_id: str,
        firm_name: str,
        sentiment: float,
        theme_id: str,
        theme_name: str,
        sentences_data: List[Dict[str, Any]],
        permno_mapping: Optional[Dict[str, int]] = None,
        earnings_date: Optional[str] = None,
        permno: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Format results for event study analysis.

        Args:
            firm_id: Firm identifier
            firm_name: Firm name
            sentiment: Aggregated sentiment score
            theme_id: Theme identifier
            theme_name: Theme name
            sentences_data: Original sentence data
            permno_mapping: Mapping from firm ID to PERMNO (deprecated, use permno param)
            earnings_date: Actual earnings call date (YYYY-MM-DD format)
            permno: PERMNO directly from firm contribution (preferred)

        Returns:
            Event dictionary for event study
        """
        # Get PERMNO (prefer direct permno param, fallback to mapping)
        if permno is not None:
            # Use PERMNO directly from firm contribution
            pass
        elif permno_mapping and firm_id in permno_mapping:
            permno = permno_mapping[firm_id]
        else:
            # NO DEFAULT MAPPINGS - these were incorrect!
            # Return None if no mapping provided
            permno = None
            logger.warning(f"No PERMNO mapping for {firm_id} - skipping")
        
        # Use actual earnings call date as event date
        if earnings_date:
            # Parse the earnings date (handle both formats)
            try:
                # Try with time component first
                if ' ' in earnings_date:
                    earnings_dt = datetime.strptime(earnings_date.split(' ')[0], '%Y-%m-%d')
                else:
                    earnings_dt = datetime.strptime(earnings_date, '%Y-%m-%d')
                event_date = earnings_dt  # Use actual earnings date
            except Exception as e:
                logger.warning(f"Failed to parse earnings_date '{earnings_date}': {e}")
                event_date = datetime.now()
        else:
            # Try to extract from transcript_date in sentences
            if sentences_data and 'transcript_date' in sentences_data[0]:
                try:
                    transcript_date = sentences_data[0]['transcript_date']
                    # Handle datetime format with time component
                    if ' ' in transcript_date:
                        earnings_dt = datetime.strptime(transcript_date.split(' ')[0], '%Y-%m-%d')
                    else:
                        earnings_dt = datetime.strptime(transcript_date, '%Y-%m-%d')
                    event_date = earnings_dt  # Use actual earnings date
                except Exception as e:
                    logger.warning(f"Failed to parse transcript_date: {e}")
                    event_date = datetime.now()
            else:
                # Fallback to current date
                logger.warning("No earnings date found - using current date")
                event_date = datetime.now()
        
        # Format date as MM/DD/YYYY for event study
        edate = event_date.strftime("%m/%d/%Y")
        
        # Only create event if we have a valid PERMNO
        if permno is None:
            return None
        
        # Create event dictionary
        event = {
            "permno": permno,
            "edate": edate,
            "sentiment": round(sentiment, 4),  # Round to 4 decimal places
            "metadata": {
                "firm_id": firm_id,
                "firm_name": firm_name,
                "theme_id": theme_id,
                "theme_name": theme_name,
                "n_sentences": len(sentences_data)
            }
        }
        
        return event
    
    def analyze_single_theme(
        self,
        theme_data: Dict[str, Any],
        aggregation_strategy: str = 'confidence_weighted'
    ) -> pd.DataFrame:
        """
        Analyze a single theme and return results as DataFrame.
        
        Useful for detailed analysis of individual themes.
        
        Args:
            theme_data: Single theme from thematic output
            aggregation_strategy: Aggregation method
            
        Returns:
            DataFrame with sentiment analysis results
        """
        results = []
        
        for firm in theme_data['firm_contributions']:
            sentences_data = firm['sentences']
            
            # Compute sentiments
            sentiment_results = self.compute_sentence_sentiments(sentences_data)
            
            # Aggregate
            firm_sentiment = self.aggregate_firm_theme_sentiment(
                sentiment_results,
                sentences_data,
                strategy=aggregation_strategy
            )
            
            # Count sentiment labels
            positive_count = sum(1 for r in sentiment_results if r.label == 'positive')
            negative_count = sum(1 for r in sentiment_results if r.label == 'negative')
            neutral_count = sum(1 for r in sentiment_results if r.label == 'neutral')
            
            # Collect detailed statistics
            result = {
                'firm_id': firm['firm_id'],
                'firm_name': firm['firm_name'],
                'cluster_id': firm['cluster_id'],
                'n_sentences': len(sentences_data),
                'sentiment_score': firm_sentiment,
                'avg_confidence': np.mean([r.score for r in sentiment_results]),
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'positive_ratio': positive_count / len(sentiment_results) if sentiment_results else 0,
                'negative_ratio': negative_count / len(sentiment_results) if sentiment_results else 0,
                'neutral_ratio': neutral_count / len(sentiment_results) if sentiment_results else 0
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def _export_theme_to_csv(
        self,
        theme_id: str,
        theme_name: str,
        events: List[Dict[str, Any]],
        csv_directory: str
    ) -> str:
        """
        Export theme events to CSV file for event study.
        
        Args:
            theme_id: Theme identifier
            theme_name: Theme name
            events: List of event dictionaries
            csv_directory: Directory to save CSV
            
        Returns:
            Path to generated CSV file
        """
        import pandas as pd
        from pathlib import Path
        
        # Create DataFrame from events
        df_data = []
        for event in events:
            # Extract core event study fields
            row = {
                'permno': event['permno'],
                'edate': event['edate'],
                'sentiment': event['sentiment']
            }
            
            # Add metadata if needed
            if 'metadata' in event:
                row['firm_name'] = event['metadata'].get('firm_name', '')
                row['firm_id'] = event['metadata'].get('firm_id', '')
                row['n_sentences'] = event['metadata'].get('n_sentences', 0)
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Create filename (sanitize theme name for filesystem)
        safe_theme_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' 
                                 for c in theme_name).strip()
        filename = f"{theme_id}_{safe_theme_name}_sentiment.csv"
        filepath = Path(csv_directory) / filename
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        
        return str(filepath)
    
    def get_sentiment_distribution(
        self,
        sentiment_results: List[SentimentResult]
    ) -> Dict[str, Any]:
        """
        Get distribution statistics for sentiment results.
        
        Args:
            sentiment_results: List of sentiment results
            
        Returns:
            Dictionary with distribution statistics
        """
        # Count labels
        positive_count = sum(1 for r in sentiment_results if r.label == 'positive')
        negative_count = sum(1 for r in sentiment_results if r.label == 'negative')
        neutral_count = sum(1 for r in sentiment_results if r.label == 'neutral')
        total_count = len(sentiment_results)
        
        # Calculate overall sentiment score
        sentiment_score = (positive_count - negative_count) / total_count if total_count > 0 else 0
        
        # Get confidence scores
        confidence_scores = [r.score for r in sentiment_results]
        
        return {
            'sentiment_score': sentiment_score,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'total_count': total_count,
            'positive_ratio': positive_count / total_count if total_count > 0 else 0,
            'negative_ratio': negative_count / total_count if total_count > 0 else 0,
            'neutral_ratio': neutral_count / total_count if total_count > 0 else 0,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'min_confidence': np.min(confidence_scores) if confidence_scores else 0,
            'max_confidence': np.max(confidence_scores) if confidence_scores else 0
        }