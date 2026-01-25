"""
Data Extractor using google-langextract.
"""
import logging
from typing import List, Dict, Any

from langextract import extract, llm_google

from pydantic import BaseModel


logger = logging.getLogger(__name__)


class Person(BaseModel):
    """Schema for a person."""
    name: str


class Organization(BaseModel):
    """Schema for an organization."""
    name: str


class DataExtractor:
    """
    Wrapper around google-langextract to extract structured entities.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataExtractor.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.extractor = extract.Extractor(
            llm=llm_google.Google(api_key=config.get("api_key")),
            default_schema={"persons": List[Person], "organizations": List[Organization]},
        )

    async def extract_entities(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Extracts entities from a list of texts.

        Args:
            texts: A list of sentences or documents.

        Returns:
            A list of dictionaries, where each dictionary contains the
            extracted entities for the corresponding input text.
        """
        if not texts:
            return []

        logger.info(f"Extracting entities from {len(texts)} texts.")
        try:
            extractions = await self.extractor.extract(contents=texts)
            return extractions
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return [{} for _ in texts]

