"""
Unit tests for DynamoDB utility classes.

Tests are written BEFORE implementation (TDD).
Uses moto for AWS mocking.
"""

import pytest
import boto3
from moto import mock_aws
from decimal import Decimal


@pytest.fixture
def mock_dynamodb_table():
    """Create a mock DynamoDB table matching our schema."""
    with mock_aws():
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")

        table = dynamodb.create_table(
            TableName="test-topics",
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
                {"AttributeName": "GSI1PK", "AttributeType": "S"},
                {"AttributeName": "GSI1SK", "AttributeType": "S"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "GSI1",
                    "KeySchema": [
                        {"AttributeName": "GSI1PK", "KeyType": "HASH"},
                        {"AttributeName": "GSI1SK", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        # Wait for table to be active
        table.meta.client.get_waiter("table_exists").wait(TableName="test-topics")

        yield table


class TestMapPhaseDynamoDBWriter:
    """Tests for MapPhaseDynamoDBWriter."""

    @mock_aws
    def test_init_connects_to_table(self):
        """Writer should connect to specified table."""
        from cloud.src.dynamodb_utils import MapPhaseDynamoDBWriter

        # Create table first
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        dynamodb.create_table(
            TableName="test-topics",
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        writer = MapPhaseDynamoDBWriter("test-topics")

        assert writer.table_name == "test-topics"

    @mock_aws
    def test_write_firm_sentences_creates_items(self):
        """write_firm_sentences should create DynamoDB items."""
        from cloud.src.dynamodb_utils import MapPhaseDynamoDBWriter
        from cloud.src.models import TranscriptSentence

        # Create table
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table = dynamodb.create_table(
            TableName="test-topics",
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        writer = MapPhaseDynamoDBWriter("test-topics")

        # Sample firm result
        firm_result = {
            "firm_id": "1001",
            "topics": [
                {
                    "topic_id": 0,
                    "sentence_ids": ["1001_T001_0000", "1001_T001_0001"],
                },
                {
                    "topic_id": 1,
                    "sentence_ids": ["1001_T001_0002"],
                },
            ],
        }

        sentences = [
            TranscriptSentence("1001_T001_0000", "AI investment.", "CEO", 0),
            TranscriptSentence("1001_T001_0001", "ML focus.", "CEO", 1),
            TranscriptSentence("1001_T001_0002", "Revenue grew.", "CFO", 2),
        ]

        writer.write_firm_sentences(firm_result, sentences)

        # Verify items were created
        response = table.scan()
        assert len(response["Items"]) == 3

    @mock_aws
    def test_sentence_item_pk_sk_format(self):
        """Sentence items should have correct PK/SK format."""
        from cloud.src.dynamodb_utils import MapPhaseDynamoDBWriter
        from cloud.src.models import TranscriptSentence

        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table = dynamodb.create_table(
            TableName="test-topics",
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        writer = MapPhaseDynamoDBWriter("test-topics")

        firm_result = {
            "firm_id": "1001",
            "topics": [{"topic_id": 0, "sentence_ids": ["1001_T001_0000"]}],
        }
        sentences = [TranscriptSentence("1001_T001_0000", "Test.", "CEO", 0)]

        writer.write_firm_sentences(firm_result, sentences)

        # Verify PK/SK format
        response = table.get_item(
            Key={"PK": "TOPIC#1001#0", "SK": "SENTENCE#1001_T001_0000"}
        )
        assert "Item" in response
        assert response["Item"]["PK"] == "TOPIC#1001#0"
        assert response["Item"]["SK"] == "SENTENCE#1001_T001_0000"

    @mock_aws
    def test_sentence_item_attributes(self):
        """Sentence items should have all required attributes."""
        from cloud.src.dynamodb_utils import MapPhaseDynamoDBWriter
        from cloud.src.models import TranscriptSentence

        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table = dynamodb.create_table(
            TableName="test-topics",
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        writer = MapPhaseDynamoDBWriter("test-topics")

        firm_result = {
            "firm_id": "1001",
            "topics": [{"topic_id": 0, "sentence_ids": ["1001_T001_0000"]}],
        }
        sentences = [TranscriptSentence("1001_T001_0000", "AI investment.", "CEO", 5)]

        writer.write_firm_sentences(firm_result, sentences)

        response = table.get_item(
            Key={"PK": "TOPIC#1001#0", "SK": "SENTENCE#1001_T001_0000"}
        )
        item = response["Item"]

        assert item["text"] == "AI investment."
        assert item["position"] == 5
        assert item["speaker_type"] == "CEO"
        assert item["firm_id"] == "1001"
        assert item["topic_id"] == 0

    @mock_aws
    def test_sentence_item_gsi1_attributes(self):
        """Sentence items should have GSI1PK/GSI1SK for firm lookups."""
        from cloud.src.dynamodb_utils import MapPhaseDynamoDBWriter
        from cloud.src.models import TranscriptSentence

        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table = dynamodb.create_table(
            TableName="test-topics",
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        writer = MapPhaseDynamoDBWriter("test-topics")

        firm_result = {
            "firm_id": "1001",
            "topics": [{"topic_id": 0, "sentence_ids": ["1001_T001_0000"]}],
        }
        sentences = [TranscriptSentence("1001_T001_0000", "Test.", "CEO", 0)]

        writer.write_firm_sentences(firm_result, sentences)

        response = table.get_item(
            Key={"PK": "TOPIC#1001#0", "SK": "SENTENCE#1001_T001_0000"}
        )
        item = response["Item"]

        assert item["GSI1PK"] == "FIRM#1001"
        assert item["GSI1SK"] == "SENTENCE#1001_T001_0000"

    @mock_aws
    def test_null_speaker_type_defaults_to_unknown(self):
        """Null speaker_type should default to UNKNOWN."""
        from cloud.src.dynamodb_utils import MapPhaseDynamoDBWriter
        from cloud.src.models import TranscriptSentence

        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table = dynamodb.create_table(
            TableName="test-topics",
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        writer = MapPhaseDynamoDBWriter("test-topics")

        firm_result = {
            "firm_id": "1001",
            "topics": [{"topic_id": 0, "sentence_ids": ["1001_T001_0000"]}],
        }
        sentences = [TranscriptSentence("1001_T001_0000", "Test.", None, 0)]  # No speaker

        writer.write_firm_sentences(firm_result, sentences)

        response = table.get_item(
            Key={"PK": "TOPIC#1001#0", "SK": "SENTENCE#1001_T001_0000"}
        )
        assert response["Item"]["speaker_type"] == "UNKNOWN"

    @mock_aws
    def test_write_multiple_topics(self):
        """Should correctly write sentences across multiple topics."""
        from cloud.src.dynamodb_utils import MapPhaseDynamoDBWriter
        from cloud.src.models import TranscriptSentence

        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table = dynamodb.create_table(
            TableName="test-topics",
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        writer = MapPhaseDynamoDBWriter("test-topics")

        firm_result = {
            "firm_id": "1001",
            "topics": [
                {"topic_id": 0, "sentence_ids": ["1001_T001_0000", "1001_T001_0001"]},
                {"topic_id": 1, "sentence_ids": ["1001_T001_0002"]},
            ],
        }
        sentences = [
            TranscriptSentence("1001_T001_0000", "S0.", "CEO", 0),
            TranscriptSentence("1001_T001_0001", "S1.", "CEO", 1),
            TranscriptSentence("1001_T001_0002", "S2.", "CFO", 2),
        ]

        writer.write_firm_sentences(firm_result, sentences)

        # Verify topic 0 sentences
        resp0 = table.get_item(Key={"PK": "TOPIC#1001#0", "SK": "SENTENCE#1001_T001_0000"})
        assert "Item" in resp0

        resp1 = table.get_item(Key={"PK": "TOPIC#1001#0", "SK": "SENTENCE#1001_T001_0001"})
        assert "Item" in resp1

        # Verify topic 1 sentence
        resp2 = table.get_item(Key={"PK": "TOPIC#1001#1", "SK": "SENTENCE#1001_T001_0002"})
        assert "Item" in resp2


class TestReducePhaseDynamoDBWriter:
    """Tests for ReducePhaseDynamoDBWriter."""

    @mock_aws
    def test_init_connects_to_table(self):
        """Writer should connect to specified table."""
        from cloud.src.dynamodb_utils import ReducePhaseDynamoDBWriter

        # Create table first
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        dynamodb.create_table(
            TableName="test-topics",
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        writer = ReducePhaseDynamoDBWriter("test-topics")

        assert writer.table_name == "test-topics"

    @mock_aws
    def test_write_themes_creates_metadata_item(self):
        """write_themes should create theme metadata items."""
        from cloud.src.dynamodb_utils import ReducePhaseDynamoDBWriter

        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table = dynamodb.create_table(
            TableName="test-topics",
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        writer = ReducePhaseDynamoDBWriter("test-topics")

        themes = [
            {
                "theme_id": "theme_20241130_001",
                "name": "AI Investment",
                "keywords": ["ai", "machine", "learning"],
                "n_firms": 3,
                "n_topics": 5,
                "topics": [],
                "metadata": {"discovery_method": "dual_bertopic"},  # No floats - DynamoDB requires Decimal
            }
        ]

        writer.write_themes(themes)

        # Verify metadata item
        response = table.get_item(
            Key={"PK": "THEME#theme_20241130_001", "SK": "METADATA"}
        )
        assert "Item" in response
        item = response["Item"]
        assert item["name"] == "AI Investment"
        assert item["n_firms"] == 3
        assert item["n_topics"] == 5

    @mock_aws
    def test_write_themes_creates_topic_link_items(self):
        """write_themes should create topic link items."""
        from cloud.src.dynamodb_utils import ReducePhaseDynamoDBWriter

        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table = dynamodb.create_table(
            TableName="test-topics",
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        writer = ReducePhaseDynamoDBWriter("test-topics")

        themes = [
            {
                "theme_id": "theme_20241130_001",
                "name": "AI Investment",
                "keywords": ["ai"],
                "n_firms": 2,
                "n_topics": 2,
                "topics": [
                    {"firm_id": "1001", "topic_id": 0, "representation": "AI Research"},
                    {"firm_id": "1002", "topic_id": 3, "representation": "ML Infrastructure"},
                ],
            }
        ]

        writer.write_themes(themes)

        # Verify topic link items
        resp1 = table.get_item(
            Key={"PK": "THEME#theme_20241130_001", "SK": "TOPIC#1001#0"}
        )
        assert "Item" in resp1
        assert resp1["Item"]["firm_id"] == "1001"
        assert resp1["Item"]["topic_id"] == 0
        assert resp1["Item"]["representation"] == "AI Research"

        resp2 = table.get_item(
            Key={"PK": "THEME#theme_20241130_001", "SK": "TOPIC#1002#3"}
        )
        assert "Item" in resp2
        assert resp2["Item"]["firm_id"] == "1002"

    @mock_aws
    def test_write_themes_gsi1_attributes(self):
        """Topic link items should have GSI1 attributes for firm lookups."""
        from cloud.src.dynamodb_utils import ReducePhaseDynamoDBWriter

        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table = dynamodb.create_table(
            TableName="test-topics",
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        writer = ReducePhaseDynamoDBWriter("test-topics")

        themes = [
            {
                "theme_id": "theme_001",
                "name": "Test",
                "keywords": [],
                "n_firms": 1,
                "n_topics": 1,
                "topics": [
                    {"firm_id": "1001", "topic_id": 0, "representation": "Test Topic"},
                ],
            }
        ]

        writer.write_themes(themes)

        response = table.get_item(
            Key={"PK": "THEME#theme_001", "SK": "TOPIC#1001#0"}
        )
        item = response["Item"]

        assert item["GSI1PK"] == "FIRM#1001"
        assert item["GSI1SK"] == "THEME#theme_001"

    @mock_aws
    def test_write_themes_returns_item_count(self):
        """write_themes should return number of items written."""
        from cloud.src.dynamodb_utils import ReducePhaseDynamoDBWriter

        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        dynamodb.create_table(
            TableName="test-topics",
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        writer = ReducePhaseDynamoDBWriter("test-topics")

        themes = [
            {
                "theme_id": "theme_001",
                "name": "Theme 1",
                "keywords": [],
                "n_firms": 2,
                "n_topics": 2,
                "topics": [
                    {"firm_id": "1001", "topic_id": 0, "representation": "T1"},
                    {"firm_id": "1002", "topic_id": 1, "representation": "T2"},
                ],
            },
            {
                "theme_id": "theme_002",
                "name": "Theme 2",
                "keywords": [],
                "n_firms": 1,
                "n_topics": 1,
                "topics": [
                    {"firm_id": "1003", "topic_id": 0, "representation": "T3"},
                ],
            },
        ]

        items_written = writer.write_themes(themes)

        # 2 themes = 2 metadata items + 3 topic links = 5 items
        assert items_written == 5

    @mock_aws
    def test_write_themes_limits_keywords(self):
        """write_themes should limit keywords to 20."""
        from cloud.src.dynamodb_utils import ReducePhaseDynamoDBWriter

        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table = dynamodb.create_table(
            TableName="test-topics",
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        writer = ReducePhaseDynamoDBWriter("test-topics")

        # Create theme with 30 keywords
        many_keywords = [f"keyword_{i}" for i in range(30)]
        themes = [
            {
                "theme_id": "theme_001",
                "name": "Test",
                "keywords": many_keywords,
                "n_firms": 0,
                "n_topics": 0,
                "topics": [],
            }
        ]

        writer.write_themes(themes)

        response = table.get_item(
            Key={"PK": "THEME#theme_001", "SK": "METADATA"}
        )
        item = response["Item"]

        # Should be limited to 20 keywords
        assert len(item["keywords"]) == 20
