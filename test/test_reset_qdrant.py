from qdrant_client.http import models as qmodels

from reset_qdrant import _reset_collection


class FakeQdrantClient:
    def __init__(self):
        self.created_indexes = []
        self.created_collection = None

    def collection_exists(self, _collection):
        return False

    def create_collection(self, **kwargs):
        self.created_collection = kwargs

    def create_payload_index(self, **kwargs):
        self.created_indexes.append((kwargs["field_name"], kwargs["field_schema"]))


def test_reset_collection_creates_runtime_payload_indexes():
    client = FakeQdrantClient()

    _reset_collection(client, "demo", 768)

    assert client.created_collection["collection_name"] == "demo"
    assert client.created_indexes == [
        ("date_numeric", qmodels.PayloadSchemaType.INTEGER),
        ("doc_hash", qmodels.PayloadSchemaType.KEYWORD),
        ("scope", qmodels.PayloadSchemaType.KEYWORD),
        ("filename", qmodels.PayloadSchemaType.KEYWORD),
    ]
