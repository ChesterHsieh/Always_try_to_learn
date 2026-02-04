"""
Tests for the Transactional Writer pattern implementation.
"""

from patterns.utils.simple_db import SimpleDB
from patterns.data_ingestion import SimpleDbSink, TransactionalWriter


def test_successful_transactional_write_commits_all_records() -> None:
    """All valid records should be committed atomically to the target table."""
    db = SimpleDB()
    table_name = "events"
    sink = SimpleDbSink(db=db, table_name=table_name)

    raw_events = [
        {"id": "e1", "amount": 10},
        {"id": "e2", "amount": 20},
    ]

    def extract() -> list[dict]:
        return list(raw_events)

    def transform(events: list[dict]) -> list[dict]:
        # Simple pass-through transform in this test
        return events

    writer = TransactionalWriter(
        sink=sink,
        extract_fn=extract,
        transform_fn=transform,
    )

    written = writer.run()

    # Verify that all records are present after commit
    assert written == 2
    assert table_name in db.data
    values = list(db.data[table_name].values())
    assert len(values) == 2
    # The order is not guaranteed, but the set of amounts should match
    assert {v["amount"] for v in values} == {10, 20}


def test_failure_during_transform_triggers_rollback() -> None:
    """If the transform step fails, no records should be visible in the table."""
    db = SimpleDB()
    table_name = "events_failure"
    sink = SimpleDbSink(db=db, table_name=table_name)

    raw_events = [
        {"id": "e1", "amount": 10},
        {"id": "e2", "amount": -5},
    ]

    def extract() -> list[dict]:
        return list(raw_events)

    def transform(events: list[dict]) -> list[dict]:
        # Simulate a validation failure
        for event in events:
            if event["amount"] < 0:
                raise ValueError("Invalid amount")
        return events

    writer = TransactionalWriter(
        sink=sink,
        extract_fn=extract,
        transform_fn=transform,
    )

    try:
        writer.run()
        # We should never get here
        assert False, "Expected writer.run() to raise"
    except ValueError:
        # Expected path
        pass

    # After rollback, either the table is empty or does not exist
    if table_name in db.data:
        assert db.data[table_name] == {}


