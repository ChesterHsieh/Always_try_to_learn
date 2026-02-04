"""
Data Ingestion Patterns

This package contains implementations of data ingestion-related patterns
from \"Data Engineering Design Patterns\" (2025), including:
- Transactional Writer
- Idempotent Writer
- Upsert Writer
- Append-Only Writer
- Change Data Capture (CDC)
"""

from .transactional_writer import SimpleDbSink, TransactionalWriter, demo_transactional_writer
from .idempotent_writer import (
    SimpleDbIdempotentSink,
    IdempotentWriter,
    demo_idempotent_writer,
)
from .upsert_writer import (
    SimpleDbUpsertSink,
    UpsertWriter,
    demo_upsert_writer,
)
from .append_only_writer import (
    SimpleDbAppendOnlySink,
    AppendOnlyWriter,
    demo_append_only_writer,
)
from .change_data_capture import (
    ChangeType,
    ChangeEvent,
    SimpleDbCDCSource,
    SimpleDbCDCSink,
    ChangeDataCapture,
    demo_change_data_capture,
)

__all__ = [
    # Transactional Writer
    "SimpleDbSink",
    "TransactionalWriter",
    "demo_transactional_writer",
    # Idempotent Writer
    "SimpleDbIdempotentSink",
    "IdempotentWriter",
    "demo_idempotent_writer",
    # Upsert Writer
    "SimpleDbUpsertSink",
    "UpsertWriter",
    "demo_upsert_writer",
    # Append-Only Writer
    "SimpleDbAppendOnlySink",
    "AppendOnlyWriter",
    "demo_append_only_writer",
    # Change Data Capture
    "ChangeType",
    "ChangeEvent",
    "SimpleDbCDCSource",
    "SimpleDbCDCSink",
    "ChangeDataCapture",
    "demo_change_data_capture",
]


