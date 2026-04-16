from dataclasses import dataclass
from datetime import datetime
from typing import Literal

RunStatus = Literal["queued", "running", "succeeded", "failed", "recovering"]
ALLOWED_STATUSES = {"queued", "running", "succeeded", "failed", "recovering"}


@dataclass
class RunContext:
    run_id: str
    pipeline_name: str
    input_path: str
    output_path: str
    start_time: datetime
    status: RunStatus = "running"

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        if not self.pipeline_name:
            raise ValueError("pipeline_name is required")
        if not self.input_path or not self.output_path:
            raise ValueError("input_path and output_path are required")
        if self.status not in ALLOWED_STATUSES:
            raise ValueError(f"invalid status: {self.status}")


def build_run_context(
    run_id: str,
    pipeline_name: str,
    input_path: str,
    output_path: str,
    *,
    status: RunStatus = "running",
) -> RunContext:
    context = RunContext(
        run_id=run_id,
        pipeline_name=pipeline_name,
        input_path=input_path,
        output_path=output_path,
        start_time=datetime.utcnow(),
        status=status,
    )
    context.validate()
    return context
