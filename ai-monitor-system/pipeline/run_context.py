from dataclasses import dataclass
from datetime import datetime


@dataclass
class RunContext:
    run_id: str
    pipeline_name: str
    input_path: str
    output_path: str
    start_time: datetime

    def validate(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        if not self.input_path or not self.output_path:
            raise ValueError("input_path and output_path are required")
