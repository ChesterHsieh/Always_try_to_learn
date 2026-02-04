from pathlib import Path

from rich.console import Console
from typer import Argument, Option, Typer

from ..process import process_files
from ..utils.generator import SampleDataPaths, create_sample_csvs
from ..utils.logging_setup import configure_logging
from ..utils.settings import AppSettings

app = Typer(help="Data pipeline CLI")
console = Console()


@app.callback()
def _init_logging(
    log_level: str = Option("INFO", "--log-level", "-l"),
) -> None:
    configure_logging(level=log_level.upper())


@app.command()
def version() -> None:
    """Show application version and key settings."""
    settings = AppSettings()
    console.print(
        {
            "app_name": settings.app_name,
            "env": settings.environment,
            "config_path": settings.config_path,
        }
    )


@app.command()
def generate_sample(out_dir: Path = Argument(Path("data"))) -> None:
    """Generate sample CSVs matching the provided image schema."""
    paths = SampleDataPaths(base_dir=out_dir)
    create_sample_csvs(paths)
    console.print({"orders": paths.orders_csv, "products": paths.products_csv})


@app.command()
def process(
    orders_csv: Path = Argument(...),
    products_csv: Path = Argument(...),
    out_dir: Path = Argument(Path("outputs")),
) -> None:
    """Process inputs and output cleaned orders and issues CSVs."""
    clean, issues = process_files(orders_csv, products_csv, out_dir)
    console.print({"clean": clean, "issues": issues})


if __name__ == "__main__":
    app()
