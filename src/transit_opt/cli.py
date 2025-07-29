"""Console script for transit_opt."""

import typer
from rich.console import Console

from transit_opt import utils

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for transit_opt."""
    console.print("Replace this message by putting your code into "
               "transit_opt.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


if __name__ == "__main__":
    app()
