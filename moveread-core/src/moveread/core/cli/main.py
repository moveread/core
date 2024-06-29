from haskellian import promise as P
import typer
from moveread.core import cli

app = typer.Typer(no_args_is_help=True)
export = typer.Typer(no_args_is_help=True)
app.add_typer(export, name="export")

Verbose = typer.Option(False, '-v', '--verbose')

@export.callback()
def doc_callback():
  """Export data in various formats"""

@export.command('pgn')
def export_pgn(path: str, verbose: bool = Verbose):
  """Export player SANs, one by line, space-delimited. Tthe same PGN will be repeated for each player."""
  P.run(cli.export_pgn)(path, verbose)

@export.command('labels')
def export_labels(path: str, verbose: bool = Verbose):
  """Export player labels, one by line, space-delimited"""
  P.run(cli.export_labels)(path, verbose)

@export.command('boxes')
def export_boxes(path: str, *, output: str = typer.Option(..., '-o', '--output'), verbose: bool = Verbose):
  """Export boxes in `files-dataset` format. (Only as many boxes as moves in the corresponding PGNs)"""
  P.run(cli.export_boxes)(path, output, verbose)

@export.command('ocr')
def export_ocr(path: str, *, output: str = typer.Option(..., '-o', '--output'), verbose: bool = Verbose):
  """Export OCR samples in `ocr-dataset` format."""
  P.run(cli.export_ocr)(path, output, verbose)
