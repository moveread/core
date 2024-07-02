from typing import Literal
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


def parse_num_boxes(num_boxes: str) -> int | Literal['auto'] | None:
  if num_boxes == 'none':
    return None
  elif num_boxes == 'auto':
    return 'auto'
  try:
    return int(num_boxes)
  except:
    raise typer.BadParameter(f'Invalid value for `--num-boxes`: "{num_boxes}". Expected "auto", "none" or an integer')

@export.command('boxes')
def export_boxes(
  path: str, *, verbose: bool = Verbose,
  output: str = typer.Option(..., '-o', '--output'),
  num_boxes: str = typer.Option('auto', '-n', '--num-boxes', help='If `"auto"`, export boxes up to the number of PGN moves; if `"none"`, export all boxes; if an integer, export at most `num_boxes` boxes	'),
):
  """Export boxes in `files-dataset` format. (Only as many boxes as moves in the corresponding PGNs)"""
  P.run(cli.export_boxes)(path, output, verbose=verbose, num_boxes=parse_num_boxes(num_boxes))

@export.command('ocr')
def export_ocr(path: str, *, output: str = typer.Option(..., '-o', '--output'), verbose: bool = Verbose):
  """Export OCR samples in `ocr-dataset` format."""
  P.run(cli.export_ocr)(path, output, verbose)