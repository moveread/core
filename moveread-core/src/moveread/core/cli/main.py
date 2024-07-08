from typing import Literal
import sys
from haskellian import promise as P
import typer
from moveread import core

app = typer.Typer(no_args_is_help=True)
export = typer.Typer(no_args_is_help=True)
app.add_typer(export, name="export")

Verbose = typer.Option(False, '-v', '--verbose')
Recursive = typer.Option(False, '-r', '--recursive')

@app.callback()
def callback(debug: bool = typer.Option(False, '--debug')):
  if debug:
    import debugpy
    debugpy.listen(5678)
    print("Waiting for debugger to attach...")
    debugpy.wait_for_client()

@export.callback()
def export_callback():
  """Export data in various formats"""

@export.command('pgn')
def export_pgn(glob: str, verbose: bool = Verbose, recursive: bool = Recursive):
  """Export player SANs, one by line, space-delimited. Tthe same PGN will be repeated for each player."""
  cores = core.glob(glob, recursive=recursive, err_stream=sys.stderr if verbose else None)
  for ds in cores:
    if verbose:
      print(f'Exporting PGNs from {ds.base_path}', file=sys.stderr)
    P.run(core.cli.export_pgn)(ds, verbose)

@export.command('labels')
def export_labels(glob: str, verbose: bool = Verbose, recursive: bool = Recursive):
  """Export player labels, one by line, space-delimited"""
  cores = core.glob(glob, recursive=recursive, err_stream=sys.stderr if verbose else None)
  for ds in cores:
    if verbose:
      print(f'Exporting PGNs from {ds.base_path}', file=sys.stderr)
    P.run(core.cli.export_labels)(ds, verbose)


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
  glob: str, *, verbose: bool = Verbose, recursive: bool = Recursive,
  output: str = typer.Option(..., '-o', '--output'),
  num_boxes: str = typer.Option('auto', '-n', '--num-boxes', help='If `"auto"`, export boxes up to the number of PGN moves; if `"none"`, export all boxes; if an integer, export at most `num_boxes` boxes	'),
):
  """Export boxes in `files-dataset` format. (Only as many boxes as moves in the corresponding PGNs)"""
  cores = core.glob(glob, recursive=recursive, err_stream=sys.stderr if verbose else None)
  for ds in cores:
    if verbose:
      print(f'Exporting boxes from {ds.base_path}', file=sys.stderr)
    P.run(core.cli.export_boxes)(ds, output, verbose=verbose, num_boxes=parse_num_boxes(num_boxes))

@export.command('ocr')
def export_ocr(
  glob: str, *, verbose: bool = Verbose, recursive: bool = Recursive,
  output: str = typer.Option(..., '-o', '--output')
):
  """Export OCR samples in `ocr-dataset` format."""
  cores = core.glob(glob, recursive=recursive, err_stream=sys.stderr if verbose else None)
  for ds in cores:
    if verbose:
      print(f'Exporting OCR samples from {ds.base_path}', file=sys.stderr)
    P.run(core.cli.export_ocr)(ds, output, verbose)
