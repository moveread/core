from typing import Literal, cast
import os
import sys
from haskellian import promise as P
import typer
import kv
from moveread import core

app = typer.Typer(no_args_is_help=True)
export = typer.Typer(no_args_is_help=True)
app.add_typer(export, name="export")


STATE = {
  'recursive': False,
  'cores': []
}

Verbose = typer.Option(False, '-v', '--verbose')
Recursive = typer.Option(False, callback=lambda _: STATE['recursive'], hidden=True)
Cores = typer.Option(None, callback=lambda _: cast(list[core.Core], STATE['cores']), hidden=True)
                         
@app.callback()
def callback(
  debug: bool = typer.Option(False, '--debug'),
  env: bool = typer.Option(False, '-e', '--env', help='Load variables from .env file'),
):
  if debug:
    import debugpy
    debugpy.listen(5678)
    print("Waiting for debugger to attach...")
    debugpy.wait_for_client()
  if env:
    from dotenv import load_dotenv
    load_dotenv()

@app.command()
def dump(
  meta: str = typer.Option('', '-m', '--meta', help='KV connection string to meta. Can also be set via a CORE_META env var'),
  blobs: str = typer.Option('', '-b', '--blobs', help='KV connection string to blobs. Can also be set via a CORE_BLOBS env var'),
  output: str = typer.Option(..., '-o', '--output', help='Output directory'),
  prefix: str = typer.Option('', '-p', '--prefix', help='Prefix for metadata keys'),
  overwrite: bool = typer.Option(False, '-f', '--force', help='Overwrite existing files'),
  verbose: bool = Verbose,
):
  """Dump an online dataset to disk"""
  meta = meta or os.getenv('CORE_META') or ''
  blobs = blobs or os.getenv('CORE_BLOBS') or ''
  if not meta or not blobs:
    raise typer.BadParameter('Both --meta and --blobs must be provided')
  games = kv.KV.of(meta, core.Game)
  if prefix:
    games = games.prefix(prefix)
  inp_core = core.Core(games, kv.KV.of(blobs))
  out_core = core.Core.at(output)
  prefix = prefix.rstrip('/') + '/'
  P.run(inp_core.dump)(out_core, prefix, overwrite=overwrite, logstream=sys.stderr if verbose else None) # type: ignore

@export.callback()
def export_callback(
  glob: str = typer.Option('', '-g', '--glob', help='Glob to match local cores'),
  meta: str = typer.Option('', '-m', '--meta', help='KV connection string to meta. Can also be set via a CORE_META env var'),
  blobs: str = typer.Option('', '-b', '--blobs', help='KV connection string to blobs. Can also be set via a CORE_BLOBS env var'),
  prefix: str = typer.Option('', '-p', '--prefix', help='Prefix for metadata keys'),
  recursive: bool = Recursive,
):
  """Export data in various formats"""
  if glob:
    STATE['cores'] = core.glob(glob, recursive=recursive)
    return

  meta = meta or os.getenv('CORE_META') or ''
  blobs = blobs or os.getenv('CORE_BLOBS') or ''
  if meta and blobs:
    games = kv.KV.of(meta, core.Game)
    if prefix:
      games = games.prefix(prefix)
    STATE['cores'] = [core.Core(games, kv.KV.of(blobs))]
  else:
    raise typer.BadParameter('Either --glob or --meta and --blobs must be provided')

@export.command('pgn')
def export_pgn(cores = Cores, verbose: bool = Verbose):
  """Export player SANs, one by line, space-delimited. Tthe same PGN will be repeated for each player."""
  for ds in cores:
    if verbose:
      print(f'Exporting PGNs from {ds}', file=sys.stderr)
    P.run(core.cli.export_pgn)(ds, verbose)

@export.command('labels')
def export_labels(cores = Cores, verbose: bool = Verbose):
  """Export player labels, one by line, space-delimited"""
  for ds in cores:
    if verbose:
      print(f'Exporting PGNs from {ds}', file=sys.stderr)
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
  cores = Cores, *, verbose: bool = Verbose,
  output: str = typer.Option(..., '-o', '--output'),
  num_boxes: str = typer.Option('auto', '-n', '--num-boxes', help='If `"auto"`, export boxes up to the number of PGN moves; if `"none"`, export all boxes; if an integer, export at most `num_boxes` boxes	'),
):
  """Export boxes in `files-dataset` format. (Only as many boxes as moves in the corresponding PGNs)"""
  for ds in cores:
    if verbose:
      print(f'Exporting boxes from {ds}', file=sys.stderr)
    P.run(core.cli.export_boxes)(ds, output, verbose=verbose, num_boxes=parse_num_boxes(num_boxes))

@export.command('ocr')
def export_ocr(
  cores = Cores, *, verbose: bool = Verbose,
  output: str = typer.Option(..., '-o', '--output')
):
  """Export OCR samples in `ocr-dataset` format."""
  for ds in cores:
    if verbose:
      print(f'Exporting OCR samples from {ds}', file=sys.stderr)
    P.run(core.cli.export_ocr)(ds, output, verbose)

if __name__ == '__main__':
  app()