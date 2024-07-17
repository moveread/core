import typer
import os
import sys
from haskellian import promise as P
import pure_cv as vc
import ocr_dataset as ods
from moveread.core import Core, cli

export_app = typer.Typer(no_args_is_help=True)

@export_app.command()
@cli.core_dep.inject
@P.run
async def ocr(
  core: Core = cli.core_dep.Depends(), verbose: bool = cli.Verbose,
  output: str = typer.Option(..., '-o', '--output', help='Output base')
):
  """Exports data in `ocr-dataset` format"""
  games = await cli.read_games(core, True)
  if games.tag == 'left':
    return
  
  total_samples = i = 0
  for i, (id, game) in enumerate(sorted(games.value)):
    base = os.path.join(output, id.replace('/', '-'))
    if (pgn := game.meta.pgn) is None:
      continue
    for j, player in enumerate(game.players):
      either = await player.ocr_samples(pgn, core.blobs)
      if either.tag == 'left':
        if verbose:
          print(f'Error in "{id}", player {j}', either.value, file=sys.stderr)
        continue
      elif either.value == []:
        if verbose:
          print(f'WARNING: No samples found in "{id}", player {j}', file=sys.stderr)
        continue

      samples = [(vc.encode(s.img, '.jpg'), s.lab) for s in either.value]
      ods.create_tar(f'{base}-{j}', samples, images_name='boxes')
      total_samples += len(samples)

    print(f'\r{i+1}/{len(games.value)} - {total_samples:05} boxes', end='', file=sys.stderr)


@export_app.command()
@cli.core_dep.inject
@P.run
async def transformer(
  core: Core = cli.core_dep.Depends(), verbose: bool = cli.Verbose,
  output: str = typer.Option(..., '-o', '--output', help='Output base')
):
  """Exports PGNs and all boxes in `ocr-dataset` format"""
  games = await cli.read_games(core, True)
  if games.tag == 'left':
    return
  
  total_samples = i = 0
  for i, (id, game) in enumerate(sorted(games.value)):
    base = os.path.join(output, id.replace('/', '-'))
    if (pgn := game.meta.pgn) is None:
      continue
    for j, player in enumerate(game.players):
      either = await player.boxes(core.blobs)
      if either.tag == 'left':
        if verbose:
          print(f'Error in "{id}", player {j}', either.value, file=sys.stderr)
        continue
      elif either.value == []:
        if verbose:
          print(f'WARNING: No samples found in "{id}", player {j}', file=sys.stderr)
        continue

      samples = [(vc.encode(img, '.jpg'), san) for img, san in zip(either.value, pgn)]
      ods.create_tar(f'{base}-{j}', samples, images_name='boxes', labels_name='pgn')
      total_samples += len(samples)

    print(f'\r{i+1}/{len(games.value)} - {total_samples:05} boxes', end='', file=sys.stderr)