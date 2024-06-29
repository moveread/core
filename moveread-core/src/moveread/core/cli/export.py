import sys
import os
import fs
import pure_cv as vc
import files_dataset as fds
import ocr_dataset as ods
import scoresheet_models as sm
from moveread.core import Core, cli

async def export_pgn(path: str, verbose: bool):
  games = await cli.read_games(path, verbose)
  if games.tag == 'left':
    return
  for id, game in sorted(games.value):
    if game.meta.pgn:
      line = ' '.join(game.meta.pgn)
      for _ in game.players:
        print(line)
    elif verbose:
      import sys
      print(f'Game "{id}" has no PGN', file=sys.stderr)


async def export_labels(path: str, verbose: bool):
  games = await cli.read_games(path, verbose)
  if games.tag == 'left':
    return
  for id, game in sorted(games.value):
    if game.meta.pgn:
      for player in game.players:
        labs = player.labels(game.meta.pgn)
        if labs.tag == 'right':
          print(' '.join(l for l in labs.value if l))
        elif verbose:
          print(f'Error exporting "{id}":', labs.value, file=sys.stderr)
        else:
          print(f'Error exporting "{id}". Run with -v to show the full error', file=sys.stderr)
    elif verbose:
      print(f'Game "{id}" has no PGN', file=sys.stderr)


async def export_boxes(path: str, output: str, verbose: bool):
  games = await cli.read_games(path, True)
  if games.tag == 'left':
    return
  
  models = sm.ModelsCache()
  ds = Core.at(path)
  total_boxes = i = 0
  for i, (id, game) in enumerate(sorted(games.value)):
    base = os.path.join(output, id.replace('/', '-'))
    pgn = game.meta.pgn
    max_boxes = len(pgn) if pgn is not None else None
    for j, player in enumerate(game.players):
      either = await player.boxes(ds.blobs, models)
      if either.tag == 'left':
        if verbose:
          print(f'Error in "{id}", player {j}', either.value, file=sys.stderr)
        else:
          print(f'Error in "{id}", player {j}. Run with -v to show full errors', file=sys.stderr)
        continue

      boxes = either.value[:max_boxes]
      total_boxes += len(boxes)
      path = f'{base}-{j}'
      os.makedirs(path, exist_ok=True)
      tarfile = f'boxes{j}.tar'
      files = [(f'box_{i}.jpg', vc.encode(box, '.jpg')) for i, box in enumerate(boxes)]
      fs.create_tarfile(files, os.path.join(path, tarfile))
      archive = fds.Archive(archive=tarfile, format='tar', num_files=len(boxes))
      meta = fds.MetaJson(files_dataset={'boxes': archive})
      with open(os.path.join(path, 'meta.json'), 'w') as f:
        f.write(meta.model_dump_json(indent=2, exclude_defaults=True))

    print(f'\r{i+1}/{len(games.value)} - {total_boxes:06} boxes', end='', file=sys.stderr)

async def export_ocr(path: str, output: str, verbose: bool):
  games = await cli.read_games(path, True)
  if games.tag == 'left':
    return
  
  models = sm.ModelsCache()
  ds = Core.at(path)
  total_samples = i = 0
  for i, (id, game) in enumerate(sorted(games.value)):
    base = os.path.join(output, id.replace('/', '-'))
    if (pgn := game.meta.pgn) is None:
      continue
    for j, player in enumerate(game.players):
      either = await player.ocr_samples(pgn, ds.blobs, models)
      if either.tag == 'left':
        if verbose:
          print(f'Error in "{id}", player {j}', either.value, file=sys.stderr)
        else:
          print(f'Error in "{id}", player {j}. Run with -v to show full errors', file=sys.stderr)
        continue

      samples = [(vc.encode(s.img, '.jpg'), s.lab) for s in either.value]
      ods.create_tar(f'{base}-{j}', samples, images_name='boxes')
      total_samples += len(samples)

    print(f'\r{i+1}/{len(games.value)} - {total_samples:05} boxes', end='', file=sys.stderr)