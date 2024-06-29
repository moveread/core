import os
import typer
from moveread import core
from haskellian import promise as P, either as E
import tf.records as tfr
import moveread.ocr as mo
import scoresheet_models as sm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def sys():
  import sys
  return sys

app = typer.Typer()
export = typer.Typer()
app.add_typer(export, name="export")



@export.command('pgn')
@P.run
async def export_pgn(path: str, verbose: bool = typer.Option(False, '-v')):
  """Export player SANs, one by line, space-delimited. Tthe same PGN will be repeated for each player."""
  games = await read_games(path, verbose)
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


@export.command('labels')
@P.run
async def export_labels(path: str, verbose: bool = typer.Option(False, '-v')):
  """Export player labels, one by line, space-delimited"""
  games = await read_games(path, verbose)
  if games.tag == 'left':
    return
  for id, game in sorted(games.value):
    if game.meta.pgn:
      for player in game.players:
        labs = player.labels(game.meta.pgn)
        if labs.tag == 'right':
          print(' '.join(l for l in labs.value if l))
        elif verbose:
          print(f'Error exporting "{id}":', labs.value, file=sys().stderr)
        else:
          print(f'Error exporting "{id}". Run with -v to show the full error', file=sys().stderr)
    elif verbose:
      print(f'Game "{id}" has no PGN', file=sys().stderr)


@export.command('boxes')
@P.run
async def export_boxes(
  path: str,
  tfrecord: str = typer.Option(help='Path to output tfrecord file'),
  meta: str = typer.Option(help='Path to meta.json'),
  verbose: bool = typer.Option(False, '-v')
):
  """Export players boxes to TFRecords. Uses GZIP compression if the path ends with '.gz'"""
  games = await read_games(path, verbose)
  if games.tag == 'left':
    return
  ds = core.Core.at(path)
  models = sm.ModelsCache()
  samples = 0
  async def records():
    nonlocal samples
    for i, (id, game) in enumerate(sorted(games.value)):
      if verbose:
        print(f'\r[{i+1}/{len(games.value)}]: "{id}"' + ' ' * 10, end='', flush=True, file=sys().stderr)
      
      pgn = game.meta.pgn
      max_boxes = len(pgn) if pgn is not None else None
      for j, player in enumerate(game.players):
        boxes = await player.boxes(ds.blobs, models)
        if boxes.tag == 'left':
          if verbose:
            print(f'Error in "{id}", player {j}', boxes.value, file=sys().stderr)
          else:
            print(f'Error in "{id}", player {j}. Run with -v to show full errors', file=sys().stderr)
        else:
          for box in boxes.value[:max_boxes]:
            samples += 1
            tensor = mo.parse_img(box)
            schema = tfr.schema(image=tfr.Tensor((256, 64, 1), 'float'))
            yield tfr.serialize(schema, image=tensor)
  
  import tensorflow as tf
  compression = 'GZIP' if tfrecord.endswith('.gz') else None
  writer =  tf.io.TFRecordWriter(tfrecord, options=compression)
  async for record in records():
    writer.write(record) # type: ignore
  writer.close()
  if verbose:
    print()

  rel_filename = os.path.relpath(tfrecord, os.path.dirname(meta))
  obj = tfr.Meta(files=[rel_filename], compression=compression, num_samples=samples, schema=mo.records.BOX_SCHEMA) # type: ignore
  meta_json = tfr.MetaJson(tfrecords_dataset=obj)
  with open(meta, 'w') as f:
    f.write(meta_json.model_dump_json(indent=2, exclude_none=True, by_alias=True))

@export.command('ocr')
@P.run
async def export_ocr(
  path: str, tfrecord: str = typer.Option(help='Path to output tfrecord file'),
  meta: str = typer.Option(help='Path to meta.json'),
  verbose: bool = typer.Option(False, '-v')
):
  """Export players samples to TFRecords. Uses GZIP compression if the path ends with '.gz'"""
  ds = core.Core.at(path)
  games = E.sequence(await ds.games.items().sync())
  if games.tag == 'left':
    print(f'Found {len(games.value)} errors')
    if verbose:
      for e in games.value:
        print(e)
    else:
      print('Run with -v to show errors')
  else:
    models = sm.ModelsCache()
    num_samples = 0
    async def records():
      nonlocal num_samples
      for i, (id, game) in enumerate(sorted(games.value)):
        if verbose:
          print(f'\r[{i+1}/{len(games.value)}]: "{id}"' + ' ' * 10, end='', flush=True, file=sys().stderr)
        if not game.meta.pgn:
          if verbose:
            print(f'"{id}" has no PGN', file=sys().stderr)
          continue
        for j, player in enumerate(game.players):
          samples = await player.ocr_samples(game.meta.pgn, ds.blobs, models)
          if samples.tag == 'left':
            if verbose:
              print(f'Error in "{id}", player {j}', samples.value, file=sys().stderr)
            else:
              print(f'Error in "{id}", player {j}. Run with -v to show full errors', file=sys().stderr)
          else:
            for box, lab in samples.value:
              num_samples += 1
              tensor = mo.parse_img(box)
              yield tfr.serialize(mo.records.SCHEMA, image=tensor, label=tf.constant(lab))
    
    import tensorflow as tf
    compression = 'GZIP' if tfrecord.endswith('.gz') else None
    writer =  tf.io.TFRecordWriter(tfrecord, options=compression)
    async for record in records():
      writer.write(record) # type: ignore
    writer.close()
    if verbose:
      print()

    rel_filename = os.path.relpath(tfrecord, os.path.dirname(meta))
    obj = tfr.Meta(files=[rel_filename], compression=compression, num_samples=num_samples, schema=mo.records.SCHEMA) # type: ignore
    meta_json = tfr.MetaJson(tfrecords_dataset=obj)
    with open(meta, 'w') as f:
      f.write(meta_json.model_dump_json(indent=2, exclude_none=True, by_alias=True))