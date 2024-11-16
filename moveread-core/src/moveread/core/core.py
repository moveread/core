from typing import Sequence, Literal, TextIO
from dataclasses import dataclass
import asyncio
from kv import KV
from ._types import Game

@dataclass
class ExistentBlobs:
  blobs: Sequence[str]
  reason: Literal['existent-blobs'] = 'existent-blobs'

@dataclass
class ExistentGame:
  reason: Literal['existent-game'] = 'existent-game'

@dataclass
class Core:
  games: KV[Game]
  blobs: KV[bytes]
  base_path: str | None = None

  @staticmethod
  def of(games_conn_str: str, blobs_conn_str: str) -> 'Core':
    return Core(KV.of(games_conn_str, Game), KV.of(blobs_conn_str, bytes))
  
  @staticmethod
  def at(path: str) -> 'Core':
    """The default, filesystem- and sqlite-based `Core`"""
    from kv import KV
    import os
    sqlite_path = os.path.join(path, 'games.sqlite')
    blobs_path = os.path.join(path, 'blobs')
    return Core(
      games=KV.of(f'sql+sqlite:///{sqlite_path}?table=games', Game),
      blobs=KV.of(f'file://{blobs_path}', bytes),
      base_path=path
    )
  
  @staticmethod
  def read(path: str) -> 'Core':
    """Try to read an existing `Core` from `path`. Raises if it didn't exist"""
    import os
    if not os.path.isdir(path):
      raise ValueError(f'Path is not a directory: {path}')
    elif not os.path.exists(os.path.join(path, 'games.sqlite')):
      raise ValueError(f'No games.sqlite found in {path}')
    elif not os.path.exists(os.path.join(path, 'blobs')):
      raise ValueError(f'No blobs directory found in {path}')
    return Core.at(path)


  async def copy(self, fromId: str, other: 'Core', toId: str, *, overwrite: bool = False) -> bool:
    """Copies `fromId` of `self` to `toId` in `other`. Returns `False` if the game already existed in `other` (and `overwrite=False`)."""
    if not overwrite and (await other.games.has(toId)):
      return False

    game = await self.games.read(fromId)

    img_tasks = [self.blobs.copy(img.url, other.blobs, img.url) for _, img in game.images]
    await asyncio.gather(*img_tasks)
    await other.games.insert(toId, game)
    return True


  async def dump(
    self, other: 'Core', prefix: str = '', *, concurrent: int = 1,
    overwrite: bool = False, logstream: TextIO | None = None
  ):
    """Copy all games from `self` to `other`."""
    if logstream:
      print('Reading keys...')
    keys = [key async for key in self.games.keys()]
    semaphore = asyncio.Semaphore(concurrent)
    skipped = 0
    done = 0
    tasks = []
    for key in keys:
      async def task(key: str):
        nonlocal skipped, done
        async with semaphore:
          if logstream:
            print(f'\rDownloading... [{done+1}/{len(keys)}] - skipped {skipped}', end='', flush=True, file=logstream)
          copied = await self.copy(key, other, prefix + key, overwrite=overwrite)
          if not copied:
            skipped += 1
          done += 1
      tasks.append(task(key))
    
    await asyncio.gather(*tasks)



def glob(glob: str, *, recursive: bool = False, err_stream: TextIO | None = None) -> list[Core]:
  """Read all cores that match a glob pattern."""
  from glob import glob as _glob
  cores = []
  for p in sorted(_glob(glob, recursive=recursive)):
    try:
      cores.append(Core.read(p))
    except Exception as e:
      if err_stream:
        print(f'Error reading dataset at {p}:', e, file=err_stream)
  return cores