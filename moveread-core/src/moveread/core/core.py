from typing import Sequence, Literal
from dataclasses import dataclass
from haskellian import Left, either as E, promise as P
from kv.api import KV, ReadError
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

  @staticmethod
  def of(games_conn_str: str, blobs_conn_str: str) -> 'Core':
    return Core(KV.of(games_conn_str), KV.of(blobs_conn_str))
  
  @staticmethod
  def at(path: str) -> 'Core':
    """The default, filesystem- and sqlite-based `Core`"""
    from kv.sqlite import SQLiteKV
    from kv.fs import FilesystemKV
    import os
    return Core(
      games=SQLiteKV.validated(Game, os.path.join(path, 'games.sqlite'), table='games'),
      blobs=FilesystemKV[bytes](os.path.join(path, 'blobs'))
    )


  @E.do[ReadError|ExistentBlobs|ExistentGame]()
  async def copy(self, fromId: str, other: 'Core', toId: str, *, overwrite: bool = False) -> None:
    """Copies `fromId` of `self` to `toId` in `other`."""
    game = (await self.games.read(fromId)).unsafe()

    if not overwrite:
      tasks = [other.blobs.has(img.url) for _, img in game.images]
      results = E.sequence(await P.all(tasks)).unsafe()
      if any(results):
        Left(ExistentBlobs([img.url for (_, img), exists in zip(game.images, results) if exists])).unsafe()

    if (await other.games.has(toId)).unsafe():
      Left(ExistentGame()).unsafe()

    img_tasks = [self.blobs.copy(img.url, other.blobs, img.url) for _, img in game.images]
    E.sequence(await P.all(img_tasks)).unsafe()
    (await other.games.insert(toId, game)).unsafe()
