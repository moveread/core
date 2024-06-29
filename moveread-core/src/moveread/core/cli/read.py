from haskellian import either as E
from moveread.core import Core

async def read_games(path: str, verbose: bool):
  ds = Core.at(path)
  games = E.sequence(await ds.games.items().sync())
  if games.tag == 'left':
    print(f'Found {len(games.value)} errors')
    if verbose:
      for e in games.value:
        print(e)
    else:
      print('Run with -v to show errors')
  
  return games