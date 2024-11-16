import sys
import typer
from haskellian import promise as P
from moveread import core
from moveread.core.cli import core_dep

list_app = typer.Typer(no_args_is_help=True)

@list_app.command()
@core_dep.inject
@P.run
async def players(game: str, core: 'core.Core' = core_dep.Depends()):
  """List players in a game"""
  g = await core.games.read(game)
  for i, _ in enumerate(g.players):
    print(i)

@list_app.command()
@core_dep.inject
@P.run
async def games(core: 'core.Core' = core_dep.Depends()):
  """List games in a core"""
  async for game in core.games.keys():
    print(game)