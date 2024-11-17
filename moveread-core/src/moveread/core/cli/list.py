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
async def games(
  core: 'core.Core' = core_dep.Depends(),
  ignore_versions: bool = typer.Option(False, '-i', '--ignore-versions', help='List keys ignoring the version schema (listning duplicates)')
):
  """List games in a core"""
  if ignore_versions:
    async for key in core.games.keys():
      print(key)
  else:
    for key in await core.keys():
      print(key)