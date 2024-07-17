import sys
from haskellian import promise as P
import typer
from moveread import core
from moveread.core.cli import Env, Debug, core_dep, Prefix, Verbose, Force, Concurrent
from .list import list_app
from .export import export_app

app = typer.Typer(no_args_is_help=True)
app.add_typer(export_app, name='export')
app.add_typer(list_app, name='list')


@app.callback()
def callback(env: bool = Env, debug: bool = Debug):
  ...

@app.command()
@core_dep.inject
def dump(
  inp_core: 'core.Core' = core_dep.Depends(),
  prefix: str = Prefix, verbose: bool = Verbose,
  output: str = typer.Option(..., '-o', '--output', help='Path to output core'),
  force: bool = Force, concurrent: int = Concurrent
):
  """Dump an online dataset to disk"""
  if prefix:
    inp_core.games = inp_core.games.prefix(prefix)
    prefix = prefix.rstrip('/') + '/'
  out_core = core.Core.at(output)
  run = P.run(inp_core.dump) # type: ignore
  run(out_core, prefix, overwrite=force, logstream=sys.stderr if verbose else None, concurrent=concurrent)

@app.command()
@core_dep.inject
def upload(
  input: str = typer.Option(..., '-i', '--input', help='Path to input local core'),
  out_core: 'core.Core' = core_dep.Depends(), concurrent: int = Concurrent,
  prefix: str = Prefix, verbose: bool = Verbose, force: bool = Force,
):
  inp_core = core.Core.read(input)
  if prefix:
    inp_core.games = inp_core.games.prefix(prefix)
    prefix = prefix.rstrip('/') + '/'
  run = P.run(out_core.dump) # type: ignore
  run(out_core, prefix, overwrite=force, logstream=sys.stderr if verbose else None, concurrent=concurrent)

