import typer
from typer_tools import Dependency, option
from moveread.core import Core

def env_callback(value: bool):
  if value:
    from dotenv import load_dotenv
    load_dotenv()

Env = option(False, '-e', '--env', is_eager=True, help='Load variables from .env file', callback=env_callback)

def debug_callback(value: bool):
  if value:
    import debugpy
    debugpy.listen(5678)
    print("Waiting for debugger to attach...")
    debugpy.wait_for_client()

Debug = option(False, '--debug', help='Enable debugging', callback=debug_callback)
Verbose = option(False, '-v', '--verbose', help='Enable verbose output')
Prefix = option('', '--prefix', help='Prefix for metadata keys')
Force = option(False, '-f', '--force', help='Overwrite existing files')
Concurrent = option(1, '-c', '--concurrent', help='Number of concurrent tasks')

def parse_core(
  path: str = typer.Option('', '-p', '--path', help='Path to local core'),
  meta: str = typer.Option('', '-m', '--meta', envvar='CORE_META', help='KV connection string to meta. Can also be set via a CORE_META env var'),
  blobs: str = typer.Option('', '-b', '--blobs', envvar='CORE_BLOBS', help='KV connection string to blobs. Can also be set via a CORE_BLOBS env var'),
):
  if path:
    return Core.read(path)
  elif meta and blobs:
    return Core.of(meta, blobs)
  else:
    raise typer.BadParameter('Either --path or --meta and --blobs must be provided')

core_dep = Dependency(parse_core)

