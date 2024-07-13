# Moveread Core Dataset

### CLI

#### Exporting

**Important**
- Exporting `boxes` will yield one box per PGN move.
- To export the labeled boxes, use `core export ocr` instead 

```bash
core export pgn -v --glob 'path/to/core' > sans.txt
core export labels -v --glob 'path/to/cores/*' > labels.txt
core export boxes -vg 'path/**/*' --recursive -o path/to/files-dataset
core export ocr -vgr '**' -o path/to/ocr-dataset
```

#### Nonlocal Core
  
```bash
core --meta 'azure+cosmos://<CONN_STR>?db=games&container=games' --blobs 'azure+blob://...' export # [...]
core --env export # [...] Loads CORE_META and CORE_BLOBS from .env file
core -e export --prefix tournament/ export labels # exports from tournament only
core -e export -p tournament/group/1 # exports from tournament/group/1 only
```

Or you can dump to local first:

```bash
core -e dump --prefix tournament -o local-tournament
```