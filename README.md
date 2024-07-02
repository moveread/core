# Moveread Core Dataset

### CLI

#### Exporting

**Important**
- Exporting `boxes` will yield one box per PGN move.
- To export the labeled boxes, use `core export ocr` instead 

```bash
core export pgn -v path/to/core > sans.txt
core export labels -v path/to/core > labels.txt
core export boxes -v path/to/core -o path/to/files-dataset
core export ocr -v path/to/core -o path/to/ocr-dataset
```
