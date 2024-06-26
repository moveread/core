from typing import Never, Iterable
from haskellian import either as E, Left
from kv.api import KV
import pure_cv as vc
import robust_extraction2 as re
import scoresheet_models as sm
import chess_notation as cn
from ._types import Image, Player

@E.do()
async def boxes(image: Image, blobs: KV[bytes], model: sm.Model | None = None, *, pads: sm.Pads = {}) -> list[vc.Img]:
  if image.meta.grid_coords and model:
    img = vc.decode((await blobs.read(image.url)).unsafe())
    return sm.extract_boxes(img, model, **image.meta.grid_coords, pads=pads)
  elif image.meta.box_contours:
    img = vc.decode((await blobs.read(image.url)).unsafe())
    return re.boxes(img, image.meta.box_contours, **pads) # type: ignore
  else:
    Left('No grid coords or box contours').unsafe()
    return Never
  

def labels(pgn: Iterable[str], meta: Player.Meta):
  notation = cn.Notation(meta.language_no_na or 'EN', meta.styles.without_na())
  labs = list(cn.styled(pgn, notation))
  for i, lab in sorted(meta.manual_labels.items()):
    if i < len(labs):
      labs[i] = lab
    elif i == len(labs):
      labs.append(lab)
  return labs