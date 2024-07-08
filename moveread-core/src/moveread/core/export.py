from typing import Never, Iterable, Any
from haskellian import either as E, Left, Right, Either
from kv import KV
import pure_cv as vc
import robust_extraction2 as re
import scoresheet_models as sm
import chess_notation as cn
import chess_utils as cu
from ._types import Image, Player

def _safe_styled_simple(pgn: Iterable[str], meta: Player.Meta) -> Iterable[str|None]:
  styles = meta.styles.without_na()
  for san in pgn:
    if (
      styles.pawn_capture is None and cn.is_pawn_capture(san)
      or styles.piece_capture is None and cn.is_piece_capture(san)
    ):
      yield None
    else:
      yield cn.style(san, styles)

def _safe_styled_validated(pgn: Iterable[str], meta: Player.Meta) -> Iterable[str|None]:
  import chess
  board = chess.Board()
  styles = meta.styles.without_na()
  for san in pgn:
    move = board.parse_san(san)
    piece = cu.captured_piece(board, move)
    board.push(move)
    if (
      styles.pawn_capture is None and cn.is_pawn_capture(san)
      or styles.piece_capture is None and cn.is_piece_capture(san)
    ):
      yield None
    else:
      yield cn.style(san, styles, captured_piece=piece)

def safe_styled(pgn: Iterable[str], meta: Player.Meta) -> Either[Any, list[str|None]]:
  try:
    return Right(list(_safe_styled_validated(pgn, meta)))
  except Exception as e:
    return Left(e)
  # if meta.styles.pawn_capture == 'PxN' or meta.styles.piece_capture == 'NxN':
  #   return _safe_styled_validated(pgn, meta)
  # else:
  #   return _safe_styled_simple(pgn, meta)

def labels(pgn: Iterable[str], meta: Player.Meta) -> Either[str, list[str|None]]:
  """Export labels from the PGN and annotations.
  - If `meta.language` is 'N/A' or None, returns `Left`
  - If some annotation is 'N/A', returns `None` on the affected moves
  - The `PGN` is cropped to `meta.end_correct`
  - If some `meta.manual_labels` exist beyond the end of the PGN, they're ignored
  """
  if meta.language == 'N/A':
    return Left('Language is N/A')
  elif meta.language is None:
    return Left('Language is None')
  
  cropped_pgn = list(pgn)[:meta.end_correct]
  styled = safe_styled(cropped_pgn, meta).unsafe()
  labs = [move and cn.translate(move, meta.language) for move in styled]
  
  for i, lab in sorted(meta.manual_labels.items()):
    if i < len(labs):
      labs[i] = lab
    elif i == len(labs):
      labs.append(lab)
    else:
      break

  return Right(labs)



@E.do()
async def boxes(image: Image, blobs: KV[bytes], model: sm.Model | None = None, *, pads: sm.Pads = {}) -> list[vc.Img]:
  if isinstance(image.meta, Image.OldMeta):
    if image.meta.grid_coords and model:
      img = vc.decode((await blobs.read(image.url)).unsafe())
      return sm.extract_boxes(img, model, **image.meta.grid_coords, pads=pads)
    elif image.meta.box_contours:
      img = vc.decode((await blobs.read(image.url)).unsafe())
      return re.boxes(img, image.meta.box_contours, **pads) # type: ignore
    else:
      return Left('No grid coords or box contours').unsafe()
  else:
    if image.meta.boxes is None:
      return Left('No boxes').unsafe()
    
    if image.meta.boxes.tag == 'box-contours':
      img = vc.decode((await blobs.read(image.url)).unsafe())
      return re.boxes(img, image.meta.boxes.contours, **pads) # type: ignore
    elif model:
      img = vc.decode((await blobs.read(image.url)).unsafe())
      return sm.extract_boxes(img, model, **image.meta.boxes.coords, pads=pads)
    else:
      return Left('No model').unsafe()