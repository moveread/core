from typing import Literal
from pydantic import BaseModel
from chess_notation import Styles, Check, Mate, Castle, PawnCapture, PieceCapture

NA = Literal['N/A']
"""Not Applicable"""

def no_na(x):
  return x if x != 'N/A' else None

class StylesNA(BaseModel):
  """Like `chess_notation.Styles`, but with possibly `'N/A'` annotations"""
  check: Check | NA | None = None
  mate: Mate | NA | None = None
  castle: Castle | NA | None = None
  pawn_capture: PawnCapture | NA | None = None
  piece_capture: PieceCapture | NA | None = None

  def without_na(self) -> Styles:
    return Styles(
      castle=no_na(self.castle), check=no_na(self.check), mate=no_na(self.mate),
      pawn_capture=no_na(self.pawn_capture), piece_capture=no_na(self.piece_capture)
    )

