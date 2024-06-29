from typing_extensions import Literal, Sequence, Iterable, TypedDict, ClassVar, Any, NamedTuple
from pydantic import BaseModel, Field
from haskellian import iter as I, either as E, Left, Right, Either, promise as P
from kv.api import KV
import pure_cv as vc
import chess_pairings as cp
from chess_notation import Language
import scoresheet_models as sm
from .labels import StylesNA, NA

Vec2 = tuple[float, float]
class Rectangle(TypedDict):
  tl: Vec2
  size: Vec2

class Corners(NamedTuple):
  tl: Vec2
  tr: Vec2
  br: Vec2
  bl: Vec2

class Image(BaseModel):
  Source: ClassVar = Literal['raw-scan', 'corrected-scan', 'camera', 'corrected-camera', 'robust-corrected'] 
  class Meta(BaseModel):
    source: 'Image.Source | None' = None
    perspective_corners: Corners | None = None
    grid_coords: Rectangle | None = None
    """Grid coords (matching some scoresheet model)"""
    box_contours: list | None = None
    """Explicit box contours (given by robust-extraction, probably)"""

  url: str
  meta: Meta = Field(default_factory=Meta)

  async def export(self, blobs: KV[bytes], model: sm.Model | None = None, *, pads: sm.Pads = {}):
    from .export import boxes
    return await boxes(self, blobs, model, pads=pads)


class Sheet(BaseModel):
  class Meta(BaseModel):
    model: str
  images: list[Image]
  meta: Meta

  async def boxes(self, blobs: KV[bytes], models: sm.ModelsCache, *, pads: sm.Pads = {}) -> Either[Any, list[vc.Img]]:
    """Export boxes of the first exportable image. Returns `Left` if none of the images are exportable"""
    model = (await models.fetch(self.meta.model)).unsafe()
    for image in self.images:
      boxes = await image.export(blobs, model, pads=pads)
      if boxes.tag == 'right':
        return boxes
    return Left('No exportable images')
  
class Sample(NamedTuple):
  img: vc.Img
  san: str
  lab: str | None

class OCRSample(NamedTuple):
  img: vc.Img
  lab: str

class Player(BaseModel):
  class Meta(BaseModel):
    language: Language | NA | None = None
    styles: StylesNA = StylesNA()
    end_correct: int | None = None
    manual_labels: dict[int, str] = {}
    @property
    def language_no_na(self) -> Language | None:
      if self.language != 'N/A':
        return self.language
      
  sheets: list[Sheet]
  meta: Meta = Field(default_factory=Meta)

  def labels(self, pgn: Iterable[str]):
    from .export import labels
    return labels(pgn, self.meta)
  
  async def boxes(self, blobs: KV[bytes], models: sm.ModelsCache, *, pads: sm.Pads = {}) -> Either[Any, list[vc.Img]]:
    """Export boxes of all exportable sheets.
    - Only the first exportable image of each sheet is taken.
    - Returns `Left` if the first sheet is not exportable. Otherwise returns as many consecutive exportable sheets as there are."""
    all_boxes = await P.all([sheet.boxes(blobs, models, pads=pads) for sheet in self.sheets])
    if not all_boxes or all_boxes[0].tag == 'left':
      return Left('No exportable sheets')
    return Right(I.flatten(E.take_while(all_boxes)).sync())
  
  @E.do()
  async def samples(self, pgn: Iterable[str], blobs: KV[bytes], models: sm.ModelsCache, *, pads: sm.Pads = {}) -> list[Sample]:
    """Export samples of all exportable sheets.
    - Only the first exportable image of each sheet is taken.
    - Returns `Left` if the first sheet is not exportable. Otherwise returns as many consecutive exportable sheets as there are.
    - Labels may be `None` if a) annotations are `N/A` or b) moves are after `end_correct`
    """
    boxes = (await self.boxes(blobs, models, pads=pads)).unsafe()
    labs = self.labels(pgn).unsafe()
    def label(idx: int, label: str | None):
      if self.meta.end_correct is None or idx < self.meta.end_correct:
        return label
    return [Sample(img, san, label(i, lab)) for i, (img, san, lab) in enumerate(zip(boxes, pgn, labs))]
  
  @E.do()
  async def ocr_samples(self, pgn: Iterable[str], blobs: KV[bytes], models: sm.ModelsCache, *, pads: sm.Pads = {}) -> list[OCRSample]:
    """Export OCR samples of all exportable sheets.
    - Skips `None` labels (due to missing annotations of index after `end_correct`)
    """
    samples = (await self.samples(pgn, blobs, models, pads=pads)).unsafe()
    return [OCRSample(s.img, s.lab) for s in samples if s.lab]


class Game(BaseModel):
  class Meta(BaseModel):
    tournament: cp.GameId | None = None
    pgn: Sequence[str] | None = None
    early: bool | None = None
    """Whether the `PGN` stops before the game actually finished"""

  players: list[Player]
  meta: Meta = Field(default_factory=Meta)

  @property
  @I.lift
  def sheets(self) -> Iterable[tuple[tuple[int, int], Sheet]]:
    for i, player in enumerate(self.players):
      for j, sheet in enumerate(player.sheets):
        yield (i, j), sheet

  @property
  @I.lift
  def images(self) -> Iterable[tuple[tuple[int, int, int], Image]]:
    for (i, j), sheet in self.sheets:
      for k, image in enumerate(sheet.images):
        yield (i, j, k), image

  async def samples(self, blobs: KV[bytes], models: sm.ModelsCache, *, pads: sm.Pads = {}) -> Either[Any, list[list[Sample]]]:
    """Export samples of all players.
    - Only the first exportable image of each sheet is taken.
    - Returns `Left` if the game's meta has no PGN or no player is exportable"""
    if not self.meta.pgn:
      return Left('No PGN')
    samples = await P.all([player.samples(self.meta.pgn, blobs, models, pads=pads) for player in self.players])
    return E.sequence(samples)
  
  @E.do()
  async def ocr_samples(self, blobs: KV[bytes], models: sm.ModelsCache, *, pads: sm.Pads = {}) -> list[list[OCRSample]]:
    """Export OCR samples of all players
    - Skips `None` labels (due to missing annotations of index after `end_correct`)
    """
    samples = (await self.samples(blobs, models, pads=pads)).unsafe()
    return [[OCRSample(s.img, s.lab) for s in player_samples if s.lab] for player_samples in samples]