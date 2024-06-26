from typing import Iterable, TypedDict, Literal, TextIO
from dataclasses import dataclass
import os
from haskellian import iter as I, Iter
import files_dataset as fds
import lines_dataset as lds
from .meta import MetaJson

Key = Literal['san', 'lab', 'img']

class Sample(TypedDict):
  san: str
  lab: str
  img: bytes

@dataclass
class Dataset:
  base_path: str
  labels: lds.Dataset
  boxes: fds.Dataset

  @staticmethod
  def read(base: str):
    with open(os.path.join(base, 'meta.json')) as f:
      meta = MetaJson.model_validate_json(f.read())
      labs = lds.Dataset(base, meta.lines_dataset)
      imgs = fds.Dataset(base, meta.files_dataset)
    return Dataset(base, labs, imgs)

  @I.lift
  def samples(self, *, labs_key: str = 'labs', sans_key: str = 'pgn', boxes_key: str = 'boxes') -> Iterable[Sample]:
    for san, lab, img in zip(self.labels.iterate(sans_key), self.labels.iterate(labs_key), self.boxes.iterate(boxes_key)):
      yield Sample(san=san, lab=lab, img=img)

  def __iter__(self):
    return self.samples()
  
  def len(self, *, labs_key: str = 'labs', sans_key: str = 'pgn', boxes_key: str = 'boxes') -> int | None:
    labs_len = self.labels.len(labs_key, sans_key)
    boxes_len = self.boxes.len(boxes_key)
    if labs_len is not None and boxes_len is not None:
      return min(labs_len, boxes_len)

def glob(glob: str, *, recursive: bool = False, err_stream: TextIO | None = None) -> list[Dataset]:
  """Read all datasets that match a glob pattern."""
  from glob import glob as _glob
  datasets = []
  for p in sorted(_glob(glob, recursive=recursive)):
    try:
      datasets.append(Dataset.read(p))
    except Exception as e:
      if err_stream:
        print(f'Error reading dataset at {p}:', e, file=err_stream)
  return datasets

def chain(datasets: Iterable[Dataset], *, labs_key: str = 'labs', sans_key: str = 'pgn', boxes_key: str = 'boxes') -> Iter[Sample]:
  """Chain multiple datasets into a single one."""
  return I.flatten([ds.samples(labs_key=labs_key, sans_key=sans_key, boxes_key=boxes_key) for ds in datasets])

def len(datasets: Iterable[Dataset], *, labs_key: str = 'labs', sans_key: str = 'pgn', boxes_key: str = 'boxes') -> int | None:
  """Total number of samples in multiple datasets (undefined lengths count as 0)"""
  return sum(ds.len(labs_key=labs_key, sans_key=sans_key, boxes_key=boxes_key) or 0 for ds in datasets)
