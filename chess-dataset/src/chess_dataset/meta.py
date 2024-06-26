from pydantic import ConfigDict
import files_dataset as fds
import lines_dataset as lds

class MetaJson(fds.MetaJson, lds.MetaJson):
  model_config = ConfigDict(extra='allow')