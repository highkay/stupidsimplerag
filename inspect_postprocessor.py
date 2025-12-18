from llama_index.core.postprocessor.types import BaseNodePostprocessor
print("Attributes of BaseNodePostprocessor:")
print([attr for attr in dir(BaseNodePostprocessor) if 'postprocess' in attr])
