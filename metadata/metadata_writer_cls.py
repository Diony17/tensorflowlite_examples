''' This code is for populating metadata in a TFLITE model.
    Mobile Embedded System Lab.    Kim, Jiwon <kim.j@yonsei.ac.kr> '''

import os
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
import tensorflow as tf

"""File information"""

model_name = 'EfficientNet-B0' # test model name
model_dir = './models/' # tflite model path
exported_model_dir = './models/export/'
model_file = model_dir + model_name + '.tflite'
exported_model_file = exported_model_dir + model_name + '.tflite'

exported_metadata_file = exported_model_dir + model_name + '-metadata.json'
cifar10_label_file = model_dir + 'cifar10-label.txt' # label list

# Copies model_file to export_path.
tf.io.gfile.copy(model_file, exported_model_file, overwrite=False)

"""Creates the metadata for an image classifier."""
# 1. 모델 메타데이터, 2-1. 입력 텐서 메타데이터, 2-2. 출력 텐서 메타데이터, 3. Subgraph 메타데이터
# A. 입출력 메타데이터를 만들어서 Subgraph 메타데이터로 집어넣고, B. 이 Subgraph 메타데이터를 모델 메타데이터로 집어넣는다.
# C. 최종적으로, 만들어진 모델 메타데이터를 FlatBuffer로 만들어 tflite 모델에 populating 한다.

# 1. Create Model Metadata
model_meta = _metadata_fb.ModelMetadataT()
model_meta.name = model_name + " image classifier"
model_meta.description = ("MOBED custom image classificiation models trained with CIFAR-10 dataset")
model_meta.version = "b0"
model_meta.author = "mobed"
model_meta.license = ("Apache License. Version 2.0 "
                      "http://www.apache.org/licenses/LICENSE-2.0.")

# 2-1. Creates input Metadata # input: 32 * 32 RGB image
input_meta = _metadata_fb.TensorMetadataT()
input_meta.name = "image"
input_meta.description = (
    "Input image to be classified. The expected image is {0} x {1}, with "
    "three channels (red, blue, and green) per pixel. Each value in the "
    "tensor is a single byte between 0 and 255.".format(32, 32))
input_meta.content = _metadata_fb.ContentT()
input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
input_meta.content.contentProperties.colorSpace = (
    _metadata_fb.ColorSpaceType.RGB)
input_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.ImageProperties)
input_normalization = _metadata_fb.ProcessUnitT()
input_normalization.optionsType = (
    _metadata_fb.ProcessUnitOptions.NormalizationOptions)
input_normalization.options = _metadata_fb.NormalizationOptionsT()
input_normalization.options.mean = [127.5]
input_normalization.options.std = [127.5]
input_meta.processUnits = [input_normalization]
input_stats = _metadata_fb.StatsT()
input_stats.max = [255]
input_stats.min = [0]
input_meta.stats = input_stats

# 2-2. Creates output Metadata
output_meta = _metadata_fb.TensorMetadataT()
output_meta = _metadata_fb.TensorMetadataT()
output_meta.name = "probability"
output_meta.description = "Probabilities of the 10 labels respectively."
output_meta.content = _metadata_fb.ContentT()
output_meta.content.contentProperties = _metadata_fb.FeaturePropertiesT()
output_meta.content.contentPropertiesType = (_metadata_fb.ContentProperties.FeatureProperties)
output_stats = _metadata_fb.StatsT()
output_stats.max = [1.0]
output_stats.min = [0.0]
output_meta.stats = output_stats
label_file = _metadata_fb.AssociatedFileT()
label_file.name = os.path.basename(cifar10_label_file)
label_file.description = "Labels for objects that the model can recognize."
label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
output_meta.associatedFiles = [label_file]

# 3. Creates subgraph Metadata 
subgraph = _metadata_fb.SubGraphMetadataT()

# A. Set subgraph Metadata with Input, Output Metadata
subgraph.inputTensorMetadata = [input_meta]
subgraph.outputTensorMetadata = [output_meta]

# B. Set model Metadata with subgraph Metadata
model_meta.subgraphMetadata = [subgraph]

# C. Perform the population
b = flatbuffers.Builder(0)
b.Finish(
    model_meta.Pack(b),
    _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
metadata_buf = b.Output()

populator = _metadata.MetadataPopulator.with_model_file(exported_model_file)
populator.load_metadata_buffer(metadata_buf)
populator.load_associated_files([cifar10_label_file])
populator.populate()
    

# Save Metadata in json file
displayer = _metadata.MetadataDisplayer.with_model_file(exported_model_file)
json_file = displayer.get_metadata_json()
with open(exported_metadata_file, "w") as f:
    f.write(json_file)
