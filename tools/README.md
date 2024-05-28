# Tools to use with the SDK

## Build the ONNX files of NeuroCorgiNet

Use `build_neurocorgi_onnx.py` to convert the weights folder for N2D2 (like `imagenet_weights/`) to a onnx file. To do so, run:
```
python build_neurocorgi_onnx.py
```
The script loads the weights of the given folder, reconstructs the model by hand and merges the BNs using the quantization algorithm.
To obtain the safetensors file, load the onnx model with the pytorch model and then use *safetensors.torch.save*.

For the COCO version, please adapt the first variables in the script to: 
```python
fq_filename = "neurocorginet_fq_coco.onnx"
chip_filename = "neurocorginet_coco.onnx"

inputs_dimensions = [1, 3, 300, 300]

weight_directory = "coco_weights"
```