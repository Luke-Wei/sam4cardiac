# SAM4Cardiac

## 1. Installation

Please refer to the [Segment-Anything](https://github.com/facebookresearch/segment-anything) repository for installation instructions.

Please note that the official PyTorch version greater than 1.7 can only run SAM and cannot convert the model to ONNX. If you wish to convert to ONNX, please install Torch2.

## 2. Getting Started

Place the "cardiac_sam_demo.ipynb" and "cardiac_sam_onnx_demo.ipynb" files into the `segment-anything-main/notebooks` directory. Also, download three weight files from the provided Google Drive link and place them in the same location.

You can download the pre-trained model and the converted ONNX file from the following link: [Google Drive](https://drive.google.com/drive/folders/1CtVO4B99sqTFlMn5RmPtR3ZoFp2czj98?usp=sharing)

The file structure should look like this:

```
segment-anything-main/notebooks/
├── cardiac_sam_demo.ipynb
├── cardiac_sam_onnx_demo.ipynb
├── sam_onnx.onnx
├── sam_onnx_quantized_example.onnx
├── sam_vit_h_4b8939.pth
```

After setting up the directory as described above, you can run the two `.ipynb` files.
