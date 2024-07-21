# import numpy as np
# from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
# from torch import nn
# from PIL import Image


# def cityscapes_palette():
#     """Palette that maps each class to RGB values."""
#     return [
#         [120, 120, 120],
#         [180, 120, 120],
#         [6, 230, 230],
#         [80, 50, 50],
#         [4, 200, 3],
#         [120, 120, 80],
#         [140, 140, 140],
#         [204, 5, 255],
#         [230, 230, 230],
#         [4, 250, 7],
#         [224, 5, 255],
#         [235, 255, 7],
#         [150, 5, 61],
#         [120, 120, 70],
#         [8, 255, 51],
#         [255, 6, 82],
#         [143, 255, 140],
#         [204, 255, 4],
#         [255, 51, 7],
#     ]


# class SegmentationModel(nn.Module):
#     palette = np.array(cityscapes_palette())

#     def __init__(self):
#         super().__init__()

#         model_name = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"

#         self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
#         self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)

#     def forward(self, x):
#         inputs = self.feature_extractor(images=x, return_tensors="pt")
#         outputs = self.model(**inputs)

#         return outputs

#     def predict2(self, image):
#         outputs = self.forward(image)

#         logits = nn.functional.interpolate(
#             outputs.logits.detach().cpu(),
#             size=image.size[::-1],  # (height, width)
#             mode="bilinear",
#             align_corners=False,
#         )

#         pred_seg = logits.argmax(dim=1)[0]
#         color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)

#         for label, color in enumerate(SegmentationModel.palette):
#             color_seg[pred_seg == label, :] = color

#         # Convert to BGR
#         color_seg = color_seg[..., ::-1]

#         # color_seg = Image.fromarray(color_seg)
#         # return color_seg

#         # Show image + mask
#         img = np.array(image) * 0.5 + color_seg * 0.3
#         img = img.astype(np.uint8)
#         img = Image.fromarray(img)

#         return img
