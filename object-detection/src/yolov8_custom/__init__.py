"""Custom YOLOv8 pipeline with belonging (ally/enemy) prediction.

Ported from KataCR's katacr/yolov8/ custom pipeline.
Adds a padding_belong class as the last output channel to predict
ally (0) vs enemy (1) for each detection.
"""
