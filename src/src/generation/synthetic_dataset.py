"""On-the-fly synthetic dataset for training YOLOv8 with KataCR's generator.

Ported from KataCR's katacr/yolov8/custom_dataset.py and custom_trainer.py.
Subclasses YOLODataset and overrides get_image_and_label() to generate unique
synthetic images on every call, letting ultralytics handle transforms/collation.

Supports optional belonging (ally/enemy) prediction via use_belonging flag.
When enabled, cls is 2-column (class_id, belonging) and labels are 6-column.
"""

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor as ThreadPool
from itertools import repeat

import numpy as np
from PIL import Image as PILImage
from ultralytics.data import YOLODataset
from ultralytics.data.dataset import (
    Instances, load_dataset_cache_file, DATASET_CACHE_VERSION,
    get_hash, TQDM, LOGGER, LOCAL_RANK, HELP_URL, NUM_THREADS,
    save_dataset_cache_file,
)
from ultralytics.utils import colorstr

from src.generation.generator import Generator
from src.generation.label_list import idx2unit, unit2idx

# Generation defaults (from KataCR's cfg.py)
TRAIN_DATASIZE = 20000
IMG_SIZE = (576, 896)  # width, height
UNIT_NUMS = 40
INTERSECT_RATIO_THRE = 0.5
MAP_UPDATE_MODE = 'dynamic'
NOISE_UNIT_RATIO = 0.25


class CRDataset(YOLODataset):
    """YOLODataset subclass that generates synthetic training images on-the-fly.

    For training: overrides get_image_and_label() to call the Generator,
    producing a unique image every call.
    For validation: delegates to the parent YOLODataset (loads from disk).

    When use_belonging=True, cls output is 2-column (class_id, belonging)
    and validation labels are read as 6-column (cls, x, y, w, h, belonging).
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        self.img_path = None
        self.use_belonging = kwargs.pop('use_belonging', False)
        if kwargs['img_path'] is not None:  # validation - load from disk
            kwargs.pop('seed', None)
            kwargs.pop('unit_nums', None)
            kwargs.pop('noise_unit_ratio', None)
            kwargs.pop('background_index', None)
            kwargs.pop('ally_classes', None)
            if self.use_belonging:
                # Custom label loading for 6-column labels
                self._init_belonging_val(*args, data=data, task=task, **kwargs)
            else:
                super().__init__(*args, data=data, task=task, **kwargs)
        else:  # training - generate on-the-fly
            self.unit_nums = kwargs.pop('unit_nums', UNIT_NUMS)
            seed = kwargs.pop('seed', None)
            noise_unit_ratio = kwargs.pop('noise_unit_ratio', NOISE_UNIT_RATIO)
            background_index = kwargs.pop('background_index', None)
            ally_classes = kwargs.pop('ally_classes', None)
            self.name_inv = {n: i for i, n in data['names'].items()}
            gen_kwargs = dict(
                seed=seed,
                background_index=background_index,
                intersect_ratio_thre=INTERSECT_RATIO_THRE,
                map_update={'mode': MAP_UPDATE_MODE, 'size': 5},
                avail_names=list(data['names'].values()),
                noise_unit_ratio=noise_unit_ratio,
            )
            if ally_classes is not None:
                gen_kwargs['ally_classes'] = ally_classes
            self.generator = Generator(**gen_kwargs)
            self.data = data
            self.augment = kwargs.get('augment', False)
            self.rect = kwargs.get('rect', True)
            self.imgsz = kwargs.get('imgsz', 960)
            self.use_segments = self.use_keypoints = self.use_obb = False
            self.transforms = self.build_transforms(hyp=kwargs.get('hyp'))

    def _init_belonging_val(self, *args, data=None, task="detect", **kwargs):
        """Initialize validation dataset with 6-column belonging labels."""
        # Don't call super().__init__ since we need custom label loading.
        # Instead, manually set up what YOLODataset needs.
        from ultralytics.data.base import BaseDataset
        # Call BaseDataset.__init__ skipping YOLODataset's label loading
        self.data = data
        self.imgsz = kwargs.get('imgsz', 960)
        self.augment = kwargs.get('augment', False)
        self.rect = kwargs.get('rect', True)
        self.use_segments = self.use_keypoints = self.use_obb = False
        self.single_cls = kwargs.get('single_cls', False)

        img_path = kwargs.get('img_path') or (args[0] if args else None)
        self.img_path = img_path

        # Build image file list
        self.im_files = sorted(
            str(p) for p in Path(img_path).glob('*.jpg')
        )
        if not self.im_files:
            self.im_files = sorted(
                str(p) for p in Path(img_path).glob('*.png')
            )

        # Build labels with belonging
        self.labels = self._get_belonging_labels()
        self.ni = len(self.labels)

        # Set up transforms
        self.transforms = self.build_transforms(hyp=kwargs.get('hyp'))

        # Required attributes
        self.batch_shapes = None
        prefix = kwargs.get('prefix', '')
        self.prefix = prefix

        # Set up batching (rect mode)
        batch_size = kwargs.get('batch_size', 16)
        if self.rect:
            self._setup_rect(batch_size)

    def _setup_rect(self, batch_size):
        """Set up rectangular training/val batching."""
        n = len(self.labels)
        if n == 0:
            return
        # Get image shapes
        shapes = []
        for lb in self.labels:
            shapes.append(lb.get('shape', lb.get('ori_shape', (960, 540))))
        shapes = np.array(shapes, dtype=np.float64)
        ar = shapes[:, 0] / shapes[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set batch shapes
        bi = np.floor(np.arange(n) / batch_size).astype(int)
        nb = bi[-1] + 1
        self.batch = bi
        s = np.array(shapes, dtype=np.float64)
        self.batch_shapes = np.zeros((nb, 2), dtype=np.int64)
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                self.batch_shapes[i] = [
                    int(round(self.imgsz * maxi / 32) * 32),
                    self.imgsz,
                ]
            elif mini > 1:
                self.batch_shapes[i] = [
                    self.imgsz,
                    int(round(self.imgsz / mini / 32) * 32),
                ]
            else:
                self.batch_shapes[i] = [self.imgsz, self.imgsz]

    def _get_belonging_labels(self):
        """Load labels with belonging from 6-column label files."""
        labels = []
        names = self.data['names']
        names_inv = {n: i for i, n in names.items()}

        for im_file in self.im_files:
            # Derive label path from image path
            lb_file = im_file.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'

            try:
                im = PILImage.open(im_file)
                shape = im.size  # (w, h)
                shape = (shape[1], shape[0])  # (h, w)
            except Exception:
                shape = (960, 540)

            if os.path.isfile(lb_file):
                with open(lb_file) as f:
                    raw = [x.split() for x in f.read().strip().splitlines() if x.strip()]

                if raw and len(raw[0]) >= 6:
                    # 6-column format: cls_id, x, y, w, h, belonging
                    cls_list = []
                    bbox_list = []
                    for parts in raw:
                        cls_id = int(parts[0])
                        x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        bel = float(parts[5])
                        cls_list.append([cls_id, bel])
                        bbox_list.append([x, y, w, h])
                    cls_arr = np.array(cls_list, dtype=np.float32)
                    bbox_arr = np.array(bbox_list, dtype=np.float32)
                elif raw and len(raw[0]) == 5:
                    # 5-column format (no belonging): cls_id, x, y, w, h
                    cls_list = []
                    bbox_list = []
                    for parts in raw:
                        cls_id = int(parts[0])
                        x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        cls_list.append([cls_id, 0.0])  # default belonging=0
                        bbox_list.append([x, y, w, h])
                    cls_arr = np.array(cls_list, dtype=np.float32)
                    bbox_arr = np.array(bbox_list, dtype=np.float32)
                else:
                    cls_arr = np.zeros((0, 2), dtype=np.float32)
                    bbox_arr = np.zeros((0, 4), dtype=np.float32)
            else:
                cls_arr = np.zeros((0, 2), dtype=np.float32)
                bbox_arr = np.zeros((0, 4), dtype=np.float32)

            labels.append({
                'im_file': im_file,
                'shape': shape,
                'cls': cls_arr,
                'bboxes': bbox_arr,
                'segments': [],
                'keypoints': None,
                'normalized': True,
                'bbox_format': 'xywh',
            })

        return labels

    def __len__(self):
        if self.img_path is not None:
            return len(self.labels)
        return TRAIN_DATASIZE

    def get_image_and_label(self, index):
        if self.img_path is not None:
            if self.use_belonging:
                return self._get_belonging_val_item(index)
            return super().get_image_and_label(index)
        self.generator.reset()
        self.generator.add_tower()
        self.generator.add_unit(self.unit_nums)
        img, box, _ = self.generator.build(box_format='cxcywh', img_size=IMG_SIZE)
        # box: (N, 6) - cx, cy, w, h, belonging, cls_id (normalized)
        bboxes = box[:, :4]

        if self.use_belonging:
            # 2-column cls: (class_id, belonging)
            belonging = box[:, 4]
            cls_ids = np.array(
                [self.name_inv[idx2unit[int(i)]] for i in box[:, 5]],
                dtype=np.float32,
            )
            cls = np.stack([cls_ids, belonging], axis=1).astype(np.float32)
        else:
            # 1-column cls: class_id only
            cls = np.array(
                [self.name_inv[idx2unit[int(i)]] for i in box[:, 5]],
                dtype=np.float32,
            ).reshape(-1, 1)

        label = {
            'im_file': None,
            'ratio_pad': (1.0, 1.0),
            'rect_shape': np.array(IMG_SIZE[::-1], np.float32),
            'ori_shape': img.shape[:2],
            'resized_shape': img.shape[:2],
            'cls': cls,
            'bbox_format': 'xywh',
            'img': img[..., ::-1],  # RGB -> BGR (ultralytics expects BGR)
            'instances': Instances(
                bboxes,
                np.zeros((0, 1000, 2), np.float32),
                None,
                'xywh',
                True,
            ),
        }
        return label

    def _get_belonging_val_item(self, index):
        """Load a validation image+label with belonging."""
        import cv2
        label_info = self.labels[index]
        im_file = label_info['im_file']

        img = cv2.imread(im_file)
        if img is None:
            raise FileNotFoundError(f"Image not found: {im_file}")

        h0, w0 = img.shape[:2]

        label = {
            'im_file': im_file,
            'ratio_pad': (1.0, 1.0),
            'ori_shape': (h0, w0),
            'resized_shape': (h0, w0),
            'cls': label_info['cls'].copy(),
            'bbox_format': 'xywh',
            'img': img,
            'instances': Instances(
                label_info['bboxes'].copy(),
                np.zeros((0, 1000, 2), np.float32),
                None,
                'xywh',
                True,
            ),
        }

        # Apply rect shape if available
        if self.batch_shapes is not None and hasattr(self, 'batch'):
            bi = self.batch[index]
            label['rect_shape'] = np.array(
                self.batch_shapes[bi], dtype=np.float32
            )
        else:
            label['rect_shape'] = np.array([h0, w0], dtype=np.float32)

        return label
