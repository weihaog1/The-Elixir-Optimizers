import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = str(Path(__file__).parent / "fonts" / "Consolas.ttf")


def plot_box_PIL(
    image: Image.Image, box_params: tuple | np.ndarray,
    text="", fontsize=12, box_color='red',
    fontpath=None,
    format='yolo',
    draw_center_point=False,
    alpha_channel=150,
  ):
  """
  (PIL) Plot the bounding box with `box_params` in `image`.

  Args:
    text: The text display in the upper left of the bounding box.
    fontpath: Path to font file. Defaults to bundled Consolas.ttf.
    format: Support three common bounding boxes type:
      - `yolo` (proportion): `(x, y, w, h)`, `(x, y)` is the center of the bbox,
      and `(w, h)` is the width and height of the bbox.
      - `coco` (pixel): `(x, y, w, h)`, `(x, y)` is the left top of the bbox,
      and `(w, h)` is the width and height of the bbox.
      - `voc` (pixel): `(x1, y1, x2, y2)`, `(x1, y1)` is the left top of the bbox,
      and `(x2, y2)` is the rigth bottom of the bbox.
    alpha_channel: If the image type is RGBA, the alpha channel will be used.
  """
  if fontpath is None:
    fontpath = FONT_PATH
  draw = ImageDraw.Draw(image)
  params, shape = np.array(box_params), image.size
  if np.max(params) <= 1.0:
    params[0] *= shape[0]
    params[2] *= shape[0]
    params[1] *= shape[1]
    params[3] *= shape[1]
  if format.lower() == 'yolo':
    x_min = int(params[0] - params[2] / 2)
    y_min = int(params[1] - params[3] / 2)
    w = int(params[2])
    h = int(params[3])
  elif format.lower() == 'coco':
    x_min = int(params[0])
    y_min = int(params[1])
    w = int(params[2])
    h = int(params[3])
  elif format.lower() == 'voc':
    x_min = int(params[0])
    y_min = int(params[1])
    w = int(params[2] - params[0])
    h = int(params[3] - params[1])
  if type(box_color) == str and box_color == 'red': box_color = (255, 0, 0)
  if type(box_color) != tuple: box_color = tuple(box_color)
  box_color = box_color + (alpha_channel,)
  draw.rectangle([x_min, y_min, x_min+w, y_min+h], outline=box_color, width=2)

  font_color = (255,255,255)  # white
  font = ImageFont.truetype(fontpath, fontsize)
  import PIL
  pil_version = int(PIL.__version__.split('.')[0])
  w_text, h_text = font.getbbox(text)[-2:] if pil_version >= 10 else font.getsize(text)
  x_text = x_min
  y_text = y_min - h_text if y_min > h_text else y_min
  draw.rounded_rectangle([x_text, y_text, x_text+w_text, y_text+h_text], radius=1.5, fill=box_color)
  draw.text((x_text, y_text), text, fill=font_color, font=font)
  if draw_center_point:
    draw.rounded_rectangle([x_min+w/2-2,y_min+h/2-2,x_min+w/2+2,y_min+h/2+2], radius=1.5, fill=(255,0,0))
  return image


def plot_cells_PIL(image: Image.Image, w: int, h: int, line_color=(0,0,0)):
  """
  (PIL) Draw the `wxh` cells division.
  """
  draw = ImageDraw.Draw(image)
  shape = image.size
  space = (shape[0] / w, shape[1] / h)
  for i in range(1, h):
    y = i * space[1]
    draw.line([(0,y), (shape[0],y)], fill=line_color)
  for i in range(1, h):
    x = i * space[0]
    draw.line([(x,0), (x,shape[1])], fill=line_color)
  return image


def get_box_colors(n):
  cmap = plt.cm.brg
  step = cmap.N // n
  colors = cmap([i for i in range(0, cmap.N, step)])
  colors = (colors[:, :3] * 255).astype(int)
  colors = [tuple(color) for color in colors]
  return colors


def build_label2colors(labels):
  if not len(labels): return {}
  labels = np.unique(labels).astype(np.int32)
  colors = get_box_colors(len(labels))
  return dict(zip(labels, colors))
