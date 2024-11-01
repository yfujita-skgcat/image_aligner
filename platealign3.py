# -*- coding: utf-8 -*-
'''
Copyright (C) 2018 Yoshihiko Fujita, yoshihiko.fujita@cira.kyoto-u.ac.jp

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
barraud@math.univ-lille1.fr

Extension for aligning multiple Cytell images.
'''

# These two lines are only needed if you don't put the script directly into
# the installation directory
import sys
sys.path.append('/usr/share/inkscape/extensions')

import inkex

import numpy as np
import logging

from lxml import etree
import re
import pprint

import xml.etree.ElementTree as ET

pp = pprint.PrettyPrinter(indent=2)

logger = logging.getLogger("platealign")
formatter = logging.Formatter('%(levelname)s - %(lineno)d - %(message)s')
# debugging (True) or deployment (False)
if False:
  # a logger for debugging/warnings
  logger.setLevel(logging.DEBUG)
  fh = logging.FileHandler(filename='/home/yfujita/work/bin/python/inkscape/platealign/platealign.log', mode='a')
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)
else:
  logger.setLevel('WARNING')
  fh = logging.StreamHandler(sys.stderr)
  fh.setFormatter(formatter)

logger.addHandler(fh)
logger.debug("\n\nLogger initialized")


# フィルタ名でファイルをソートするときに使用する関数
# それっぽいキーワードにマッチしたらそのindexを数値で返す
# 基本的に透過光 -> 短い波長から という順で値が大きくなるようにしてソート関数で使えるようにする
# 複数のフィルタ名にマッチする場合(tagBFP-iRFP670)合算する?
# 方針としてはフィルタ名(一部keyword)にマッチしたら2^n を合算して返却することで同じ数値にならないようにする.
# BFP-iRFP670 と iRFP670-BFP は同じ値になる
def filter2index(filter_name):
  index = 0
  filters = [
      'CH1', 'GC', 'PH', 'BF', 'DIA', 'Transillumination', 'Bright_Field', 'Lamp', 'Through', 'PhaseContrast',
      'CH2', 'DAPI', 'BFP', 'BP447',
      'CH3', 'NIBA', 'FITC', 'GFP', 'GFP_N', 'BP525',
      'CH4', 'WIGA', 'Cy3', 'tagRFP', 'RFP_W', 'mCherry', 'BP617',
      'CH5', 'Cy5', 'CY5', 'iRFP670', 'BP685'
  ]

  filters = [".*" + s for s in filters]
  for fi, fname in enumerate(filters):
    # logger.debug(str(fi) + ":検索文字列: " + fname + ", フィルタ名: " + filter_name)
    if re.match(fname, filter_name, flags=re.IGNORECASE):
      # logger.debug("fi=" + str(fi))
      index = index + 2 ** fi
      # logger.debug("fname=" + fname)
      # logger.debug("current index=" + str(index))
  # logger.debug("filter_name=" + filter_name)
  # logger.debug("index=" + str(index))
  return index


# class WellImageSet:
#   """
#   1 Wellの画像セットのクラス
#   - 画像のIDのリスト
#   - IDから画像へのdictonary
#   - IDからフィルタ名へのdictonary
#   - IDからfieldへのdictonary
#   """

#   def __init__(self, row, col, svg, filter_align="vertical", angle=0, width=0, filter_space=0, field_space=0):
#     self.svg = svg
#     self.row = row
#     self.width = width
#     self.col = col
#     self.angle = angle
#     self.filter_align = filter_align
#     self.filter_space = filter_space
#     self.field_space = field_space
#     self.image_ids = set()
#     self.id2image = {}
#     self.id2filter = {}
#     self.id2field = {}
#     self.id2time = {}
#     self.id2zpos = {}
#     self.filter2images = {}
#     self.field2images = {}
#     self.time2images = {}
#     self.zpos2images = {}

#   # 画像を追加
#   def push(self, image):
#     image_id = image.get("id")
#     self.image_ids.add(image_id)
#     fname = ImageAlignEffect.get_filter_name(image)
#     row, col, fld, time, zpos, wav = ImageAlignEffect.get_params_from_name(fname)
#     if row != self.row or col != self.col:
#       logger.error("row, col mismatch: " + fname)
#       raise ValueError("row, col mismatch: " + fname)
#     self.id2image[image_id] = image
#     self.id2filter[image_id] = wav
#     self.id2field[image_id] = fld
#     self.id2tiem[image_id] = time
#     self.id2zpos[image_id] = zpos
#     self.filter2images.setdefault(wav, set()).add(image)
#     self.field2images.setdefault(fld, set()).add(image)
#     self.time2images.setdefault(time, set()).add(image)
#     self.zpos2images.setdefault(zpos, set()).add(image)

#     width = float(image.get("width"))
#     height = float(image.get("height"))
#     if self.angle == 90 or self.angle == 270:
#       width, height = height, width
#     if self.width != 0.0:
#       height = height * self.width / width
#       width = self.width
#     image.set("width", str(width))
#     image.set("height", str(height))

#     if self.angle == 0 and "transform" in image.attrib:
#       del image.attrib["transform"]
#     elif self.angle == 90:
#       image.set("transform", "matrix(0,1,-1,0,0,0)")
#     elif self.options.angle == 180:
#       image.set("transform", "scale(-1,-1)")
#     elif self.options.angle == 270:
#       image.set("transform", "matrix(0,-1,1,0,0,0)")


#     # self.id2width[image_id] = width
#     # self.id2height[image_id] = height

#   def nfield(self):
#     n = len(set(self.id2field.values()))
#     return 1 if n == 0 else n

#   def ntime(self):
#     n = len(set(self.id2tiem.values()))
#     return 1 if n == 0 else n

#   def nzpos(self):
#     n = len(set(self.id2zpos.values()))
#     return 1 if n == 0 else n

#   # 図のレイアウトはfilterの並びの直角方向にfield, zpos, time が並ぶ
#   # zpos, time が複数ある場合は、zpos -> time の順で並べる
#   def row_size(self):
#     """
#     1行の画像数を返す
#     :param col_param: "field" or "filter"
#     :return:
#     """
#     if self.filter_align == "vertical":
#       return self.nfilter()
#     elif self.filter_align == "horizontal":
#       return self.nfield() * self.nzpos() * self.ntime()
#     else:
#       raise ValueError("filter_align must be vertical or horizontal")

#   def col_size(self):
#     """
#     1列の画像数を返す
#     :param col_param: "field" or "filter"
#     :return:
#     """
#     if self.filter_align == "vertical":
#       return self.nfield() * self.nzpos() * self.ntime()
#     elif self.filter_align == "horizontal":
#       return self.nfilter()
#     else:
#       raise ValueError("filter_align must be vertical or horizontal")

#   # well 内の画像を並べたときのx, yのサイズを返す
#   def geometory(self):
#     return self.width(), self.height()


#   def height(self):
#     axis = "height" if self.angle == 0 or self.angle == 180 else "width"
#     if self.align == "vertical":
#       # フィルタを縦に並べる場合は、それぞれのフィルタに含まれる全画像の高さの最大値の合計が高さになる
#       return sum(max(float(image.get(axis)) + (self.nfilter - 1) * self.filter_space for image in self.filter2images[filter]) for filter in set(self.id2filter.values()))
#     elif self.align == "horizontal":
#       # フィルタを横に並べる場合は、同一のtime, zpos, filedをもつ全画像の高さの最大値の合計が高さになる
#       max_axis = [max(float(image.get(axis)) for image in self.field2images[field] & self.time2images[time] & self.zpos2images[zpos])
#                   for field in set(self.id2field.values())
#                   for time in set(self.id2time.values())
#                   for zpos in set(self.id2zpos.values())]
#       total_axis = sum(max_axis) + (len(max_axis) - 1) * self.field_space
#       return total_axis
#       # return sum(max(float(image.get(axis)) for image in (self.field2images[field] & self.time2images[time] & self.zpos2images[zpos])) for field in set(self.id2field.values()) for time in set(self.id2time.values()) for zpos in set(self.id2zpos.values()))
#     else:
#       raise ValueError("align must be vertical or horizontal")

#   def width(self):
#     axis = "width" if self.angle == 0 or self.angle == 180 else "height"
#     if self.align == "vertital":
#       # フィルタを縦に並べる場合は、同一のtime, zpos, filedをもつ全画像の幅の最大値の合計が幅になる
#       max_axis = [max(float(image.get(axis)) for image in self.field2images[field] & self.time2images[time] & self.zpos2images[zpos])
#                   for field in set(self.id2field.values())
#                   for time in set(self.id2time.values())
#                   for zpos in set(self.id2zpos.values())]
#       total_axis = sum(max_axis) + (len(max_axis) - 1) * self.field_space
#       return total_axis
#     elif self.align == "horizontal":
#       # フィルタを横に並べる場合は、それぞれのフィルタに含まれる全画像の幅の最大値の合計が幅になる
#       return sum(max(float(image.get(axis)) + (self.nfilter - 1) * self.filter_space for image in self.filter2images[filter]) for filter in set(self.id2filter.values()))
#     else:
#       raise ValueError("align must be vertical or horizontal")

#   def dimention(self, dim_type):
#     if dim_type not in ["height", "width"]:
#       raise ValueError("dim_type must be height or width")
#     axis = dim_type if self.angle in [0, 180] else "height" if dim_type == "width" else "width"
#     # if self.angle == 0 or self.angle == 180:
#     #   axis = dim_type
#     # else:
#     #   if dim_type == "width":
#     #     axis = "height"
#     #   else:
#     #     axis = "width"
#     if self.align == "vertical":
#         return sum(
#             max(float(image.get(axis)) for image in self.filter2images[filter])
#             for filter in set(self.id2filter.values())
#         )
#     elif self.align == "horizontal":
#         return sum(
#             max(float(image.get(axis)) for image in (self.field2images[field] & self.time2images[time] & self.zpos2images[zpos]))
#             for field in set(self.id2field.values())
#             for time in set(self.id2time.values())
#             for zpos in set(self.id2zpos.values())
#         )
#     else:
#         raise ValueError("align must be 'vertical' or 'horizontal'")


class ImageAlignEffect(inkex.Effect):  # class 宣言の引数は継承
  """
  Either draw marks for alignment, or align images
  """
  # ファイルのwell数、フィルタ名などを見つけるための正規表現を先にコンパイルしておく.
  # ここはmap後にlistにしないとイテレーターとして扱われる。そうすると、一度取得した値を繰り返し使えないので、
  # list化を忘れずに
  file_regexps = list(map(lambda r: re.compile(r, flags=re.IGNORECASE), [
    r'^.*?(?P<ROW>[A-Z])-(?P<COL>\d+)_fld_(?P<FLD>\d+)_wv_(?P<FLT>[^.]+).*$',
    r'^.*?(?P<ROW>[A-Z])%20-%20(?P<COL>\d+)\(fld%20(?P<FLD>\d+)%20wv%20(?P<FLT>[^)]+).*$',
    r'^.*?(?P<ROW>[A-Z])(?P<COL>\d+)-W\d+-P(?P<FLD>\d+)-Z(?P<ZPOS>\d+)-T(?P<TIME>\d+)-(?P<FLT>[^.]+)',
    r'^.*?(?P<ROW>[A-Z])(?P<COL>\d+)F(?P<FLD>\d+)T(?P<TIME>\d+)Z(?P<ZPOS>...)(?P<FLT>[^.]+)\..*$',
    r'^.*?(?P<ROW>[A-Z])(?P<COL>\d+)F(?P<FLD>\d+)T(?P<TIME>\d+)Z(?P<ZPOS>...)(?P<FLT>C\d+)\..*$',
    r'^.*?(?P<ROW>[A-Z])[-_]?(?P<COL>\d+)[-_].*[-_](?P<FLD>\d+)_w\d+(?P<FLT>BF|BFP|NIBA|WIGA|CY5)\..*$',
    r'^.*?(?P<ROW>[A-Z])[-_]?(?P<COL>\d+).*_w\d+(?P<FLT>BF|BFP|NIBA|WIGA|CY5)\..*$',
    r'^.*?(?P<ROW>\d+).*_w\d+(?P<FLT>BF|BFP|NIBA|WIGA|CY5)\..*$',
    r'^.*?(?P<ROW>[A-Z])[-_]?(?P<COL>\d+)[-_]?F(?P<FLD>\d+).*(?P<FLT>C[\d-]+)\..*$',
    r'^.*?(?P<ROW>[A-Z])(?P<COL>\d+)_\d+_(?P<FLD>\d+)_\d+_(?P<FLT>[^.]+)\..*$',
    r'^(?P<ROW>[A-Z])(?P<COL>\d+)-(?P<FLT>[^.]+)\..*$',
    r'^(?P<ROW>[A-Z])(?P<COL>\d+).*_(?P<FLT>[^._]+)\..*$',
    r'^.*?(?P<ROW>[A-Z])(?P<COL>\d+)[-_].*',
    r'^.*?(?P<ROW>[A-Z])(?P<COL>\d+)\.(jpg|tif|png).*'
  ]))

  transform_matrix = {
      0: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
      90: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
      180: np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
      270: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
  }

  def __init__(self):
    """
    Setting stuff up
    """
    inkex.Effect.__init__(self)

    # desc = u'{0} [Args] [Options]\n'
    # self.parser = ArgumentParser(description=desc)

    self.arg_parser.add_argument(
        '-x', '--x', action='store',
        type=int, dest='x', default='0',
        help='Top left x position.'
    )

    self.arg_parser.add_argument(
        '-y', '--y',
        action='store',
        type=int, dest='y', default='0',
        help='Top left y position.'
    )

    self.arg_parser.add_argument(
        '-a', '--angle',
        action='store',
        type=int, dest='angle', default='0',
        help='Rotate images.'
    )

    self.arg_parser.add_argument(
        '-d', '--direction', action='store',
        type=str, dest='direction', default='horizontal',
        help='Set direction of the each filter image.'
    )

    self.arg_parser.add_argument(
        '--filterspace', action='store',
        type=int, dest='filterspace', default='2',
        help='Space between each filter'
    )

    self.arg_parser.add_argument(
        '--fieldspace', action='store',
        type=int, dest='fieldspace', default='5',
        help='Space between each field'
    )

    self.arg_parser.add_argument(
        '-v', '--vspace', action='store',
        type=int, dest='vspace', default='5',
        help='Set vertical space between images.'
    )

    self.arg_parser.add_argument(
        '--hspace', action='store',
        type=int, dest='hspace', default='5',
        help='Set horizontal space between images.'
    )

    self.arg_parser.add_argument(
        '-w', '--width', action='store',
        type=float, dest='width', default='5',
        help='Scaling width of images.'
    )

    self.arg_parser.add_argument(
        '--stamp', action='store',
        type=inkex.Boolean, dest='stamp', default=False,
        help='Stamp layout paramters.'
    )

    self.arg_parser.add_argument(
        '--label', action='store',
        type=inkex.Boolean, dest='label', default=False,
        help='Stamp layout labels.'
    )

    self.arg_parser.add_argument(
        "-s", "--selectedonly",
        action="store", type=inkex.Boolean,
        dest="selectedonly", default=False,
        help="Align only selected images"
    )
    self.arg_parser.parse_args()
    self.page_elements = None
    self.align_text_label = "plate_align_text"
    self.align_stamp_label = "plate_align_stamp"
    self.image_ids = None

  color_list = ['#0000ff', '#00ff00', '#ff0000', '#ff00ff', '#00ffff', '#ffff00', '#8800ff', '#0088ff', '#00ff88', '#ff0088']

  # 画像が所属しているpageを取得する. 所属していない場合は最初のpageを返す
  def get_page_obj(self, img_obj):
    if self.page_elements is None:
      # get <inkscae:page/> element
      # namespace = {'inkscape': 'http://www.inkscape.org/namespaces/inkscape'}
      # get pages if exists
      logger.debug("get page elements")
      self.page_elements = self.document.getroot().findall('.//inkscape:page', inkex.NSS)
    # 画像のオリジナル位置
    ori_y = float(img_obj.get('y'))
    ori_x = float(img_obj.get('x'))
    # transform されている場合は、transfrom後の位置を計算する
    if "transform" in img_obj.attrib:
      transform_str = img_obj.attrib["transform"]
      logger.debug("transform_str=" + transform_str)
      m = re.match(r"rotate\((-?\d+)\)", transform_str)
      if m:
        angle = int(m.group(1))
        if angle < 0:
          angle += 360
      m = re.match(r"scale\(-1(,\s*-1)?\)", transform_str)
      if m:
        angle = 180
      ori_x, ori_y = self.transform_rotate_xy(ori_x, ori_y, angle=angle)

    for page in self.page_elements:
      px = float(page.get('x'))
      py = float(page.get('y'))
      pw = float(page.get('width'))
      ph = float(page.get('height'))
      if px <= ori_x <= px + pw and py <= ori_y <= py + ph:
        return page
    if self.page_elements is None or len(self.page_elements) == 0:
      return None
    return self.page_elements[0]

  # x, y に画像が所属していたpageのoffsetを加算する
  def add_page_offset(self, x, y, img_obj):
    page = self.get_page_obj(img_obj)
    if page is None:
      return x, y
    px = float(page.get('x'))
    py = float(page.get('y'))
    return x + px, y + py

  # 'plate_align_text' をラベルに持つtextを削除する
  def delete_plate_align_text(self):
    text_objects = self.document.getroot().xpath('//svg:text', namespaces=inkex.NSS)
    if self.options.selectedonly:
      text_objects = [text_obj for text_obj in text_objects if text_obj.get('id') in self.options.ids]

    for text_obj in text_objects:
      if self.options.label and text_obj.get(inkex.addNS('label', 'inkscape')) == self.align_text_label:
        text_obj.getparent().remove(text_obj)
      if self.options.stamp and text_obj.get(inkex.addNS('label', 'inkscape')) == self.align_stamp_label:
        text_obj.getparent().remove(text_obj)

  # 画像のidを取得する(選択されている場合は選択されている画像のみ)
  def get_image_ids(self):
    if self.image_ids is not None:
      return self.image_ids
    self.image_ids = [img_obj.get("id") for img_obj in self.document.getroot().xpath('//svg:image', namespaces=inkex.NSS)]
    if self.options.selectedonly:
      self.image_ids = [img_id for img_id in self.image_ids if img_id in self.options.ids]
    return self.image_ids

  # 一番左上の画像とテキストのx, y を返す
  def get_top_left_corner(self, page_id, filter_direction="horizontal"):
    page_obj = self.svg.getElementById(page_id)

    return [0, 0, 0, 0]

  #  filt1  filt2  filt3
  # [img11, img12, img13,...]
  # [img21, -----, img23,...]
  # [img31, -----, img33,...]
  # または
  # filt1 [img11, img12, img13,...]
  # filt2 [img21, -----, img23,...]
  # filt3 [img31, -----, img33,...]
  # を並べる
  # ただし img11 = [z1t1,z1t2,...z2t1,z2t2,... ] になっている
  def align_images(self, images, x, y, width, height, filter_direction="horizontal"):

    col_keys = []
    for i, row in sorted(images.items()):
      col_keys.extend(row.keys())

    if filter_direction == "horizontal":
      # col_keys = sorted(set(col_keys), key=lambda x:filter2index[x])
      col_keys = sorted(set(col_keys), key=filter2index)  # col 側がFilter名
      row_keys = sorted(images.keys())
      xspace = self.options.filterspace
      yspace = self.options.fieldspace
    else:
      col_keys = sorted(set(col_keys))
      # row_keys = sorted(images.keys(), key=lambda x:filter2index[x])
      row_keys = sorted(images.keys(), key=filter2index)  # row 側がfilter名
      xspace = self.options.fieldspace
      yspace = self.options.filterspace

    # logger.debug(row_keys)
    # logger.debug(col_keys)
    # field か フィルタ(縦軸)
    for i, row_key in enumerate(row_keys):
      # field か フィルタ(横軸)
      for j, col_key in enumerate(col_keys):
        if col_key in images[row_key]:
          z_keys = sorted(images[row_key][col_key].keys())
          # z-stack
          for z, z_key in enumerate(z_keys):
            t_keys = sorted(images[row_key][col_key][z_key].keys())
            # time
            for t, t_key in enumerate(t_keys):
              img_obj = images[row_key][col_key][z_key][t_key]
              xpos = x + j * xspace + j * len(t_keys) * len(z_keys) * width + (z * len(t_keys) + t) * width
              ypos = y + i * (height + yspace)
              # xpos, ypos に画像が所属していたpageのoffsetを加算する
              # xpos, ypos = self.add_page_offset(xpos, ypos, img_obj)
              xpos, ypos = self.inverse_transform_rotate_xy(xpos, ypos)
              img_obj.set("x", str(xpos))
              img_obj.set("y", str(ypos))

  def inverse_transform_rotate_xy(self, x, y, angle=None):
    if angle is None:
      mat = ImageAlignEffect.transform_matrix[self.options.angle]
    else:
      mat = ImageAlignEffect.transform_matrix[angle]
    xy = np.array([x, y, 1])
    xy_t = xy.T
    new_xy = np.dot(np.linalg.inv(mat), xy_t)
    return new_xy[0], new_xy[1]

  def transform_rotate_xy(self, x, y, angle=None):
    if angle is None:
      mat = ImageAlignEffect.transform_matrix[self.options.angle]
    else:
      mat = ImageAlignEffect.transform_matrix[angle]
    xy = np.array([x, y, 1])
    xy_t = xy.T
    original_xy = np.dot(mat, xy_t)
    return original_xy[0], original_xy[1]

  @classmethod
  def get_image_fname(cls, img_obj):
    filename = None
    for attr in img_obj.keys():
      # logger.debug("attr=" + attr)
      if re.search("label$", attr):
        filename = img_obj.get(attr).split("/")[-1]
        break
      if re.search("href$", attr):
        data = img_obj.get(attr)
        if re.search("^data:", data):
          continue
        filename = data.split("/")[-1]
        break
    return filename

  @classmethod
  def get_params_from_name(cls, fname):
    if fname is None:
      raise ValueError("fname is None")
    row = None
    col = None
    fld = None
    wav = None
    time = None
    zpos = None
    for reg in ImageAlignEffect.file_regexps:
      match_result = None
      match_result = reg.match(fname)
      if match_result is None:
        continue
      # logger.debug("MATCH: filename=" + str(filename) + "  REGEXP=" + str(reg))
      group_dict = match_result.groupdict()
      row = group_dict['ROW'] if 'ROW' in group_dict.keys() else "A"
      col = group_dict['COL'].zfill(2) if 'COL' in group_dict.keys() else "01"
      fld = group_dict['FLD'].zfill(4) if 'FLD' in group_dict.keys() else "0001"
      time = group_dict['TIME'].zfill(4) if 'TIME' in group_dict.keys() else "0001"
      zpos = group_dict['ZPOS'].zfill(3) if 'ZPOS' in group_dict.keys() else "0001"
      wav = group_dict['FLT'] if 'FLT' in group_dict.keys() else "BF"
      break
    return row, col, fld, time, zpos, wav

  def categorize_images(self, image_ids, direction="horizontal"):
    max_actual_width = 0
    max_actual_height = 0
    images = {}
    if self.options.angle == 90 or self.options.angle == 270:
      vertical_param = "width"
      horizontal_param = "height"
    else:
      vertical_param = "height"
      horizontal_param = "width"
    for img_id in image_ids:
      img_obj = self.svg.getElementById(img_id)

      if "transform" in img_obj.attrib:
        del img_obj.attrib["transform"]
      if self.options.angle == 0:
        pass
        # del img_obj.attrib["transform"]
      elif self.options.angle == 90:
        img_obj.set("transform", "matrix(0,1,-1,0,0,0)")
      elif self.options.angle == 180:
        img_obj.set("transform", "scale(-1,-1)")
      elif self.options.angle == 270:
        img_obj.set("transform", "matrix(0,-1,1,0,0,0)")

      # 90回転していた場合、実際のwidthとheightが逆転するので
      actual_width = float(img_obj.get(horizontal_param))
      actual_height = float(img_obj.get(vertical_param))
      # if self.options.angle == 90 or self.options.angle == 270:
      #   image_width, image_height = image_height, image_width
      if self.options.width != 0.0:
        actual_height = actual_height * self.options.width / actual_width
        actual_width = self.options.width
      img_obj.set(horizontal_param, str(actual_width))
      img_obj.set(vertical_param, str(actual_height))

      if max_actual_width < actual_width:
        max_actual_width = actual_width
      if max_actual_height < actual_height:
        max_actual_height = actual_height

      filename = None
      filename = ImageAlignEffect.get_image_fname(img_obj)

      if filename is None:
        continue

      row, col, fld, time, zpos, wav = ImageAlignEffect.get_params_from_name(filename)

      # images[row] = {} if not images.has_key(row) else images[row]
      if row is None:
        continue
      # images[row] = {} if not row in images.keys() else images[row]
      if row not in images.keys():
        images[row] = {}
      if col is None:
        continue
      # images[row][col] = {} if not col in images[row].keys() else images[row][col]
      if col not in images[row].keys():
        images[row][col] = {}
      if fld is None or wav is None:
        continue
      if time is None:
        continue
      if zpos is None:
        continue
      # logger.debug("direction=" + direction)
      if direction == "horizontal":
        if fld not in images[row][col].keys():
          images[row][col][fld] = {}
        if wav not in images[row][col][fld].keys():
          images[row][col][fld][wav] = {}
        if zpos not in images[row][col][fld][wav].keys():
          images[row][col][fld][wav][zpos] = {}
        images[row][col][fld][wav][zpos][time] = img_obj
      else:
        if wav not in images[row][col].keys():
          images[row][col][wav] = {}
        if fld not in images[row][col][wav].keys():
          images[row][col][wav][fld] = {}
        if zpos not in images[row][col][wav][fld].keys():
          images[row][col][wav][fld][zpos] = {}
        images[row][col][wav][fld][zpos][time] = img_obj
    return images, max_actual_width, max_actual_height

  # 画像の並びの構造は以下のとおりになっている
  # プレートは以下の通り
  #   1  2  3  4 ...
  # A
  # B
  # C
  # D
  # ...
  # まずこのrow, colから画像が存在するところだけ取り出す.
  # そしてあるrow, colについて
  #    filter1, filter2, filter3,...
  # fld1
  # fld2
  # fld3
  # ...
  # となっているので入れ子構造になっているように見える.
  # ただし、プレートと同じようにすると、あるrow, col内ではfilterは揃った縦列に並ぶが
  # 異なるrow, col同士を比べると、filterの縦列が揃わない可能性がある.
  # だから、filterが横に並ぶ場合は同じcolを全部見て位置を決めなければならない
  # 同じcolに含まれるfilterを全部拾ってuniqとって、数をindexにするのか
  # table は dictionary
  # ウェルテーブル
  #    1,  2,  3  ...
  # A: [A1, A2, A3, ...]
  # B: [B1, B2, B3, ...]
  # C: [C1, C2, C3, ...]
  # 各 A1, A2, A3.. がウェル内テーブル、
  #    1,  2,  3  ...
  # a: [a1, a2, a3, ...]
  # b: [b1, b2, b3, ...]
  # c: [c1, c2, c3, ...]
  # という構造をとっている.
  # なので、例えば2番目の横方向の幅を調べる場合は、
  # A2,B2,C2 について、len(a), len(b), len(c)を調べていけば良い
  # さらにa1, a2, それぞれについて、 a1 = [ zpos1, zpos2, zpos3 ], zpos1 = [t1, t2, ...] になっている
  #
  # 一方B番目の縦方向の幅を調べるには
  # B1, B2, B3 について、len(B1), len(B2), len(B3)を調べていけば良い
  def get_max_ncol_at(self, table, ci):
    max_n = 0
    # logger.debug("ci = " + str(ci))
    for ri in table.keys():  # 行のindex
      # logger.debug("ri=" + ri)
      # if not ci in table[ri].keys():
      #   logger.debug("ci=" + ci)
      #   logger.debug("str(table[ri].keys()=" + str(table[ri].keys()))

      if ci in table[ri].keys():
        # logger.debug("str(table[ri][ci]=" + str(table[ri][ci]))
        # images[row][col][fld][wav][zpos][time] = img_obj
        # images[row][col][wav][fld][zpos][time] = img_obj
        for fwnum1 in table[ri][ci].keys():  # ウェル内テーブルの行のindex, fwnum1 = fld or wav number
          # if max_n < len(table[ri][ci][fwnum1]): # 行の長さ
          #   max_n = len(table[ri][ci][fwnum1])
          # logger.debug("str(table[ri][ci][fwnum1]=" + str(table[ri][ci][fwnum1]))
          number_of_fwnum2_time_slice = 0
          for fwnum2 in table[ri][ci][fwnum1].keys():  # fwnum2 = fid or wav number
            # nslice = len(table[ri][ci][fwnum1][fwnum2])
            # logger.debug("str(table[ri][ci][fwnum1].keys()=" + str(table[ri][ci][fwnum1].keys()))
            # logger.debug("str(table[ri][ci][fwnum1][nslice]=" + str(table[ri][ci][fwnum1][nslice]))
            # logger.debug("str(table[ri][ci][fwnum1][fwnum2].keys()=" + str(table[ri][ci][fwnum1][fwnum2].keys()))
            for nslice in table[ri][ci][fwnum1][fwnum2].keys():
              number_of_fwnum2_time_slice += len(table[ri][ci][fwnum1][fwnum2][nslice])  # time
          if max_n < number_of_fwnum2_time_slice:  # 行の長さ
            max_n = number_of_fwnum2_time_slice
            # logger.debug("max_n=" + str(max_n))
    return max_n

  def get_max_ncol_in_each_col(self, table):
    max_ncol = []
    for i in range(1, 25):
      max_ncol.append(self.get_max_ncol_at(table, str(i).zfill(2)))
    return max_ncol

  # table が画像のHashTable[row][col][wav][fld][zpos][time]
  def get_max_nrow_at(self, table, ri):
    max_n = 0
    # logger.debug("table.keys() = " + str(table.keys()))
    if ri not in table.keys():  # row のインデックスがあるかチェック => なければそのrow のhorizon は0
      return max_n
    # logger.debug("ri=" + str(ri))
    for coli in table[ri].keys():  # 列のindex.
      if max_n < len(table[ri][coli]):
        max_n = len(table[ri][coli])

      # logger.debug("coli=" + str(coli))
      # # z slice と time は単純に右に繋いでいくのでそれぞれの画像の数を足す
      # # fld or wav の数 x それぞれのtime-zsliceの数を足す(掛け算ではなく、それぞれ別のtime-zslice数である可能性ありなので全部足す)
      # num_tz_images = 0
      # for wav_fld_i in table[ri][coli].keys():
      #   # logger.debug("wav_fld_i=" + str(wav_fld_i))
      #   logger.debug("table[ri][coli][wav_fld_i]=" + str(table[ri][coli][wav_fld_i]))
      #   for zi in table[ri][coli][wav_fld_i].keys():
      #     # logger.debug("zi=" + str(zi))
      #     num_tz_images = num_tz_images + len(table[ri][coli][wav_fld_i][zi])
      #     for ti in table[ri][coli][wav_fld_i][zi].keys():
      #       logger.debug("ti=" + str(ti))
      # logger.debug("num_tz_images=" + str(num_tz_images))
      # if max_n < num_tz_images:
      #   max_n = num_tz_images
    return max_n

  def get_max_nrow_in_each_row(self, table):
    max_nrow = []
    for i in range(65, 89):  # A-Z まで
      max_nrow.append(self.get_max_nrow_at(table, chr(i)))
    return max_nrow

  # 画像のIDを所属するページごとに返す
  def group_image_ids_by_page(self, image_ids):
    page_ids2image_ids = {}
    for image_id in image_ids:
      image_obj = self.svg.getElementById(image_id)
      page = self.get_page_obj(image_obj)
      if page is None:
        continue
      page_ids2image_ids.setdefault(page.get('id'), []).append(image_id)
    # return {"NoPage": image_ids} if page_ids2images_ids is empty
    if len(page_ids2image_ids) == 0:
      return {"NoPage": image_ids}
    return page_ids2image_ids

  # 多分このeffect()が affect() で呼ばれるんだと思う.
  def effect(self):

    self.delete_plate_align_text()
    image_ids = self.get_image_ids()
    page_ids2image_ids = self.group_image_ids_by_page(image_ids)

    # 画像を回転させる場合の係数
    x_coeffi = {
        0: 0,
        90: 1,
        180: 1,
        270: 0
    }
    y_coeffi = {
        0: 0,
        90: 0,
        180: 1,
        270: 1
    }

    for page_id, image_ids in page_ids2image_ids.items():
      logger.debug("===== page id : " + str(page_id))
      logger.debug("image_ids=" + str(image_ids))
      images, max_image_width, max_image_height = self.categorize_images(image_ids, direction=self.options.direction)
      if page_id == "NoPage":
        px = 0
        py = 0
      else:
        page = self.svg.getElementById(page_id)
        px = float(page.get('x'))
        py = float(page.get('y'))

      # 各行の縦方向の画像の枚数の最大値
      max_nrow = self.get_max_nrow_in_each_row(images)
      # remove 0 from max_nrow
      logger.debug("max_nrow=" + str(max_nrow))
      max_nrow = [x for x in max_nrow if x != 0]
      logger.debug("max_nrow=" + str(max_nrow))
      # 各列の横方向の画像の枚数の最大値
      max_ncol = self.get_max_ncol_in_each_col(images)
      # remove 0 from max_ncol
      logger.debug("max_ncol=" + str(max_ncol))
      max_ncol = [x for x in max_ncol if x != 0]
      logger.debug("max_ncol=" + str(max_ncol))

      # 後で印字するようの配列
      row_label_list = {}
      row_label_x = float('inf')
      col_label_list = {}
      col_label_y = float('inf')

      # 左上から並べていくんだが左上を(0,0)として
      # row, col, fld, wav それぞれにインデックスをつけて並べていくのが良い
      # row_chunk_count = -1        # この数字は実際にデータが存在した行をどれぐらい処理したかカウントするため(行間インターバルの個数)
      # for rowi, row in sorted(images.items()):

      logger.debug("sorted(images.keys())=" + str(sorted(images.keys())))
      row_keys = sorted(images.keys())
      col_keys = [images[row_key].keys() for row_key in row_keys]
      col_keys = sorted(set(item for sublist in col_keys for item in sublist))
      # i はwellの行index
      # for i, row_key in enumerate(sorted(images.keys())):
      for i, row_key in enumerate(row_keys):
        logger.debug("row_key=" + str(row_key))
        total_nrow_images = sum(max_nrow[n] for n in range(i))
        logger.debug("total_nrow_images=" + str(total_nrow_images))

        logger.debug(f"{row_key} sorted(images[row_key].keys())=" + str(sorted(images[row_key].keys())))
        # j はwellの列index
        # for j, col_key in enumerate(sorted(images[row_key].keys())):
        for j, col_key in enumerate(col_keys):
          logger.debug("col_key=" + str(col_key))
          total_ncol_images = sum(max_ncol[n] for n in range(j))
          logger.debug("total_ncol_images=" + str(total_ncol_images))

          # 行にカラムが存在しない場合はスキップ
          if col_key not in images[row_key]:
            continue

          # well に含まれる画像(filter, field など複数含まれる)
          well_images = images[row_key][col_key]
          if self.options.direction == "horizontal":
            x = px + self.options.x + max_image_width * x_coeffi[self.options.angle] + total_ncol_images * (max_image_width + self.options.filterspace) + j * (self.options.hspace - self.options.filterspace)
            y = py + self.options.y + max_image_height * y_coeffi[self.options.angle] + total_nrow_images * (max_image_height + self.options.fieldspace) + i * (self.options.vspace - self.options.fieldspace)
          else:
            x = px + self.options.x + max_image_width * x_coeffi[self.options.angle] + total_ncol_images * (max_image_width + self.options.fieldspace) + j * (self.options.hspace - self.options.fieldspace)
            y = py + self.options.y + max_image_height * y_coeffi[self.options.angle] + total_nrow_images * (max_image_height + self.options.filterspace) + i * (self.options.vspace - self.options.filterspace)

          text_x = x - max_image_width * x_coeffi[self.options.angle]
          text_y = y - max_image_height * y_coeffi[self.options.angle]

          # x, y をpixel単位に変換
          text_x = inkex.units.convert_unit(text_x, "px")
          text_y = inkex.units.convert_unit(text_y, "px")
          x = inkex.units.convert_unit(x, "px")
          y = inkex.units.convert_unit(y, "px")
          # logger.debug("x=" + str(x) + ", y=" + str(y))
          # logger.debug("text_x=" + str(text_x) + ", text_y=" + str(text_y))

          # logger.debug("col_image_count=" + str(col_image_count))
          self.align_images(well_images, x, y, max_image_width, max_image_height, filter_direction=self.options.direction)

          # if col_key not in col_label_list:
          # text = inkex.etree.Element(inkex.addNS('text', 'svg'))
          text = etree.Element(inkex.addNS('text', 'svg'))
          text.text = col_key
          text.set('x', str(text_x + max_ncol[j] * max_image_width / 2 - 8))
          if col_label_y > text_y:
            col_label_y = text_y
          # text.set('y', str(y-10))
          text.set(inkex.addNS('label', 'inkscape'), 'plate_align_text')
          style = {
              'stroke': 'none',
              'stroke-width': '1',
              'fill': '#000000',
              'font-family': 'Arial',
              'font-weight': 'normal',
              'font-style': 'normal',
              'font-strech': 'normal',
              'font-variant': 'normal',
              'font-size': '16px'
          }
          # text.set('style', simplestyle.formatStyle(style))
          text.set('style', str(inkex.Style(style)))
          col_label_list[col_key] = text

          # if row_key not in row_label_list:
          # text = inkex.etree.Element(inkex.addNS('text', 'svg'))
          text = etree.Element(inkex.addNS('text', 'svg'))
          text.text = row_key
          if row_label_x > text_x:
            row_label_x = text_x
          # text.set('x', str(x-16))
          text.set('y', str(text_y + max_nrow[i] * max_image_height / 2 + 8))
          text.set(inkex.addNS('label', 'inkscape'), 'plate_align_text')
          style = {
              'stroke': 'none',
              'stroke-width': '1',
              'fill': '#000000',
              'font-family': 'Arial',
              'font-weight': 'normal',
              'font-style': 'normal',
              'font-strech': 'normal',
              'font-variant': 'normal',
              'font-size': '16px'
          }
          # text.set('style', simplestyle.formatStyle(style))
          text.set('style', str(inkex.Style(style)))
          row_label_list[row_key] = text

      # ラベルを印字する
      if self.options.label:
        # parent = self.current_layer
        parent = self.svg.get_current_layer()
        # for _text in col_label_list.itervalues():
        for _text in col_label_list.values():
          _text.set('y', str(col_label_y - 10))
          parent.append(_text)
        # for _text in row_label_list.itervalues():
        for _text in row_label_list.values():
          _text.set('x', str(row_label_x - 16))
          parent.append(_text)

      if self.options.stamp:  # 並び方のパラメータを印字
        # parent = self.current_layer
        parent = self.svg.get_current_layer()
        # text = inkex.etree.Element(inkex.addNS('text', 'svg'))
        text = etree.Element(inkex.addNS('text', 'svg'))
        text.text = "x" + str(self.options.x) + "y" + str(self.options.y) + "w" + str(self.options.width) + "hs" + str(self.options.hspace) + "vs" + str(self.options.vspace)
        stamp_x = px + self.options.x + 2
        stamp_y = py + self.options.y - 26  # font が10pt だから10pt + 2 かな?
        text.set('x', str(stamp_x))
        text.set('y', str(stamp_y))  # font が10pt だから10pt + 2 かな?
        text.set(inkex.addNS('label', 'inkscape'), 'plate_align_stamp')
        style = {
          'stroke': 'none',
          'stroke-width': '1',
          'fill': '#000000',
          'font-family': 'Arial',
          'font-weight': 'normal',
          'font-style': 'normal',
          'font-strech': 'normal',
          'font-variant': 'normal',
          'font-size': '12.5px'
        }
        # text.set('style', simplestyle.formatStyle(style))
        text.set('style', str(inkex.Style(style)))

        parent.append(text)


# Create effect instance and apply it.
# logger.debug(len(sys.argv))

if len(sys.argv) == 1:
  # sys.argv = [ './platealign.py', '--angle=0', '--direction=horizontal', '--hspace=10', '--vspace=20', '--width=384', '/home/yfujita/work/bin/python/inkscape/platealign/test.svg' ]
  sys.argv = ['./platealign3.py', '--id=image4757', '--angle=90', '--direction=vertical', '--hspace=10', '--vspace=20', '--filterspace=2', '--fieldspace=5', '--width=384', '/home/yfujita/work/bin/python/inkscape/platealign/test.svg']

# logger.debug( "Started with: {}.".format( str( sys.argv ) ) )
effect = ImageAlignEffect()
# effect.affect(args=sys.argv)
effect.run()
