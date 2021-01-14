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
sys.path.append( '/usr/share/inkscape/extensions' )

import inkex

import simplestyle
import simpletransform
import numpy as np
import math
import logging

from argparse import ArgumentParser

# import synfig_prepare

from lxml import etree
import re
import pprint

pp = pprint.PrettyPrinter(indent=2)

logger = logging.getLogger( "platealign" )
formatter = logging.Formatter('%(levelname)s - %(lineno)d - %(message)s')
# debugging (True) or deployment (False)
if False:
    #a logger for debugging/warnings
    logger.setLevel( logging.DEBUG )
    fh = logging.FileHandler(filename='/home/yfujita/work/bin/python/inkscape/platealign/platealign.log', mode='a')
    fh.setLevel( logging.DEBUG )
    fh.setFormatter(formatter)
else:
    logger.setLevel( 'WARNING' )
    fh = logging.StreamHandler( sys.stderr )
    fh.setFormatter(formatter)

logger.addHandler( fh )
logger.debug( "\n\nLogger initialized" )

# def new_affect(self, args=sys.argv[1:], output=True):
#     """Affect an SVG document with a callback effect"""
#     self.svg_file = args[-1]
#     self.getoptions(args)
#     self.parse()
#     self.getposinlayer()
#     self.getselected()
#     self.getdocids()
#     self.effect()
#     if output: self.output()

# フィルタ名でファイルをソートするときに使用する関数
# それっぽいキーワードにマッチしたらそのindexを数値で返す
# 基本的に透過光 -> 短い波長から という順で値が大きくなるようにしてソート関数で使えるようにする
# 複数のフィルタ名にマッチする場合(tagBFP-iRFP670)合算する?
# 方針としてはフィルタ名(一部keyword)にマッチしたら2^n を合算して返却することで同じ数値にならないようにする.
# BFP-iRFP670 と iRFP670-BFP は同じ値になる
def filter2index(filter_name):
    index = 0
    filters = ['C1', 'CH1', 'PH', 'BF', 'Transillumination', 'Bright_Field',
            'C2', 'CH2', 'DAPI', 'BFP',
            'C3', 'CH3', 'NIBA', 'FITC', 'GFP',
            'C4', 'CH4', 'WIGA', 'Cy3', 'tagRFP', 'mCherry',
            'C5', 'CH5', 'Cy5', 'iRFP670']
    filters = [".*" + s for s in filters]
    for fi, fname in enumerate(filters):
        logger.debug(str(fi) + ":検索文字列: " + fname + ", フィルタ名: " + filter_name)
        if re.match(fname, filter_name, flags=re.IGNORECASE):
            # logger.debug("fi=" + str(fi))
            index = index + 2 ** fi
            # logger.debug("fname=" + fname)
            # logger.debug("current index=" + str(index))
    logger.debug("filter_name=" + filter_name)
    logger.debug("index=" + str(index))
    return index



class ImageAlignEffect( inkex.Effect ):  # class 宣言の引数は継承
    """
    Either draw marks for alignment, or align images
    """
    # ファイルのwell数、フィルタ名などを見つけるための正規表現を先にコンパイルしておく.
    # ここはmap後にlistにしないとイテレーターとして扱われる。そうすると、一度取得した値を繰り返し使えないので、
    # list化を忘れずに
    file_regexps = list(map(lambda r: re.compile(r, flags=re.IGNORECASE), [
        '^.*?(?P<ROW>[A-Z])-(?P<COL>\d+)_fld_(?P<FLD>\d+)_wv_(?P<FLT>[^.]+).*$',
        '^.*?(?P<ROW>[A-Z])%20-%20(?P<COL>\d+)\(fld%20(?P<FLD>\d+)%20wv%20(?P<FLT>[^)]+).*$',
        '^.*?(?P<ROW>[A-Z])(?P<COL>\d+)-W\d+-P(?P<FLD>\d+)-Z\d+-T\d+-(?P<FLT>[^.]+)',
        '^.*?(?P<ROW>[A-Z])[-_]?(?P<COL>\d+)[-_].*[-_](?P<FLD>\d+)_w\d+(?P<FLT>BF|BFP|NIBA|WIGA|CY5)\..*$',
        '^.*?(?P<ROW>[A-Z])[-_]?(?P<COL>\d+).*_w\d+(?P<FLT>BF|BFP|NIBA|WIGA|CY5)\..*$',
        '^.*?(?P<ROW>\d+).*_w\d+(?P<FLT>BF|BFP|NIBA|WIGA|CY5)\..*$',
        '^.*?(?P<ROW>[A-Z])[-_]?(?P<COL>\d+)[-_]?F(?P<FLD>\d+).*(?P<FLT>C[\d-]+)\..*$',
        '^.*?(?P<ROW>[A-Z])(?P<COL>\d+)_\d+_(?P<FLD>\d+)_\d+_(?P<FLT>[^.]+)\..*$',
        '^(?P<ROW>[A-Z])(?P<COL>\d+)-(?P<FLT>[^.]+)\..*$',

        '^.*?(?P<ROW>[A-Z])(?P<COL>\d+)[-_].*',
        '^.*?(?P<ROW>[A-Z])(?P<COL>\d+)\.(jpg|tif|png).*'
            ]))

    def __init__( self ):
        """
        Setting stuff up
        """
        inkex.Effect.__init__( self )

        # desc = u'{0} [Args] [Options]\n'
        # self.parser = ArgumentParser(description=desc)

        self.arg_parser.add_argument( '-x', '--x', action = 'store',
          type = int, dest = 'x', default = '0',
          help = 'Top left x position.' )

        self.arg_parser.add_argument( '-y', '--y', action = 'store',
          type = int, dest = 'y', default = '0',
          help = 'Top left y position.' )

        self.arg_parser.add_argument( '-a', '--angle', action = 'store',
          type = int, dest = 'angle', default = '0',
          help = 'Rotate images.' )

        self.arg_parser.add_argument( '-d', '--direction', action = 'store',
          type = str, dest = 'direction', default = 'horizontal',
          help = 'Set direction of the each filter image.' )

        self.arg_parser.add_argument( '--filterspace', action = 'store',
          type = int, dest = 'filterspace', default = '2',
          help = 'Space between each filter' )

        self.arg_parser.add_argument(  '--fieldspace', action = 'store',
          type = int, dest = 'fieldspace', default = '5',
          help = 'Space between each field' )

        self.arg_parser.add_argument( '-v', '--vspace', action = 'store',
          type = int, dest = 'vspace', default = '5',
          help = 'Set vertical space between images.' )

        self.arg_parser.add_argument( '--hspace', action = 'store',
          type = int, dest = 'hspace', default = '5',
          help = 'Set horizontal space between images.' )

        self.arg_parser.add_argument( '-w', '--width', action = 'store',
          type = float, dest = 'width', default = '5',
          help = 'Scaling width of images.' )

        self.arg_parser.add_argument( '--stamp', action = 'store',
          type = inkex.Boolean, dest = 'stamp', default = False,
          help = 'Stamp layout paramters.' )

        self.arg_parser.add_argument( '--label', action = 'store',
          type = inkex.Boolean, dest = 'label', default = False,
          help = 'Stamp layout labels.' )

        self.arg_parser.add_argument("-s", "--selectedonly",
            action="store", type=inkex.Boolean, 
            dest="selectedonly", default=False,
            help="Align only selected images")
          # help = 'Id of object to be aligned (use groups for multipe)' )
        # parser.add_argument( '-n', '--markernum', action = 'store',
        #   type = 'int', dest = 'markernum', default = '1',
        #   help = 'No. of images to align' )
        self.arg_parser.parse_args()

    color_list = [ '#0000ff', '#00ff00', '#ff0000', '#ff00ff', '#00ffff', '#ffff00', '#8800ff', '#0088ff', '#00ff88', '#ff0088' ] 

    #  filt1  filt2  filt3
    # [img11, img12, img13,...]
    # [img21, -----, img23,...]
    # [img31, -----, img33,...]
    # または
    # filt1 [img11, img12, img13,...]
    # filt2 [img21, -----, img23,...]
    # filt3 [img31, -----, img33,...]
    # を並べる
    def align_images(self, images, x, y, width, height, filter_direction="horizontal"):
        col_keys = []
        for i, row in sorted(images.items()):
            col_keys.extend(row.keys())

        if filter_direction == "horizontal":
            # col_keys = sorted(set(col_keys), key=lambda x:filter2index[x])
            col_keys = sorted(set(col_keys), key=filter2index)
            row_keys = sorted(images.keys())
            xspace = self.options.filterspace
            yspace = self.options.fieldspace
        else:
            col_keys = sorted(set(col_keys))
            # row_keys = sorted(images.keys(), key=lambda x:filter2index[x])
            row_keys = sorted(images.keys(), key=filter2index)
            xspace = self.options.fieldspace
            yspace = self.options.filterspace


        for i, row_key in enumerate(row_keys):
            for j, col_key in enumerate(col_keys):
                # if images[row_key].has_key(col_key):
                if col_key in images[row_key]:
                    img_obj = images[row_key][col_key]
                    xpos = x + j * (width  + xspace)
                    ypos = y + i * (height + yspace)
                    xpos, ypos = self.transform_rotate_xy(xpos, ypos)
                    img_obj.set("x", str(xpos))
                    img_obj.set("y", str(ypos))

        # row_n = len(row_keys)
        # col_n = len(col_keys)
        # row_label = (y + row_n * (height + yspace))/2
        # col_label = (x + col_n * (height + yspace))/2
        # logger.debug("row_label=" + str(row_label))
        # logger.debug("row_keys=" + str(row_keys))
        # logger.debug("col_keys=" + str(col_keys))

    def transform_rotate_xy(self, x, y):
        transform_matrix = {
                0:    np.array([[ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]]),
                90:   np.array([[ 0, -1,  0], [ 1,  0,  0], [ 0,  0,  1]]),
                180:  np.array([[-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]]),
                270:  np.array([[ 0,  1,  0], [-1,  0,  0], [ 0,  0,  1]]),
                }
        mat = transform_matrix[self.options.angle]
        xy = np.array([x, y, 1])
        xy_t = xy.T
        new_xy = np.dot(np.linalg.inv(mat), xy_t)
        return new_xy[0], new_xy[1]


    def get_image_fname(self, img_obj):
        filename = None
        for attr in img_obj.keys():
            logger.debug("attr=" + attr)
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


    def categorize_images(self, image_ids, direction="horizontal"):
        max_image_width  = 0
        max_image_height = 0
        images = {}
        rows = []
        cols = []
        if self.options.angle == 90 or  self.options.angle == 270:
            vertical_param   = "width"
            horizontal_param = "height"
        else:
            vertical_param   = "height"
            horizontal_param = "width"
        for img_id in image_ids:
            img_obj = self.svg.getElementById(img_id)
            # if self.options.angle == 0 and img_obj.attrib.has_key("transform"):
            if self.options.angle == 0 and "transform" in img_obj.attrib:
                del img_obj.attrib["transform"]
            elif self.options.angle == 90:
                img_obj.set("transform", "matrix(0,1,-1,0,0,0)")
            elif self.options.angle == 180:
                img_obj.set("transform", "scale(-1,-1)")
            elif self.options.angle == 270:
                img_obj.set("transform", "matrix(0,-1,1,0,0,0)")
            image_width  = float(img_obj.get(horizontal_param))
            image_height = float(img_obj.get(vertical_param))
            if self.options.width != 0.0:
                new_width = self.options.width
                new_height = image_height * new_width / image_width
                img_obj.set(horizontal_param, str(new_width))
                img_obj.set(vertical_param, str(new_height))
                image_width = new_width
                image_height = new_height

            if max_image_width < image_width:
                max_image_width = image_width
            if max_image_height < image_height:
                max_image_height = image_height

            filename = self.get_image_fname(img_obj)

            if filename == None:
                continue

            # 正規表現でマッチング
            for reg in ImageAlignEffect.file_regexps:
                match_result = reg.match(filename)
                if match_result == None:
                    logger.debug("NOTMATCH: filename=" + str(filename) + "  REGEXP=" + str(reg))
                    continue
                logger.debug("MATCH: filename=" + str(filename) + "  REGEXP=" + str(reg))
                group_dict = match_result.groupdict()
                row = group_dict['ROW']          if 'ROW' in  group_dict else "A" 
                col = group_dict['COL'].zfill(2) if 'COL' in  group_dict else "01"
                fld = group_dict['FLD'].zfill(2) if 'FLD' in  group_dict else "01"
                wav = group_dict['FLT']          if 'FLT' in  group_dict else "Transillumination-Blank1"
                logger.debug("reg=" + str(reg))
                logger.debug("filename=" + str(filename))
                logger.debug("row=" + str(row))
                logger.debug("col=" + str(col))
                logger.debug("fld=" + str(fld))
                logger.debug("wav=" + str(wav))
                break



            # images[row] = {} if not images.has_key(row) else images[row]
            if row == None:
                continue
            images[row] = {} if not row in images else images[row]
            # images[row][col] = {} if not images[row].has_key(col) else images[row][col]
            if col == None:
                continue
            images[row][col] = {} if not col in images[row] else images[row][col]
            if fld == None or wav == None:
                continue
            if direction == "horizontal":
                # images[row][col][fld] = {} if not images[row][col].has_key(fld) else images[row][col][fld]
                images[row][col][fld] = {} if not fld in images[row][col] else images[row][col][fld]
                images[row][col][fld][wav] = img_obj
            else:
                # images[row][col][wav] = {} if not images[row][col].has_key(wav) else images[row][col][wav]
                images[row][col][wav] = {} if not wav in images[row][col] else images[row][col][wav]
                images[row][col][wav][fld] = img_obj

        # logger.debug("images = " + pp.pformat(images))
        return images, max_image_width, max_image_height

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
    #      filter1, filter2, filter3,...
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
    #      1,  2,  3  ...
    # A: [A1, A2, A3, ...]
    # B: [B1, B2, B3, ...]
    # C: [C1, C2, C3, ...]
    # 各 A1, A2, A3.. が
    #      1,  2,  3  ...
    # a: [a1, a2, a3, ...]
    # b: [b1, b2, b3, ...]
    # c: [c1, c2, c3, ...]
    # という構造をとっている.
    # なので、例えば2番目の横方向の幅を調べる場合は、
    # A2,B2,C2 について、len(a), len(b), len(c)を調べていけば良い
    #
    # 一方B番目の縦方向の幅を調べるには
    # B1, B2, B3 について、len(B1), len(B2), len(B3)を調べていけば良い
    def get_vertical_max_width(self, table, ci):
        max_n = 0
        # logger.debug("ci = " + str(ci))
        for i in table.keys(): # 行のindex
            # if table[i].has_key(ci):
            if ci in table[i]:
                for j in table[i][ci].keys(): # 行のindex
                    if max_n < len(table[i][ci][j]): # 行の長さ
                        max_n = len(table[i][ci][j])
        return max_n

    def get_vertical_max_width_table(self, table):
        col2hnum = {}
        for i in range(1, 25):
            col2hnum[str(i).zfill(2)] = self.get_vertical_max_width(table, str(i).zfill(2))
        return col2hnum

    def get_horizontal_max_height(self, table, ri):
        max_n = 0
        # if not table.has_key(ri):
        if not ri in table:
            return max_n
        for i in table[ri].keys(): # 列のindex
            if max_n < len(table[ri][i]):
                max_n = len(table[ri][i])
        return max_n


    def get_horizontal_max_height_table(self, table):
        row2vnum = {}
        for i in range(65, 89):  # A-Z まで
            row2vnum[chr(i)] = self.get_horizontal_max_height(table, chr(i))
        return row2vnum


    # 多分このeffect()が affect() で呼ばれるんだと思う.
    def effect( self ):

        if self.options.selectedonly:
            rp = re.compile('text\d+')
            text_ids = self.options.ids
            text_ids = [x for x in text_ids if rp.match(x)]
            for text_id in text_ids:
                # text_obj = self.getElementById(text_id)
                text_obj = self.svg.getElementById(text_id)
                if self.options.label and text_obj.get(inkex.addNS('label', 'inkscape')) == 'plate_align_text':
                    text_obj.getparent().remove(text_obj)
                if self.options.stamp and text_obj.get(inkex.addNS('label', 'inkscape')) == 'plate_align_stamp':
                    text_obj.getparent().remove(text_obj)
        else:
            if self.options.label:
                # 最初に inkscape:label が plate_align_text のやつを削除する
                # text.set(inkex.addNS('label', 'inkscape'), 'plate_align_text')
                for _text in self.document.getroot().xpath('//svg:text', namespaces=inkex.NSS):
                    if _text.get(inkex.addNS('label', 'inkscape')) == 'plate_align_text':
                        _text.getparent().remove(_text)
                        # self.document.getroot().remove(_text)

            if self.options.stamp:
                for _text in self.document.getroot().xpath('//svg:text', namespaces=inkex.NSS):
                    if _text.get(inkex.addNS('label', 'inkscape')) == 'plate_align_stamp':
                        _text.getparent().remove(_text)
                        # self.document.getroot().remove(_text)

        if self.options.selectedonly:
            rp = re.compile('image\d+')
            image_ids = self.options.ids
            image_ids = [x for x in image_ids if rp.match(x)]
        else:
            image_ids = map(lambda x: x.get("id"), self.document.getroot().xpath('//svg:image', namespaces=inkex.NSS))


        images, max_image_width, max_image_height = self.categorize_images(image_ids, direction=self.options.direction)

        for row in range(65, 89):
            chr(row)



        row2vnum = self.get_horizontal_max_height_table(images)
        col2hnum = self.get_vertical_max_width_table(images)
        # logger.debug("row2vnum = " + pp.pformat(row2vnum))
        # logger.debug("col2hnum = " + pp.pformat(col2hnum))

        x_coeffi = {
                0:  0,
                90: 1,
                180: 1,
                270: 0
                }
        y_coeffi = {
                0:  0,
                90: 0,
                180: 1,
                270: 1
                }

        # 後で印字するようの配列
        row_label_list = {}
        row_label_x = float('inf')
        col_label_list = {}
        col_label_y = float('inf')


        # 左上から並べていくんだが左上を(0,0)として
        # row, col, fld, wav それぞれにインデックスをつけて並べていくのが良い
        # row_image_count = -row2vnum["A"]  # この数字は実際の画像の数(画像間intervalなどのため)
        # row_chunk_count = -1              # この数字は実際にデータが存在した行をどれぐらい処理したかカウントするため(行間インターバルの個数)
        row_image_count = 0
        row_chunk_count = 0
        current_y = 0
        # for rowi, row in sorted(images.items()):
        for i, row_key in enumerate(sorted(row2vnum.keys())):
            if row2vnum[row_key] < 1:
                continue


            # col_image_count = -col2hnum["01"]
            # col_chunk_count = -1
            col_image_count = 0
            col_chunk_count = 0
            current_x = 0
            for j, col_key in enumerate(sorted(col2hnum.keys())):
                if col2hnum[col_key] < 1:
                    continue

                # if images[row_key].has_key(col_key):
                if col_key in images[row_key]:
                    img_table = images[row_key][col_key]
                    # x は col_image_count * width 
                    if self.options.direction == "horizontal":
                        x = self.options.x + max_image_width * x_coeffi[self.options.angle] + col_image_count * (max_image_width  + self.options.filterspace) + col_chunk_count * (self.options.hspace - self.options.filterspace)
                        y = self.options.y + max_image_height * y_coeffi[self.options.angle] + row_image_count * (max_image_height + self.options.fieldspace)  + row_chunk_count * (self.options.vspace - self.options.fieldspace)
                    else:
                        x = self.options.x + max_image_width * x_coeffi[self.options.angle] + col_image_count * (max_image_width  + self.options.fieldspace)  + col_chunk_count * (self.options.hspace - self.options.fieldspace)
                        y = self.options.y + max_image_height * y_coeffi[self.options.angle] + row_image_count * (max_image_height + self.options.filterspace) + row_chunk_count * (self.options.vspace - self.options.filterspace)
                    self.align_images(img_table, x, y, max_image_width, max_image_height, filter_direction=self.options.direction)

                    text_x = x - max_image_width * x_coeffi[self.options.angle]
                    text_y = y - max_image_height * y_coeffi[self.options.angle]
                    
                    # if not col_label_list.has_key(col_key):
                    if not col_key in col_label_list:
                        # text = inkex.etree.Element(inkex.addNS('text', 'svg'))
                        text = etree.Element(inkex.addNS('text', 'svg'))
                        text.text = col_key
                        text.set('x', str(text_x + col2hnum[col_key] * max_image_width/2 - 8 ))
                        if col_label_y > text_y:
                            col_label_y = text_y
                        # text.set('y', str(y-10))
                        text.set(inkex.addNS('label', 'inkscape'), 'plate_align_text')
                        style = {
                                'stroke': 'none',
                                'stroke-width' : '1',
                                'fill' : '#000000',
                                'font-family': 'Arial',
                                'font-weight': 'normal',
                                'font-style': 'normal',
                                'font-strech': 'normal',
                                'font-variant': 'normal',
                                'font-size': '16px' #
                                }
                        # text.set('style', simplestyle.formatStyle(style))
                        text.set('style', str(inkex.Style(style)))
                        col_label_list[col_key] = text

                    # if not row_label_list.has_key(row_key):
                    if not row_key in row_label_list:
                        # text = inkex.etree.Element(inkex.addNS('text', 'svg'))
                        text = etree.Element(inkex.addNS('text', 'svg'))
                        text.text = row_key
                        if row_label_x > text_x:
                            row_label_x = text_x
                        # text.set('x', str(x-16))
                        text.set('y', str(text_y + row2vnum[row_key] * max_image_height/2 + 8))
                        text.set(inkex.addNS('label', 'inkscape'), 'plate_align_text')
                        style = {
                                'stroke': 'none',
                                'stroke-width' : '1',
                                'fill' : '#000000',
                                'font-family': 'Arial',
                                'font-weight': 'normal',
                                'font-style': 'normal',
                                'font-strech': 'normal',
                                'font-variant': 'normal',
                                'font-size': '16px' #
                                }
                        # text.set('style', simplestyle.formatStyle(style))
                        text.set('style', str(inkex.Style(style)))
                        row_label_list[row_key] = text

                col_image_count += col2hnum[col_key]
                col_chunk_count += 1

            row_image_count += row2vnum[row_key]
            row_chunk_count += 1
        # inkact = synfig_prepare.InkscapeActionGroup()
        # inkact.select_id(self.options.ids)
        # inkact.run_document()

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


        if self.options.stamp: # 並び方のパラメータを印字
            # parent = self.current_layer
            parent = self.svg.get_current_layer()
            # text = inkex.etree.Element(inkex.addNS('text', 'svg'))
            text = etree.Element(inkex.addNS('text', 'svg'))
            text.text = "x" + str(self.options.x) + "y" + str(self.options.y) + "w" + str(self.options.width) + "hs" + str(self.options.hspace) + "vs" + str(self.options.vspace)
            text.set('x', str(self.options.x + 2))
            text.set('y', str(self.options.y - 26)) # font が10pt だから10pt + 2 かな?
            text.set(inkex.addNS('label', 'inkscape'), 'plate_align_stamp')
            style = {
                    'stroke': 'none',
                    'stroke-width' : '1',
                    'fill' : '#000000',
                    'font-family': 'Arial',
                    'font-weight': 'normal',
                    'font-style': 'normal',
                    'font-strech': 'normal',
                    'font-variant': 'normal',
                    'font-size': '12.5px' # 10pt
                    }
            # text.set('style', simplestyle.formatStyle(style))
            text.set('style', str(inkex.Style(style)))

            parent.append(text)


# Create effect instance and apply it.
# logger.debug(len(sys.argv))

if len(sys.argv) == 1:
    # sys.argv = [ './platealign.py', '--angle=0', '--direction=horizontal', '--hspace=10', '--vspace=20', '--width=384', '/home/yfujita/work/bin/python/inkscape/platealign/test.svg' ]
    sys.argv = [ './platealign3.py', '--id=image4757', '--angle=90', '--direction=vertical', '--hspace=10', '--vspace=20', '--filterspace=2', '--fieldspace=5', '--width=384', '/home/yfujita/work/bin/python/inkscape/platealign/test.svg' ]

# logger.debug( "Started with: {}.".format( str( sys.argv ) ) )
effect = ImageAlignEffect()
# effect.affect(args=sys.argv)
effect.run()
