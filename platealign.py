# -*- coding: utf-8 -*-
'''
Copyright (C) 2017 Matyas Kocsis, matyilona@openmailbox.org

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

Extension for aligning multiple bitmap images, by selecting the same points on them.
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

from lxml import etree
import re
import pprint

pp = pprint.PrettyPrinter(indent=2)

#a logger for debugging/warnings
logger = logging.getLogger( "platealign" )
# logger.setLevel( 'WARNING' )
logger.setLevel( logging.DEBUG )
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# sh = logging.StreamHandler( sys.stderr )
# fh = logging.FileHandler(filename='/home/yfujita/work/bin/python/inkscape/platealign/platealign.log', mode='w')
fh = logging.FileHandler(filename='/home/yfujita/work/bin/python/inkscape/platealign/platealign.log', mode='a')
fh.setLevel( logging.DEBUG )
fh.setFormatter(formatter)

logger.addHandler( fh )
logger.debug( "Logger initialized" )

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

class ImageAlignEffect( inkex.Effect ):
    """
    Either draw marks for alignment, or align images
    """
    def __init__( self ):
        """
        Setting stuff up
        """
        inkex.Effect.__init__( self )

        self.OptionParser.add_option( '-a', '--angle', action = 'store',
          type = 'int', dest = 'angle', default = '0',
          help = 'Rotate images.' )
        self.OptionParser.add_option( '-d', '--direction', action = 'store',
          type = 'string', dest = 'direction', default = 'horizontal',
          help = 'Set direction of the each filter image.' )
        self.OptionParser.add_option("-s", "--selectedonly",
            action="store", type="inkbool", 
            dest="selectedonly", default=False,
            help="Align only selected images")
          # help = 'Id of object to be aligned (use groups for multipe)' )
        # self.OptionParser.add_option( '-n', '--markernum', action = 'store',
        #   type = 'int', dest = 'markernum', default = '1',
        #   help = 'No. of images to align' )

    color_list = [ '#0000ff', '#00ff00', '#ff0000', '#ff00ff', '#00ffff', '#ffff00', '#8800ff', '#0088ff', '#00ff88', '#ff0088' ] 

    # 多分このeffect()が affect() で呼ばれるんだと思う.
    def effect( self ):

        if self.options.selectedonly:
            image_ids = self.options.ids
        else:
            image_ids = map(lambda x: x.get("id"), self.document.getroot().xpath('//svg:image', namespaces=inkex.NSS))

        # self.document=document #not that nice... oh well
        # path = '//svg:image'
        # for node in self.document.getroot().xpath(path, namespaces=inkex.NSS):
        #     self.embedImage(node)
        logger.debug("image_ids = " + str(image_ids))

        images = {}

        image_array = {}

        for img_id in image_ids:
            img_obj = self.getElementById(img_id)
            # img_obj.set("x", "0")
            # logger.debug(str(etree.tostring(img_obj)))
            # logger.debug(img_obj.get("id"))
            # logger.debug("inkex.NSS = " + str(inkex.NSS))
            for attr in img_obj.keys():
                # logger.debug("attr = " + attr)
                if re.search("href$", attr):
                    # logger.debug("MATCH attr = " + attr)
                    # filename = re.sub("^[^/]", "", img_obj.get(attr))
                    filename = img_obj.get(attr).split("/")[-1]
                    break

            # logger.debug(str(img_obj.attrib))
            # logger.debug(str(img_obj.get("{http://www.w3.org/1999/xlink}href")))
            # filename = re.sub("[^/]+", "", img_obj.get("xlink:href"))
            if filename == None:
                continue

            matchObj = re.match("([A-Z])-(\d+)_fld_(\d+)_wv_([^.]+)", filename)
            row = matchObj.group(1)
            col = matchObj.group(2)
            # rowi = ord(row) - 65
            # coli = int(col) - 1
            fld = matchObj.group(3).zfill(2)
            # fldi = int(fld) - 1
            wav = matchObj.group(4)
            # logger.debug("row = " + row)
            # logger.debug("col = " + col)
            # logger.debug("rowi= " + str(rowi))
            # logger.debug("coli= " + str(coli))
            # logger.debug("fld = " + fld)
            # logger.debug("wav = " + wav)
            images[row] = {} if images.has_key(row) is not None else images[row]
            images[row][col] = {} if images[row].has_key(col) is not None else images[row][col]
            images[row][col][fld] = {} if images[row][col].has_key(fld) is not None else images[row][col][fildi]
            # images[row][col][fld][wav] = img_obj
            images[row][col][fld][wav] = filename

        logger.debug(pp.pformat(images))

        for rowi, row in sorted(images.items()):
            # row = images[rowi]
            for coli, col in sorted(row.items()):
                # col = images[rowi][coli]
                for fldi, fld in sorted(col.items()):
                    # fld = images[rowi][coli][fldi]
                    for wav, fname in sorted(fld.items()):
                        logger.debug("wav = " + wav)
                        logger.debug("filename = " + fname)




        # logger.debug(self.options.angle)
        # logger.debug(self.options.direction)
        # logger.debug("self.selectedid = " + str(self.options.ids))
        # logger.debug("self.getdocids() = " + str(self.getdocids()))
        # logger.debug("doc_ids = " + str(self.doc_ids))
        # logger.debug("getElementById(\"image4207\") = " + str(self.getElementById("image4207")))
        # img_obj = self.getElementById("image4207")
        # logger.debug(etree.tostring(img_obj))
        # logger.debug(str(img_obj.get("x")))
        # img_obj.set("x", "0")
        # logger.debug(str(img_obj.get("x")))
        # u = simpletransform.composeParents( img, simpletransform.parseTransform( None ) ) + [[ 0, 0, 1 ]]

    # def getMarker( self, ind1, ind2 ):
    #     """
    #     Make a marker ind1: index of image to align, ind2: number of marker
    #     every marker is a group with an arrow and a label
    #     """
    #     #create a group element to place the text and arrow in
    #     group = inkex.etree.Element( 'g' )
    #     group.set( 'id', 'alignmarkgroup%i%i' % ( ind1, ind2 ) )
    #     #style for both the arrow and the label
    #     style = { 'stroke' : 'none', 'stroke-width' : '0', 'fill-opacity' : '.5', 'fill' : self.color_list[ ind1 ] }
    #     style = simplestyle.formatStyle( style )
    #     #path for the arrow
    #     arrow_path = 'm 0,0 0,5 5,0 z'
    #     inkex.etree.SubElement( group, 'path', { 'style' : style, 'd' : arrow_path } ).set( 'id', 'alignmark%i%i' % ( ind1, ind2 ) )
    #     inkex.etree.SubElement( group, 'text', { 'style' : style } ).text = str( ind2 )
    #     return( group )
    #
    # def cleanup( self ):
    #     """
    #     Delet previous align layers and markers
    #     """
    #     la = self.getElementById( "alignlayera" )
    #     lb = self.getElementById( "alignlayerb" )
    #     if la is not None:
    #         la.getparent().remove(la)
    #     if lb is not None:
    #         lb.getparent().remove(lb)
    #
    #     #some marks might have been moved to other layers
    #     for i in range( 3 ):
    #         for j in range( 1, self.options.markernum + 1 ):
    #             marker_group = self.getElementById( "alignmarkgroup%i%i" % (i, j) )
    #             if marker_group is not None:
    #                 marker_group.getparent().remove( marker_group )
    #
    #
    # def putMarkers( self ):
    #     """
    #     Put 3 markers on svg, for every image, and 3 for target
    #     """
    #     #delet old marks
    #     self.cleanup()
    #     #set up layers for markers
    #     svg = self.document.getroot()
    #     layera = inkex.etree.SubElement( svg, "g" )
    #     layera.set( 'id', 'alignlayera' )
    #     layera.set( inkex.addNS( "label", "inkscape" ), "Align Layer A" )
    #     layera.set( inkex.addNS( "groupmode", "inkscape" ), "layer" )
    #     layerb = inkex.etree.SubElement( svg, "g" )
    #     layerb.set( 'id', 'alignlayerb' )
    #     layerb.set( inkex.addNS( "label", "inkscape" ), "Align Layer B" )
    #     layerb.set( inkex.addNS( "groupmode", "inkscape" ), "layer" )
    #     #draw markers
    #     for i in range( 3 ):
    #         ma = self.getMarker( 0, i )
    #         ma.set( 'transform', 'translate(%i  0)' % 20 * i )
    #         layera.append( ma )
    #         for j in range( 1, self.options.markernum + 1 ):
    #             mb = self.getMarker( j, i )
    #             mb.set( 'transform', 'translate(%i %i)' % ( i*20, j*30 ) )
    #             layerb.append( mb )
    #
    # def alignImage( self ):
    #     """
    #     Align image based on the real backtransformed coords of the markers
    #     """
    #     points = { }
    #     for i in range( self.options.markernum + 1):
    #         points[ i ] = [ ]
    #         for j in range( 3 ):
    #             r = self.getElementById( 'alignmark%i%i' % ( i, j ) )
    #             #get the point marker points to from boundingbox's corner
    #             if r is not None:
    #                 bb = simpletransform.computeBBox( [r] )
    #             else:
    #                 raise( BaseException( "Can't find alignmark%i%i" % ( i, j ) ) )
    #             x = bb[0]
    #             y = bb[3]
    #             #use 3x3 matrix convention, numpy wants that, simleTransform honeybadger
    #             point = [ x, y, 1 ]
    #             #get transforms on point
    #             a = simpletransform.composeParents( r, simpletransform.parseTransform( None ) ) 
    #             #apply to stored coords not real etree object
    #             simpletransform.applyTransformToPoint( a , point )
    #             points[i].append( point )
    #        
    #         if i != 0:
    #             #get 3x3 numpy transform matrix
    #             a = np.matrix( points[ 0 ], dtype = float ).getT()
    #             b = np.matrix( points[ i ], dtype = float ).getT()
    #             t = ( a * b.getI() )
    #             for j in range( 3 ):
    #                 r = self.getElementById( 'alignmarkgroup%i%i' % ( i, j ) )
    #                 #transform markes 
    #                 simpletransform.applyTransformToNode( t.tolist(), r )
    #             #get the image we're working on
    #             img = self.getElementById( self.options.alignid + str( i ) )
    #             if img is not None:
    #                 #parents transform will always get apply after nodes own, but t should be applied to already transformed coordinates
    #                 u = simpletransform.composeParents( img, simpletransform.parseTransform( None ) ) + [[ 0, 0, 1 ]]
    #                 u = np.matrix( u, dtype = float )
    #                 v =  u.getI() * t * u
    #                 simpletransform.applyTransformToNode( v.tolist() , img )
    #             else:
    #                 logger.warning( "Could not find %s" % self.options.alignid + str( i ) )
    #
    # def effect( self ):
    #     """
    #     Effect behaviour.
    #     Overrides base class method.
    #     """
    #     logger.debug( "Effecting started" )
    #     if self.options.start == 'start':
    #         self.putMarkers( )
    #     else:
    #         self.alignImage( )



# Create effect instance and apply it.
logger.debug(len(sys.argv))
# if len(sys.argv) == 1:
#     sys.argv = [ './platealign.py', '--angle=0', '--direction=horizontal', '/home/yfujita/work/bin/python/inkscape/platealign/test.svg' ]
    # inkex.Effect.svg_file = '/home/yfujita/work/bin/python/inkscape/platealign/test.svg'





if len(sys.argv) == 1:
    sys.argv = [ './platealign.py', '--angle=0', '--direction=horizontal', '/home/yfujita/work/bin/python/inkscape/platealign/test.svg' ]

logger.debug( "Started with: {}.".format( str( sys.argv ) ) )
effect = ImageAlignEffect()
# effect.affect = new_affect
effect.affect(args=sys.argv)
