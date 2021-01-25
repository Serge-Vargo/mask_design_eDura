''' Starting with Sympy for polygon generation and dxf drawing '''

import functools

import sympy as sp
import numpy as np

from dxfwrite import DXFEngine as dxf

'''
Code to generate 2D & 3D models,
Assumptions:
    - Dimensions are in microns
    - Cartesian x-y coordinate system (2D) |__
    - Right-hand x-y-z coordinate system (3D) |.__
'''

nslice = 2000   #differential length

def rectangle(edge_center = (0,0), w = 30, l = 10):
    x,y = edge_center
    vertices = [(x-w/2,y),(x+w/2,y),(x+w/2,y+l),(x-w/2,y+l)]
    print("vertices")
    print(type(vertices))
    print(type(vertices[0]))
    print(vertices)
    return sp.geometry.Polygon(*vertices)

def polygon_to_polyline(poly, layer):
    points = [(v.x,v.y) for v in poly.vertices]
    pl = dxf.polyline(points = points, layer=layer)
    pl.close()
    return pl

# def points_polyline():

# def array_electrodes(poly, pitch)

def MaskDesign(w = 10, l = 10): #Label parameters
    label = f"W{w}L{l}"
    drawing = dxf.drawing(f"init/{label}.dxf")

    WG_LAYER = "waveguide"

    drawing.add(dxf.text(label, insert =(-50,-550), height = 40, layer = WG_LAYER))
    device = rectangle((0,0),30,30)
    print("device")
    print(type(device))
    print(device)

    polyline = polygon_to_polyline(device, WG_LAYER)
    print("polyline")
    print(type(polyline))
    print(polyline)

    drawing.add(polyline)

    print("Exporting...")
    drawing.save()
    print("Done")
    return drawing



rect = rectangle((5,5), 30,30)
print(type(rect))
print(rect)
print(rect.area)

MaskDesign()
