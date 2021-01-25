''' Generate the mask for a 32 electrode NHP eDura '''

import functools

import ezdxf as dxf
import numpy as np
import sd_utils_v1_2 as ed

'''
Code to generate 2D & 3D models,
Assumptions:
    - Dimensions are in microns
    - Cartesian x-y coordinate system (2D) |__
'''
# ==== Workspace variables ====================================================
# File parameters
FILE_ROOT = 'gen_v0_256el\eDura_v0_NHP_'
# Layers
WAFER_INT = 'wafer_int'     # internal outline wafer layer
WAFER_EXT = 'wafer_ext'     # external outline wafer layer
EDURA_INT = 'eDura_outline_int'
EDURA_EXT = 'eDura_outline_ext'
METAL = "metal"
ETCH = "etch"

# Parameters
# 4-in wafer
D_WAFER = 100000   #(um) diameter of 4 in wafer
center = [0,0]     #() center of wafer
FLAT_B = 32500     #(um) bottom flat length of the wafer
FLAT_T = 18000     #(um) top flat length of the wafer
GAP_WAFER = 5000   #(um) gap of wafer outline for mask (aligner)

# eDura outline
W_A = 10000        #(um) width of arm
L_A = 20000        #(um) length of arm  + clearance for the ZIF connector
D_B = 20000        #(um) diameter of body
GAP_EDURA = 800    #(um) thickness of eDura outline (PDMS cut tolerance pruposes)

# electrodes
N_TRODES = 256     #() number of electrodes
TRODE_PITCH = 500  #(um) eletrode pitch
TRODE_DIAM = 30    #(um) diameter of electrode
# TRODE_DIAM_METAL = TRODE_DIAM + 10      #(um) gap between via and metal
TRODE_DIAM_METAL = 50      #(um) gap between via and metal

# traces
TRACE_PITCH = 70  #(um) trace pitch
TRACE_WIDTH = 10   #(um) trace pitch
# body_trace_spacing = 80
body_trace_spacing = None # for uniform pitch

# bond pads - ACF
PAD_WIDTH = 350    #(um) pad width
PAD_HEIGHT = 350   #(um) pad height
GAP_ETCH = 10
PAD_SPACE_X = 350  #(um) pad space X
PAD_SPACE_Y = 350 + 375 #(um) pad space Y
EDGE_TOL = 1000    #(um) pad edge tolerance
MEET_X = 15000     #(um) where pads and electrodes meet
N_PAD_Y = 8        #() number of bond pads in y-direction on each side
N_PAD_X = (N_TRODES//2) // N_PAD_Y #() number of bond pads in x-direction on each side

# arbitrary in case of odd number of electrodes
# N_PAD_Y = 6
# N_PAD_X = 4

# # bond pads - UW compatible =========================
# PAD_WIDTH = 1250    #(um) pad width
# PAD_HEIGHT = 1250   #(um) pad height
# PAD_SPACE_X = 1290  #(um) pad space X
# PAD_SPACE_Y = 3720  #(um) pad space Y
# EDGE_TOL = 1000     #(um) pad edge tolerance
# MEET_X = 15000     #(um) where pads and electrodes meet
# N_PAD_Y = 2        #() number of bond pads in y-direction on each side
# N_PAD_X = (N_TRODES//2) // N_PAD_Y #() number of bond pads in x-direction on each side
# # arbitrary in case of odd number of electrodes
# # N_PAD_Y = 6
# # N_PAD_X = 4


# ==== Create new Mask File & Layers ==========================================
mask = dxf.new()
msp = mask.modelspace()

mask.layers.new(WAFER_INT, dxfattribs={'color': 8})
mask.layers.new(WAFER_EXT, dxfattribs={'color': 8})
mask.layers.new(EDURA_INT, dxfattribs={'color': 6})
mask.layers.new(EDURA_EXT, dxfattribs={'color': 6})
mask.layers.new(METAL, dxfattribs={'color': 4})
mask.layers.new(ETCH, dxfattribs={'color': 3})

# ==== Build Layers ===========================================================
''' Wafer Outline '''
wafer_points, wafer_points_gap = ed.wafer_outline_points(D_WAFER, FLAT_B, FLAT_T, GAP_WAFER)
msp.add_lwpolyline (wafer_points, 'xyseb', {'closed':'True', 'layer': WAFER_INT})
msp.add_lwpolyline (wafer_points_gap, 'xyseb', {'closed':'True', 'layer': WAFER_EXT})

''' eDura Outline '''
top_right_intersection = ed.arm_body_intersection(D_B, W_A)       # 1st vertice of right arm
outline_points = ed.edura_outline_points(top_right_intersection, D_B, L_A, W_A)
msp.add_lwpolyline(outline_points, 'xyseb', {'closed':'True', 'layer': EDURA_INT})

# Add gap
top_r_inter_gap = ed.arm_body_intersection(D_B + GAP_EDURA, W_A + GAP_EDURA)
outline_points_gap = ed.edura_outline_points(top_r_inter_gap, D_B + GAP_EDURA,
                                             L_A + GAP_EDURA/4, W_A + GAP_EDURA)
                                                   # gap/4 is to adjust side edges spacing
msp.add_lwpolyline(outline_points_gap, 'xyseb', {'closed':'True', 'layer': EDURA_EXT})


''' Electrodes '''
points, p_x, p_y, n_trodes = ed.points_hex_lattice_hex_grid(N_TRODES, center,
                                                            TRODE_PITCH, D_B)

for row in points:
    # print("\n points in row:\n", row)
    for pt in row:
        msp.add_circle(pt, TRODE_DIAM_METAL/2, {'layer': METAL})
        msp.add_circle(pt, TRODE_DIAM/2, {'layer': ETCH})

print("\nNumber of total electrodes:", n_trodes)

''' Traces of Electrodes '''
# Body (from electrodes outwards) ==============================================
diag, hor, end_points = ed.traces_electrodes_out(points, p_x, p_y, TRACE_WIDTH,
                           TRODE_DIAM_METAL, body_trace_spacing)
for row in diag:
    # print("\n Diagonal trace points in row:\n", row)
    for trace in row:
        msp.add_lwpolyline(trace, 'xyseb', {'layer': METAL})

for row in hor:
    # print("\n Horizontal trace points in row:\n", row)
    for trace in row:
        msp.add_lwpolyline(trace, 'xyseb', {'layer': METAL})

# for idx, row in enumerate(end_points):
#     print("\n Row: ", idx, "\n End points in row:")
#     for pt in row:
#         print(pt)

# ======= (from body to arms (finish-up)) ======================================
# Run 1
hor_traces, dia_traces, end_points = ed.traces_center_to_arms(end_points, TRACE_PITCH,
                                        TRACE_WIDTH, p_y, p_x, body_trace_spacing)

for idx, row in enumerate(hor_traces):
    print("\n Horizontal Traces: ", len(row))
    for trace in row:
        msp.add_lwpolyline(trace, 'xyseb', {'layer': METAL})
        # print(trace)

for idx, row in enumerate(dia_traces):
    print("\n Diagonal Traces: ", len(row))
    for trace in row:
        msp.add_lwpolyline(trace, 'xyseb', {'layer': METAL})
        # print(trace)

# Run 2 - Stair case
hor_traces, diag_traces = ed.traces_stair_case_to_arms(end_points, TRACE_PITCH,
                                                      TRACE_WIDTH, p_y, p_x)

for idx, row in enumerate(hor_traces):
    print("\n Horizontal Traces - Stair Case: ", len(row))
    for trace in row:
        msp.add_lwpolyline(trace, 'xyseb', {'layer': METAL})
        # print(trace)

for idx, row in enumerate(diag_traces):
    print("\n Diagonal Traces - Stair Case: ", len(row))
    for trace in row:
        msp.add_lwpolyline(trace, 'xyseb', {'layer': METAL})
        # print(trace)

''' Bond Pads '''
pts_pads_r, pts_pads_l = ed.bond_pads_points(N_PAD_X, N_PAD_Y, PAD_WIDTH, PAD_HEIGHT,
                                             PAD_SPACE_X, PAD_SPACE_Y,
                                             top_right_intersection, L_A, W_A, EDGE_TOL)

for idx_row, row in enumerate(pts_pads_r):
    for idx, pt in enumerate(row):
        p_r = ed.square_ver(pts_pads_r[idx_row][idx], PAD_WIDTH + GAP_ETCH, PAD_HEIGHT + GAP_ETCH)
        p_l = ed.square_ver(pts_pads_l[idx_row][idx], PAD_WIDTH + GAP_ETCH, PAD_HEIGHT + GAP_ETCH)

        msp.add_polyline2d(p_r, {'closed':'True','layer': METAL})
        msp.add_polyline2d(p_l, {'closed':'True','layer': METAL})

        p_r_via = ed.square_ver(pts_pads_r[idx_row][idx], PAD_WIDTH, PAD_HEIGHT)
        p_l_via = ed.square_ver(pts_pads_l[idx_row][idx], PAD_WIDTH, PAD_HEIGHT)

        msp.add_polyline2d(p_r_via, {'closed':'True','layer': ETCH})
        msp.add_polyline2d(p_l_via, {'closed':'True','layer': ETCH})

''' Bond Pad traces '''
diag_r, hor_r = ed.bond_pads_traces(pts_pads_r, TRACE_PITCH, TRACE_WIDTH, N_PAD_X, N_PAD_Y,
                                    PAD_WIDTH, PAD_HEIGHT, PAD_SPACE_X, PAD_SPACE_Y, MEET_X)

for trace in diag_r:
    msp.add_lwpolyline(trace, 'xyseb', {'layer': METAL})
for trace in hor_r:
    msp.add_lwpolyline(trace, 'xyseb', {'layer': METAL})

diag_l, hor_l = ed.bond_pads_traces(pts_pads_l, TRACE_PITCH, TRACE_WIDTH, N_PAD_X, N_PAD_Y,
                                    PAD_WIDTH, PAD_HEIGHT, PAD_SPACE_X, PAD_SPACE_Y, MEET_X)

for trace in diag_l:
    msp.add_lwpolyline(trace, 'xyseb', {'layer': METAL})
for trace in hor_l:
    msp.add_lwpolyline(trace, 'xyseb', {'layer': METAL})

''' Output Mask '''
file_name = FILE_ROOT + f"ne{N_TRODES}diam{TRODE_DIAM}w{W_A//1000}l{L_A//1000}.dxf"
# file_name = FILE_ROOT + f"ne{N_TRODES}pch{p_x}w{W_A//1000}l{L_A//1000}.dxf"
ed.mask_out(mask, file_name)
