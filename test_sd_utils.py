''' Test Code for sd_utils.py '''
import numpy as np
import sd_utils as ed
import ezdxf as dxf

# ==== Create new Mask File & Layers ==========================================
mask = dxf.new()
msp = mask.modelspace()

''' Test wafer_outline_points() '''
D_WAFER = 100000    #(um) wafer diamter
gap_wafer = 5000    #(um) wafer outline thickness
WAFER_INT = 'wafer_int'
WAFER_EXT = 'wafer_ext'
mask.layers.new(WAFER_INT, dxfattribs={'color': 8})
mask.layers.new(WAFER_EXT, dxfattribs={'color': 8})

wafer_points, wafer_points_gap = ed.wafer_outline_points()
msp.add_lwpolyline (wafer_points, 'xyseb', {'closed':'True', 'layer': WAFER_INT})
msp.add_lwpolyline (wafer_points_gap, 'xyseb', {'closed':'True', 'layer': WAFER_EXT})

file_name = "test\\t1_wafer_test.dxf"
ed.mask_out(mask, file_name)

#==============================================================================

''' Test edura_outline_points() '''
D_B = 20000 #(um) diameter of body of eDura
W_A = 10000 #(um) width of arm
H_A = 20000 #(um) length of arm
EDURA_INT = 'eDura_outline_int'
EDURA_EXT = 'eDura_outline_ext'
mask.layers.new(EDURA_INT, dxfattribs={'color': 6})
mask.layers.new(EDURA_EXT, dxfattribs={'color': 6})

top_right_intersection = ed.arm_body_intersection(D_B, W_A)       # 1st vertice of right arm
outline_points = ed.edura_outline_points(top_right_intersection, D_B, H_A, W_A)
msp.add_lwpolyline(outline_points, 'xyseb', {'closed':'True', 'layer': EDURA_INT})
# Add gap
gap = 800
top_right_intersection_gap = ed.arm_body_intersection(D_B + gap, W_A + gap)
                                            # gap/4 is to adjust side edges spacing
outline_points_gap = ed.edura_outline_points(top_right_intersection_gap, D_B + gap,
                                             H_A +gap/4, W_A + gap)
msp.add_lwpolyline(outline_points_gap, 'xyseb', {'closed':'True', 'layer': EDURA_EXT})

# file_name = "test\\t2_eDura_outline_test.dxf"
# ed.mask_out(mask, file_name)

#==============================================================================

''' Test edura_sq_outline_points() '''
# S_B = 3000 #(um) side of body of eDura
# W_A = 1000 #(um) width of arm
# H_A = 6000 #(um) length of arm
# EDURA_INT = 'eDura_outline_int'
# EDURA_EXT = 'eDura_outline_ext'
# mask.layers.new(EDURA_INT, dxfattribs={'color': 6})
# mask.layers.new(EDURA_EXT, dxfattribs={'color': 6})
#
# top_right_intersection = ed.arm_sq_body_intersection(S_B, W_A)       # 1st vertice of right arm
# outline_points = ed.edura_sq_outline_points(top_right_intersection, S_B, H_A, W_A)
# msp.add_lwpolyline(outline_points, 'xyseb', {'closed':'True', 'layer': EDURA_INT})
# # Add gap
# gap = 100
# top_right_intersection_gap = ed.arm_sq_body_intersection(S_B + gap, W_A + gap)
                                                  # gap/4 is to adjust side edges spacing
# outline_points_gap = ed.edura_sq_outline_points(top_right_intersection_gap, S_B + gap,
#                         H_A +gap/4, W_A + gap) # gap/4 is to adjust side edges spacing
# msp.add_lwpolyline(outline_points_gap, 'xyseb', {'closed':'True', 'layer': EDURA_EXT})
#
# file_name = "test\\t3_eDura_sq_outline_test.dxf"
# ed.mask_out(mask, file_name)

# =============================================================================

''' Test points_square_grid() '''
# N_TRODES = 64      #() number of electrodes
# TRODE_PITCH = 700  #(um) eletrode pitch
# TRODE_DIAM = 50    #(um) electrode diameter
# GAP_ETCH = 10      #(um)
# center = [0,0]     #() center of wafer
# METAL = "metal"
# ETCH = "etch"
# mask.layers.new(METAL, dxfattribs={'color': 4})
# mask.layers.new(ETCH, dxfattribs={'color': 3})
#
# trodes_side, points = ed.points_square_grid(N_TRODES, center, TRODE_PITCH)
# p_x = p_y = TRODE_PITCH
#
# for row in points:
#     print("\npoints in each row:\n", row)
#     for pt in row:
#         msp.add_circle(pt, TRODE_DIAM/2 + GAP_ETCH, {'layer': METAL})
#         msp.add_circle(pt, TRODE_DIAM/2, {'layer': ETCH})
#
# file_name = "test\\t4_square_grid_test.dxf"
# ed.mask_out(mask, file_name)

# =============================================================================

''' Test pts_hexagonal_lattice_sq_grid '''
# N_TRODES = 36      #() number of electrodes
# TRODE_PITCH = 400  #(um) eletrode pitch
# TRODE_DIAM = 20    #(um) electrode diameter
# GAP_ETCH = 10      #(um)
# center = [0,0]     #() center of wafer
# METAL = "metal"
# ETCH = "etch"
# mask.layers.new(METAL, dxfattribs={'color': 4})
# mask.layers.new(ETCH, dxfattribs={'color': 3})
#
# trodes_side, points, p_x, p_y = ed.points_hexagonal_lattice_sq_grid(N_TRODES,
                                   # center, TRODE_PITCH)
#
# for row in points:
#     print("\npoints in each row:\n", row)
#     for pt in row:
#         msp.add_circle(pt, TRODE_DIAM/2 + GAP_ETCH, {'layer': METAL})
#         msp.add_circle(pt, TRODE_DIAM/2, {'layer': ETCH})
#
# file_name = "test\\t5_hexagonal_lattice_square_grid_test.dxf"
# ed.mask_out(mask, file_name)

# =============================================================================

''' Test pts_hexagonal_lattice_hex_grid '''
N_TRODES = 32      #() number of electrodes
TRODE_PITCH = 500  #(um) eletrode pitch
TRODE_DIAM = 50    #(um) electrode diameter
GAP_ETCH = 5       #(um)
center = [0,0]     #() center of wafer
D_B = 20000        #(um) diameter of body of eDura
METAL = "metal"
ETCH = "etch"
mask.layers.new(METAL, dxfattribs={'color': 4})
mask.layers.new(ETCH, dxfattribs={'color': 3})

points, p_x, p_y, n_trodes = ed.points_hex_lattice_hex_grid(N_TRODES, center,
                                                            TRODE_PITCH, D_B)

for row in points:
    # print("\n points in row:\n", row)
    for pt in row:
        msp.add_circle(pt, TRODE_DIAM/2 + GAP_ETCH, {'layer': METAL})
        msp.add_circle(pt, TRODE_DIAM/2, {'layer': ETCH})

print("\nNumber of total electrodes:", n_trodes)

file_name = "test\\t6_hexagonal_lattice_hexagonal_grid_test.dxf"
ed.mask_out(mask, file_name)

#===============================================================================

''' Test traces_electrodes_out() '''
TRACE_WIDTH = 10  #(um) trace with
offset_traces = False
body_trace_spacing = None
diag, hor, end_points = ed.traces_electrodes_out(points, p_x, p_y, TRACE_WIDTH,
                           TRODE_DIAM + GAP_ETCH, offset_traces, spacing_trace)

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

file_name = "test\\t7_electrodes_out_traces_hex_test.dxf"
ed.mask_out(mask, file_name)

# ==============================================================================

''' Test traces_center_to_arms_all() '''
TRACE_PITCH = 400  #(um) trace pitch
OUTLINE_TOLERANCE = 100 #(um) tolerance bewtween traces and device outline
all_points = True
print("\n\n ===== Traces center to arms ===== ")

hor_traces, dia_traces, end_points, symm_left = ed.traces_center_to_arms_symm(end_points,
                                        TRACE_PITCH, TRACE_WIDTH,
                                        p_y, p_x, offset_traces, all_points)

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

file_name = "test\\t8_center_to_arms_traces_test.dxf"
ed.mask_out(mask, file_name)

# ==============================================================================

''' Test stair_case_traces_to_arms_all() '''

print("\n\n ===== Traces - Stair Case ===== ")

hor_traces, dia_traces = ed.traces_stair_case_to_arms_test(end_points, TRACE_PITCH,
                            TRACE_WIDTH, p_y, p_x, offset_traces, all_points, symm_left)

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

file_name = "test\\t9_stair_case_traces_to_arms_test.dxf"
ed.mask_out(mask, file_name)

# ==============================================================================

''' Test bond_pads_points()'''

PAD_WIDTH = 350     #(um) pad width
PAD_HEIGHT = 350    #(um) pad height
PAD_SPACE_X = 350   #(um) pad space X
PAD_SPACE_Y = 350   #(um) pad space Y
EDGE_TOL = 700      #(um) pad edge tolerance
MEET_X = 1000      #(um) where pads and electrodes meet

n_pad_y = 4
n_pad_x = (N_TRODES//2) // n_pad_y

pts_pads_r, pts_pads_l = ed.bond_pads_points(n_pad_x, n_pad_y, PAD_WIDTH, PAD_HEIGHT,
                                         PAD_SPACE_X, PAD_SPACE_Y, top_right_intersection, H_A, W_A, EDGE_TOL)

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

file_name = "test\\t10_bond_pads_points.dxf"
ed.mask_out(mask, file_name)

# ==============================================================================

''' Test bond_pad_tracing '''
diag_r, hor_r = ed.bond_pads_traces_test(pts_pads_r, TRACE_PITCH, TRACE_WIDTH, n_pad_x, n_pad_y,
                                     PAD_WIDTH, PAD_HEIGHT, PAD_SPACE_X, PAD_SPACE_Y, MEET_X, offset_traces)

for trace in diag_r:
    msp.add_lwpolyline(trace, 'xyseb', {'layer': METAL})
for trace in hor_r:
    msp.add_lwpolyline(trace, 'xyseb', {'layer': METAL})

diag_l, hor_l = ed.bond_pads_traces_test(pts_pads_l, TRACE_PITCH, TRACE_WIDTH, n_pad_x, n_pad_y,
                                     PAD_WIDTH, PAD_HEIGHT, PAD_SPACE_X, PAD_SPACE_Y, MEET_X, offset_traces)

for trace in diag_l:
    msp.add_lwpolyline(trace, 'xyseb', {'layer': METAL})
for trace in hor_l:
    msp.add_lwpolyline(trace, 'xyseb', {'layer': METAL})

file_name = "test\\t11_bond_pads_traces.dxf"
ed.mask_out(mask, file_name)

# ==============================================================================
