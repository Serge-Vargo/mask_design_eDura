''' Helper Functions for Mask Design of eDura '''

import functools

import numpy as np

''' mask file output '''
def mask_out(file, name):
    print("\n\nExporting...")
    file.saveas(name)
    print("Done")
    return

''' generate an array of points [x,y] defining the wafer outline '''
def wafer_outline_points(diam=100000, flat_bot=32500, flat_top=18000, gap=5000):
    points_outline = []
    points_outline_gap = []

    # angles of right arc of wafer
    theta = np.arcsin((flat_bot/2)/(diam/2))
    st_angle_right = -1 * (np.pi/2 - theta)
    phi = np.arcsin((flat_top/2)/(diam/2))
    end_angle_right = np.pi/2 - phi
    angles_r = np.linspace(st_angle_right, end_angle_right)
    # right arc
    arc_right_x = (diam/2*np.cos(angles_r)).tolist()
    arc_right_y = (diam/2*np.sin(angles_r)).tolist()
    arc_right = []
    for i in range(len(arc_right_x)):
        arc_right.append([arc_right_x[i], arc_right_y[i]]) # both lists must be the same length
    # angles of right arc of wafer
    st_angle_left = np.pi/2 + phi
    end_angle_left = (3*np.pi/2 - theta)
    angles_l = np.linspace(st_angle_left, end_angle_left)
    # left arc
    arc_left_x = (diam/2*np.cos(angles_l)).tolist()
    arc_left_y = (diam/2*np.sin(angles_l)).tolist()
    arc_left = []
    for i in range(len(arc_left_x)):
        arc_left.append([arc_left_x[i], arc_left_y[i]])   # both lists must be the same length
    # top line
    top_x_right = flat_top/2
    top_x_left = -top_x_right
    top_y = np.sqrt((diam/2)**2 - top_x_right**2)
    top_line = [[top_x_right, top_y], [top_x_left, top_y]] # points ordered from right to left
    # bottom line
    bottom_x_right = flat_bot/2
    bottom_x_left = -bottom_x_right
    bottom_y = -np.sqrt((diam/2)**2 - bottom_x_right**2)
    bottom_line = [[bottom_x_left, bottom_y], [bottom_x_right, bottom_y]] # points ordered from left to right

    # Add gap (same as before but with added gap, need to recalculate points)
    # top line with gap
    top_x_right_gap = top_x_right + gap*np.sin(phi) # - gap*sin(theta/2) b/c the sign of sine is "negative"
    top_x_left_gap = - top_x_right_gap
    top_y_gap = np.sqrt((diam/2 + gap)**2 - top_x_right_gap**2)  # use circle formula x^2 + y^2 = r^2
    top_line_gap = [[top_x_right_gap, top_y_gap], [top_x_left_gap, top_y_gap]]
    # bottom line with gap
    bottom_x_right_gap = bottom_x_right + gap*np.sin(theta) # - gap*sin(theta/2) b/c the sign of sine is "negative"
    bottom_x_left_gap = - bottom_x_right_gap
    bottom_y_gap = - np.sqrt((diam/2 + gap)**2 - bottom_x_right_gap**2)  # use circle formula x^2 + y^2 = r^2
    bottom_line_gap = [[bottom_x_right_gap, bottom_y_gap], [bottom_x_left_gap, bottom_y_gap]]
    # right arc with gap (same angles)
    arc_right_x_gap = ((diam/2 + gap)*np.cos(angles_r)).tolist()
    arc_right_y_gap = ((diam/2 + gap)*np.sin(angles_r)).tolist()
    arc_right_gap = []
    for i in range(len(arc_right_x_gap)):
        arc_right_gap.append([arc_right_x_gap[i], arc_right_y_gap[i]]) # both lists must be the same length
    # left arc with gap (same angles)
    arc_left_x_gap = ((diam/2 + gap)*np.cos(angles_l)).tolist()
    arc_left_y_gap = ((diam/2 + gap)*np.sin(angles_l)).tolist()
    arc_left_gap = []
    for i in range(len(arc_left_x_gap)):
        arc_left_gap.append([arc_left_x_gap[i], arc_left_y_gap[i]])   # both lists must be the same length

    points_outline = arc_right + top_line + arc_left + bottom_line
    points_outline_gap = arc_right_gap + top_line_gap + arc_left_gap + bottom_line_gap

    return points_outline, points_outline_gap


''' Top right intersection of circular body and rectangular arm of eDura device'''
def arm_body_intersection(diam=20000, width=15000, center=[0,0]):
    r = diam/2
    y = width/2
    x = np.sqrt(r**2 - y**2)
    return [x,y]


'''
eDura outline:
    Starting from the top right intersection point between body and arm, this
    function forms an array of points in the correct order to form a polyline
    for the edura outline
'''
def edura_outline_points(top_right_intersection=[0,0], diam=20000, l=20000, w=10000):
    points_outline = []
    # right arm
    x_tr,y_tr = top_right_intersection
    # x_tr,y_tr = arm_body_intersection(diam, w)    # use function directly
    vertices_right_arm = [[x_tr,y_tr],[x_tr+l,y_tr],[x_tr+l,y_tr-w],[x_tr,y_tr-w]]
    # angles of top arc of body
    end_angle = np.arctan2(y_tr,x_tr)   # angle at which top right intersection occurs
    start_angle = np.pi - end_angle
    angles = np.linspace(start_angle, end_angle)
    # top arc
    arc_top_x = (diam/2*np.cos(angles)).tolist()    # top arc points from left to right in x
    arc_top_y = (diam/2*np.sin(angles)).tolist()
    arc_top = []
    for i in range(len(arc_top_x)):
        arc_top.append([arc_top_x[i], arc_top_y[i]])
    # bottom arc
    arc_bottom_x = (-diam/2*np.cos(angles)).tolist() # bottom arc points from right to left (reverse)
    arc_bottom_y = (-diam/2*np.sin(angles)).tolist()
    arc_bottom = []
    for i in range(len(arc_bottom_x)):
        arc_bottom.append([arc_bottom_x[i], arc_bottom_y[i]])
    # left arm
    x_bl,y_bl = [-x_tr, -y_tr] #bottom left vertice
    vertices_left_arm = [[x_bl,y_bl],[x_bl-l,y_bl],[x_bl-l,y_bl+w],[x_bl,y_bl+w]]
    # array of points in order
    points_outline = arc_top + vertices_right_arm + arc_bottom + vertices_left_arm
    return points_outline


''' Top right intersection of circular body and rectangular arm of eDura device'''
def arm_sq_body_intersection(side=5000, width=7000, center=[0,0]):
    x = side/2
    y = width/2
    return [x,y]


'''
eDura outline:
    Starting from the top right intersection point between body and arm, this
    function forms an array of points in the correct order to form a polyline
    for the edura outline
'''
def edura_sq_outline_points(top_right_intersection=[0,0], side=5000, l=10000, w=7000):
    points_outline = []
    # edge
    edge = side/2 - top_right_intersection[1]
    # right arm
    x_tr,y_tr = top_right_intersection
    vertices_right_arm = [[x_tr,y_tr],[x_tr+l,y_tr],[x_tr+l,y_tr-w],[x_tr,y_tr-w]]
    # bottom body
    x_br,y_br = [x_tr, -y_tr]  #bottom right vertice
    vertices_bott_body = [[x_br,y_br],[x_br,y_br-edge],[x_br-side,y_br-edge],[x_br-side,y_br]]
    # left arm
    x_bl,y_bl = [-x_tr, -y_tr] #bottom left vertice
    vertices_left_arm = [[x_bl,y_bl],[x_bl-l,y_bl],[x_bl-l,y_bl+w],[x_bl,y_bl+w]]
    # top body
    x_tl,y_tl = [-x_tr, y_tr]  #top left vertice
    vertices_top_body = [[x_tl,y_tl],[x_tl,y_tl + edge],[x_tl+side,y_tl + edge],[x_tl+side,y_tl]]
    # array of points in order
    points_outline = vertices_right_arm + vertices_bott_body + vertices_left_arm + vertices_top_body
    return points_outline


'''
Square Lattice in Square grid.
    Returns list of lists of points [x, y] in each row from top left to bottom right.
    It supports any number of electrodes that forms a square ( sqrt(number_electrodes) = integer )
            powers of 4 (4, 16, 64, 256, 1024 . . .)
            9 * powers of 4 (9, 36, 144, 576, 2304 . . .)
            25 * powers of 4 (25, 100, 400, 1600 . . .)
            49 * powers of 4 (49, 196, 784, 3136 . . .)
            etc ...
'''
def points_square_grid(number_electrodes=16, center=[0,0], pitch=500):
    x_o, y_o = center
    electrodes_side = int(np.sqrt(number_electrodes)) # number of points on side of grid

    # get "centered" top corner coordinates
    if electrodes_side % 2 == 0: # even
        center_offset = pitch/2     # offset center between two electrodes
        corner_offset = electrodes_side//2 - 1          # number of points between center and top left corner
    else: # odd
        center_offset = 0           # center at middle electode
        corner_offset = electrodes_side//2              # number of points between center and top left corner
    top_left_x = x_o - center_offset + (corner_offset)*(-pitch)
    top_left_y = y_o + center_offset + (corner_offset)*(+pitch)

    # [x, y] points of grid
    points_x = top_left_x * np.ones(electrodes_side)    # start with all at top left corner
    points_y = top_left_y * np.ones(electrodes_side)
    increment_pitch = pitch*np.arange(electrodes_side)  # pitch increments
    points_x = points_x + increment_pitch               # adjust points with pitch
    points_y = points_y - increment_pitch

    points = [] # all points of each row
    for j in range(len(points_y)):
        pts_row = []    # list of points in each row
        for i in range(len(points_x)):
            pts_row.append([points_x[i], points_y[j]])
        points.append(pts_row)

    return electrodes_side, points


'''
Hexagonal Lattice in Square Grid.
    Returns list of lists of points [x,y] in each row from top left to bottom right.
    Same as square grid but shifting every other row and column by (1/2)*pitch
    and sqrt(3)/2*pitch respectively and adjusting the center of the grid
'''
def points_hexagonal_lattice_sq_grid(number_electrodes=16, center=(0,0), pitch=500):
    x_o, y_o = center

    electrodes_side = int(np.sqrt(number_electrodes)) # number of points on side of grid

    pitch_x = pitch
    pitch_y = pitch * (np.sqrt(3)/2)
    # get "centered" top corner coordinates
    top_left_x = x_o + (electrodes_side)*(-pitch_x//2) + pitch_x/4
    top_left_y = y_o + (electrodes_side)*(+pitch_y//2) - pitch_y/2

    # (x, y) points of grid
    points_x1 = top_left_x * np.ones(electrodes_side)    # start with all at top left corner
    points_y = top_left_y * np.ones(electrodes_side)
    increment_pitch_x = pitch_x*np.arange(electrodes_side)  # pitch increments
    increment_pitch_y = pitch_y*np.arange(electrodes_side)  # pitch increments
    # offset = pitch/2 * np.ones(electrodes_side)       # offset array triangular
    offset_x = pitch_x/2   # offset in x
    offset_y = pitch_y/2 # offset array hexagonal lattice

    points_x1 = points_x1 + increment_pitch_x               # adjust points with pitch
    points_x2 = points_x1 + offset_x
    points_y = points_y - increment_pitch_y

    points = [] # sent points in 1D list as 2 element list [x, y]
    for j in range(len(points_y)):
        pts_row = []
        for i in range(len(points_x1)):
            if j % 2 == 0:
                pts_row.append([points_x2[i], points_y[j]])
            else:
                pts_row.append([points_x1[i], points_y[j]])
        points.append(pts_row)

    return electrodes_side, points, pitch_x, pitch_y


'''
Hexagonal Lattice in Hexagon Grid.
    Returns list of lists of points [x, y] in each row.
    Method:
    1st -> calculate number of electrodes in center row = sqrt(n_electrodes) + 2
    2nd -> pitch in x and y are such to mantain hexagonal/honeycomb lattice
    3rd -> every row has one less electrode with respect to its neighbour
           starting from the center and moving outwards.
'''
def points_hex_lattice_hex_grid(n_electrodes=16, center=(0,0), pitch=500, dura_diam=20000):
    #Center row
    n_electrodes_center = int(np.sqrt(n_electrodes)) + 2  # number of electrodes in center row
    # print("\nNumber of electrodes in center row:", n_electrodes_center)
    opt_pitch_x = dura_diam//(n_electrodes_center + 1)    # optimal pitch in x
    opt_pitch_y = opt_pitch_x*(np.sqrt(3)/2)              # optimal pitch in y = pitch_x * sin(60)

    if n_electrodes_center % 2 == 0: # even
        center_offset_x = opt_pitch_x/2     # offset center between two electrodes
    else: # odd
        center_offset_x = 0                 # center at middle electode
    # coordinate point [x,y=0] of electrode at left edge of center row
    edge_electrode_center_x = opt_pitch_x *(-1)*int(n_electrodes_center/2) + center_offset_x

    # find number of electrodes per row for bottom half
    n_electrodes_per_row = []   # (initialize) number of electrodes per row
    n_rows_half = 0             # number of rows in one half
    sum_electrodes = n_electrodes_center   # running sum of electrodes in rows
    for i in range(n_electrodes_center - 1,0, -1):
        sum_electrodes = sum_electrodes + 2*i
        n_electrodes_per_row.append(i)          # number of electrodes per row for bottom half
        n_rows_half = n_rows_half + 1
        if n_electrodes - sum_electrodes < int(n_electrodes*0.05):  # when number of electrodes is within 5% of total number electrodes stop
            break
    n_electrodes_per_row = n_electrodes_per_row[::-1]  + [n_electrodes_center] + n_electrodes_per_row # reversed for top half + center row + bottom half

    points = []     # array of all electrode coordinate points [x,y]

    for idx, n_electrodes in enumerate(n_electrodes_per_row):
        pts_row = []
        points_x = (edge_electrode_center_x + (n_electrodes_center - n_electrodes)*opt_pitch_x//2) * np.ones(n_electrodes)
        increment_pitch = opt_pitch_x*np.arange(n_electrodes)  # pitch increments
        points_x = points_x + increment_pitch
        if idx <= n_rows_half:
            points_y = ((n_electrodes_center - n_electrodes)*opt_pitch_y) * np.ones(n_electrodes)   # y-coordinate points in top half are positive
        else:
            points_y = (-(n_electrodes_center - n_electrodes)*opt_pitch_y) * np.ones(n_electrodes)  # y-coordinate points in bottom half are negative
        for j in range(n_electrodes):
            pts_row.append([points_x[j], points_y[j]])

        points.append(pts_row)

    return points, opt_pitch_x, opt_pitch_y, sum_electrodes


'''
Traces from electrode sites out towards the edge of device-body
    Returns list of lists of points [x, y] in each row.
    Method:
    1st -> Diagonal traces towards the left or right
    2nd -> Horizontal traces towards outside of arm
'''
def traces_electrodes_out(points, pitch_x, pitch_y, trace_w, trode_diam, offset, spacing_trace):
    diag_traces = []
    hor_traces = []
    end_points = []

    for idx_row, row in enumerate(points):
        # go left or right
        # print("\npoints in row (before): ", row)
        if idx_row % 2 == 0: # if even, go left
            dir = -1            # -x direction
            row = row[::-1]     # reverse order of points
        else:                # if odd, go right
            dir = 1     # +x direction

        # print("\npoints in row (after): ", row)
        # build list of lists of trace information [init_pt, end_pt] = [ [x1, y1, width1, width1] [x2, y2, width2, width2] ]
        if spacing_trace == None:
            spacing_y = (pitch_y)//(len(row))
        else:
            spacing_y = spacing_trace

        spacing_x = spacing_y/(np.sqrt(3))
        diag_traces_row = []
        hor_traces_row = []
        end_points_row = []

        for idx, pt in enumerate(row):
            #diagonal traces
            if pt == row[-1]:    # if no diagonal trace needed (no need for offset)
                offset_x = 0; offset_y = 0
            else:
                if offset:       # Add offset to make diagonal and horizontal traces coincide
                    offset_x = dir*trace_w/2*(np.sqrt(3)/2)
                    offset_y = trace_w/4
                else:
                    offset_x = 0; offset_y = 0

            init_point_d = pt
            end_point_d = [row[idx][0] + dir*((len(row)-1)-idx)*spacing_x + offset_x, \
                           row[idx][1] - ((len(row)-1)-idx)*spacing_y - offset_y]
            diag_traces_row.append([ init_point_d + [trace_w, trace_w] , \
                                     end_point_d + [trace_w, trace_w] ])
            #horizontal traces
            init_point_h = [end_point_d[0] - offset_x, end_point_d[1] + offset_y]
            # default end_point
            # end_point_h = [row[idx][0] + dir*((len(row)-1)-idx)*pitch_x + dir*pitch_x/2, \
            #                init_point_h[1]]
            if idx_row < len(points)//2:     # This if-else statement is used to ease the routing of traces
                end_point_h = [row[idx][0] + dir*((len(row)-1)-idx)*pitch_x + dir*pitch_x/2, \
                               init_point_h[1]]
            else:                            # For bottom half traces add more spacing in x direction for clearance
                end_point_h = [row[idx][0] + dir*((len(row)-1)-idx)*pitch_x + dir*pitch_x/8, \
                               init_point_h[1]]

            hor_traces_row.append([ init_point_h + [trace_w, trace_w] , \
                                     end_point_h + [trace_w, trace_w] ])

            end_points_row.append(end_point_h)

        diag_traces.append(diag_traces_row)
        hor_traces.append(hor_traces_row)
        end_points.append(end_points_row[::-1]) # reverse for points from top to bottom

    return diag_traces, hor_traces, end_points


'''
Organize points in quadrants to help with tracing
    Returns a 4 item list (1 list per quadrant) of lists of rows of points [x, y]
    in each Quadrant.
    (1st list --> Q II | 2nd list --> Q III | 3rd list --> Q IV | 4th list --> Q I)
    Method:
    --> Identify list of points in quadrant via if statements and group together
'''
def organize_in_quadrants(points):
    #==========================================================================
    # Quadrants are used for processing the coordinates of traces more accurately
    # The order in which points are grouped in quadrants is: Q II, Q III, Q IV, Q I
    rows_qii = []  # rows in 2nd quadrant
    rows_qiii = [] # rows in 3rd quadrant
    rows_qiv = []  # rows in 4th quadrant
    rows_qi = []   # rows in 1st quadrant

    n_pts_in_q = [0,0,0,0] # (initialize) number of pts in quadrants
    n_rows_in_q = [0,0,0,0] # (initialize) number of rows in quadrants

    for idx_row, row in enumerate(points):
        if idx_row % 2 == 0: # if even, is in the left side (-x)
            if row[0][1] > 0:   # 2nd Quadrant
                rows_qii.append(row)
                n_pts_in_q[0] = n_pts_in_q[0] + len(row)
                n_rows_in_q[0] = n_rows_in_q[0] + 1
            elif row[0][1] <= 0:   # 3rd Quadrant
                row = row[::-1] # reverse order of points
                rows_qiii.append(row)
                n_pts_in_q[1] = n_pts_in_q[1] + len(row)
                n_rows_in_q[1] = n_rows_in_q[1] + 1
        else:                # if odd, is in right (+x)
            if row[0][1] > (0):   # 1st Quadrant
                rows_qi.append(row)
                n_pts_in_q[2] = n_pts_in_q[2] + len(row)
                n_rows_in_q[2] = n_rows_in_q[2] + 1
            elif row[0][1] <= (0):   # 4th Quadrant
                row = row[::-1] # reverse order of points
                rows_qiv.append(row)
                n_pts_in_q[3] = n_pts_in_q[3] + len(row)
                n_rows_in_q[3] = n_rows_in_q[3] + 1

    rows_all_q = []
    rows_all_q.append(rows_qii); rows_all_q.append(rows_qiii[::-1]) # reverse for outside in direction
    rows_all_q.append(rows_qiv[::-1]); rows_all_q.append(rows_qi)

    rows_symm = []
    rows_not_symm = []
    print("\nnumber of points per quadrant: ", n_pts_in_q)
    print(n_pts_in_q[0] == n_pts_in_q[1])
    if n_pts_in_q[0] == n_pts_in_q[1] :
        print("Symmetry in left side")
        rows_symm.append(rows_qii); rows_symm.append(rows_qiii[::-1])
        rows_not_symm.append(rows_qiv[::-1]); rows_not_symm.append(rows_qi)
    else:
        print("Symmetry in right side")
        # rows_symm.append(rows_qiv[::-1]); rows_symm.append(rows_qi)
        rows_symm.append(rows_qiv[::-1]); rows_symm.append(rows_qi)
        rows_not_symm.append(rows_qii); rows_not_symm.append(rows_qiii[::-1])

    return rows_all_q, rows_symm, rows_not_symm


'''
Traces from body of device towards arms (assuming symmetry):
    Returns list of lists of lwpolyline data [pi, pf, width_i, width_f] for
    each row of traces.
    Method:
    1st -> Diagonal traces towards the left or right
    2nd -> Horizontal traces extending towards outside of arm
'''
def traces_center_to_arms_symm(points, trace_p, trace_w, p_y, p_x, offset, all_points):

    rows_all_q, rows_symm, rows_not_symm = organize_in_quadrants(points)

    if all_points:
        points = rows_all_q
        symm_left = None
        hor_dir = [-1, -1, +1, +1] # left (-x), left (-x), right (+x), right (+x)
        ver_dir = [-1, +1, +1, -1] # down (-y), up (+y), up (+y), down (-y)
    else:
        points = rows_symm
        if points[0][0] == rows_all_q[0][0] : # if QII
            print("Left side")
            symm_left = True
            hor_dir = [-1, -1] # left (-x), left (-x)
            ver_dir = [-1, +1] # up (+y), down (-y)
        else:
            print("Right side")
            symm_left = False
            hor_dir = [+1, +1] # right (+x), right (+x)
            ver_dir = [+1, -1] # up (+y), down (-y)

    #==========================================================================
    # Processing of traces 1st run
    hor_traces = []
    diag_traces = []
    hor_traces_row = []
    diag_traces_row = []
    end_points = []  # initiallize end_points for each quadrant

    for idx_q, quad in enumerate(points):
        print("\n\nQuadrant #: ", idx_q + 1)
        end_points_quad = []
        for idx_r, row in enumerate(quad):
            print("idx_row: ", idx_r)
            print("len of row: ", len(quad))
            py_i = np.absolute(row[1][1] - row[0][1]) # initial trace pitch
            py_f = trace_p

            print("First y-point of first row: ", row[0][1])
            if row[0][1] < 0: # if bottom half traces
                print("Bottom Half")
                spacing_y = p_y + (py_i)*(len(row) - 1) - py_f*(len(row) - 1) - py_f/2
            # if top half
            else:
                # if last row
                if idx_r == (len(quad) - 1):
                        spacing_y = p_y - (py_f * len(row)) + py_f/2
                else:
                    # spacing_y = (2*py_i - py_f) * len(row)
                    spacing_y = 2*p_y  - (py_f * len(row)) #- py_f

            if spacing_y < 0:   # if spacing is negative
                    spacing_x = (-1)*spacing_y/(np.sqrt(3))
            else:
                spacing_x = spacing_y/(np.sqrt(3)) # default

            pts_yo = row[0][1] * np.ones(len(row))
            increment_pitch_trodes_y = py_i * np.arange(len(row))
            pts_yo = pts_yo + ver_dir[idx_q] * increment_pitch_trodes_y

            pts_y1 = (pts_yo[0] + ver_dir[idx_q]*spacing_y) * np.ones(len(row))
            increment_pitch_trace_y = py_f * np.arange(len(row))
            pts_y1 = pts_y1 + ver_dir[idx_q]*increment_pitch_trace_y

            pts_xo = row[0][0] * np.ones(len(row))
            # pts_xo = row[0][0] * np.ones(len(row)) + hor_dir[idx_q]*py_i/np.sqrt(3)*np.arange(len(row))

            pts_x1 = pts_xo + hor_dir[idx_q]*spacing_x* np.ones(len(row))
            increment_spacing_x = (py_i/np.sqrt(3) - py_f/np.sqrt(3))  * np.arange(len(row))
            if spacing_y < 0:   # if spacing is negative
                    pts_x1 = pts_x1 + hor_dir[idx_q]*increment_spacing_x
            else:
                pts_x1 = pts_x1 - hor_dir[idx_q]*increment_spacing_x # default

            if offset:
                offset_x = hor_dir[idx_q]*trace_w/2*(np.sqrt(3)/2)
                offset_y = ver_dir[idx_q]*trace_w/4
            else:
                offset_x = 0; offset_y = 0

            end_points_row = []
            for idx, pt in enumerate(row):
                # horizontal traces
                # init_point_h = [pt[0] - offset_x, pt[1] - offset_y]
                # end_point_h = [pts_xo[idx] - offset_x, pts_yo[idx] - offset_y]
                #
                # hor_traces_row.append([ init_point_h + [trace_w, trace_w] , \
                #                         end_point_h + [trace_w, trace_w] ])

                # diagonal traces
                init_point_d = [pts_xo[idx] - offset_x, pts_yo[idx] - offset_y]
                # init_point_d = [end_point_h[0] - offset_x, end_point_h[1] - offset_y]
                end_point_d = [pts_x1[idx] + 2*offset_x, pts_y1[idx] + 2*offset_y]
                diag_traces_row.append( [init_point_d + [trace_w, trace_w] , \
                                        end_point_d + [trace_w, trace_w] ])

                # horizontal traces
                init_point_h = [end_point_d[0] - offset_x, end_point_d[1] - offset_y]
                # init_point_h = [end_point_d[0], end_point_d[1]]
                if row[0][1] < 0 and (idx_r == len(quad) - 1): #( quad[idx_r][1] == quad[0][1] ):
                     # end_point_h = [hor_dir[idx_q]*(p_x - p_x/8 + p_x/2) + pts_xo[idx], init_point_h[1]]
                     end_point_h = [hor_dir[idx_q]*(p_x - p_x/8 + 0) + pts_xo[idx], init_point_h[1]]
                     # if (idx_q == 1 and not symm_right) or (idx_q == 2 and symm_right):
                     #     end_point_h = [hor_dir[idx_q]*(p_x - p_x/8 + 0) + pts_xo[idx], init_point_h[1]]
                else:
                    end_point_h = [hor_dir[idx_q]*p_x + pts_xo[idx], init_point_h[1]]

                hor_traces_row.append([ init_point_h + [trace_w, trace_w] , \
                                        end_point_h + [trace_w, trace_w] ])

                end_points_row.append(end_point_h)
            end_points_quad.append(end_points_row)
        end_points.append(end_points_quad)

    hor_traces.append(hor_traces_row)
    diag_traces.append(diag_traces_row)

    return hor_traces, diag_traces, end_points, symm_left

'''
Traces from body of device towards arms:
    Returns list of lists of lwpolyline data [pi, pf, width_i, width_f] for
    each row of traces.
    Method:
    1st -> Diagonal traces towards the left or right
    2nd -> Horizontal traces towards outside of arm
'''
def traces_stair_case_to_arms(points, trace_p, trace_w, p_y, p_x, offset, all_points, symm_left):

    if all_points:
        # points = rows_all_q
        hor_dir = [-1, -1, +1, +1] # left (-x), left (-x), right (+x), right (+x)
        ver_dir = [-1, +1, +1, -1] # down (-y), up (+y), up (+y), down (-y)
    else:
        # points = rows_symm
        if symm_left: # if QII
            print("Left side")
            hor_dir = [-1, -1] # left (-x), left (-x)
            ver_dir = [-1, +1] # up (+y), down (-y)
        else:
            print("Right side")
            hor_dir = [+1, +1] # right (+x), right (+x)
            ver_dir = [+1, -1] # up (+y), down (-y)

    #==========================================================================
    # Processing of traces 2nd run (Stair case)
    hor_traces = []
    diag_traces = []
    hor_traces_row = []
    diag_traces_row = []
    end_points = []  # initiallize end_points for each quadrant
    recur_end_pts = []
    last_row = False

    for idx_q, quad in enumerate(points):
        print("\n\nQuadrant #: ", idx_q + 1)
        end_points_quad = []
        end_points_row = []
        recur_points_quad = []
        recur_points_row = []
        for idx_r, row in enumerate(quad):
            print("idx row: ", idx_r)
            print("length of quad: ", len(quad))

            py_i = np.absolute(row[1][1] - row[0][1]) # initial trace pitch
            py_f = trace_p

            if idx_r == len(quad) - 1:
                print("Last Row: ", idx_r)
                for pt in row:
                    end_points_row.append(pt)
                break
            else:
                if row[0][1] < 0: # if bottom half traces
                    spacing_y = 2*p_y - py_f*(len(row) + 2)
                    # if assymetric side, don't extend diagonal traces too much
                    # if (idx_q == 1 and not symm_right) or (idx_q == 2 and symm_right) and (quad[idx_r][1] == quad[0][1]):
                    #     print("Left side is assymetric")
                    #     spacing_y = p_y - (py_f * (len(row) + 2))
                else: # if top half traces
                    if idx_r == len(quad) - 2: # and (symm_right and idx_q==1): # or (not symm_right and idx_q==2) ):
                        # spacing_y = (p_y - py_f) * len(row)
                        spacing_y = p_y - (py_f * (len(row) + 2)) + py_f/2
                        print("idx_q", idx_q)
                    # not the last row or second to last row
                    else:
                        spacing_y = 2*p_y - (py_f * (len(row) + 2))

            if spacing_y < 0:   # if spacing is negative
                    spacing_x = (-1)*spacing_y/(np.sqrt(3))
            else:
                spacing_x = spacing_y/(np.sqrt(3)) # default

            pts_yo = row[0][1] * np.ones(len(row))
            increment_pitch_trace_y = py_f * np.arange(len(row))
            pts_yo = pts_yo + ver_dir[idx_q] * increment_pitch_trace_y

            pts_y1 = pts_yo + (ver_dir[idx_q]*spacing_y*np.ones(len(row)))

            # pts_xo = row[0][0] * np.ones(len(row))
            pts_xo = row[0][0] * np.ones(len(row)) + hor_dir[idx_q]*py_i/np.sqrt(3)*np.arange(len(row))

            pts_x1 = pts_xo + hor_dir[idx_q]*spacing_x* np.ones(len(row))
            # increment_spacing_x = (py_i/np.sqrt(3) - py_f/np.sqrt(3))  * np.arange(len(row))
            # pts_x1 = pts_x1 - hor_dir[idx_q]*increment_spacing_x
            # if spacing_y < 0:   # if spacing is negative
            #         pts_x1 = pts_x1 + hor_dir[idx_q]*increment_spacing_x
            # else:
            #     pts_x1 = pts_x1 - hor_dir[idx_q]*increment_spacing_x
            # # pts_x1 = pts_x1 - hor_dir[idx_q]*increment_spacing_x

            if offset:
                offset_x = hor_dir[idx_q]*trace_w/2*(np.sqrt(3)/2)
                offset_y = ver_dir[idx_q]*trace_w/4
            else:
                offset_x = 0; offset_y = 0

            for idx, pt in enumerate(row):
                # horizontal traces
                init_point_h = [pt[0] - offset_x, pt[1] - offset_y]
                end_point_h = [pts_xo[idx] - offset_x, pts_yo[idx] - offset_y]

                hor_traces_row.append([ init_point_h + [trace_w, trace_w] , \
                                        end_point_h + [trace_w, trace_w] ])

                # diagonal traces
                # init_point_d = [pts_xo[idx] - offset_x, pts_yo[idx] - offset_y]
                init_point_d = [end_point_h[0] - offset_x, end_point_h[1] - offset_y]
                end_point_d = [pts_x1[idx] + 2*offset_x, pts_y1[idx] + 2*offset_y]
                diag_traces_row.append( [init_point_d + [trace_w, trace_w] , \
                                        end_point_d + [trace_w, trace_w] ])

                # horizontal traces
                init_point_h = [end_point_d[0] - offset_x, end_point_d[1] - offset_y]
                # if row[0][1] < 0 and (idx_r == len(quad) - 1):
                #      end_point_h = [hor_dir[idx_q]*(p_x - p_x/8 + 0) + pts_xo[idx], init_point_h[1]]
                # else:
                #     end_point_h = [hor_dir[idx_q]*p_x + pts_xo[idx], init_point_h[1]]
                end_point_h = [hor_dir[idx_q]*p_x + pts_xo[idx], init_point_h[1]]

                hor_traces_row.append([ init_point_h + [trace_w, trace_w] , \
                                        end_point_h + [trace_w, trace_w] ])

                end_points_row.append(end_point_h)
            end_points_quad.append(end_points_row)
        end_points.append(end_points_quad)

    hor_traces.append(hor_traces_row)
    diag_traces.append(diag_traces_row)

    return hor_traces, diag_traces

'''
Traces from body of device towards arms:
    Returns list of lists of lwpolyline data [pi, pf, width_i, width_f] for
    each row of traces.
    Method:
    1st -> Diagonal traces towards the left or right
    2nd -> Horizontal traces towards outside of arm
'''
def traces_stair_case_to_arms_test(points, trace_p, trace_w, p_y, p_x, offset, all_points, symm_left):

    if all_points:
        # points = rows_all_q
        hor_dir = [-1, -1, +1, +1] # left (-x), left (-x), right (+x), right (+x)
        ver_dir = [-1, +1, +1, -1] # down (-y), up (+y), up (+y), down (-y)
    else:
        # points = rows_symm
        if symm_left: # if QII
            print("Left side")
            hor_dir = [-1, -1] # left (-x), left (-x)
            ver_dir = [-1, +1] # up (+y), down (-y)
        else:
            print("Right side")
            hor_dir = [+1, +1] # right (+x), right (+x)
            ver_dir = [+1, -1] # up (+y), down (-y)

    #==========================================================================
    # Processing of traces 2nd run (Stair case)
    hor_traces = []
    diag_traces = []
    hor_traces_row = []
    diag_traces_row = []
    end_points = []  # initiallize end_points for each quadrant
    recur_end_pts = []
    last_row = False

    # for idx_q in range(len(points[1])): #
    #     print("Length of Q ", idx_q, "is: ", len(points[idx_q]))

    for n_iterations in range(len(points[1]) - 1):
        print("Number of iterations: ", n_iterations)
        if n_iterations == 0:
            points = points
            print("Length of longest quadrant: ", len(points[1]))
            # print(points)
        else:
            points = recur_end_pts
            recur_end_pts = []
            print("Length of longest quadrant: ", len(points[1]))
            # print(points)

        for idx_q, quad in enumerate(points):
            print("\n\nQuadrant #: ", idx_q + 1)
            end_points_quad = []
            recur_points_quad = []
            for idx_r, row in enumerate(quad):
                print("idx row: ", idx_r)
                print("length of quad: ", len(quad))

                py_i = np.absolute(row[1][1] - row[0][1]) # initial trace pitch
                py_f = trace_p

                if idx_r == len(quad) - 1:
                    print("Last Row: ", idx_r)
                    for pt in row:
                        end_points_row.append(pt)
                    break
                else:
                    if row[0][1] < 0: # if bottom half traces
                        spacing_y = 2*p_y - py_f*(len(row) + 2*(n_iterations + 1))
                    else: # if top half traces
                        if idx_r == len(quad) - 2: # and (symm_right and idx_q==1): # or (not symm_right and idx_q==2) ):
                            # spacing_y = (p_y - py_f) * len(row)
                            spacing_y = p_y - (py_f * (len(row) + 2*(n_iterations + 1))) + py_f/2
                        # not the last row or second to last row
                        else:
                            spacing_y = 2*p_y - (py_f * (len(row) + 2*(n_iterations + 1)))

                    if spacing_y < 0:   # if spacing is negative
                            spacing_x = (-1)*spacing_y/(np.sqrt(3))
                    else:
                        spacing_x = spacing_y/(np.sqrt(3))
                    # spacing_x = spacing_y/(np.sqrt(3))

                    pts_yo = row[0][1] * np.ones(len(row))
                    increment_pitch_trace_y = py_f * np.arange(len(row))
                    pts_yo = pts_yo + ver_dir[idx_q] * increment_pitch_trace_y

                    pts_y1 = pts_yo + (ver_dir[idx_q]*spacing_y*np.ones(len(row)))

                    pts_xo = row[0][0] * np.ones(len(row))

                    pts_x1 = pts_xo + hor_dir[idx_q]*spacing_x* np.ones(len(row))

                    if offset:
                        offset_x = hor_dir[idx_q]*trace_w/2*(np.sqrt(3)/2)
                        offset_y = ver_dir[idx_q]*trace_w/4
                    else:
                        offset_x = 0; offset_y = 0

                    end_points_row = []
                    recur_points_row = []
                    for idx, pt in enumerate(row):
                        # diagonal traces
                        init_point_d = [pts_xo[idx] - offset_x, pts_yo[idx] - offset_y]
                        end_point_d = [pts_x1[idx] + 2*offset_x, pts_y1[idx] + 2*offset_y]
                        diag_traces_row.append( [init_point_d + [trace_w, trace_w] , \
                                                end_point_d + [trace_w, trace_w] ])

                        # horizontal traces
                        init_point_h = [end_point_d[0] - offset_x, end_point_d[1] - offset_y]
                        if row[0][1] < 0 and (idx_r == len(quad) - 1):
                             # end_point_h = [hor_dir[idx_q]*(p_x - p_x/8 + p_x/2) + pts_xo[idx], init_point_h[1]]
                             end_point_h = [hor_dir[idx_q]*(p_x - p_x/8 + 0) + pts_xo[idx], init_point_h[1]]
                        else:
                            end_point_h = [hor_dir[idx_q]*p_x + pts_xo[idx], init_point_h[1]]

                        hor_traces_row.append([ init_point_h + [trace_w, trace_w] , \
                                                end_point_h + [trace_w, trace_w] ])

                        end_points_row.append(end_point_h)
                        recur_points_row.append(end_point_h)
                        print("Recurring rows:\n", recur_points_row)
                    end_points_quad.append(end_points_row)
                recur_points_quad.append(recur_points_row)
                end_points.append(end_points_quad)
            recur_end_pts.append(recur_points_quad)
        print("Recurring points:\n", recur_end_pts)

    hor_traces.append(hor_traces_row)
    diag_traces.append(diag_traces_row)

    return hor_traces, diag_traces

''' Intersection of trace with device outline '''
def outline_trace_intersection(diam=20000, coordinate=10000, center=[0,0]):
    r = diam/2
    cor = coordinate
    p = np.sqrt(r**2 - cor**2)
    return p

'''
Version 1: from edge of outline. Traces from body of device towards arms:
    Returns list of lists of points [x, y] in each row.
    Method:
    1st -> Horizontal traces towards outside of arm
    2nd -> Diagonal traces towards the left or right
'''
def traces_center_to_arms_v1(points, tolerance, top_right_intersection, pitch, trace_w, diam, center, offset):
    #==========================================================================
    # Quadrants are used for processing of coordinates of traces

    # The order in which points are grouped in quadrants is: Q II, Q III, Q IV, Q I
    points_qii = []  # points in 2nd quadrant
    points_qiii = [] # points in 3rd quadrant
    points_qiv = []  # points in 4th quadrant
    points_qi = []   # points in 1st quadrant
    n_pts_in_q = [0,0,0,0] # (initialize) number of traces in quadrants

    for idx_row, row in enumerate(points):
        if idx_row % 2 == 0: # if even, is in the left side (-x)
            if row[0][1] > (0 + trace_w):   # 2nd Quadrant
                points_qii.append(row)
                n_pts_in_q[0] = n_pts_in_q[0] + len(row)
            elif row[0][1] < (0 + trace_w):   # 3rd Quadrant
                row = row[::-1] # reverse order of points
                points_qiii.append(row)
                n_pts_in_q[1] = n_pts_in_q[1] + len(row)
        else:                # if odd, is in right (+x)
            if row[0][1] > (0 + trace_w):   # 1st Quadrant
                points_qi.append(row)
                n_pts_in_q[2] = n_pts_in_q[2] + len(row)
            elif row[0][1] < (0 + trace_w):   # 4th Quadrant
                row = row[::-1] # reverse order of points
                points_qiv.append(row)
                n_pts_in_q[3] = n_pts_in_q[3] + len(row)

    points_all_q = []
    points_all_q.append(points_qii); points_all_q.append(points_qiii)
    points_all_q.append(points_qiv); points_all_q.append(points_qi)
    print("\nnumber of points per quadrant: ", n_pts_in_q)

    hor_dir = [-1, -1, +1, +1] # left (-x), left (-x), right (+x), right (+x)
    ver_dir = [-1, +1, +1, -1] # down (-y), up (+y), up (+y), down (-y)
    corners = [ [-top_right_intersection[0], top_right_intersection[1]], \
                [-top_right_intersection[0], -top_right_intersection[1]],\
                [top_right_intersection[0], -top_right_intersection[1]], \
                top_right_intersection \
               ]
    print("corners:\n", corners)

    #==========================================================================
    # Processing of traces
    hor_traces = []
    diag_traces = []
    hor_traces_row = []
    diag_traces_row = []
    for idx_q, quad in enumerate(points_all_q):
        print("\n\nQuadrant #: ", idx_q)
        # print("Contents: ", quad)
        print("Length: ", len(quad))

        for row in quad:
            if ver_dir[idx_q]*row[0][1] < corners[idx_q][1] and ver_dir[idx_q]*(-1)*row[0][1] > corners[idx_q][1]:
                x_limit_h = outline_trace_intersection(diam, row[0][1], center) - tolerance
                x_limit_d = top_right_intersection[0]
                y_limit_h = outline_trace_intersection(diam, row[0][0], center) - tolerance
                y_limit_d = top_right_intersection[1] - tolerance
                print("X limit of h: ", x_limit_h, "and d: ", x_limit_d)
                print("Y limit of h: ", y_limit_h, "and d: ", y_limit_d)
                spacing_yo = y_limit_h - y_limit_d
                spacing_xo = spacing_yo/(np.sqrt(3))
                print("Spacing_yo: ", spacing_yo, "Spacing_xo: ", spacing_xo)

                pts_yo = row[0][1] * np.ones(len(row))
                increment_pitch_trodes_y = np.absolute(row[1][1] - row[0][1]) * np.arange(len(row))
                print("Pitch traces of electrodes:\n", increment_pitch_trodes_y)
                pts_yo = pts_yo + ver_dir[idx_q] * increment_pitch_trodes_y
                print("Points yo:\n", pts_yo)
                print("Actual yo:\n", row)

                pts_y1 = (-1)*ver_dir[idx_q]*(y_limit_d) * np.ones(len(row))
                increment_pitch_trace_y = pitch * np.arange(1,len(row)+1, 1)
                print("Pitch traces of trace:\n", increment_pitch_trace_y)
                pts_y1 = pts_y1 + ver_dir[idx_q]*increment_pitch_trace_y
                print("Points y1:\n", pts_y1)

                pts_xo = hor_dir[idx_q] * x_limit_h * np.ones(len(row))
                print("Points xo (before):\n", pts_xo)
                # increment_spacing_x = hor_dir[idx_q] * pitch * np.arange(1,len(row)+1, 1)
                increment_spacing_x = (increment_pitch_trodes_y - np.absolute(row[1][1] - row[0][1]) )/(np.sqrt(3))
                # increment_spacing_x = np.absolute(row[1][1] - row[0][1])/(np.sqrt(3))
                pts_xo = pts_xo + hor_dir[idx_q]*increment_spacing_x/(np.sqrt(3))
                # pts_xo = pts_xo - hor_dir[idx_q]*np.absolute(pts_y1 - pts_yo)/(np.sqrt(3))
                print("Points xo (after):\n", pts_xo)
                print("Actual xo:\n", row)
                pts_x1 = pts_xo + hor_dir[idx_q]*np.absolute(pts_y1 - pts_yo)/(np.sqrt(3))

                print("Points xo:\n", pts_xo)
                print("Points x1:\n ", pts_x1)

                if offset:
                    offset_x = hor_dir[idx_q]*trace_w/2*(np.sqrt(3)/2)
                    offset_y = ver_dir[idx_q]*trace_w/4
                else:
                    offset_x = 0; offset_y = 0

                for idx, pt in enumerate(row):
                    # horizontal traces
                    init_point_h = pt
                    end_point_h = [pts_xo[idx], pts_yo[idx]]
                    hor_traces_row.append([ init_point_h + [trace_w, trace_w] , \
                                            end_point_h + [trace_w, trace_w] ])
                    # diagonal traces
                    init_point_d = [end_point_h[0] - offset_x, end_point_h[1] - offset_y]
                    end_point_d = [pts_x1[idx], pts_y1[idx]]
                    # end_point_d = [hor_dir[idx_q]*x_limit_d, pts_y1[idx]]
                    diag_traces_row.append( [init_point_d + [trace_w, trace_w] , \
                                            end_point_d + [trace_w, trace_w] ])

                    # horizontal traces
                    init_point_h = end_point_d
                    end_point_h = [hor_dir[idx_q]*10000, end_point_d[1]]
                    hor_traces_row.append([ init_point_h + [trace_w, trace_w] , \
                                            end_point_h + [trace_w, trace_w] ])


    hor_traces.append(hor_traces_row)
    diag_traces.append(diag_traces_row)

    return hor_traces, diag_traces

def bond_pads_points(n_pad_x=4, n_pad_y=4, pad_w=350, pad_h=350,
                     pad_space_x=150, pad_space_y=150,
                     arm_ver=[0,0], arm_l=20000, arm_w=10000, edge_tolerance=700):

    edge_arm_r = [arm_ver[0] + arm_l,0]

    corner_offset_y = pad_space_y/2 + pad_h*(n_pad_y/2) + pad_space_y*(n_pad_y/2-1) - pad_space_x/2
    corner_offset_x = edge_tolerance + pad_w/2

    top_r_pad = [edge_arm_r[0] - corner_offset_x, edge_arm_r[1] + corner_offset_y]
    top_l_pad = [-top_r_pad[0], top_r_pad[1]]

    pitch_x = pad_w + pad_space_x
    increment_p_x = pitch_x*np.arange(n_pad_x)  # pitch increments
    pitch_y = pad_h + pad_space_y
    increment_p_y = pitch_y*np.arange(n_pad_y)  # pitch increments

    points_x_r = top_r_pad[0] * np.ones(n_pad_x)
    points_x_r = points_x_r - increment_p_x
    points_y_r = top_r_pad[1] * np.ones(n_pad_y)
    points_y_r = points_y_r - increment_p_y

    points_x_l = top_l_pad[0] * np.ones(n_pad_x)
    points_x_l = points_x_l + increment_p_x
    points_y_l = top_l_pad[1] * np.ones(n_pad_y)
    points_y_l = points_y_l - increment_p_y

    pts_pads_r = []
    pts_pads_l = []

    for j in range(len(points_y_r)):
        pts_row_r = []; pts_row_l = []
        for i in range(len(points_x_r)):
            pts_row_r.append([points_x_r[i], points_y_r[j]])
            pts_row_l.append([points_x_l[i], points_y_l[j]])
        pts_pads_r.append(pts_row_r)
        pts_pads_l.append(pts_row_l)

    return pts_pads_r, pts_pads_l

def square_ver(point=[0,0], w=350, h=350,):
    x,y = point
    st_c = [x-w/2, y+h/2] #first corner (top left)
    vertices = [st_c, [x+w/2, y+h/2], [x+w/2, y-h/2], [x-w/2, y-h/2], st_c]
    return vertices

def bond_pads_traces(points=[0,0], trace_w=10, dir='left', n_pad_x=4, n_pad_y=4,
                      pad_w=350, pad_h=350, pad_space_x=150, pad_space_y=150,
                      final_x=1000, offset = False):
    diag_traces = []
    hor_traces = []
    spacing_yo = (pad_space_y)//(n_pad_x + 1)
    # print("Spacing of y:\n", spacing_y)
    spacing_xo = spacing_yo

    if offset:
        offset_x = hor_dir[idx_q]*trace_w/2*(np.sqrt(3)/2)
        offset_y = ver_dir[idx_q]*trace_w/4
    else:
        offset_x = 0; offset_y = 0

    if dir == 'left':
        for i in range(-len(points),0):
            #Clear paths 1st
            #diagonal traces
            # print("i:", i,"\nScaling of height:", i//n_pad_x)
            init_point_d = points[i]
            end_point_d = [points[i][0] + (i//n_pad_x)*spacing_xo - pad_w//2, \
                           points[i][1] - (i//n_pad_x)*spacing_yo + pad_h//2]
            diag_traces.append([ init_point_d + [trace_w, trace_w] , \
                                  end_point_d + [trace_w, trace_w] ])
            #horizontal
            init_point_h = [end_point_d[0] + trace_w//2/(np.sqrt(3)/2), end_point_d[1]]
            end_point_h = [final_x, init_point_h[1]]
            hor_traces.append([ init_point_h + [trace_w, trace_w] , \
                                 end_point_h + [trace_w, trace_w] ])

    if dir == 'right':
        for i in range(-len(points),0):
            #diagonal traces
            init_point_d = points[i]
            end_point_d = [points[i][0] - (i//n_pad_x)*spacing_x + pad_w//2, \
                           points[i][1] - (i//n_pad_x)*spacing_y + pad_h//2]
            diag_traces.append([ init_point_d + [trace_w, trace_w] , \
                                  end_point_d + [trace_w, trace_w] ])
            #horizontal
            init_point_h = [end_point_d[0] - trace_w//2/(np.sqrt(3)/2), end_point_d[1]]
            end_point_h = [-final_x, init_point_h[1]]
            hor_traces.append([ init_point_h + [trace_w, trace_w] , \
                                 end_point_h + [trace_w, trace_w] ])

    return diag_traces, hor_traces

def bond_pads_traces_test(points=[0,0], trace_pitch=400, trace_w=10, n_pad_x=4, n_pad_y=4,
                          pad_w=350, pad_h=350, pad_space_x=150, pad_space_y=150,
                          final_x=1000, offset = False):
    diag_traces = []
    hor_traces = []

    if points[0][0][0] < 0: # if points on left side of device
        print("\n\nLeft side")
        hor_dir = +1       # go right (+x)
    else:
        print("\n\nRight side")
        hor_dir = -1       # go left (-x)
    ver_dir = +1       # go up (+y)

    spacing_yo = (pad_space_y)//(n_pad_x + 1)
    # print("Spacing of y:\n", spacing_y)
    spacing_xo = spacing_yo
    end_xo = points[0][0][0] + hor_dir*n_pad_x*(pad_w + pad_space_x)

    print("end_xo: ", end_xo)
    print("init point: ", points[0][0][0])

    if offset:
        offset_x = hor_dir*trace_w/2*(np.sqrt(3)/2)
        offset_y = ver_dir*trace_w/4
    else:
        offset_x = 0; offset_y = 0

    end_points_top = []; end_points_bott = []
    for idx_row, row in enumerate(points):

        if idx_row >= n_pad_y//2:
            ver_dir = -1
        end_points_o_row = []
        for idx, pt in enumerate(row):
            #diagonal traces
            # print("i:", i,"\nScaling of height:", i//n_pad_x)
            init_point_d = pt
            end_point_d = [row[idx][0] + hor_dir*((len(row)-1)-idx)*spacing_xo + + hor_dir*pad_w/2 + offset_x, \
                           row[idx][1] + ver_dir*((len(row)-1)-idx)*spacing_yo + ver_dir*pad_h/2 + offset_y]
            diag_traces.append([ init_point_d + [trace_w, trace_w] , \
                                  end_point_d + [trace_w, trace_w] ])
            #horizontal
            init_point_h = [end_point_d[0] + offset_x, end_point_d[1]]
            # end_point_h = [init_point_h[0] + hor_dir*final_x, init_point_h[1]]
            end_point_h = [end_xo, init_point_h[1]]
            hor_traces.append([ init_point_h + [trace_w, trace_w] , \
                                 end_point_h + [trace_w, trace_w] ])

            # end_points_o_row.append(end_point_h)
            if idx_row >= n_pad_y//2:
                end_points_bott.append(end_point_h)
            else:
                end_points_top.append(end_point_h)

    end_points_bott.sort()

    # Traces to body ===========================================================
    n_pads_top = n_pad_x*n_pad_y//2
    py_i = np.absolute(end_points_top[1][1] - end_points_top[0][1])
    py_f_top = (trace_pitch / 2) * np.ones(n_pads_top)
    increment_pitch_trace_y = trace_pitch * np.arange(n_pads_top)
    py_f_top = py_f_top + increment_pitch_trace_y
    py_f_top = py_f_top[::-1]
    py_f_bott = (-1)*py_f_top

    for idx_pt, pt in enumerate(end_points_top):

        yo = pt[1]
        y1 = py_f_top[idx_pt]

        xo = pt[0] + hor_dir*py_i/np.sqrt(3)*(idx_pt)
        if y1 - yo >= 0:
            x1 = xo + hor_dir*(y1-yo)/np.sqrt(3)
        else:
            x1 = xo +  hor_dir*(-1)*(y1-yo)/np.sqrt(3)

        #horizontal traces
        init_point_h = pt
        end_point_h = [xo + offset_x, pt[1] + offset_y]
        hor_traces.append([ init_point_h + [trace_w, trace_w] , \
                              end_point_h + [trace_w, trace_w] ])

        #diagonal traces
        init_point_d = end_point_h
        end_point_d = [x1 + offset_x, \
                       y1 + offset_y]
        diag_traces.append([ init_point_d + [trace_w, trace_w] , \
                              end_point_d + [trace_w, trace_w] ])

    for idx_pt, pt in enumerate(end_points_bott):

        yo = pt[1]
        y1 = py_f_bott[idx_pt]

        xo = pt[0] + hor_dir*py_i/np.sqrt(3)*(idx_pt)
        if y1 - yo >= 0:
            x1 = xo + hor_dir*(y1-yo)/np.sqrt(3)
        else:
            x1 = xo +  hor_dir*(-1)*(y1-yo)/np.sqrt(3)

        #horizontal traces
        init_point_h = pt
        end_point_h = [xo + offset_x, pt[1] + offset_y]
        hor_traces.append([ init_point_h + [trace_w, trace_w] , \
                              end_point_h + [trace_w, trace_w] ])

        #diagonal traces
        # print("i:", i,"\nScaling of height:", i//n_pad_x)
        init_point_d = end_point_h
        end_point_d = [x1 + offset_x, \
                       y1 + offset_y]
        diag_traces.append([ init_point_d + [trace_w, trace_w] , \
                              end_point_d + [trace_w, trace_w] ])

    return diag_traces, hor_traces


'''
Vertical Trace from Electrode
'''
def vertical_trace(point=[0,0], dir='left', idx_colum=0,
                   trace_pitch=20, trace_width=10,trode_pitch=500, trode_diam=40):
    init_point = point
    if dir == 'left':
        vertical_space = 2*trode_diam + 2*trace_width + (idx_colum + 1)*trace_pitch
        end_point = [point[0], point[1] - vertical_space]
        # if vertical trace > troce pitch --> traces are going to cross!
    elif dir == 'right':
        vertical_space = trode_pitch/2 - 2*trode_diam - 2*trace_width - (idx_colum + 1)*trace_pitch
        end_point = [point[0], point[1] - vertical_space]

    ver_polyline = [init_point + [trace_width, trace_width] , \
                    end_point + [trace_width, trace_width]]
    #format for ezdxf.add_lwpolyline[[x1,y1, start_width, end_width], [x2,y2, start_width, end_width]]
    return ver_polyline, end_point

'''
Arc trace to route trace outwards towards the arms
'''
def arc_trace(point, trace_width, radius, msp, layer, dir='left'):
    if dir == 'left':
        center = [point[0] - trace_width/2 - radius, point[1]]
        msp.add_arc(center, radius, 0, -90, False, layer)
        msp.add_arc(center, radius + trace_width, 0, -90, False, layer)
        end_point = [point[0] - (radius + trace_width/2),  point[1] - (radius + trace_width/2)]
    elif dir == 'right':
        center = [point[0] + trace_width/2 + radius, point[1]]
        msp.add_arc(center, radius, 180, 270, True, layer)
        msp.add_arc(center, radius + trace_width, 180, 270, True, layer)
        end_point = [point[0] + (radius + trace_width/2),  point[1] - (radius + trace_width/2)]
    return end_point

'''
Horizontal trace to route trace outwards towards the arms
'''
def horizontal_trace(point=[0,0], dir='left', idx_col=0, side_electrodes=4,
                     trace_width=10, trode_pitch=500, ):
    init_point = point
    if dir == 'left':
        hor_space = (idx_col + 1) * trode_pitch/2 + (idx_col* trode_pitch/2)
        end_point = [point[0] - hor_space, point[1]]
    elif dir == 'right':
        hor_space = (side_electrodes - idx_col)*trode_pitch/2 + (side_electrodes - idx_col - 1) * trode_pitch/2
        end_point = [point[0] + hor_space, point[1]]

    hor_pline = [init_point + [trace_width, trace_width] , \
                 end_point + [trace_width, trace_width]]
    #format for ezdxf.add_lwpolyline[[x1,y1, start_width, end_width], [x2,y2, start_width, end_width]]
    return hor_pline, end_point
