from PIL import Image
import sys
from numpy import array,zeros,searchsorted,amax,int8,int16,int32
from scipy.misc import imsave,imresize
from sys import argv
integer = int32
try:
    import scipy
    import scipy.misc
except ImportError as e:
    print(e)

directions = {"up":[0,-1],"down":[0,+1],"left":[-1,0],"right":[+1,0],"center":[0,0]}

def test_array_bounds(array_to_test,x,y):
    y_size,x_size = array_to_test.shape
    #print "TEST",x_size,y_size,(x >= 0),(x < x_size),(y >= 0),(y < y_size)
    return (x >= 0) and (x < x_size) and (y >= 0) and (y < y_size)

def test_dot(array_to_test,x,y):
    return test_array_bounds(array_to_test,x,y) and array_to_test[y,x]

def find_nearby_dot(dotmap,x,y):
    height,width = top.shape
    m = test_dot(dotmap,x,y)
    if m:
        return x,y
    else:
        for r in range(1,max(height,width)):
            for offset in range(0,r+1):
                for test in [[x-offset,y-r],
                    [x-offset,y+r],
                    [x+offset,y-r],
                    [x+offset,y+r],
                    [x+r,y+offset],
                    [x-r,y+offset],
                    [x+r,y-offset],
                    [x-r,y-offset]]:
                    m = test_dot(dotmap,test[0],test[1])
                    if m:
                        return test
        r += 1
    return None

def get_direction_results(dotmap,x,y):
    results = {}
    for direction in directions:
        x_delta,y_delta = directions[direction]
        next_x = x + x_delta
        next_y = y + y_delta
        if test_array_bounds(dotmap,next_x,next_y):
            results[direction] = dotmap[next_y,next_x]
        else:
            results[direction] = None
    return results

def next_edge(dotmap, x, y, seek=True, clockwise=True):
    #assumes the directions in the "directions" variable is set correctly for the map
    #if that assumption is wrong, it may go the wrong way around
    positions = [[x,y]]
    go = None
    results = get_direction_results(dotmap,x,y)
    if clockwise:
        test_directions = ["down","left","up","right"]
    else:
        test_directions = ["down","right","up","left"]
    dir_a,dir_b,dir_c,dir_d = test_directions
    if not results[dir_a] and results[dir_b]:
        go = dir_b
    elif not results[dir_b] and results[dir_c]:
        go = dir_c
    elif not results[dir_c] and results[dir_d]:
        go = dir_d
    elif not results[dir_d] and results[dir_a]:
        go = dir_a
    elif seek:
        for direction in results:
            if results[direction]:
                x_delta,y_delta = directions[direction]
                return x+x_delta, y+y_delta
    if not go:
        return None
    x_delta,y_delta = directions[go]
    return x+x_delta, y+ y_delta

def follow_edge(dotmap,x,y):
    positions = []
    if test_array_bounds(dotmap,x,y) and dotmap[y,x]:
        positions.append([x,y])
        dotmap[y,x] = False
    position = next_edge(dotmap,x,y)
    while position:
        if position:
            x,y = position
            dotmap[y,x] = False
        positions.append(position)
        position = next_edge(dotmap,x,y)
    return x,y,positions

def next_zigzag(dotmap, x, y):
    #assumes the directions in the "directions" variable is set correctly for the map
    #if that assumption is wrong, it may go the wrong way around
    positions = [[x,y]]
    go = None
    results = get_direction_results(dotmap,x,y)
    test_directions = ["left","down","right","up"]
    dir_a,dir_b,dir_c,dir_d = test_directions
    if not results[dir_a] and results[dir_b] and results[dir_c]:
        go = dir_a
    elif not results[dir_c] and results[dir_a] or results[dir_b]:
        go = dir_c
    else:
        for direction in results:
            if results[direction]:
                x_delta,y_delta = directions[direction]
                return x+x_delta, y+y_delta
    if not go:
        return None
    x_delta,y_delta = directions[go]
    return x+x_delta, y+ y_delta

def zigzag(dotmap,x,y):
    positions = []
    position = next_zigzag(dotmap,x,y)
    while position:
        if position:
            x,y = position
            dotmap[y,x] = False
        positions.append(position)
        position = next_zigzag(dotmap,x,y)
    return x,y,positions

def downsample_to_bit_diameter(image, scale, bit_diameter):
    #print image.shape
    height,width = image.shape
    scaled_width = width*scale
    scaled_height = height*scale
    output = zeros((scaled_height,scaled_width),dtype=integer)
    #coordinates are height,width
    #print "output is %s" % str(output.shape)
    print "target width, target height, scale = ",scaled_width,scaled_height,scale
    #print "downsample_to_bit_diamter width=%i height=%i" % (width,height)
    #print "downsample_to_bit_diameter shape:", output.shape
    #print len(image[::bit_diameter,::bit_diameter].tolist()[0])
    #help(searchsorted)
    #scale
    for y in range(int(scaled_height)):
        for x in range(int(scaled_width)):
            #print "%s:%s,%s:%s = %s" % ( (y)/scale,(y+bit_diameter)/scale,x/scale,(x+bit_diameter)/scale,amax(image[(y)/scale:(y+bit_diameter)/scale,x/scale:(x+bit_diameter)/scale]))
            left = max( (x-bit_diameter/2)/scale,  0)
            right = min( (x+bit_diameter/2)/scale,  width)
            top = max( (y-bit_diameter/2)/scale, 0)
            bottom = min( (y+bit_diameter/2)/scale, height)
            output[y,x] = amax(image[top:bottom,left:right])
    #print "downsample_to_bit_diameter shape:", output.shape
    return output


def downsample(image, image_x_axis, image_y_axis, x_bounds, y_bounds, x_resolution, y_resolution):
    x_resolution, y_resolution = int(round(x_resolution)), int(round(y_resolution))
    x_bounds = np.searchsorted(image_x_axis, x_bounds)
    y_bounds = np.searchsorted(image_y_axis, y_bounds)
    #y_bounds = image.shape[0] + 1 - y_bounds[::-1]
    subset = image[y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1]]
    x_downsample_factor = max(round(subset.shape[1] / x_resolution / 3.), 1)
    y_downsample_factor = max(round(subset.shape[0] / y_resolution / 3.), 1)
    subset = subset[::y_downsample_factor,::x_downsample_factor]
    image = scipy.misc.imresize(subset, (y_resolution,x_resolution), interp='nearest')
    bounds = image_x_axis[x_bounds[0]:x_bounds[1]]
    dw = np.max(bounds) - np.min(bounds)
    bounds = image_y_axis[y_bounds[0]:y_bounds[1]]
    dh = np.max(bounds) - np.min(bounds)
    return {'data': image,
            'offset_x': image_x_axis[x_bounds[0]],
            'offset_y': image_y_axis[y_bounds[0]],
            'dw': dw,
            'dh': dh,
            'subset': subset,
    }


def make_top(bottom):
    return zeros(bottom.shape,dtype=integer)

def test_xy(top,bottom,x,y):
    #print "x: %i\ty: %i" % (x,y)
    height,width = top.shape
    if x >= 0 and y >= 0 and y < height and x < width:
        #print bottom[x,y],top[x,y],bottom[x,y] - top[x,y]
        try:
            result = bottom[y,x] - top[y,x]
        except:
            result = None
    else:
        result = None
    #if result:
    #    print "RESULT:",x,y,result
    return result

def scan_column(top,bottom,y):
    #so damned slow
    x=0
    positions = []
    debug = False
    height,width = top.shape
    #print test_xy(top,bottom,10,10)
    while x < width and x >= 0:
        depth = test_xy(top,bottom,x,y)
        #print top[x,y],bottom[x,y],depth
        if depth:
            if debug: print "POSITION: %s" % x
            top[x,y] += 1
            positions.append([x,y,top[y,x]])
        l_depth = test_xy(top,bottom,x-1,y)
        r_depth = test_xy(top,bottom,x+1,y)
        if r_depth == None and l_depth != None and l_depth > 0:
            if debug: print "LEFT"
            x -= 1
        elif depth <= 1 and l_depth != None and l_depth > 0:
            if debug: print "LEFT"
            x -= 1
        elif depth and not r_depth:
            if debug: print "STAY"
        else:
            if debug: print "RIGHT"
            x += 1
    return positions

def test_multi_xy(top,bottom,x,y):
    measure = {"up":0,"down":0,"left":0,"right":0,"center":0}
    measure_total = 0
    for direction in directions:
        offset = directions[direction]
        m = test_xy(top,bottom,x+offset[0],y+offset[1])
        if m != None:
            measure_total += m
            measure[direction] = m
    measure["overall"] = measure_total
    return measure

def test_multi_xy_matching_depth(top,bottom,x,y,depth,test_depth_vs_top=True):
    height,width = top.shape
    measure = {}
    overall = False
    stress = 0
    for direction in directions:
        offset = directions[direction]
        try:
            test_y = y+offset[1]
            test_x = x+offset[0]
            if test_y >= 0 and test_x >= 0 and test_y < height and test_x < width:
                top_measure = top[test_y,test_x]
                bottom_measure = bottom[y+offset[1],x+offset[0]]
                if test_depth_vs_top:
                    measure[direction] = (bottom_measure >= depth) and (depth > top_measure)
                    if measure[direction]: stress += 1
                else:
                    measure[direction] = (bottom_measure >= depth) and (bottom_measure > top_measure)
                    if measure[direction]: stress += 1
            else:
                measure[direction] = None
            #print direction,bottom_measure,depth,top_measure,(bottom_measure >= depth),(depth > top_measure)
            if measure[direction]:
                overall = True
        except:
            measure[direction] = False
    measure["overall"] = overall
    measure["stress"] = stress
    return measure

def follow_edge1(top,bottom,nx,ny):
    x = nx
    y = ny
    #something is wrong with this....
    positions = []
    pending_positions = []
    dir = 0
    last_dir = 0
    measure = test_multi_xy(top,bottom,x,y)
    direction_keys = ["up","right","down","left","center"]
    while measure["overall"]:
        #print measure
        if measure["center"]:
            top[y,x] += 1
            if last_dir != dir:
                positions.append([ "line", pending_positions[0], pending_positions[-1] ])
                #print [ "line", pending_positions[0], pending_positions[-1] ]
                pending_positions = []
            pending_positions.append([x,y,top[y,x]])
            #print "APPEND",[x,y,top[y][x]]
        last_dir = dir
        #print "%s,%s,%s while bottom is %s" % (x,y,top[y][x],bottom[y][x])
        found = False
        #print "at %s,%s and facing %s with cutting = %s" % (x,y,direction_keys[dir],measure["center"])
        while not found:
            direction = direction_keys[dir]
            #print direction,measure[direction]
            if direction == "center":
                dir = 0
                direction = "up"
                found = True
            if measure[direction]:
                x += directions[direction][0]
                y += directions[direction][1]
                found = True
                if direction == "center":
                    dir = 0
            else:
                dir += 1
                if direction == "center":
                    found = True #but not really
        measure = test_multi_xy(top,bottom,x,y)
    if pending_positions:
        #positions.append("#add pending positions")
        positions.append([ "line", pending_positions[0], pending_positions[-1] ])
        x = pending_positions[-1][0]
        y = pending_positions[-1][1]
        #positions.append("#line %s to %s" % (pending_positions[0],pending_positions[-1]))
    #positions.append("#ending at %s,%s" % (x,y))
    return top,positions,x,y

def follow_edge2(top,bottom,x,y):
    positions = []
    pending_positions = []
    dir = 0
    last_dir = 0
    direction="up"
    direction_keys = ["up","right","down","left","center"]
    depth = top[y,x]+1
    measure = test_multi_xy_matching_depth(top,bottom,x,y,depth)
    #print "A",measure
    while measure["overall"]:
        #print "B",measure
        if measure["center"]:
            top[y,x] += 1
            if last_dir != dir or direction == "center":
                positions.append([ "line", pending_positions[0], pending_positions[-1] ])
                #print [ "line", pending_positions[0], pending_positions[-1] ]
                pending_positions = []
            pending_positions.append([x,y,top[y,x]])
        #print "APPEND",[x,y,top[y][x]]
        last_dir = dir
        found = False
        while not found:
            direction = direction_keys[dir]
            #print direction,measure[direction]
            if direction == "center":
                dir = 0
                direction = "up"
                found = True
            if measure[direction]:
                x += directions[direction][0]
                y += directions[direction][1]
                found = True
                if direction == "center":
                    dir = 0
            else:
                dir += 1
                if direction == "center":
                    found = True #but not really
        measure = test_multi_xy_matching_depth(top,bottom,x,y,depth)
        if not measure["overall"]:
            depth += 1
        measure = test_multi_xy_matching_depth(top,bottom,x,y,depth)
    if pending_positions:
        positions.append([ "line", pending_positions[0], pending_positions[-1] ])
        x = pending_positions[-1][0]
        y = pending_positions[-1][1]
    return top,positions,x,y

def follow_edge3(top,bottom,x,y):
    positions = []
    pending_positions = []
    max_stress = 0
    direction = last_dir = "left"
    left = False
    up = False
    horizontal = True
    depth = top[y,x]+1
    measure = test_multi_xy_matching_depth(top,bottom,x,y,depth)
    #print "A",measure
    while measure["overall"]:
        #print "B",measure
        left = True
        up = True
        if measure["center"]:
            depth = top[y,x]+1
            top[y][x] += 1
            if last_dir != direction:
                positions.append([ "line_with_stress", pending_positions[0], pending_positions[-1], max_stress ])
                #print [ "line", pending_positions[0], pending_positions[-1] ]
                pending_positions = []
                max_stress = 0
            pending_positions.append([x,y,top[y,x]])
            max_stress = max(max_stress,measure["stress"])
            #print "APPEND",[x,y,top[y][x]],measure["stress"]
        last_dir = direction
        if measure["left"]:
            direction = "left"
        elif measure["right"]:
            direction = "right"
        elif measure["up"]:
            direction = "up"
        elif measure["down"]:
            direction = "down"
        else:
            while not measure["overall"]:
                depth += 1
                measure = test_multi_xy_matching_depth(top,bottom,x,y,depth)
            if measure["left"]:
                direction = "left"
            elif measure["right"]:
                direction = "right"
            elif measure["up"]:
                direction = "up"
            elif measure["down"]:
                direction = "down"
        x += directions[direction][0]
        y += directions[direction][1]
        measure = test_multi_xy_matching_depth(top,bottom,x,y,depth)
    if pending_positions:
        positions.append([ "line_with_stress", pending_positions[0], pending_positions[-1], max_stress ])
        x = pending_positions[-1][0]
        y = pending_positions[-1][1]
    return top,positions,x,y

def follow_edge4(top,bottom,xin,yin):
    x = xin
    y = yin
    positions = []
    #positions.append("#starting at %s,%s" % (x,y))
    pending_positions = []
    direction = last_dir = "left"
    last_directions = ["down","right","up","left"]
    depth = top[y,x]+1
    measure = test_multi_xy_matching_depth(top,bottom,x,y,depth)
    shift = False
    max_stress = 0
    #print "A",measure
    while measure["overall"]:
        #print "B",measure
        if measure["center"]:
            depth = top[y,x]+1
            top[y,x] += 1
            if last_dir != direction:
                positions.append([ "line_with_stress", pending_positions[0], pending_positions[-1], max_stress ])
                #print [ "line", pending_positions[0], pending_positions[-1] ]
                pending_positions = []
                max_stress = 0
            pending_positions.append([x,y,top[y,x]])
            max_stress = max(max_stress,measure["stress"])
            #print "APPEND",[x,y,top[y][x]], measure["overall"]
        last_dir = direction
        if shift:
            alt = last_directions.pop(-3)
            last_directions.append(alt)
            shift = False
        elif measure[last_directions[-1]]:
            #keep going that direction
            pass
        elif measure[last_directions[-2]]:
            alt = last_directions.pop(-2)
            last_directions.append(alt)
            shift = True
        #we don't check for -3 when not in shift mode because that would mean it was L/R when we are going R/L
        elif measure[last_directions[-4]]:
            alt = last_directions.pop(-4)
            last_directions.append(alt)
        direction = last_directions[-1]
        x += directions[direction][0]
        y += directions[direction][1]
        measure = test_multi_xy_matching_depth(top,bottom,x,y,depth)
    if pending_positions:
        #positions.append("#add pending positions")
        positions.append([ "line_with_stress", pending_positions[0], pending_positions[-1], max_stress ])
        #positions = positions + pending_positions
        x = pending_positions[-1][0]
        y = pending_positions[-1][1]
        #positions.append("#line %s to %s" % (pending_positions[0],pending_positions[-1]))
    #positions.append("#ending at %s,%s" % (x,y))
    return top,positions,x,y

def find_nearby_cut(top,bottom,x,y):
    valid = False
    m = test_xy(top,bottom,x,y)
    #print x,y,m
    if m != None:
        if m > 0:
            return x,y
    r = 1
    valid = True
    while valid:
        valid = False
        for offset in range(-r,r+1):
            #print r,offset
            m = test_xy(top,bottom,x+offset,y-r)
            if m != None:
                valid = True
                if m > 0:
                    return x+offset,y-r
            m = test_xy(top,bottom,x+offset,y+r)
            if m != None:
                valid = True
                if m > 0:
                    return x+offset,y+r
            m = test_xy(top,bottom,x+r,y+offset)
            if m != None:
                valid = True
                if m > 0:
                    return x+r,y+offset
            m = test_xy(top,bottom,x-r,y+offset)
            if m != None:
                valid = True
                if m > 0:
                    return x-r,y+offset
        r += 1
    return None

def find_nearby_cut2(top,bottom,x,y):
    height,width = top.shape
    valid = False
    m = test_xy(top,bottom,x,y)
    #print x,y,m
    if m != None:
        if m > 0:
            return x,y
    r = 1
    valid = True
    while valid:
        valid = False
        for offset in range(0,r+1):
            #print r,offset
            m = test_xy(top,bottom,x-offset,y-r)
            if m != None:
                valid = True
                if m > 0:
                    return x-offset,y-r
            m = test_xy(top,bottom,x-offset,y+r)
            if m != None:
                valid = True
                if m > 0:
                    return x-offset,y+r
            m = test_xy(top,bottom,x+offset,y-r)
            if m != None:
                valid = True
                if m > 0:
                    return x+offset,y-r
            m = test_xy(top,bottom,x+offset,y+r)
            if m != None:
                valid = True
                if m > 0:
                    return x+offset,y+r
            m = test_xy(top,bottom,x+r,y+offset)
            if m != None:
                valid = True
                if m > 0:
                    return x+r,y+offset
            m = test_xy(top,bottom,x-r,y+offset)
            if m != None:
                valid = True
                if m > 0:
                    return x-r,y+offset
            m = test_xy(top,bottom,x+r,y-offset)
            if m != None:
                valid = True
                if m > 0:
                    return x+r,y-offset
            m = test_xy(top,bottom,x-r,y-offset)
            if m != None:
                valid = True
                if m > 0:
                    return x-r,y-offset
        r += 1
    return None

def cut_to_gcode(cuts,x=0,y=0,z=0, cut_speed=500, z_cut_speed=300, z_rapid_speed=400, rapid_speed=700, safe_distance=2,unit="mm",offsets=[0,0,0],minimum_stress=1):
    #if the next cut location is more than one space away, go to safe_distance before moving
    #if the next cut location is diagonal, go to safe_distance before moving
    #if the next cut depth is not as deep as the current depth, change to the new depth before moving
    #after moving to the next cut location, make sure we are at the right depth
    #if the next cut is a string, it might be "seek" which will tell us we might need to switch to safe_distance before moving

    calculated_cut_speeds = []
    for stress in range(6):
        calculated_speed = ( stress*cut_speed + (5-stress)*rapid_speed )/ 5
        calculated_cut_speeds.append(calculated_speed)
        
    print "calculated cut speeds by stress:",calculated_cut_speeds
    gcode = []
    if unit.lower() == "mm":
        gcode.append("G21")
    elif unit.lower() == "inch":
        gcode.append("G20")
    gcode.append("G17 G90 G64 M3 S3000 M7")
    gcode.append("G0 X0 Y0 F%f" % rapid_speed)
    gcode.append("G0 Z0 F%f" % z_rapid_speed)
    gcode.append("G1 X0 Y0 F%f" % cut_speed)
    gcode.append("G1 Z0 F%f" % z_cut_speed)
    for cut in cuts:
        if type(cut) == str:
            command = "#"
        elif type(cut[0]) == str:
            command = cut[0]
        else:
            command = "dot"
        calculated_speed = cut_speed #default to the slow cuts
        #print "CUT[0]:",cut[0]
        if cut == "seek":
            #gcode.append("#seek")
            continue
        elif command == "#" and ( cut.startswith("#") or cut.startswith("(") ):
            gcode.append("(%s)" % cut)
            continue
        elif command in ["line","line_with_stress"]: #start cut from point A and end it later in the loop
            start = cut[1]
            end = cut[2]
            if end[2] <> start[2]:
                print "Z mismatch on %s!" % cut

            travel = abs(start[0]-x) + abs(start[1]-y)
            if command == "line_with_stress":
                stress = min( cut[3] + minimum_stress, 5 )
                calculated_speed = calculated_cut_speeds[stress]
            else:
                calculated_speed = cut_speed
            #gcode.append("#start line")
            if travel > 1:
                gcode.append("G0 F%.3f Z%.3f" % (z_rapid_speed,safe_distance))
                gcode.append("G0 F%.3f X%.3f Y%.3f" % (rapid_speed,offsets[0]+start[0],offsets[1]+start[1]))
                gcode.append("G1 F%.3f Z%.3f" % (z_cut_speed,offsets[2]-start[2]))
                z = start[2]
            elif z > start[2]: #we must go up
                #print z,offsets[2]-cut[2]
                gcode.append("G0 F%.3f Z%.3f" % (z_rapid_speed,offsets[2]-start[2]))
                gcode.append("G1 F%.3f X%.3f Y%.3f" % (calculated_speed,offsets[0]+start[0],offsets[1]+start[1]))
                z = start[2]
            else:
                gcode.append("G1 F%.3f X%.3f Y%.3f" % (calculated_speed,offsets[0]+start[0],offsets[1]+start[1]))
                z = start[2]
            gcode.append("G1 F%.3f X%.3f Y%.3f" % (calculated_speed,offsets[0]+end[0],offsets[1]+end[1]))
            if end[2] != z:
                gcode.append("G1 F%.3f Z%.3f" % (z_cut_speed,offsets[2]-end[2]))
            x,y,z = end
        elif command == "stress_segment": #cut from the last point to this point
            end = cut[1]
            if end[2] <> z:
                print "Z mismatch on %s!" % cut
            stress = min( cut[2] + minimum_stress, 5 )
            calculated_speed = calculated_cut_speeds[stress]
            #gcode.append("#start line")
            gcode.append("G1 F%.3f X%.3f Y%.3f" % (calculated_speed, offsets[0]+end[0], offsets[1]+end[1]))
            if end[2] != z:
                gcode.append("G1 F%.3f Z%.3f" % (z_cut_speed,offsets[2]-end[2]))
            x,y,z = end
        elif command in ["dot","stress_dot"]:
            if command == "stress_dot":
                stress = min( cut[2] + minimum_stress, 5 )
                calculated_speed = calculated_cut_speeds[stress]
            else:
                calculated_speed = cut_speed
            start = cut
            travel = abs(start[0]-x) + abs(start[1]-y)
            if travel > 1:
                gcode.append("G0 F%.3f Z%.3f" % (z_rapid_speed,safe_distance))
                gcode.append("G0 F%.3f X%.3f Y%.3f" % (rapid_speed,offsets[0]+start[0],offsets[1]+start[1]))
                z = safe_distance
            elif z > start[2]: #we must go up
                #print z,offsets[2]-cut[2]
                gcode.append("G0 F%.3f Z%.3f" % (z_rapid_speed,offsets[2]-start[2]))
                gcode.append("G1 F%.3f X%.3f Y%.3f" % (calculated_speed,offsets[0]+start[0],offsets[1]+start[1]))
                z = start[2]
            else:
                gcode.append("G1 F%.3f X%.3f Y%.3f" % (calculated_speed,offsets[0]+start[0],offsets[1]+start[1]))
                z = start[2]
            if z != start[2]:
                gcode.append("G1 F%.3f Z%.3f" % (z_cut_speed,offsets[2]-start[2]))
            x,y,z = start
        else:
            print "UNHANDLED COMMAND: %s" % command
            gcode.append("UNHANDLED COMMAND: %s" % command)
        #gcode.append("(done with this command)")
    return gcode

def convert_image_to_int8_image(input):
    height,width = input.shape
    output = zeros(input.shape)
    output.dtype = int8
    for x in range(width):
        for y in range(height):
            output[x,y] = int(input[x,y])
    return output
    
#cut_image = Image.open("Best Mom Ever Heart3.gif")

if len(argv) < 7:
    print 'USAGE: python bucket_mill.py "Best Mom Ever Heart3.gif" W 200 20 3 edge1 test.gcode'
    print 'input file = "Best Mom Ever Heart3.gif"' #argv[1]
    print 'W or H or Width or Height = "W"' #argv[2]
    print 'measurement in mm of width or height as selected above = "200"' #argv[3]
    print 'measurement in mm of the target thickness = "20"' #argv[4]
    print 'measurement in mm of billing bit = "3"' #argv[5]
    print 'milling pattern to use (scan or edge1 or edge2 or edge3) = "edge1"' #argv[6]
    print 'optionally defines output gcode file = "test.gcode". Note that it is otherwise the input file + ".rough-cut.gcode"' #argv[7]

input_file = argv[1]
dimension_restricted = argv[2]
dimension_measurement = argv[3]
thickness = argv[4]
bit_diameter = argv[5]
pattern = argv[6]
output_filename = argv[1].rsplit(".")[0] + ".rough-cut.gcode"
if len(argv) == 8:
    output_filename = argv[7]
output_file = open(output_filename, "w")

print "input file: %s" % input_file
print "dimension_restricted: %s" % dimension_restricted
print "dimension_measurement: %s" % dimension_measurement
print "thickness: %s" % thickness
print "bit diameter: %s" % bit_diameter
print "pattern: %s" % pattern
print "output file name: %s" % output_filename

cut_image = Image.open(input_file)
cut_image.convert("L") # Convert image to grayscale

cut_image = cut_image.transpose(Image.FLIP_TOP_BOTTOM) #The bottom left is 0,0 in the CNC, but the upper left is 0,0 in the image

width, height = cut_image.size
#help(cut_image)
#print cut_image.tobytes()
print len(cut_image.getdata())
print "cut image width,height,overall size:",width,height,width*height
x=0

row = []

bottom = array(cut_image)
print "bottom shape:",bottom.shape
shape = (None,None)
#print bottom.tolist()[100]
if dimension_restricted.upper() in ["W","WIDTH"]:
    bottom = downsample_to_bit_diameter(bottom,float(dimension_measurement)/width,int(bit_diameter))
    print "scale = %s" % (float(dimension_measurement)/width)
elif dimension_restricted.upper() in ["H","HEIGHT"]:
    bottom = downsample_to_bit_diameter(bottom,float(dimension_measurement)/height,int(bit_diameter))
    print "scale = %s" % (float(dimension_measurement)/height)

print "min-max before:",bottom.min(),bottom.max()
#print bottom.tolist()[10]
bottom = int(thickness) -1 - bottom * int(thickness) / 256
print "min-max after:",bottom.min(),bottom.max()
#bottom = convert_image_to_int8_image(bottom)
#bottom.dtype = int8
print "target shape:",shape
print "bottom shape:",bottom.shape
#for row in bottom.tolist():
#    print row

top = make_top(bottom)
print "THE TOP IS MADE"
#for row in top.tolist():
#    print row
print "bottom shape:",bottom.shape, "top shape:",top.shape
print "test shape:",shape

#exit()
#help(bottom)
#print dir(bottom)

#print bottom.tolist()
#imsave("test.jpg",bottom)
#exit()

#print bottom.tolist()
#help(bottom)

#top = imresize(bottom,(225,210)) #in this way, we can scale down the image quickly to the dimensions in milimeters
#print top.tolist()
#exit()

cut_positions = []

if pattern.upper() == "SCAN":
    print "TEST ONE COLUMN AT A TIME"
    cut_positions = []
    total_cuts = 0
    for y in range(height):    
        print "scanning column %i" % y
        new_positions = scan_column(top,bottom,y)
        #cut_positions_by_row.append(new_positions)
        cut_positions = cut_positions + new_positions
        total_cuts += len(new_positions)
        #print new_positions

    #print cut_positions

    #print "TOP\t\t\t\t\tBOTTOM"
    #for row in top.tolist():
    #    print row

    print "CUTS"
    #for y in range(height):
    #    total_cuts += len(cut_positions_by_row[y])
    #    cuts_removed_from_rows = cuts_removed_from_rows + cut_positions[y]
        #print cut_positions[y]
    
    print "We had %i cuts." % total_cuts

elif pattern.upper() == "EDGE1":
    print
    print "TEST AN EDGE FOLLOW METHOD"
    cut_positions = []
    start_position = find_nearby_cut2(top,bottom,0,0)
    print "Start position:",start_position
    while start_position:
        cut_positions = cut_positions + ["seek"]
        x,y = start_position
        #cut_positions.append("#start at %s,%s" % (x,y))
        top,new_cuts,x,y = follow_edge1(top,bottom,x,y)
        cut_positions = cut_positions + new_cuts
        #cut_positions.append("#seek starting at %s,%s" % (x,y))
        start_position = find_nearby_cut2(top,bottom,x,y)
        
        
    #print "TOP\t\t\t\t\tBOTTOM"
    #for y in range(len(top)):
    #    print top[y],bottom[y]
    print "CUTS"
    #for i in range(20):
    #    print cut_positions[i]
    print "We had %i cuts and %i seeks." % (len(cut_positions)-cut_positions.count("seek"), cut_positions.count("seek"))
    print

elif pattern.upper() == "EDGE2":
    print "TEST ANOTHER EDGE FOLLOW METHOD, BUT WITH DEPTH MATCHING"
    cut_positions = []
    start_position = find_nearby_cut2(top,bottom,0,0)
    print start_position
    while start_position:
        cut_positions = cut_positions + ["seek"]
        x,y = start_position
        #print start_position
        top,new_cuts,x,y = follow_edge2(top,bottom,x,y)
        cut_positions = cut_positions + new_cuts
        start_position = find_nearby_cut2(top,bottom,x,y)
        
    #print "TOP\t\t\t\t\tBOTTOM"
    #for y in range(len(top)):
    #    print top[y],bottom[y]
    #print "CUTS"
    #print cut_positions
    print "We had %i cuts and %i seeks." % (len(cut_positions)-cut_positions.count("seek"), cut_positions.count("seek"))

elif pattern.upper() == "EDGE3":
    print "TEST ANOTHER EDGE FOLLOW METHOD, BUT WITH DEPTH MATCHING"
    cut_positions = []
    start_position = find_nearby_cut2(top,bottom,0,0)
    print start_position
    while start_position:
        cut_positions = cut_positions + ["seek"]
        x,y = start_position
        #print start_position
        top,new_cuts,x,y = follow_edge3(top,bottom,x,y)
        cut_positions = cut_positions + new_cuts
        start_position = find_nearby_cut2(top,bottom,x,y)
        
    #print "TOP\t\t\t\t\tBOTTOM"
    #for y in range(len(top)):
    #    print top[y],bottom[y]
    #print "CUTS"
    #print cut_positions
    print "We had %i cuts and %i seeks." % (len(cut_positions)-cut_positions.count("seek"), cut_positions.count("seek"))

elif pattern.upper() == "EDGE4":
    print "TEST ANOTHER EDGE FOLLOW METHOD, BUT WITH DEPTH MATCHING"
    cut_positions = []
    start_position = find_nearby_cut2(top,bottom,0,0)
    print start_position
    while start_position:
        cut_positions = cut_positions + ["seek"]
        x,y = start_position
        #print start_position
        top,new_cuts,x,y = follow_edge4(top,bottom,x,y)
        cut_positions = cut_positions + new_cuts
        #cut_positions.append("#seek starting at %s,%s" % (x,y))
        start_position = find_nearby_cut2(top,bottom,x,y)
        
    #print "TOP\t\t\t\t\tBOTTOM"
    #for y in range(len(top)):
    #    print top[y],bottom[y]
    #print "CUTS"
    #print cut_positions
    print "We had %i cuts and %i seeks." % (len(cut_positions)-cut_positions.count("seek"), cut_positions.count("seek"))
elif pattern.upper() == "EDGE5":
    print "TEST ANOTHER EDGE FOLLOW METHOD, ONE LAYER AT A TIME"
    cut_positions = []
    layer = bottom > top
    start_position = find_nearby_dot(layer,0,0)
    depth = 0
    layer = bottom > top
    while start_position:
        cut_positions = cut_positions + ["seek"]
        x,y = start_position
        #print start_position
        x,y,positions = follow_edge(layer,x,y)
        for position in positions:
            cx,cy = position
            cut_positions.append([cx,cy,depth])
        if (layer == False).all():
            print top.max(),"vs",bottom.max()
            depth += 1
            top = top.clip(depth,bottom)
            layer = bottom > top    
        start_position = find_nearby_dot(layer,x,y)
    print "We had %i cuts and %i seeks." % (len(cut_positions)-cut_positions.count("seek"), cut_positions.count("seek"))
elif pattern.upper() == "EDGE6":
    print "TEST ANOTHER EDGE FOLLOW METHOD, ONE LAYER AT A TIME"
    cut_positions = []
    layer = bottom > top
    start_position = find_nearby_dot(layer,0,0)
    depth = 0
    layer = bottom > top
    while start_position:
        #print top.max(),"vs",bottom.max()
        cut_positions = cut_positions + ["seek"]
        x,y = start_position
        #print start_position
        #print "DEBUG",layer.max() and start_position
        #while layer.max() and start_position:
        print "START: %s,%s,%s" % (x,y,depth), (layer==True).sum(),"to go"
            #print layer.max(),layer.shape
        x,y,positions = zigzag(layer,x,y)
        for position in positions:
            cx,cy = position
            cut_positions.append([cx,cy,depth])
        #print len(positions)
        #print positions
        #cut_positions.append("#seek starting at %s,%s" % (x,y))
            #print layer
            #print start_position
        if (layer == False).all():
            depth += 1
            top = top.clip(depth,bottom)
            layer = bottom > top
        start_position = find_nearby_dot(layer,x,y)

    #print "TOP\t\t\t\t\tBOTTOM"
    #for y in range(len(top)):
    #    print top[y],bottom[y]
    #print "CUTS"
    #print cut_positions
    print "We had %i cuts and %i seeks." % (len(cut_positions)-cut_positions.count("seek"), cut_positions.count("seek"))

print "TEST GCODE GENERATION"
gcode = cut_to_gcode(cut_positions)
for line in gcode:
    output_file.write(line+"\r\n")
output_file.close()