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
    if not test_array_bounds(array_to_test,x,y):
        return None
    return array_to_test[y,x]

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

def seek_dot_on_z(dotmap,p1,p2,this_depth):
    x1,y1 = p1
    x2,y2 = p2
    y1_horizontal_move = True
    y2_horizontal_move = True
    x1_vertical_move = True
    x2_vertical_move = True
    #print "Testing %s to %s" % (p1,p2)
    if not test_dot(dotmap,x2,y2):
        #print "failed x2,y2"
        return None
    if x1 < x2:
        xstep = 1
    else:
        xstep = -1
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1,x2,xstep):
        y1_horizontal_move = y1_horizontal_move and test_dot(dotmap,x,y1)
        y2_horizontal_move = y2_horizontal_move and test_dot(dotmap,x,y2)
    for y in range(y1,y2,ystep):
        x1_vertical_move = x1_vertical_move and test_dot(dotmap,x1,y)
        x2_vertical_move = x2_vertical_move and test_dot(dotmap,x2,y)
    positions = []
    #print x1_vertical_move,x2_vertical_move,y1_horizontal_move,y2_horizontal_move
    if x1_vertical_move and y2_horizontal_move:
        positions.append([ "line",[x1,y1,this_depth],[x1,y2,this_depth] ])
        positions.append([ "line",[x1,y2,this_depth],[x2,y2,this_depth] ])
    elif x2_vertical_move and y1_horizontal_move:
        positions.append([ "line",[x1,y1,this_depth],[x2,y1,this_depth] ])
        positions.append([ "line",[x2,y1,this_depth],[x2,y2,this_depth] ])
    #print positions
    travel = abs(x1-x2) + abs(y1-y2)
    return positions

def get_direction_results(dotmap,x,y):
    stress = 0
    results = {}
    for direction in directions:
        x_delta,y_delta = directions[direction]
        next_x = x + x_delta
        next_y = y + y_delta
        result = results[direction] = test_dot(dotmap,next_x,next_y)
        if result:
            stress += 1
    results["overall"] = stress != 0
    results["stress"] = stress
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
        for direction in test_directions:
            if results[direction]:
                x_delta,y_delta = directions[direction]
                return x+x_delta, y+y_delta, results["stress"]
    if not go:
        return None
    x_delta,y_delta = directions[go]
    return x+x_delta, y+ y_delta, results["stress"]

def trace(dotmap,x,y):
    positions = []
    if test_array_bounds(dotmap,x,y) and dotmap[y,x]:
        positions.append([x,y])
        dotmap[y,x] = False
    position = next_edge(dotmap,x,y)
    while position:
        if position:
            x,y,stress = position
            dotmap[y,x] = False
        positions.append(position)
        position = next_edge(dotmap,x,y)
    return x,y,positions

def zigzag(layer,xin,yin,this_depth):
    x = xin
    y = yin
    positions = []
    #positions.append("#starting at %s,%s" % (x,y))
    pending_positions = []
    direction = last_dir = "left"
    last_directions = ["down","right","up","left"]
    measure = get_direction_results(layer,x,y)
    shift = False
    stress = measure["stress"]
    #print "A",measure
    save_points = [[x,y,this_depth]]
    while measure["overall"]:
        last_stress = stress
        stress = measure["stress"]
        #print "B",measure
        if measure["center"]:
            layer[y,x] = False
            if last_dir != direction or stress != last_stress:
                if len(pending_positions) == 1:
                    positions.append( ["dot",pending_positions[0]] )
                    #print "DOT"
                else:
                    positions.append([ "line_with_stress", pending_positions[0], pending_positions[-1], last_stress ])
                #print [ "line", pending_positions[0], pending_positions[-1] ]
                pending_positions = []
            save_points = [[x,y,this_depth]] + save_points[:10]
            pending_positions.append([x,y,this_depth])
            #print "APPEND",[x,y,this_depth], measure["stress"]
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
        measure = get_direction_results(layer,x,y)
        if not measure["overall"]:
            #print "BACKTRACING!"
            backtrack = []
            for dot in save_points:
                testx,testy,testz = dot #testz is not used
                measure = get_direction_results(layer,testx,testy)
                backtrack.append(dot)
                if measure["overall"]:
                    #print "BACKED UP to %s" % dot
                    positions.append([ "line_with_stress", pending_positions[0], pending_positions[-1], last_stress ])
                    #positions.append("#BACK UP to %s" % dot)
                    for bt in backtrack:
                        positions.append(["stress_dot",bt,0]) #no stress because we are just retracing our steps
                    pending_positions = [dot]
                    x,y = testx,testy
                    alt = last_directions.pop(-2)
                    last_directions.append(alt)
                    x += directions[direction][0]
                    y += directions[direction][1]
                    shift = True
                    break
    if pending_positions:
        #positions.append("#add pending positions")
        positions.append([ "line_with_stress", pending_positions[0], pending_positions[-1], last_stress ])
        #positions = positions + pending_positions
        x,y,z = pending_positions[-1] #discard the z
        #positions.append("#line %s to %s" % (pending_positions[0],pending_positions[-1]))
    #positions.append("#ending at %s,%s" % (x,y))
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
            cut = ["dot",cut]
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
            if len(end) < 3 or len(start) < 3:
                print "BAD LINE:",start,end
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
            start = cut[1]
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
    deduped_gcode = []
    last_line = ""
    for line in gcode:
        if line != last_line:
            deduped_gcode.append(line)
        last_line = line
    return deduped_gcode

def convert_image_to_int8_image(input):
    height,width = input.shape
    output = zeros(input.shape)
    output.dtype = int8
    for x in range(width):
        for y in range(height):
            output[x,y] = int(input[x,y])
    return output

if __name__ == "__main__":
    #cut_image = Image.open("Best Mom Ever Heart3.gif")

    if len(argv) < 7:
        print 'USAGE: python bucket_mill.py "Best Mom Ever Heart3.gif" W 200 20 3 trace test.gcode'
        print 'input file = "Best Mom Ever Heart3.gif"' #argv[1]
        print 'W or H or Width or Height = "W"' #argv[2]
        print 'measurement in mm of width or height as selected above = "200"' #argv[3]
        print 'measurement in mm of the target thickness = "20"' #argv[4]
        print 'measurement in mm of billing bit = "3"' #argv[5]
        print 'milling pattern to use (trace or zigzag) = "trace"' #argv[6]
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


    if pattern.upper() == "ZIGZAG":
        print "ZIGZAG ONE LAYER AT A TIME"
        cut_positions = []
        layer = bottom > top
        start_position = find_nearby_dot(layer,0,0)
        depth = 0
        layer = bottom > top
        while start_position:
            #cut_positions = cut_positions + ["#seek"]
            x,y = start_position
            #print start_position
            #print "%i to go" % (layer == False).sum()
            x,y,positions = zigzag(layer,x,y,depth)
            cut_positions = cut_positions + positions
            if (layer == False).all():
                print depth,"vs",bottom.max()
                depth += 1
                top = top.clip(depth,bottom)
                layer = bottom > top
                start_position = find_nearby_dot(layer,x,y)
            else:
                start_position = find_nearby_dot(layer,x,y)
                if True: #I want to be able to turn this on and off for testing
                    seek = seek_dot_on_z( bottom > depth, (x,y), start_position, depth)
                    if seek:
                        #print "SEEK ON Z!"
                        cut_positions = cut_positions + seek
                    #else:
                        #print "FAIL SEEK IN Z!"
        print "We had %i cuts and %i seeks." % (len(cut_positions)-cut_positions.count("seek"), cut_positions.count("seek"))
    elif pattern.upper() == "TRACE":
        print "TRACE AROUND THE EDGES, ONE LAYER AT A TIME"
        cut_positions = []
        layer = bottom > top
        start_position = find_nearby_dot(layer,0,0)
        depth = 0
        layer = bottom > top
        while start_position:
            cut_positions = cut_positions + ["seek"]
            x,y = start_position
            #print start_position
            x,y,positions = trace(layer,x,y)
            for position in positions:
                if len(position) == 3:
                    cx,cy,stress = position
                    cut_positions.append(["stress_dot",[cx,cy,depth],stress])
                else:
                    cx,cy = position
                    cut_positions.append(["dot",[cx,cy,depth]])
            if (layer == False).all():
                print top.max(),"vs",bottom.max()
                depth += 1
                top = top.clip(depth,bottom)
                layer = bottom > top
                start_position = find_nearby_dot(layer,x,y)
            else:
                start_position = find_nearby_dot(layer,x,y)
                if True: #I want to be able to turn this on and off for testing
                    seek = seek_dot_on_z( bottom > depth, (x,y), start_position, depth)
                    if seek:
                        #print "SEEK ON Z!"
                        cut_positions = cut_positions + seek
                    #else:
                        #print "FAIL SEEK IN Z!"
        print "We had %i cuts and %i seeks." % (len(cut_positions)-cut_positions.count("seek"), cut_positions.count("seek"))

    print "TEST GCODE GENERATION"
    gcode = cut_to_gcode(cut_positions)
    for line in gcode:
        output_file.write(line+"\r\n")
    output_file.close()