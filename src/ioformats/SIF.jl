# SIF.jl, adds an imread function for Andor .sif images
# 2013 Ronald S. Rock, Jr.

import Images.imread
function imread{S<:IO}(stream::S, ::Type{Images.AndorSIF})
    # line 1
    seek(stream, 0)
    l = strip(readline(stream))
    l == "Andor Technology Multi-Channel File" || error("Not an Andor file: " * l)

    # line 2
    l = strip(readline(stream))
    l == "65538 1" || error("Unknown Andor version number at line 2: " * l)

    # line 3: TInstaImage thru "Head model"
    l = strip(readline(stream))
    fields = split(l)
    fields[1] == "65547" || fields[1] == "65558" ||
        error("Unknown TInstaImage version number at line 3: " * fields[1])

    ixon = Dict{Any,Any}()
    ixon["data_type"] = int(fields[2])
    ixon["active"] = int(fields[3])
    ixon["structure_vers"] = int(fields[4]) # (== 1)
    # date is recored as seconds counted from 1970.1.1 00:00:00
    ixon["date"] = int(fields[5]) # need to convert to actual date
    ixon["temperature"] = max(int(fields[6]), int(fields[48]))
    ixon["temperature_stable"] = int(fields[6]) != -999
    ixon["head"] = fields[7]
    ixon["store_type"] = fields[8]
    ixon["data_type"] = fields[9]
    ixon["mode"] = fields[10]
    ixon["trigger_source"] = fields[11]
    ixon["trigger_level"] = fields[12]
    ixon["exposure_time"] = float(fields[13])
    ixon["frame_delay"] = float(fields[14])
    ixon["integration_cycle_time"] = float(fields[15])
    ixon["no_integrations"] = int(fields[16])
    ixon["sync"] = fields[17]
    ixon["kin_cycle_time"] = float(fields[18])
    ixon["pixel_readout_time"] = float(fields[19])
    ixon["no_points"] = int(fields[20])
    ixon["fast_track_height"] = int(fields[21])
    ixon["gain"] = int(fields[22])
    ixon["gate_delay"] = float(fields[23])
    ixon["gate_width"] = float(fields[24])
    ixon["gate_step"] = float(fields[25])
    ixon["track_height"] = int(fields[26])
    ixon["series_length"] = int(fields[27])
    ixon["read_pattern"] = fields[28]
    ixon["shutter_delay"] = fields[29]
    ixon["st_center_row"] = int(fields[30])
    ixon["mt_offset"] = int(fields[31])
    ixon["operation_mode"] = fields[32]
    ixon["flipx"] = fields[33]
    ixon["flipy"] = fields[34]
    ixon["clock"] = fields[35]
    ixon["aclock"] = fields[36]
    ixon["MCP"] = fields[37]
    ixon["prop"] = fields[38]
    ixon["IOC"] = fields[39]
    ixon["freq"] = fields[40]
    ixon["vert_clock_amp"] = fields[41]
    ixon["data_v_shift_speed"] = float(fields[42])
    ixon["output_amp"] = fields[43]
    ixon["pre_amp_gain"] = float(fields[44])
    ixon["serial"] = int(fields[45])
    ixon["num_pulses"] = int(fields[46])
    ixon["m_frame_transfer_acq_mode"] = int(fields[47])
    ixon["unstabilized_temperature"] = int(fields[48])
    ixon["m_baseline_clamp"] = int(fields[49])
    ixon["m_pre_scan"] = int(fields[50])
    ixon["m_em_real_gain"] = int(fields[51])
    ixon["m_baseline_offset"] = int(fields[52])
    _ = fields[53]
    _ = fields[54]
    ixon["sw_vers1"] = int(fields[55])
    ixon["sw_vers2"] = int(fields[56])
    ixon["sw_vers3"] = int(fields[57])
    ixon["sw_vers4"] = int(fields[58])

    # line 4
    ixon["camera_model"] = strip(readline(stream))

    # line 5, something like camera dimensions??
    _ = readline(stream)

    # line 6
    ixon["original_filename"] = strip(readline(stream))

    # line 7
    l = strip(readline(stream))
    fields = split(l)
    fields[1] == "65538" || error("Unknown TUserText version number in line 7: $fields[1]")
    usertextlen = int(fields[2]) # don't need?

    # line 8
    usertext = strip(readline(stream))
    # ixon["usertext"] = usertext # Not useful

    # line 9 TShutter
    l = strip(readline(stream)) # Weird!

    # line 10 TCalibImage
    l = strip(readline(stream))

    # lines 11-22
    _ = strip(readline(stream))
    _ = strip(readline(stream))
    _ = strip(readline(stream))
    _ = strip(readline(stream))
    _ = strip(readline(stream))
    _ = strip(readline(stream))
    _ = strip(readline(stream))
    _ = strip(readline(stream))
    _ = strip(readline(stream))
    _ = strip(readline(stream))
    _ = strip(readline(stream))

    # what a bizarre file format here
    # length of the next string is in this line
    next_str_len = int(strip(readline(stream)))
    # and here is the next string, followed by the length
    # of the following string, with no delimeter in between!
    l = strip(readline(stream))
    next_str_len = int(l[(next_str_len + 1):end])
    # lather, rinse, repeat...
    l = strip(readline(stream))
    next_str_len = int(l[(next_str_len + 1):end])
    l = strip(readline(stream))
    l = l[(next_str_len + 1):end]
    fields = split(l)
    fields[1] == "65538" || error("Unknown version number at image dims record")
    ixon["image_format_left"] = int(fields[2])
    ixon["image_format_top"] = int(fields[3])
    ixon["image_format_right"] = int(fields[4])
    ixon["image_format_bottom"] = int(fields[5])
    frames = int(fields[6])
    ixon["frames"] = frames
    ixon["num_subimages"] = int(fields[7])
    ixon["total_length"] = int(fields[8]) # in pixels across all frames
    ixon["single_frame_length"] = int(fields[9])

    # Now at the first (and only) subimage
    l = strip(readline(stream))
    fields = split(l)
    fields[1] == "65538" || error("unknown TSubImage version number: " * fields[1])
    left = int(fields[2])
    top = int(fields[3])
    right = int(fields[4])
    bottom = int(fields[5])
    vertical_bin = int(fields[6])
    horizontal_bin = int(fields[7])
    subimage_offset = int(fields[8])

    # calculate frame width, height, with binning
    width = right - left + 1
    mod = width%horizontal_bin
    width = int((width - mod)/horizontal_bin)
    height = top - bottom + 1
    mod = height%vertical_bin
    height = int((height - mod)/vertical_bin)

    ixon["left"] = left
    ixon["top"] = top
    ixon["right"] = right
    ixon["bottom"] = bottom
    ixon["vertical_bin"] = vertical_bin
    ixon["horizontal_bin"] = horizontal_bin
    ixon["subimage_offset"] = subimage_offset

    # rest of the header is a timestamp for each frame
    # (actually, just a bunch of zeros). Skip
    for i = 1:frames
        _ = readline(stream)
    end
    offset = position(stream) # start of the actual pixel data, 32-bit float, little-endian

    pixels = read(stream, Float32, width, height, frames)
    prop = Images.@Dict(
        "colorspace" => "Gray",
        "spatialorder" => ["y", "x"],
        "ixon" => ixon,
        "suppress" => Set({"ixon"}),
        "pixelspacing" => [1, 1]
    )
    if frames > 1
        prop["timedim"] = 3
    end
    Image(float64(pixels), prop)
end
