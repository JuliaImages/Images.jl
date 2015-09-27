# SIF.jl, adds an imread function for Andor .sif images
# 2013 Ronald S. Rock, Jr.
import FileIO: skipmagic, stream, @format_str, Stream
function load(fs::Stream{format"AndorSIF"})
    # line 1
    skipmagic(fs, 0)
    io = stream(fs)
    # line 2
    l = strip(readline(io))
    l == "65538 1" || error("Unknown Andor version number at line 2: " * l)

    # line 3: TInstaImage thru "Head model"
    l = strip(readline(io))
    fields = split(l)
    fields[1] == "65547" || fields[1] == "65558" ||
        error("Unknown TInstaImage version number at line 3: " * fields[1])

    ixon = Dict{Any,Any}()
    ixon["data_type"] = round(Int,fields[2])
    ixon["active"] = convert(Int,fields[3])
    ixon["structure_vers"] = round(Int,fields[4]) # (== 1)
    # date is recored as seconds counted from 1970.1.1 00:00:00
    ixon["date"] = round(Int,fields[5]) # need to convert to actual date
    ixon["temperature"] = max(round(Int,fields[6]), round(Int,fields[48]))
    ixon["temperature_stable"] = round(Int,fields[6]) != -999
    ixon["head"] = fields[7]
    ixon["store_type"] = fields[8]
    ixon["data_type"] = fields[9]
    ixon["mode"] = fields[10]
    ixon["trigger_source"] = fields[11]
    ixon["trigger_level"] = fields[12]
    ixon["exposure_time"] = float(fields[13])
    ixon["frame_delay"] = float(fields[14])
    ixon["integration_cycle_time"] = float(fields[15])
    ixon["no_integrations"] = round(Int,fields[16])
    ixon["sync"] = fields[17]
    ixon["kin_cycle_time"] = float(fields[18])
    ixon["pixel_readout_time"] = float(fields[19])
    ixon["no_points"] = round(Int,fields[20])
    ixon["fast_track_height"] = round(Int,fields[21])
    ixon["gain"] = round(Int,fields[22])
    ixon["gate_delay"] = float(fields[23])
    ixon["gate_width"] = float(fields[24])
    ixon["gate_step"] = float(fields[25])
    ixon["track_height"] = round(Int,fields[26])
    ixon["series_length"] = round(Int,fields[27])
    ixon["read_pattern"] = fields[28]
    ixon["shutter_delay"] = fields[29]
    ixon["st_center_row"] = round(Int,fields[30])
    ixon["mt_offset"] = round(Int,fields[31])
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
    ixon["serial"] = round(Int,fields[45])
    ixon["num_pulses"] = round(Int,fields[46])
    ixon["m_frame_transfer_acq_mode"] = round(Int,fields[47])
    ixon["unstabilized_temperature"] = round(Int,fields[48])
    ixon["m_baseline_clamp"] = round(Int,fields[49])
    ixon["m_pre_scan"] = round(Int,fields[50])
    ixon["m_em_real_gain"] = round(Int,fields[51])
    ixon["m_baseline_offset"] = round(Int,fields[52])
    _ = fields[53]
    _ = fields[54]
    ixon["sw_vers1"] = round(Int,fields[55])
    ixon["sw_vers2"] = round(Int,fields[56])
    ixon["sw_vers3"] = round(Int,fields[57])
    ixon["sw_vers4"] = round(Int,fields[58])

    # line 4
    ixon["camera_model"] = strip(readline(io))

    # line 5, something like camera dimensions??
    _ = readline(io)

    # line 6
    ixon["original_filename"] = strip(readline(io))

    # line 7
    l = strip(readline(io))
    fields = split(l)
    fields[1] == "65538" || error("Unknown TUserText version number in line 7: $fields[1]")
    usertextlen = round(Int,fields[2]) # don't need?

    # line 8
    usertext = strip(readline(io))
    # ixon["usertext"] = usertext # Not useful

    # line 9 TShutter
    l = strip(readline(io)) # Weird!

    # line 10 TCalibImage
    l = strip(readline(io))

    # lines 11-22
    _ = strip(readline(io))
    _ = strip(readline(io))
    _ = strip(readline(io))
    _ = strip(readline(io))
    _ = strip(readline(io))
    _ = strip(readline(io))
    _ = strip(readline(io))
    _ = strip(readline(io))
    _ = strip(readline(io))
    _ = strip(readline(io))
    _ = strip(readline(io))

    # what a bizarre file format here
    # length of the next string is in this line
    next_str_len = round(Int,strip(readline(io)))
    # and here is the next string, followed by the length
    # of the following string, with no delimeter in between!
    l = strip(readline(io))
    next_str_len = round(Int,l[(next_str_len + 1):end])
    # lather, rinse, repeat...
    l = strip(readline(io))
    next_str_len = round(Int,l[(next_str_len + 1):end])
    l = strip(readline(io))
    l = l[(next_str_len + 1):end]
    fields = split(l)
    fields[1] == "65538" || error("Unknown version number at image dims record")
    ixon["image_format_left"] = round(Int,fields[2])
    ixon["image_format_top"] = round(Int,fields[3])
    ixon["image_format_right"] = round(Int,fields[4])
    ixon["image_format_bottom"] = round(Int,fields[5])
    frames = round(Int,fields[6])
    ixon["frames"] = frames
    ixon["num_subimages"] = round(Int,fields[7])
    ixon["total_length"] = round(Int,fields[8]) # in pixels across all frames
    ixon["single_frame_length"] = round(Int,fields[9])

    # Now at the first (and only) subimage
    l = strip(readline(io))
    fields = split(l)
    fields[1] == "65538" || error("unknown TSubImage version number: " * fields[1])
    left = round(Int,fields[2])
    top = round(Int,fields[3])
    right = round(Int,fields[4])
    bottom = round(Int,fields[5])
    vertical_bin = round(Int,fields[6])
    horizontal_bin = round(Int,fields[7])
    subimage_offset = round(Int,fields[8])

    # calculate frame width, height, with binning
    width = right - left + 1
    mod = width%horizontal_bin
    width = round(Int,(width - mod)/horizontal_bin)
    height = top - bottom + 1
    mod = height%vertical_bin
    height = round(Int,(height - mod)/vertical_bin)

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
        _ = readline(io)
    end
    offset = position(io) # start of the actual pixel data, 32-bit float, little-endian

    pixels = read(io, Float32, width, height, frames)
    prop = Compat.@compat Dict(
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
