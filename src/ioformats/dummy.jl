import Images.imread, Images.imwrite

function imread{S<:IO}(stream::S, ::Type{Images.Dummy})
    pixels = reshape(uint8(1:12), 3, 4)
    Image(pixels, ["colorspace" => "Gray", "spatialorder" => "xy"])
end

function imwrite(img, filename::String, ::Type{Images.Dummy})
    stream = open(filename, "w")
    println(stream, "Dummy Image")
    close(stream)
end
