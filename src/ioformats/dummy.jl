import Images.imread, Images.imwrite

function imread{S<:IO}(stream::S, ::Type{Images.Dummy})
    pixels = reshape(uint8(1:12), 3, 4)
    Image(pixels, Images.@Dict("colorspace" => "Gray", "spatialorder" => "xy"))
end

function imwrite(img, filename::String, ::Type{Images.Dummy})
    open(filename, "w") do stream
        println(stream, "Dummy Image")
    end
end
