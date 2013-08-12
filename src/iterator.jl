parent(img::AbstractImage) = parent(data(img))
first_index(A::Array) = 1
first_index(A::SubArray) = A.first_index

function iterate_spatial(img::AbstractArray)
    sz = size(img)
    s = strides(img)
    colorsz = 1
    colorstride = 0
    timesz = 1
    timestride = 0
    cd = colordim(img)
    if cd != 0
        colorsz = sz[cd]
        colorstride = s[cd]
    end
    td = timedim(img)
    if td != 0
        timesz = sz[td]
        timestride = s[td]
    end
    cspatial = setdiff(1:ndims(img), [cd, td])
    first_index(data(img)), sz[cspatial], s[cspatial], colorsz, colorstride, timesz, timestride
end
