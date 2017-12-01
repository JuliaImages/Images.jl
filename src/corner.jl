"""
```
corners = imcorner(img; [method])
corners = imcorner(img, threshold, percentile; [method])
```

Performs corner detection using one of the following methods -

    1. harris
    2. shi_tomasi
    3. kitchen_rosenfeld

The parameters of the individual methods are described in their documentation. The
maxima values of the resultant responses are taken as corners. If a threshold is
specified, the values of the responses are thresholded to give the corner pixels.
The threshold is assumed to be a percentile value unless `percentile` is set to false.
"""
function imcorner(img::AbstractArray; method::Function = harris, args...)
    responses = method(img; args...)
    corners = similar(img, Bool)
    fill!(corners, false)
    maxima = map(CartesianIndex, findlocalmaxima(responses))
    for m in maxima corners[m] = true end
    corners
end

function imcorner(img::AbstractArray, threshold; method::Function = harris, args...)
    responses = method(img; args...)
    map(i -> i > threshold, responses)
end

function imcorner(img::AbstractArray, thresholdp::Percentile; method::Function = harris, args...)
    responses = method(img; args...)
    threshold = StatsBase.percentile(vec(responses), thresholdp.p)
    map(i -> i > threshold, responses)
end

"""
```
corners = imcorner_subpixel(img; [method])
         -> Vector{HomogeneousPoint{Float64,3}}
corners = imcorner_subpixel(img, threshold, percentile; [method])
         -> Vector{HomogeneousPoint{Float64,3}}
```

Same as [`imcorner`](@ref), but estimates corners to sub-pixel precision.

Sub-pixel precision is achieved by interpolating the corner response values using
the 4-connected neighbourhood of a maximum response value. 
See [`corner2subpixel`](@ref) for more details of the interpolation scheme. 

"""
function imcorner_subpixel(img::AbstractArray; method::Function = harris, args...)
    responses = method(img; args...)
    corner_indicator = similar(img, Bool)
    fill!(corner_indicator, false)
    maxima = map(CartesianIndex, findlocalmaxima(responses))
    for m in maxima corner_indicator[m] = true end
    corners = corner2subpixel(responses,corner_indicator)
end

function imcorner_subpixel(img::AbstractArray, threshold; method::Function = harris, args...)
    responses = method(img; args...)
    corner_indicator = map(i -> i > threshold, responses)
    corners = corner2subpixel(responses,corner_indicator)
end

function imcorner_subpixel(img::AbstractArray, thresholdp::Percentile; method::Function = harris, args...)
    responses = method(img; args...)
    threshold = StatsBase.percentile(vec(responses), thresholdp.p)
    corner_indicator = map(i -> i > threshold, responses)
    corners = corner2subpixel(responses,corner_indicator)
end


"""
```
corners = corner2subpixel(responses::AbstractMatrix,corner_indicator::AbstractMatrix{Bool})
        -> Vector{HomogeneousPoint{Float64,3}}
```
Refines integer corner coordinates to sub-pixel precision.

The function takes as input a matrix representing corner responses
and a boolean indicator matrix denoting the integer coordinates of a corner
in the image. The output is a vector of type [`HomogeneousPoint`](@ref)
storing the sub-pixel coordinates of the corners.

The algorithm computes a correction factor which is added to the original
integer coordinates. In particular, a univariate quadratic polynomial is fit
separately to the ``x``-coordinates and ``y``-coordinates of a corner and its immediate
east/west, and north/south neighbours. The fit is achieved using a local
coordinate system for each corner, where the origin of the coordinate system is
a given corner, and its immediate neighbours are assigned coordinates of  minus
one and plus one.

The corner and its two neighbours form a system of three equations. For example,
let  ``x_1 = -1``,  ``x_2 = 0`` and  ``x_3 = 1`` denote the local ``x`` coordinates
of the west, center and east pixels and let the vector ``\\mathbf{b} = [r_1, r_2, r_3]``
denote the corresponding corner response values. With

```math 
    \\mathbf{A} = 
        \\begin{bmatrix}
            x_1^2 & x_1  & 1  \\\\
            x_2^2 & x_2  & 1 \\\\
            x_3^2 & x_3  & 1 \\\\
        \\end{bmatrix},
```
the coefficients of the quadratic polynomial can be found by solving the
system of equations ``\\mathbf{b} = \\mathbf{A}\\mathbf{x}``. 
The result is given by ``x = \\mathbf{A}^{-1}\\mathbf{b}``.

The vertex of the quadratic polynomial yields a sub-pixel estimate of the
true corner position. For example, for a univariate quadratic polynomial
``px^2 + qx + r``, the ``x``-coordinate of the vertex is ``\\frac{-q}{2p}``. 
Hence, the refined sub-pixel coordinate is equal to:
 ``c +  \\frac{-q}{2p}``, where ``c`` is the integer coordinate. 

!!! note
    Corners on the boundary of the image are not refined to sub-pixel precision.

"""
function corner2subpixel(responses::AbstractMatrix, corner_indicator::AbstractMatrix{Bool})
    row_range, col_range = indices(corner_indicator)
    row, col, _ = findnz(corner_indicator)
    ncorners = length(row)
    corners = fill(HomogeneousPoint((0.0,0.0,0.0)),ncorners)
    invA = @SMatrix [0.5 -1.0 0.5; -0.5 0.0 0.5; 0.0 1.0 -0.0]
    for k = 1:ncorners
        # Corners on the perimeter of the image will not be interpolated.
        if  (row[k] == first(row_range) || row[k] == last(row_range) || 
             col[k] == first(col_range) || col[k] == last(col_range))
            y = convert(Float64,row[k])
            x = convert(Float64,col[k])
            corners[k] = HomogeneousPoint((x,y,1.0))
        else
            center, north, south, east, west =
                                unsafe_neighbourhood_4(responses,row[k],col[k])
            # Solve for the coefficients of the quadratic equation.
            a, b, c = invA* @SVector [west, center, east]
            p, q, r = invA* @SVector [north, center, south]
            # Solve for the first coordinate of the vertex.
            u = -b/(2.0a)
            v = -q/(2.0p)
            corners[k] = HomogeneousPoint((col[k]+u,row[k]+v,1.0))
        end
    end
    return corners
end

"""
```
unsafe_neighbourhood_4(matrix::AbstractMatrix,r::Int,c::Int)
```

Returns the value of a matrix at given coordinates together with the values
of the north, south, east and west neighbours.

This function does not perform bounds checking. It is up to the user to ensure
that the function is not called with indices that are on the boundary of the
matrix.

"""
function unsafe_neighbourhood_4(matrix::AbstractMatrix,r::Int,c::Int)
    center = matrix[r,c]
    north = matrix[r-1,c]
    south = matrix[r+1,c]
    east = matrix[r,c+1]
    west = matrix[r,c-1]
    return center, north, south, east, west
end


"""
```
harris_response = harris(img; [k], [border], [weights])
```

Performs Harris corner detection. The covariances can be taken using either a mean
weighted filter or a gamma kernel.
"""
function harris(img::AbstractArray; k::Float64 = 0.04, args...)
    cov_xx, cov_xy, cov_yy = gradcovs(img, args...)
    corner = map((xx, yy, xy) -> xx * yy - xy ^ 2 - k * (xx + yy) ^ 2, cov_xx, cov_yy, cov_xy)
    corner
end

"""
```
shi_tomasi_response = shi_tomasi(img; [border], [weights])
```

Performs Shi Tomasi corner detection. The covariances can be taken using either a mean
weighted filter or a gamma kernel.
"""
function shi_tomasi(img::AbstractArray; border::AbstractString = "replicate", args...)
    cov_xx, cov_xy, cov_yy = gradcovs(img, border; args...)
    corner = map((xx, yy, xy) -> ((xx + yy) - (sqrt((xx - yy) ^ 2 + 4 * xy ^ 2))) / 2, cov_xx, cov_yy, cov_xy)
    corner
end

"""
```
kitchen_rosenfeld_response = kitchen_rosenfeld(img; [border])
```

Performs Kitchen Rosenfeld corner detection. The covariances can be taken using either a mean
weighted filter or a gamma kernel.
"""
function kitchen_rosenfeld(img::AbstractArray; border::AbstractString = "replicate")
    meth = KernelFactors.sobel
    (grad_x, grad_y) = imgradients(img, meth, border)
    (grad_xx, grad_xy) = imgradients(grad_x, meth, border)
    (grad_yx, grad_yy) = imgradients(grad_y, meth, border)
    map(kr, grad_x, grad_y, grad_xx, grad_xy, grad_yy)
end

function kr(x::T, y::T, xx::T, xy::T, yy::T) where T<:Real
    num = xx*y*y + yy*x*x - 2*xy*x*y
    denom = x*x + y*y
    ifelse(denom == 0, zero(num)/one(denom), -num/denom)
end

function kr(x::Real, y::Real, xx::Real, xy::Real, yy::Real)
    xp, yp, xxp, xyp, yyp = promote(x, y, xx, xy, yy)
    kr(xp, yp, xxp, xyp, yyp)
end

kr(x::RealLike, y::RealLike, xx::RealLike, xy::RealLike, yy::RealLike) =
    kr(gray(x), gray(y), gray(xx), gray(xy), gray(yy))

function kr(x::AbstractRGB, y::AbstractRGB, xx::AbstractRGB, xy::AbstractRGB, yy::AbstractRGB)
    krrgb = RGB(kr(red(x), red(y), red(xx), red(xy), red(yy)),
                kr(green(x), green(y), green(xx), green(xy), green(yy)),
                kr(blue(x),  blue(y),  blue(xx),  blue(xy),  blue(yy)))
    gray(convert(Gray, krrgb))
end

"""
    fastcorners(img, n, threshold) -> corners

Performs FAST Corner Detection. `n` is the number of contiguous pixels
which need to be greater (lesser) than intensity + threshold (intensity - threshold)
for a pixel to be marked as a corner. The default value for n is 12.
"""
function fastcorners(img::AbstractArray{T}, n::Int = 12, threshold::Float64 = 0.15) where T
    img_padded = padarray(img, Fill(0, (3,3)))
    corner = falses(size(img))
    R = CartesianRange(size(img))
    idx = map(CartesianIndex{2}, [(0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3),
            (0, -3), (-1, -3), (-2, -2), (-3, -1), (-3, 0), (-3, 1), (-2, 2), (-1, 3)])

    idxidx = [1, 5, 9, 13]
    for I in R
        bright_threshold = img_padded[I] + threshold
        dark_threshold = img_padded[I] - threshold
        if n >= 12
            sum_bright = 0
            sum_dark = 0
            for k in idxidx
                pixel = img_padded[I + idx[k]]
                if pixel > bright_threshold
                    sum_bright += 1
                elseif pixel < dark_threshold
                    sum_dark += 1
                end
            end
            if sum_bright < 3 && sum_dark < 3
                continue
            end
        end
        consecutive_bright = 0
        consecutive_dark = 0

        for i in 1:15 + n
            k = mod1(i, 16)
            pixel = img_padded[I + idx[k]]
            if pixel > bright_threshold
                consecutive_dark = 0
                consecutive_bright += 1
            elseif pixel < dark_threshold
                consecutive_bright = 0
                consecutive_dark += 1
            end

            if consecutive_dark == n || consecutive_bright == n
                corner[I] = true
                break
            end
        end
    end
    corner
end

function gradcovs(img::AbstractArray, border::AbstractString = "replicate"; weights::Function = meancovs, args...)
    (grad_x, grad_y) = imgradients(img, KernelFactors.sobel, border)

    cov_xx = dotc.(grad_x, grad_x)
    cov_xy = dotc.(grad_x, grad_y)
    cov_yy = dotc.(grad_y, grad_y)

    weights(cov_xx, cov_xy, cov_yy, args...)
end

function meancovs(cov_xx, cov_xy, cov_yy, blockSize::Int = 3)

    box_filter_kernel = centered((1 / (blockSize * blockSize)) * ones(blockSize, blockSize))

    filt_cov_xx = imfilter(cov_xx, box_filter_kernel)
    filt_cov_xy = imfilter(cov_xy, box_filter_kernel)
    filt_cov_yy = imfilter(cov_yy, box_filter_kernel)

    filt_cov_xx, filt_cov_xy, filt_cov_yy
end

function gammacovs(cov_xx, cov_xy, cov_yy, gamma::Float64 = 1.4)
    kernel = KernelFactors.gaussian((gamma, gamma))

    filt_cov_xx = imfilter(cov_xx, kernel)
    filt_cov_xy = imfilter(cov_xy, kernel)
    filt_cov_yy = imfilter(cov_yy, kernel)

    filt_cov_xx, filt_cov_xy, filt_cov_yy
end
