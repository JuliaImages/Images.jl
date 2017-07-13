### Edge and Gradient Related Image Operations ###

# Phase (angle of steepest gradient ascent), calculated from X and Y gradient images
"""
    phase(grad_x, grad_y) -> p

Calculate the rotation angle of the gradient given by `grad_x` and
`grad_y`. Equivalent to `atan2(-grad_y, grad_x)`, except that when both `grad_x` and
`grad_y` are effectively zero, the corresponding angle is set to zero.
"""
function phase{T<:Number}(grad_x::T, grad_y::T, tol=sqrt(eps(T)))
    atan2(-grad_y, grad_x) * ((abs(grad_x) > tol) | (abs(grad_y) > tol))
end
phase(grad_x::Number,   grad_y::Number)   = phase(promote(grad_x, grad_y)...)
phase(grad_x::NumberLike, grad_y::NumberLike) = phase(gray(grad_x), gray(grad_y))

phase(grad_x::AbstractRGB, grad_y::AbstractRGB) = phase(vecsum(grad_x), vecsum(grad_y))

magnitude_phase(grad_x::NumberLike, grad_y::NumberLike) =
    hypot(grad_x, grad_y), phase(grad_x, grad_y)

function magnitude_phase(grad_x::AbstractRGB, grad_y::AbstractRGB)
    gx, gy = vecsum(grad_x), vecsum(grad_y)
    magnitude_phase(gx, gy)
end

vecsum(c::AbstractRGB) = float(red(c)) + float(green(c)) + float(blue(c))

## TODO? orientation seems nearly redundant with phase, deprecate?

"""
    orientation(grad_x, grad_y) -> orient

Calculate the orientation angle of the strongest edge from gradient images
given by `grad_x` and `grad_y`.  Equivalent to `atan2(grad_x, grad_y)`.  When
both `grad_x` and `grad_y` are effectively zero, the corresponding angle is set to
zero.
"""
function orientation{T<:Number}(grad_x::T, grad_y::T, tol=sqrt(eps(T)))
    atan2(grad_x, grad_y) * ((abs(grad_x) > tol) | (abs(grad_y) > tol))
end
orientation(grad_x::Number,   grad_y::Number)   = orientation(promote(grad_x, grad_y)...)
orientation(grad_x::NumberLike, grad_y::NumberLike) = orientation(gray(grad_x), gray(grad_y))

orientation(grad_x::AbstractRGB, grad_y::AbstractRGB) = orientation(vecsum(grad_x), vecsum(grad_y))

# Magnitude of gradient, calculated from X and Y image gradients
"""
```
m = magnitude(grad_x, grad_y)
```

Calculates the magnitude of the gradient images given by `grad_x` and `grad_y`.
Equivalent to ``sqrt(grad_x.^2 + grad_y.^2)``.

Returns a magnitude image the same size as `grad_x` and `grad_y`.
"""
magnitude(grad_x::AbstractArray, grad_y::AbstractArray) = hypot.(grad_x, grad_y)

Base.hypot(x::AbstractRGB, y::AbstractRGB) = hypot(vecsum(x), vecsum(y))

phase(grad_x::AbstractArray, grad_y::AbstractArray) = phase.(grad_x, grad_y)

# Orientation of the strongest edge at a point, calculated from X and Y gradient images
# Note that this is perpendicular to the phase at that point, except where
# both gradients are close to zero.

orientation{T}(grad_x::AbstractArray{T}, grad_y::AbstractArray{T}) = orientation.(grad_x, grad_y)

# Return both the magnitude and phase in one call
"""
    magnitude_phase(grad_x, grad_y) -> m, p

Convenience function for calculating the magnitude and phase of the gradient
images given in `grad_x` and `grad_y`.  Returns a tuple containing the magnitude
and phase images.  See `magnitude` and `phase` for details.
"""
function magnitude_phase{T}(grad_x::AbstractArray{T}, grad_y::AbstractArray{T})
    m = similar(grad_x, eltype(T))
    p = similar(m)
    for I in eachindex(grad_x, grad_y)
        m[I], p[I] = magnitude_phase(grad_x[I], grad_y[I])
    end
    m, p
end

# Return the magnitude and phase of the gradients in an image
function magnitude_phase(img::AbstractArray, method::Function=KernelFactors.ando3, border::AbstractString="replicate")
    grad_x, grad_y = imgradients(img, method, border)
    return magnitude_phase(grad_x, grad_y)
end

# Return the x-y gradients and magnitude and phase of gradients in an image
"""
```
grad_y, grad_x, mag, orient = imedge(img, kernelfun=KernelFactors.ando3, border="replicate")
```

Edge-detection filtering. `kernelfun` is a valid kernel function for
[`imgradients`](@ref), defaulting to [`KernelFactors.ando3`](@ref).
`border` is any of the boundary conditions specified in `padarray`.

Returns a tuple `(grad_x, grad_y, mag, orient)`, which are the horizontal
gradient, vertical gradient, and the magnitude and orientation of the strongest
edge, respectively.
"""
function imedge(img::AbstractArray, kernelfun=KernelFactors.ando3, border::AbstractString="replicate")
    grad_x, grad_y = imgradients(img, kernelfun, border)
    mag = magnitude(grad_x, grad_y)
    orient = orientation(grad_x, grad_y)
    return (grad_x, grad_y, mag, orient)
end

# Thin edges
"""
```
thinned = thin_edges(img, gradientangle, [border])
thinned, subpix = thin_edges_subpix(img, gradientangle, [border])
thinned, subpix = thin_edges_nonmaxsup(img, gradientangle, [border]; [radius::Float64=1.35], [theta=pi/180])
thinned, subpix = thin_edges_nonmaxsup_subpix(img, gradientangle, [border]; [radius::Float64=1.35], [theta=pi/180])
```

Edge thinning for 2D edge images.  Currently the only algorithm available is
non-maximal suppression, which takes an edge image and its gradient angle, and
checks each edge point for local maximality in the direction of the gradient.
The returned image is non-zero only at maximal edge locations.

`border` is any of the boundary conditions specified in `padarray`.

In addition to the maximal edge image, the `_subpix` versions of these functions
also return an estimate of the subpixel location of each local maxima, as a 2D
array or image of `Graphics.Point` objects.  Additionally, each local maxima is
adjusted to the estimated value at the subpixel location.

Currently, the `_nonmaxsup` functions are identical to the first two function
calls, except that they also accept additional keyword arguments.  `radius`
indicates the step size to use when searching in the direction of the gradient;
values between 1.2 and 1.5 are suggested (default 1.35).  `theta` indicates the
step size to use when discretizing angles in the `gradientangle` image, in
radians (default: 1 degree in radians = pi/180).

Example:

```
g = rgb2gray(rgb_image)
gx, gy = imgradients(g)
mag, grad_angle = magnitude_phase(gx,gy)
mag[mag .< 0.5] = 0.0  # Threshold magnitude image
thinned, subpix =  thin_edges_subpix(mag, grad_angle)
```
"""
thin_edges{T}(img::AbstractArray{T,2}, gradientangles::AbstractArray, border::AbstractString="replicate") =
    thin_edges_nonmaxsup(img, gradientangles, border)
thin_edges_subpix{T}(img::AbstractArray{T,2}, gradientangles::AbstractArray, border::AbstractString="replicate") =
    thin_edges_nonmaxsup_subpix(img, gradientangles, border)

# Code below is related to non-maximal suppression, and was ported to Julia from
# http://www.csse.uwa.edu.au/~pk/research/matlabfns/Spatial/nonmaxsup.m
# (Please conserve the original copyright below.)

# NONMAXSUP - Non-maxima suppression
#
# Usage:
#          (im,location) = nonmaxsup(img, gradientangles, radius);
#
# Function for performing non-maxima suppression on an image using
# gradient angles.  Gradient angles are assumed to be in radians.
#
# Input:
#   img - image to be non-maxima suppressed.
#
#   gradientangles - image containing gradient angles around each pixel in radians
#                    (-pi,pi)
#
#   radius  - Distance in pixel units to be looked at on each side of each
#             pixel when determining whether it is a local maxima or not.
#             This value cannot be less than 1.
#             (Suggested value about 1.2 - 1.5)
#
# Returns:
#   im        - Non maximally suppressed image.
#   location  - `Graphics.Point` image holding subpixel locations of edge
#               points.
#
# Notes:
#
# This function uses bilinear interpolation to estimate
# intensity values at ideal, real-valued pixel locations on each side of
# pixels to determine if they are local maxima.

# Copyright (c) 1996-2013 Peter Kovesi
# Centre for Exploration Targeting
# The University of Western Australia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# The Software is provided "as is", without warranty of any kind.

# December  1996 - Original version
# September 2004 - Subpixel localization added
# August    2005 - Made Octave compatible
# October   2013 - Final thinning applied to binary image for Octave
#                  compatbility (Thanks to Chris Pudney)
# June      2014 - Ported (and modified significantly) to Julia (Kevin Squire)

import .Point

if !applicable(zero, Point)
    import Base.zero
    zero(::Type{Point}) = Point(0.0,0.0)
end

# Used to encode the sign, integral, and fractional components of
# an offset from a coordinate
immutable CoordOffset
    s::Int      # sign
    i::Int      # integer part
    f::Float64  # fractional part
end

CoordOffset(x::Float64) = ((frac,i) = modf(x); CoordOffset(sign(frac), round(Int, i), abs(frac)))
(-)(off::CoordOffset) = CoordOffset(-off.s,-off.i, off.f)
(*)(x::Number, off::CoordOffset) = x*(off.i + off.s*off.f)
(*)(off::CoordOffset, x::Number) = x*(off.i + off.s*off.f)
(+)(x::Number, off::CoordOffset) = x + off.i + off.s*off.f
(+)(off::CoordOffset, x::Number) = x + off.i + off.s*off.f

# Precalculate x and y offsets relative to centre pixel for each orientation angle
function _calc_discrete_offsets(θ, radius)

    θ_count = round(Int, 2π/θ)
    θ = 2π/θ_count
    angles = (0:θ_count)*θ

    # x and y offset of points at specified radius and angles
    # from each reference position.

    xoffs = [CoordOffset( x) for x in  radius * cos.(angles)]
    yoffs = [CoordOffset(-y) for y in  radius * sin.(angles)]

    return θ, xoffs, yoffs
end

_discretize_angle(angle::AbstractFloat, invθ) =
    angle < 0 ? round(Int, (angle + 2π)*invθ)+1 : round(Int, angle*invθ)+1

# Interpolate the value of an offset from a particular pixel
#
# Returns (interpolated value, min_value of adjacent pixels in direction of offset)
#
# The second value is made available to eliminate double edges; if the value at
# (x,y) is less than the value or values adjacent to it in the direction of the
# gradient (xoff,yoff), then it is not a local maximum

function _interp_offset(img::AbstractArray, x::Integer, y::Integer, xoff::CoordOffset, yoff::CoordOffset, Ix, Iy, pad)
    fx = Ix[x + xoff.i + pad]
    fy = Iy[y + yoff.i + pad]
    cx = Ix[x + xoff.i + xoff.s + pad]
    cy = Iy[y + yoff.i + yoff.s + pad]

    tl = img[fy,fx]    # Value at bottom left integer pixel location.
    tr = img[fy,cx]    # bottom right
    bl = img[cy,fx]    # top left
    br = img[cy,cx]    # top right

    upperavg = tl + xoff.f * (tr - tl)  # Now use bilinear interpolation to
    loweravg = bl + xoff.f * (br - bl)  # estimate value at x,y

    min_adjacent = (fx == x) & (fy == y) ? min(tr,bl) : tl

    return (upperavg + yoff.f * (loweravg - upperavg), min_adjacent)
end

# Core edge thinning algorithm using nonmaximal suppression
function thin_edges_nonmaxsup_core!{T}(out::AbstractArray{T,2}, location::AbstractArray{Point,2},
                                       img::AbstractArray{T,2}, gradientangles::AbstractMatrix, radius, border, theta)
    calc_subpixel = !isempty(location)

    # Error checking
    size(img) == size(gradientangles) == size(out) || error("image, gradient angle, and output image must all be the same size")
    calc_subpixel && size(location) != size(img) && error("subpixel location has a different size than the input image")
    radius < 1.0 && error("radius must be >= 1")

    # Precalculate x and y offsets relative to centre pixel for each orientation angle
    θ, xoffs, yoffs = _calc_discrete_offsets(theta, radius)
    iθ = 1/θ

    # Indexes to use for border handling
    pad = ceil(Int, radius)
    Ix = Images.padindexes(img, 2, pad, pad, border)
    Iy = Images.padindexes(img, 1, pad, pad, border)

    # Now run through the image interpolating grey values on each side
    # of the centre pixel to be used for the non-maximal suppression.

    (height,width) = size(img)

    for x = 1:width, y = 1:height
        (c = img[y,x]) == 0 && continue  # For thresholded images

        or = _discretize_angle(gradientangles[y,x],iθ)   # Disretized orientation
        v1, n1 = _interp_offset(img, x, y, xoffs[or], yoffs[or], Ix, Iy, pad)

        if (c > v1) & (c >= n1) # We need to check the value on the other side...
            v2, n2 = _interp_offset(img, x, y, -xoffs[or], -yoffs[or], Ix, Iy, pad)

            if (c > v2) & (c >= n2)  # This is a local maximum.
                                     # Record value in the output image.
                if calc_subpixel
                    # Solve for coefficients of parabola that passes through
                    # [-1, v2]  [0, img] and [1, v1].
                    # v = a*r^2 + b*r + c

                    # c = img[y,x]
                    a = (v1 + v2)/2 - c
                    b = a + c - v2

                    # location where maxima of fitted parabola occurs
                    r = -b/2a
                    location[y,x] = Point(x + r*xoffs[or], y + r*yoffs[or])

                    if T<:AbstractFloat
                        # Store the interpolated value
                        out[y,x] = a*r^2 + b*r + c
                    else
                        out[y,x] = c
                    end
                else
                    out[y,x] = c
                end
            end
        end
    end

    out
end


# Main function call when subpixel location of edges is not needed
function thin_edges_nonmaxsup!(out, img, gradientangles, border::AbstractString="replicate";
                               radius::Float64=1.35, theta=pi/180)
    thin_edges_nonmaxsup_core!(out, Matrix{Point}(0,0), img, gradientangles, radius, border, theta)
end

function thin_edges_nonmaxsup(img, gradientangles, border::AbstractString="replicate";
                                 radius::Float64=1.35, theta=pi/180)
    (height,width) = size(img)
    out = zeros(eltype(img), height, width)
    thin_edges_nonmaxsup_core!(out, Matrix{Point}(0,0), img, gradientangles, radius, border, theta)
end

# Main function call when subpixel location of edges is desired
function thin_edges_nonmaxsup_subpix!(out, location, img, gradientangles,
                                      border::AbstractString="replicate";
                                      radius::Float64=1.35, theta=pi/180)
    eltype(location) != Point && error("Preallocated subpixel location array/image must have element type Graphics.Point")

    thin_edges_nonmaxsup_core!(out, location, img, gradientangles, radius, border, theta)
    img, location
end

function thin_edges_nonmaxsup_subpix(img, gradientangles,
                                     border::AbstractString="replicate";
                                     radius::Float64=1.35, theta=pi/180)
    (height,width) = size(img)
    out = zeros(eltype(img), height, width)
    location = zeros(Point, height, width)
    thin_edges_nonmaxsup_core!(out, location, img, gradientangles, radius, border, theta)
    out, location
end

"""
```
canny_edges = canny(img, sigma = 1.4, upperThreshold = 0.80, lowerThreshold = 0.20)
```

Performs Canny Edge Detection on the input image.

Parameters :

  (upper, lower) :  Bounds for hysteresis thresholding
  sigma :           Specifies the standard deviation of the gaussian filter

"""
function canny{T<:NumberLike, N<:Union{NumberLike,Percentile{NumberLike}}}(img_gray::AbstractMatrix{T}, threshold::Tuple{N,N}, sigma::Number = 1.4)
    img_grayf = imfilter(img_gray, KernelFactors.IIRGaussian((sigma,sigma)), NA())
    img_grad_y, img_grad_x = imgradients(img_grayf, KernelFactors.sobel)
    img_mag, img_phase = magnitude_phase(img_grad_x, img_grad_y)
    img_nonMaxSup = thin_edges_nonmaxsup(img_mag, img_phase)
    if N<:Percentile{}
        upperThreshold ,lowerThreshold = StatsBase.percentile(img_nonMaxSup[:], [threshold[i].p for i=1:2])
    else
        upperThreshold, lowerThreshold = threshold
    end
    img_thresholded = hysteresis_thresholding(img_nonMaxSup, upperThreshold, lowerThreshold)
    edges = map(i -> i >= 0.9, img_thresholded)
    edges
end

canny(img::AbstractMatrix, args...) = canny(convert(Array{Gray}, img), args...)

function hysteresis_thresholding{T}(img_nonMaxSup::AbstractArray{T, 2}, upperThreshold::Number, lowerThreshold::Number)
    img_thresholded = map(i -> i > lowerThreshold ? i > upperThreshold ? 1.0 : 0.5 : 0.0, img_nonMaxSup)
    queue = CartesianIndex{2}[]
    R = CartesianRange(size(img_thresholded))

    I1, Iend = first(R), last(R)
    for I in R
      if img_thresholded[I] == 1.0
        img_thresholded[I] = 0.9
        push!(queue, I)
        while !isempty(queue)
          q_top = shift!(queue)
          for J in CartesianRange(max(I1, q_top - I1), min(Iend, q_top + I1))
            if img_thresholded[J] == 1.0 || img_thresholded[J] == 0.5
              img_thresholded[J] = 0.9
              push!(queue, J)
            end
          end
        end
      end
    end
    img_thresholded
end

function padindexes{T,n}(img::AbstractArray{T,n}, dim, prepad, postpad, border::AbstractString)
    M = size(img, dim)
    I = Vector{Int}(M + prepad + postpad)
    I = [(1 - prepad):(M + postpad);]
    if border == "replicate"
        I = min.(max.(I, 1), M)
    elseif border == "circular"
        I = 1 .+ mod.(I .- 1, M)
    elseif border == "symmetric"
        I = [1:M; M:-1:1][1 .+ mod.(I .- 1, 2 * M)]
    elseif border == "reflect"
        I = [1:M; M-1:-1:2][1 .+ mod.(I .- 1, 2 * M - 2)]
    else
        error("unknown border condition")
    end
    I
end
