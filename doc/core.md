# Julia Images Guide

## The basic types

### Plain arrays

Images can be plain arrays, which are interpreted to be in "Matlab format": the
first two dimensions are height (`h`) and width (`w`), a storage order here
called "vertical-major". This ordering is inspired by the column/row index order
of matrices and the desire to have a displayed image look like what one sees
when a matrix is written out in text.

If you're working with RGB color, your best approach is to encode color as a
`ColorValue`, as defined in the `Color` package.
That package provides many utility functions for analyzing and manipulating
colors.
Alternatively, you can use a third dimension (often of size 3) or
encode your images as either `Uint32` or `Int32` and work with
24-bit color. (If you need to use `Uint32` simply to store pixel intensities,
you should not use plain arrays.)

It's worth noting that these conventions are sometimes inconvenient.
For example, the `x` coordinate (horizontal) is second and the `y` coordinate
(vertical) is first; in other words, one uses `img[y,x]` to address a pixel that
is displayed at a
particular `x,y` position. This often catches newcomers (and sometimes even
old-timers) by surprise. 
Moreover, most image file formats, cameras, and graphics libraries such as
Cairo use "horizontal-major" storage of images, and have the color dimension
first (fastest). This therefore necessitates a
permutation of the dimensions of the raw data array.

The conventions for plain arrays are "baked in" via a few simple utility
functions in the file
`core.jl`; if you really need to use plain arrays but want to work with
different conventions, you can (locally) change these defaults with just a few
lines. Algorithms which have been written generically should continue
to work.

However, a more flexible approach is to use one of the self-documenting image
types.

### Image types

All image types should descend from `AbstractImage`, an abstract base type used
to indicate that an array is to be interpreted as an image. If you're writing a
custom image type, it is more likely that you'll want to derive from either
`AbstractImageDirect` or `AbstractImageIndexed`. The former is for direct images
(where intensity at a pixel is represented directly), the latter for indexed
images (where intensity is looked up in a colormap table).

In practice, it is assumed that `AbstractImages` have at least two fields,
called `data` and `properties`. These are the only two fields in the first
concrete image type, called `Image`:

```julia
type Image{T,N,A<:AbstractArray} <: AbstractImageDirect{T,N}
    data::A
    properties::Dict
end
```

`data` stores the actual image data, and is an `AbstractArray`. This fact alone
is the basis for a great deal of customizability: `data` might be a plain
`Array` stored in memory, a `SubArray`, a memory-mapped array (which is still
just an `Array`), a custom type that stores additional information about
"missing data" (like bad pixels or dropped frames), or a custom type that
seamlessly presents views of a large number of separate files.
One concrete example in the Images codebase is the color `Overlay` [type](overlays.md).
If you have a
suitably-defined `AbstractArray` type, you can probably use `Image` without
needing to create alternative `AbstractImageDirect` types.


`properties` is a dictionary, with `String` keys, that allows you to
annotate images. More detail about this point can be found below.

The only other concrete image type is for indexed images:

```julia
type ImageCmap{T,N,A<:AbstractArray,C<:AbstractArray} <:
AbstractImageIndexed{T,N}
    data::A
    cmap::C
    properties::Dict
end
```
The `data` array here just encodes the index used to look up the color in the
`cmap` field.

## Addressing image data

For any valid image type, `data(img)` returns the array that corresponds to the
image.
This works when `img` is a plain `Array` (in which case no operation is
performed) as well as for an `Image` (in which case it returns `img.data`).
This is our first example of how to write generic algorithms.

If `img` is an `Image`, then `img[i,j]` looks up the value `img.data[i,j]`.
Assignment, `sub`, and `slice` work similarly. In other words, for indexing an
`Image` works just as if you were using plain arrays.

If you are indexing over an extended region and want to get back an `Image`,
rather than a value or an `Array`, then you
will want to use `getindexim`, `subim`, and `sliceim`. For the first two, the
resulting image will share everything but the `data` field with the original
image; if you make modifications in one, the other will also be affected. For
`sliceim`, because it can change the dimensionality some adjustments to
`properties` are needed; in this case a copy is made.

One of the properties (see below) that you can grant to images is
`spatialorder`, which provides a name for each spatial dimension in the image.
Using this feature, you can cut out regions or slices from images in the
following ways:

```julia
A = img["x", 200:400, "y", 500:700]
imgs = sliceim(img, "z", 14)      # cuts out the 14th frame in a stack
```
These routines "do the right thing" no matter what storage order is being used.

## Image properties and accessor functions

The `properties` dictionary can contain any information you want to store along
with your images. Typically, each property is also affiliated with an accessor
function of the same name.

Let's illustrate this with one of the default properties, `"colorspace"`. The
value of this property is a string, such as `"RGB"` or `"Gray"` or `"HSV"`. You
can extract the value of this field using a function:
```
cs = colorspace(img)
```
The reason to have a function, rather than just looking it up in the
`properties` dictionary, is that we can provide defaults. For example, images
represented as plain `Array`s don't have a `properties` dictionary; if we are to
write generic code, we don't want to have to wonder whether this information is
available. So for plain arrays, there are a number of defaults specified for the
output of the `colorspace` function, depending on the element type and size of
the array.

Here is a list of the properties supported in `core.jl`:

- `colorspace`: "RGB", "RGBA", "Gray", "Binary", "24bit", "Lab", "HSV", etc.
- `colordim`: the array dimension used to store color information, or 0 if there
is no dimension corresponding to color
- `timedim`: the array dimension used for time (i.e., sequence), or 0 for single
images
- `limits`: (minvalue,maxvalue) for this type of image (e.g., (0,255) for Uint8
images, even if pixels do not reach these values)
- `pixelspacing`: the spacing between adjacent pixels along spatial dimensions
- `spatialorder`: a string naming each spatial dimension, in the storage order
of
the data array. Names can be arbitrary, but the choices "x" and "y" have special
meaning (horizontal and vertical, respectively, irrespective of storage order).
If supplied, you must have one entry per spatial dimension.

If you specify their values in the `properties` dictionary, your values will be
used; if not, hopefully-reasonable defaults will be chosen.


## Writing generic algorithms

There are many examples that you can use as a guide. Here we work through the
example of `imfilter` (in
`algorithms.jl`), which was originally designed to work on just arrays, and show
how to convert it into an algorithm that works with most image types.
(One important limitation is that Julia's `imfilter` algorithm currently works
only with 2d images; generalizing it to multiple dimensions is a different
topic.) First, let's assume that we've re-named the old array-based algorithm as
`_imfilter`. Then a more general variant of `imfilter` can be written in the
following way:

```
function imfilter{T}(img::AbstractArray{T}, filter::Matrix{T}, border::String,
value)
    assert2d(img)      # Julia's imfilter was written for 2d images, so enforce
this
    cd = colordim(img) # find out which dimension corresponds to color, if any
    local A
    if cd == 0         # no explicit array dimension for color
        A = _imfilter(data(img), filter, border, value)
    else
        A = similar(data(img))                    # allocate the output
        coords = RangeIndex[1:size(img,i) for i = 1:ndims(img)]   # indexes covering the whole
array
        for i = 1:size(img, cd)
            coords[cd] = i                        # slice along the color
dimension
            simg = slice(img, coords...)
            tmp = _imfilter(simg, filter, border, value)   # filter this color
channel
            A[coords...] = tmp[:]                 # store the result
        end
    end
    share(img, A)       # output image has the same properties as the input
end
```
Despite the fact that this allows the color channel to be any one of the array
dimensions, it is hardly any more complicated than its predecessor which
assumed that the third dimension corresponded to color.

Other examples showing how to generalize array-based code can be found in
`algorithms.jl`, `core.jl`, and `io.jl`.
