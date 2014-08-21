# Images.jl

An image processing library for [Julia](http://julialang.org/).

[![Status](http://iainnz.github.io/packages.julialang.org/badges/Images_0.3.svg)](http://iainnz.github.io/packages.julialang.org/badges/Images_0.3.svg) [![Coverage Status](https://coveralls.io/repos/timholy/Images.jl/badge.png?branch=master)](https://coveralls.io/r/timholy/Images.jl?branch=master)

## Installation

Install via the package manager,

```
Pkg.add("Images")
```

It's helpful to have ImageMagick installed on your system, as Images relies on it for reading and writing many common image types.
For unix platforms, adding the Images package should install ImageMagick for you automatically.
**On Windows, currently you need to install ImageMagick manually** if you want to read/write most image file formats.
More details about manual installation and troubleshooting can be found in the [installation help](doc/install.md).

## Image viewing

If you're using the IJulia notebook, images will be displayed [automatically](http://htmlpreview.github.com/?https://github.com/timholy/Images.jl/blob/master/ImagesDemo.html).

Julia code for the display of images can be found in [ImageView](https://github.com/timholy/ImageView.jl).
Installation of this package is recommended but not required.

## TestImages

When testing ideas or just following along with the documentation, it can be useful to have some images to work with.
The [TestImages](https://github.com/timholy/TestImages.jl) package bundles several "standard" images for you.
To load one of the images from this package, say
```
using TestImages
img = testimage("mandrill")
```
The examples below will assume you're loading a particular file from your disk, but you can substitute those
commands with `testimage`.

## Getting started

For these examples you'll need to install both `Images` and `ImageView`.
Load the code for these packages with

```julia
using Images, ImageView
```

### Loading your first image

You likely have a number of images already at your disposal, and you can use these, TestImages.jl, or
run `readremote.jl` in the `test/` directory.
(This requires an internet connection.)
These will be deposited inside an `Images` directory inside your temporary directory
(e.g., `/tmp` on Linux systems). The `"rose.png"` image in this example comes from the latter.

Let's begin by reading an image from a file:
```
julia> img = imread("rose.png")
RGB Image with:
  data: 70x46 Array{RGB{Ufixed8},2}
  properties:
    spatialorder:  x y
```
If you're using Images through IJulia, rather than this text output you probably see the image itself.
This is nice, but often it's quite helpful to see the structure of these Image objects.
This happens automatically at the REPL, or within IJulia you can call
```
show(img)
```
to see the output above.

As you can see, this is an RGB image. It is stored as a two-dimensional `Array` of `RGB{Ufixed8}`.
To see what this pixel type is, we can do the following:
```
julia> img[1,1]
RGB{Ufixed8}(Ufixed8(0.188),Ufixed8(0.184),Ufixed8(0.176))
```
This extracts the first pixel, the one visually at the upper-left of the image. You can see that
an `RGB` (which comes from the [Color](https://github.com/JuliaLang/Color.jl) package) is a triple of values.
The `Ufixed8` type (which comes from the [FixedPointNumbers](https://github.com/JeffBezanson/FixedPointNumbers.jl) package)
represents fractional numbers, having values between 0 and 1 inclusive, using just 1 byte (8 bits).
If you've previously used other image processing libraries, you may be used to thinking of two basic
image types, floating point-valued and integer-valued. In those libraries, `1.0` commonly means "saturated"
for floating point-valued images, whereas for a `Uint8` image 255 means saturated.
`Images.jl` unifies these two types so that `1` always means saturated, making it easier to write
generic algorithms and visualization packages, while still allowing one to use efficient (or C-compatible)
raw representations.

You can see that this image has `properties`, in this case just the single property `"spatialorder"`.
`["x", "y"]` indicates that, after color, the image data are in "horizontal-major" order,
meaning that a pixel at spatial location `(x,y)` would be addressed as `img[x,y]`.
`["y", "x"]` would indicate vertical-major. Consequently, this image is 70 pixels wide and 46 pixels high.

Note that the image was loaded in "non-permuted" form, i.e., following the direct representation on disk.
If you prefer to work with plain arrays, you can convert it:
```
julia> imA = convert(Array, img);

julia> summary(imA)
"46x70 Array{RGB{Ufixed8},2}"
```
You can see that this permuted the dimensions into vertical-major order, but
preserved this as an `Array{RGB}`. If you prefer to extract into an array of the
elementary type in color-last order (typical of Matlab), you can use
```
julia> imA = separate(img)
RGB Image with:
  data: 46x70x3 Array{Ufixed8,3}
  properties:
    colorspace: RGB
    colordim: 3
    spatialorder:  x y
```
You can see that two new properties were added: `"colordim"`, which specifies which dimension of the array
is used to encode color, and `"colorspace"`. Compare this to
```
julia> imA = reinterpret(Ufixed8, img)
RGB Image with:
  data: 3x70x46 Array{Ufixed8,3}
  properties:
    colorspace: RGB
    colordim: 1
    spatialorder:  x y
```
`convert(Array, img)` and `separate(img)` make copies of the data,
whereas `reinterpret` just gives you a new view of the same underlying memory as `img`.


## Further documentation ##

Detailed documentation about the design of the library
and the available functions
can be found in the `doc/` directory. Here are some of the topics available:

- [Getting started](doc/usage.md), a short demonstration
- The [core](doc/core.md), i.e., the representation of images
- [I/O](doc/extendingIO.md) and custom image file formats
- [Function reference](doc/function_reference.md)
- [Overlays](doc/overlays.md), a type for combining multiple grayscale arrays into a single color array

# Credits

Elements of this package descend from "image.jl"
that once lived in Julia's `extras/` directory.
That file had several authors, of which the primary were
Jeff Bezanson, Stefan Kroboth, Tim Holy, Mike Nolta, and Stefan Karpinski.
This repository has been quite heavily reworked;
the current package maintainer is Tim Holy.
