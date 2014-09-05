# Images.jl

An image processing library for [Julia](http://julialang.org/).

[![Status](http://iainnz.github.io/packages.julialang.org/badges/Images_0.3.svg)](http://iainnz.github.io/packages.julialang.org/badges/Images_0.3.svg) [![Coverage Status](https://coveralls.io/repos/timholy/Images.jl/badge.png?branch=master)](https://coveralls.io/r/timholy/Images.jl?branch=master)

## Installation

Install via the package manager,

```
Pkg.add("Images")
```

It's helpful to have ImageMagick installed on your system, as Images relies on it for reading and writing many common image types.
ImageMagick _should_ be installed for your automatically. In case of trouble,
more details about manual installation and troubleshooting can be found in the [installation help](doc/install.md).

## Package interactions

A few other packages define overlapping functions or types (`PyPlot` defines `imread`, and `Winston` defines `Image`).
When using the two packages together, you can always specify which version you want with `Images.imread("myimage.png")`.

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
Depending on your task, it's also very useful to have two other packages
loaded, `Color` and `FixedPointNumbers`.
Load the code for all of these packages with

```julia
using Images, Color, FixedPointNumbers, ImageView
```

### Loading your first image: how images are represented

You likely have a number of images already at your disposal, and you can use these, TestImages.jl, or
run `readremote.jl` in the `test/` directory.
(This requires an internet connection.)
These will be deposited inside an `Images` directory inside your temporary directory
(e.g., `/tmp` on Linux systems). The `"rose.png"` image in this example comes from the latter.

Let's begin by reading an image from a file:
```
julia> img = imread("rose.png")
RGB Image with:
  data: 70x46 Array{RGB{UfixedBase{Uint8,8}},2}
  properties:
    IMcs: sRGB
    spatialorder:  x y
    pixelspacing:  1 1
```
If you're using Images through IJulia, rather than this text output you probably see the image itself.
This is nice, but often it's quite helpful to see the structure of these Image objects.
This happens automatically at the REPL, or within IJulia you can call
```
show(img)
```
to see the output above.

The first line tells you that this is an RGB image.
It is stored as a two-dimensional `Array` of `RGB{UfixedBase{Uint8,8}}`.
To see what this pixel type is, we can do the following:
```
julia> img[1,1]
RGB{Ufixed8}(0.188,0.184,0.176)
```
This extracts the first pixel, the one visually at the upper-left of the image. You can see that
an `RGB` (which comes from the [Color](https://github.com/JuliaLang/Color.jl) package) is a triple of values.
The `Ufixed8` number type (which comes from the
[FixedPointNumbers](https://github.com/JeffBezanson/FixedPointNumbers.jl) package), and whose long
name is `UfixedBase{Uint8,8}`)
represents fractional numbers, those that can encode values that lie between 0 and 1, using just 1 byte (8 bits).
If you've previously used other image processing libraries, you may be used to thinking of two basic
image types, floating point-valued and integer-valued. In those libraries, "saturated"
(the color white for an RGB image) would be
represented by `1.0` for floating point-valued images, 255 for a `Uint8` image,
and `0x0fff` for an image collected by a 12-bit camera.
`Images.jl`, via Color and FixedPointNumbers, unifies these so that `1` always means saturated, no
matter whether the element type is `Float64`, `Ufixed8`, or `Ufixed12`.
This makes it easier to write generic algorithms and visualization code,
while still allowing one to use efficient (and C-compatible) raw representations.

You can see that this image has `properties`, of which there are three:
`"IMcs"`, `"spatialorder"` and `"pixelspacing"`.
We'll talk more about the latter two in the next section.
The `"IMcs"` is really for internal use by ImageMagick; it says that the colorspace
is `"sRGB"`, although (depending on which version of the library you have)
you may see it say `"RGB"`.
Such differences are due to [changes](http://www.imagemagick.org/script/color-management.php)
in how ImageMagick handles colorspaces, and the fact that both older
and newer versions of the library are still widespread.

Given an Image `img`, you can access the underlying array with `A = data(img)`.
Images is designed to work with either plain arrays or with Image types---in general, though,
you're probably best off leaving things as an Image, particularly if you work
with movies, 3d images, or other more complex objects.
Likewise, you can retrieve the properties using `props = properties(img)`.

### Storage order and changing the representation of images

In the example above, the `"spatialorder"` property has value `["x", "y"]`.
This indicates that the image data are in "horizontal-major" order,
meaning that a pixel at spatial location `(x,y)` would be addressed as `img[x,y]`
rather than `img[y,x]`. `["y", "x"]` would indicate vertical-major.
Consequently, this image is 70 pixels wide and 46 pixels high.

Images returns this image in horizontal-major order because this is how it was stored on disk.
Because the Images package is designed to scale to terabyte-sized images, a general philosophy
is to work with whatever format users provide without forcing changes to the raw array representation.
Consequently, when you load an image, its representation will match that used in the file.

Of course, if you prefer to work with plain arrays, you can convert it:
```
julia> imA = convert(Array, img);

julia> summary(imA)
"46x70 Array{RGB{UfixedBase{Uint8,8}},2}"
```
You can see that this permuted the dimensions into vertical-major order, consistent
with the column-major order with which Julia stores `Arrays`. Note that this
preserved the element type, returning an `Array{RGB}`.
If you prefer to extract into an array of plain numbers in color-last order
(typical of Matlab), you can use
```
julia> imsep = separate(img)
RGB Image with:
  data: 46x70x3 Array{UfixedBase{Uint8,8},3}
  properties:
    IMcs: sRGB
    colorspace: RGB
    colordim: 3
    spatialorder:  y x
    pixelspacing:  1 1
```
You can see that `"spatialorder"` was changed to reflect the new layout, and that
two new properties were added: `"colordim"`, which specifies which dimension of the array
is used to encode color, and `"colorspace"` so you know how to interpret these colors.

Compare this to
```
julia> imr = reinterpret(Ufixed8, img)
RGB Image with:
  data: 3x70x46 Array{UfixedBase{Uint8,8},3}
  properties:
    IMcs: sRGB
    colorspace: RGB
    colordim: 1
    spatialorder:  x y
    pixelspacing:  1 1
```
`reinterpret` gives you a new view of the same underlying memory as `img`, whereas
`convert(Array, img)` and `separate(img)` create new arrays if the memory-layout
needs alteration.

You can go back to using ColorValues to encode your image this way:
```
julia> imcomb = convert(Image{RGB}, imsep)
RGB Image with:
  data: 46x70 Array{RGB{UfixedBase{Uint8,8}},2}
  properties:
    IMcs: sRGB
    spatialorder:  y x
    pixelspacing:  1 1
```
or even change to a new colorspace like this:
```
julia> imhsv = convert(Image{HSV}, float32(img))
HSV Image with:
  data: 70x46 Array{HSV{Float32},2}
  properties:
    IMcs: sRGB
    spatialorder:  x y
    pixelspacing:  1 1
```
Many of the colorspaces supported by Color need a wider range of values than `[0,1]`,
so it's necessary to convert to floating point.

If you say `view(imhsv)`, you may be surprised to see something that looks
like the original RGB image. Since the colorspace is known, it converts
to RGB before rendering it. If, for example, you wanted to see what a
"pure-V" image looks like, you can do this:
```julia
imv = share(imhsv, [HSV(0, 0, imhsv[i,j].v) for i = 1:size(imhsv,1),j = 1:size(imhsv,2)])
view(imv)
```
and a pure-H image like this:
```julia
imh = share(imhsv, [HSV(imhsv[i,j].h, 0.5, 0.5) for i = 1:size(imhsv,1),j = 1:size(imhsv,2)])
view(imh)
```
(Hue without saturation or value generates gray or black, so we used a constant different from zero for these parameters.)

![raw](doc/figures/rose_hsv.png)

### Other properties, and usage of Units

The `"pixelspacing"` property informs ImageView that this image has an aspect ratio 1.
In scientific or medical imaging, you can use actual units to encode this property,
for example through the [SIUnits](https://github.com/Keno/SIUnits.jl) package.
For example, if you're doing microscopy you might specify
```
using SIUnits
img["pixelspacing"] = [0.32Micro*Meter,0.32Micro*Meter]
```
If you're performing three-dimensional imaging, you might set different values for the
different axes:
```
using SIUnits.ShortUnits
mriscan["pixelspacing"] = [0.2mm, 0.2mm, 2mm]
```

ImageView includes facilities for scale bars, and by supplying your pixel spacing
you can ensure that the scale bars are accurate.

### A brief demonstration of image processing

Now let's work through a more sophisticated example:
```
using Images, TestImages, ImageView
img = testimage("mandrill")
view(img)
# Let's do some blurring
kern = ones(Float32,7,7)/49
imgf = imfilter(img, kern)
view(imgf)
# Let's make an oversaturated image
imgs = 2imgf
view(imgs)
```
![processing](doc/figures/mandrill.jpg)


## Further documentation ##

Detailed documentation about the design of the library
and the available functions
can be found in the `doc/` directory. Here are some of the topics available:

- The [core](doc/core.md), i.e., the representation of images
- [I/O](doc/extendingIO.md) and custom image file formats
- [Function reference](doc/function_reference.md)
- [Overlays](doc/overlays.md), a type for combining multiple grayscale arrays into a single color array
- [Changes](doc/fixing_breakages.md) describes some recent changes and how to fix likely breakages in old code

# Credits

Elements of this package descend from "image.jl"
that once lived in Julia's `extras/` directory.
That file had several authors, of which the primary were
Jeff Bezanson, Stefan Kroboth, Tim Holy, Mike Nolta, and Stefan Karpinski.
This repository has been quite heavily reworked;
the current package maintainer is Tim Holy, and
important contributions have been made by Ron Rock,
Kevin Squire, Lucas Beyer, Elliot Saba, Isaiah Norton,
Daniel Perry, Waldir Pimenta, Tobias Knopp,
Jason Merrill, Dahua Lin, and several others.
