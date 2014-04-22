# Images.jl

An image processing library for [Julia](http://julialang.org/).

[![Status](http://iainnz.github.io/packages.julialang.org/badges/Images_0.3.svg)](http://iainnz.github.io/packages.julialang.org/badges/Images_0.3.svg)

## Aims

Images are very diverse.
You might be working with a single photograph, or you
might be processing MRI scans from databases of hundreds of subjects.
In the
former case, you might not need much information about the image; perhaps just
the pixel data itself suffices.
In the latter case, you probably need to
know a lot of extra details, like the patient's ID number and characteristics of
the image like the physical size of a voxel in all three dimensions.

Even the raw pixel data can come in several different flavors:
- For example, you might represent each pixel as a `Uint32` because you are encoding red, green, and blue in separate 8-bit words within each integer---visualization libraries like Cairo use these kinds of representations, and you might want to interact with those libraries efficiently.
Alternatively, perhaps you're an astronomer and your camera has such high precision that 16 bits aren't enough to encode grayscale intensities.
- If you're working with videos (images collected over time), you might have arrays that are too big to load into memory at once.
You still need to be able to "talk about" the array as a whole, but it may not be trivial to adjust the byte-level representation to match some pre-conceived storage order.

To handle this diversity, we've endeavored to take a "big tent" philosophy.
We avoid imposing a strict programming model, because we don't want to make life
difficult for people who have relatively simple needs.
If you do all your image
processing with plain arrays (as is typical in Matlab, for example), that should
work just fine---you just have to respect certain conventions, like a
`m`-by-`n`-by-`3` array always means an RGB image with the third dimension
encoding color.
You can call the routines that are in this package, and write
your own custom algorithms that assume the same format.

But if your images don't fit neatly into these assumptions, you can choose to
represent your images using other schemes; you can then tag them with enough
metadata that there's no ambiguity about the meaning of anything.
The algorithms
in this package are already set to look for certain types of metadata, and
adjust their behavior accordingly.

One of the potential downsides of flexibility is complexity---it makes it harder
to write generic algorithms that work with all these different representations.
We've tried to mitigate this downside by providing many short utility functions
that abstract away much of the complexity.
Many algorithms require just a
handful of extra lines to work generically.
Or if you just want to get
something running, it usually only takes a couple of lines of code to assert
that the input is in the format you expect.

## Installation

Install via the package manager,

```
Pkg.add("Images")
```

It's helpful to have ImageMagick installed on your system, as Images relies on it for reading and writing many common image types.
For unix platforms, adding the Images package should install ImageMagick for you automatically.
If this fails, try installing it [manually](http://www.imagemagick.org/download/binaries/).
Depending on where it installs, you may need to set the `MAGICK_HOME` environment variable to help Julia find the library (or set your `DL_LOAD_PATH`).

Note that on older RedHat-based distributions, the packaged version of the library may be too old.
If that is the case, a newer library may be [required](http://dl.nux.ro/rpm/nux-imagemagick.repo).
You may need to edit the `releasever` parameter to match your installation.

On Macs, there is now experimental support for reading images using the built-in OS X frameworks.
For many common image types, this reader will be tried before ImageMagick.  This reader
is now enabled by default on Macs; if you need to disable it in favor of ImageMagick,
just comment out line 105 of `src/io.jl`, which reads `img = imread(filename, OSXNative)`.

On Windows it is mandatory to have ImageMagick previously installed because the installer requires user interaction so it cannot be done by the package alone. Get the current version from http://www.imagemagick.org/script/binary-releases.php#windows (e.g. ImageMagick-6.8.8-7-Q16-x86-dll.exe) and make sure that the "Install development headers and libraries for C and C++" checkbox is selected. You may choose to let the installer add the installation directory to the system path or provide it separately. In the later case you may add it to your `.juliarc.jl` file as (for example) `push!(Base.DL_LOAD_PATH, "C:/programs/ImageMagick-6.8.8"`)

When manual intervention is necessary, you may need to restart Julia for the necessary changes to take effect.

## Image viewing

If you're using the IJulia notebook, images will be displayed [automatically](http://htmlpreview.github.com/?https://github.com/timholy/Images.jl/blob/master/ImagesDemo.html).

Julia code for the display of images has been moved to [ImageView](https://github.com/timholy/ImageView.jl).

## Documentation ##

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
