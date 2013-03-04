# Images.jl

An image processing library for [Julia](http://julialang.org/).

## Aims

We've endeavored to take a "big tent" philosophy:

- If you do all your image processing with plain old arrays (as is typical in Matlab, for example), that should work just fine. But if you really wished that each image came along with enough information about the camera settings to convert those digital counts to photons, or if you need to know the spacing between slices in those MRI scans, that too is possible.
- You might be in the habit of storing image data as Uint32 because you like to work with 24bit RGB color. Alternatively, maybe you're an astronomer and you use Uint32s simply because your monochrome camera has more than 16 bits. It's straightforward to set things up so that algorithms, I/O, and display to do the right thing.
- You may need to work with indexed images (i.e., images with a colormap), transparency, 3D images, image sequences, and/or 5D images (3 spatial dimensions, a color channel, and time).
- You may have custom file formats that you need to be able to easily import and manipulate with standard tools. The way the data are stored on disk may differ: for example, perhaps you have a single file containing RGB color stored in the order (color, horizontal, vertical), or perhaps you have two color channels each stored in a separate file. Perhaps these files are too large to load into memory at once. Despite these complexities, you should be able to set yourself up to work easily with your image data and apply generic algorithms.

We've tried to set up the core data representation so that all of these things are possible, and can usually leverage a common set of algorithms.

## Documentation ##

Detailed documentation about the design of the library and the available functions can be found in the `doc/` directory.

## Status/TODOs

### Data representation:

The core seems to be largely finished. Obviously, any custom types need to be added individually.

### Configuration

It's helpful to have ImageMagick and some kind of viewer (`feh`, `gwenview`) available. Detection of these extenal packages has improved slightly.

### I/O

- A framework for generic I/O, using the magic bytes of the file, is already in place. It's also designed to be readily extensible for custom image formats, so `imread(myfile)` should always work.
- At the moment the large majority of formats are imported via ImageMagick. In recent times the ImageMagick wrapper has increased in sophistication to better-preseve the image's native format. This includes reading alpha channels and (unfinished) support for indexed images. Image writing is currently broken but will be fixed soon.
- PPM/PGM/PBM is supported directly
- A wrapper of libpng was once written, but it is currently disabled because of changes in Julia itself. It may be less important now because of the improved ImageMagick support, but anyone wishing to polish this up can do so. It's unclear whether the C wrapper is still essential.
- TIFF reading can be found at https://github.com/rephorm/TIFF.jl and should be incorporated

### Display

Needed: direct support for Cairo/Winston/Gadfly. The flexibility in storage format may make display via Cairo more efficient (it will no longer be necessary to take a transpose, since you can work with images in Cairo's native storage order if you prefer).

### Algorithms

Currently none of the algorithms have been ported to the new framework. This should not be terribly difficult, but it desperately needs doing.


# Credits

Many elements of this package descend from "image.jl" that once lived in Julia's `extras/` directory. That file had several authors, of which the primary were Jeff Bezanson, Stefan Kroboth, Tim Holy, Mike Nolta, and Stefan Karpinski. This repository has been quite heavily reworked; the current package maintainer is Tim Holy.
