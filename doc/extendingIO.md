## Extending Image I/O

It is not difficult to extend the Images library to handle custom
image files.  For an example, here are the essential steps for code
that reads a `.sif` file, a file format used by Andor Technology for
their scientific CCD cameras.

### Implementation for personal use

All of the following code is found in
one file, `readSIF.jl`.

First, we create a new type for the image, whose parent is
`Images.ImageFileType`.  Next, we register the `.sif` extension and
the "magic bytes" to the new type.  The magic bytes are found at
the start of the file and are used to uniquely identify the format.

```julia
# readSIF.jl, adds an imread function for Andor .sif images

using Images

type AndorSIF <: Images.ImageFileType end
add_image_file_format(".sif", b"Andor Technology Multi-Channel File", AndorSIF)
```

Next, we create an imread function to handle the new image type.  But
first, we have to explicitly import the imread function so that we can
extend it:

```julia
import Images.imread
function imread{S<:IO}(stream::S, ::Type{AndorSIF})
    seek(stream, 0)
    l = strip(readline(stream))
    l == "Andor Technology Multi-Channel File" || error("Not an Andor file: " * l)
```

When this imread is called, the magic bytes have already been read
from the stream.  In this case, it's easier to parse the header if we start over
from the beginning of the file with the `seek(stream, 0)` statement.

Next, we parse the header information.  Many of the fields in this
file are space delimited within lines.  If something unexpected is
read, we spit out an error (very important when the file format
changes, pulling out the rug from underneath you).  This particular file
type has a large number of fields.  For completeness, we will store
all of these fields in a `Dict` named `ixon`:

```julia
    # ...skipped a few uninteresting fields before here
    l = strip(readline(stream))
    fields = split(l)
    fields[1] == "65547" || fields[1] == "65558" ||
        error("Unknown TInstaImage version number at line 3: " * fields[1])

    ixon = {"data_type" => int(fields[2])}
    ixon["active"] = int(fields[3])
    ixon["structure_vers"] = int(fields[4]) # (== 1)
    # ...and skip a whole bunch of uninteresting fields after here
```

We keep reading and parsing the image header until we reach the first
byte of the actual image data.  Along the way, we collect variables
`width`, `height`, and `frames` (this image file is actually a 3D
array, an image sequence over time).  Now we are ready to read the
actual pixel data:

```julia
    pixels = read(stream, Float32, width, height, frames)
```

Finally, we wrap up by setting the properties, including the new
`ixon` collection, and return the `Image`:

```julia
    prop = {"colorspace" => "Gray", "spatialorder" => "yxt", "ixon" => ixon}
    Image(pixels, prop)
end # imread()
```

### Contributing a file format to Images

To make your file format available to others, only a few changes are needed.
First, move your "registration" code from `readSIF.jl` to Images'
`io.jl`:

```julia
type AndorSIF <: ImageFileType end
add_image_file_format(".sif", b"Andor Technology Multi-Channel File", AndorSIF, "readSIF.jl")
```
Note that we've added one more argument to this function, the `"readSIF.jl"`
string. By supplying a filename, you're setting it up so that the code to handle
SIF images is loaded automatically the first time you try to read or write a SIF
file.

The final step is to take the remainder of `readSIF.jl` and add it
to the `ioformats` directory (inside `src/`). Submit a pull request, and once
merged your format will be usable by anyone.
