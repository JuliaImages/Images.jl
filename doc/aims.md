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
