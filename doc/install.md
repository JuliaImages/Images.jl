### Manual installation on Windows

Modern versions of Images should install ImageMagick automatically even on Windows.
However, if this fails, get the current version from http://www.imagemagick.org/script/binary-releases.php#windows (e.g. ImageMagick-6.8.8-7-Q16-x86-dll.exe) and make sure that the "Install development headers and libraries for C and C++" checkbox is selected.
You may choose to let the installer add the installation directory to the system path or provide it separately.
In the later case you may add it to your `.juliarc.jl` file as (for example) `push!(Base.DL_LOAD_PATH, "C:/programs/ImageMagick-6.8.8"`)

**When manual intervention is necessary, you need to restart Julia for the necessary changes to take effect.**

### Fixing broken installations on Macs

Before asking for help, please try the following sequence:
```julia
using Homebrew
Homebrew.rm("imagemagick")
Homebrew.update()
Homebrew.add("imagemagick")
Pkg.build("Images")
```

## Reading images on Macs

On Macs, there is now support for reading images using the built-in OS X frameworks.
For many common image types, this reader will be tried before ImageMagick.  This reader
is now enabled by default on Macs; if you need to disable it in favor of ImageMagick,
just comment out the line of `src/io.jl` which reads `img = imread(filename, OSXNative)`.

## Debugging installation problems

Images depends on ImageMagick, and this dependency is the main source of trouble installing Images.
If this fails, try the following steps:
- `Pkg.status()`: does it report that Images is "dirty"? If so, you have made changes that are likely preventing you
  from installing the latest version of Images. You might want to `git stash` your changes or,
  if there's nothing you want to keep, just delete the entire Images package folder and `Pkg.add("Images")` freshly.
- Try `Pkg.update()` to see if your problem has already been fixed.
- Try `Pkg.build()` to force a new build even if you're running the latest version.

If these don't work for you, please report the problem to the Images issue tracker.

You can also try [manual installation](http://www.imagemagick.org/download/binaries/).
Depending on where it installs, you may need to set the `MAGICK_HOME` environment variable to help Julia find the library (or set your `DL_LOAD_PATH`).

Note that on older RedHat-based distributions, the packaged version of the library may be too old.
If that is the case, a newer library may be [required](http://dl.nux.ro/rpm/nux-imagemagick.repo).
You may need to edit the `releasever` parameter to match your installation.
