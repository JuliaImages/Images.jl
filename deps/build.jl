using BinDeps

@BinDeps.setup

mpath = get(ENV, "MAGICK_HOME", "") # If MAGICK_HOME is defined, add to library search path
if !isempty(mpath)
    push!(DL_LOAD_PATH, mpath)
    push!(DL_LOAD_PATH, joinpath(mpath,"lib"))
end
libnames = ["libMagickWand"]
suffixes = ["", "-Q16", "-6.Q16", "-Q8"]
options = ["", "HDRI"]
aliases = vec(libnames.*transpose(suffixes).*reshape(options,(1,1,length(options))))
libwand = library_dependency("libwand", aliases = aliases)

@linux_only begin
    provides(AptGet, "libmagickwand4", libwand)
    provides(Yum, "ImageMagick", libwand)
end

# @windows_only begin
#     libwand = library_dependency("CORE_RL_wand_")
# 
#     const OS_ARCH = WORD_SIZE == 64 ? "x86_64" : "x86"
# 
#     if WORD_SIZE == 32
#         provides(Binaries,URI("http://www.imagemagick.org/download/binaries/ImageMagick-6.8.8-6-Q16-windows-dll.exe"),libwand,os = :Windows)
#     else
#         provides(Binaries,URI("http://www.imagemagick.org/download/binaries/ImageMagick-6.8.8-6-Q16-windows-x64-dll.exe"),libwand,os = :Windows)
#     end
# end

@osx_only begin
    if Pkg.installed("Homebrew") === nothing
            error("Homebrew package not installed, please run Pkg.add(\"Homebrew\")")
    end
    using Homebrew
    provides( Homebrew.HB, "imagemagick", libwand, os = :Darwin, onload =
    """
    function __init__()
        ENV["MAGICK_CONFIGURE_PATH"] = joinpath("$(Homebrew.prefix("imagemagick"))","lib","ImageMagick","config-Q16")
        ENV["MAGICK_CODER_MODULE_PATH"] = joinpath("$(Homebrew.prefix("imagemagick"))", "lib","ImageMagick","modules-Q16","coders")
    end
    """ )
end

@BinDeps.install [:libwand => :libwand]
