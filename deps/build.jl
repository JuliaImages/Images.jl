using BinDeps

@BinDeps.setup

mpath = get(ENV, "MAGICK_HOME", "") # If MAGICK_HOME is defined, add to library search path
if !isempty(mpath)
    push!(DL_LOAD_PATH, mpath)
    push!(DL_LOAD_PATH, joinpath(mpath,"lib"))
end
libnames = ["libMagickWand", "CORE_RL_wand_"]
suffixes = ["", "-Q16", "-6.Q16", "-Q8"]
options = ["", "HDRI"]
extensions = ["", ".so.4", ".so.5"]
aliases = vec(libnames.*transpose(suffixes).*reshape(options,(1,1,length(options))).*reshape(extensions,(1,1,1,length(extensions))))
libwand = library_dependency("libwand", aliases = aliases)

@linux_only begin
    provides(AptGet, "libmagickwand4", libwand)
    provides(AptGet, "libmagickwand5", libwand)
    provides(Pacman, "imagemagick", libwand)
    provides(Yum, "ImageMagick", libwand)
end

# TODO: remove me when upstream is fixed
@windows_only push!(BinDeps.defaults, BuildProcess)

@windows_only begin
    const OS_ARCH = (WORD_SIZE == 64) ? "x64" : "x86"

    # Will need to be updated for releases
    # TODO: checksums: we have gpg
    magick_exe = "ImageMagick-6.8.9-7-Q16-$(OS_ARCH)-dll.exe"

    magick_tmpdir = BinDeps.downloadsdir(libwand)
    magick_url = "http://www.imagemagick.org/download/binaries/$(magick_exe)"
    magick_libdir = joinpath(BinDeps.libdir(libwand), OS_ARCH)

    innounp_url = "https://julialang.s3.amazonaws.com/bin/winnt/extras/innounp.exe"

    provides(BuildProcess,
        (@build_steps begin
            CreateDirectory(magick_tmpdir)
            CreateDirectory(magick_libdir)
            FileDownloader(magick_url, joinpath(magick_tmpdir, magick_exe))
            FileDownloader(innounp_url, joinpath(magick_tmpdir, "innounp.exe"))
            @build_steps begin
                ChangeDirectory(magick_tmpdir)
                info("Installing ImageMagick library")
                `innounp.exe -q -y -b -e -x -d$(magick_libdir) $(magick_exe)`
            end
        end),
        libwand,
        os = :Windows,
        unpacked_dir = magick_libdir)
end

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

# Save the library version; by checking this now, we avoid a runtime dependency on libwand
# See https://github.com/timholy/Images.jl/issues/184#issuecomment-55643225
module CheckVersion
include("deps.jl")
if isdefined(:__init__)
    __init__()
end
p = ccall((:MagickQueryConfigureOption, libwand), Ptr{Uint8}, (Ptr{Uint8},), "LIB_VERSION_NUMBER")
vstr = string("v\"", join(split(bytestring(p), ',')[1:3], '.'), "\"")
open("deps.jl", "a") do file
    write(file, "const libversion = $vstr\n")
end
end
