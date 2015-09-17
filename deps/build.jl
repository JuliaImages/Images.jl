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
extensions = ["", ".so.2", ".so.4", ".so.5"]
aliases = vec(libnames.*transpose(suffixes).*reshape(options,(1,1,length(options))).*reshape(extensions,(1,1,1,length(extensions))))
libwand = library_dependency("libwand", aliases = aliases)

@linux_only begin
    kwargs = Any[(:onload, "ccall((:MagickWandGenesis,libwand), Void, ())")]
    provides(AptGet, "libmagickwand4", libwand; kwargs...)
    provides(AptGet, "libmagickwand5", libwand; kwargs...)
    provides(AptGet, "libmagickwand-6.q16-2", libwand; kwargs...)
    provides(Pacman, "imagemagick", libwand; kwargs...)
    provides(Yum, "ImageMagick", libwand; kwargs...)
end

# TODO: remove me when upstream is fixed
@windows_only push!(BinDeps.defaults, BuildProcess)

@windows_only begin
    const OS_ARCH = (WORD_SIZE == 64) ? "x64" : "x86"

    # TODO: checksums: we have gpg
    # Extract the appropriate filename to download
    magick_base = "http://www.imagemagick.org/download/binaries"
    binariesfn = download(magick_base)
    str = readall(binariesfn)
    pattern = "ImageMagick-6.9.*?-Q16-$(OS_ARCH)-dll.exe"
    m = match(Regex(pattern), str)
    magick_exe = convert(ASCIIString, m.match)

    magick_tmpdir = BinDeps.downloadsdir(libwand)
    magick_url = "$(magick_base)/$(magick_exe)"
    magick_libdir = joinpath(BinDeps.libdir(libwand), OS_ARCH)

    innounp_url = "https://bintray.com/artifact/download/julialang/generic/innounp.exe"

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
        unpacked_dir = magick_libdir,
        preload =
            """
            ENV["MAGICK_CONFIGURE_PATH"] = \"$(escape_string(magick_libdir))\"
            ENV["MAGICK_CODER_MODULE_PATH"] = \"$(escape_string(magick_libdir))\"
            """,
        onload = "ccall((:MagickWandGenesis,libwand), Void, ())")
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
        ENV["PATH"] = joinpath("$(Homebrew.prefix("imagemagick"))", "bin") * ":" * ENV["PATH"]
        ccall((:MagickWandGenesis,libwand), Void, ())
    end
    """ )
end

@BinDeps.install Dict([(:libwand, :libwand)])

# Save the library version; by checking this now, we avoid a runtime dependency on libwand
# See https://github.com/timholy/Images.jl/issues/184#issuecomment-55643225
module CheckVersion
include("deps.jl")
if isdefined(:__init__)
    __init__()
end
p = ccall((:MagickQueryConfigureOption, libwand), Ptr{Uint8}, (Ptr{Uint8},), "LIB_VERSION_NUMBER")
vstr = string("v\"", join(split(bytestring(p), ',')[1:3], '.'), "\"")
open(joinpath(dirname(@__FILE__),"versioninfo.jl"), "w") do file
    write(file, "const libversion = $vstr\n")
end
end
