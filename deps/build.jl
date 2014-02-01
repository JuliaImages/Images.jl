using BinDeps

@BinDeps.setup

@linux_only begin
    libnames = ["libMagickWand"]
    suffixes = ["", "-Q16", "-6.Q16", "-Q8"]
    options = ["", "HDRI"]
    aliases = vec(libnames.*transpose(suffixes).*reshape(options,(1,1,length(options))))
    libwand = library_dependency("libMagickWand", aliases=aliases)
    provides(AptGet, "libmagickwand4", libwand)
    provides(Yum, "ImageMagick", libwand)
end

@windows_only begin
    libwand = library_dependency("CORE_RL_wand_")

    const OS_ARCH = WORD_SIZE == 64 ? "x86_64" : "x86"

    if WORD_SIZE == 32
            provides(Binaries,URI("http://www.imagemagick.org/download/binaries/ImageMagick-6.7.7-6-Q16-windows-dll.exe"),libwand,os = :Windows)
    else
            provides(Binaries,URI("http://www.imagemagick.org/download/binaries/ImageMagick-6.7.7-6-Q16-windows-x64-dll.exe"),libwand,os = :Windows)
    end
end

@osx_only begin
    if Pkg.installed("Homebrew") === nothing
            error("Homebrew package not installed, please run Pkg.add(\"Homebrew\")")
    end
    using Homebrew
    libwand = library_dependency("libMagickWand-6.Q16")
    provides( Homebrew.HB, "imagemagick", libwand, os = :Darwin )
end

@BinDeps.install
