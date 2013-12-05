# Julia wrapper for header: /usr/include/turbojpeg.h

# Based on TurboJPEG:
# Copyright (C)2009-2012 D. R. Commander.  All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the libjpeg-turbo Project nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS",
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Automatically generated using Clang.jl wrap_c, version 0.0.0
#   using Clang.wrap_c
#   context = wrap_c.init(; output_file="turbojpeg.jl",
#                         header_library=x->"turbojpeg",
#                         header_wrapped=(x,y)->contains(y, "turbojpeg"),
#                         common_file="turbojpeg_h.jl", clang_diagnostics=true,
#                         clang_args=["-v"], clang_includes=["/cmn/julia/usr/bin/"])
#   context.options.wrap_structs = true
#   wrap_c.wrap_c_headers(context, ["/usr/include/turbojpeg.h"])

################################################################################

module TurboJPEG
# Flags
module Flags
    const BOTTOMUP = 2
    const FORCEMMX = 8
    const FORCESSE = 16
    const FORCESSE2 = 32
    const FORCESSE3 = 128
    const FASTUPSAMPLE = 256
    const NOREALLOC = 1024
    const FASTDCT = 2048
    const ACCURATEDCT = 4096
end
module Sampling
    const S444 = 0
    const S422 = 1
    const S420 = 2
    const SGRAY = 3
    const S440 = 4
end
# Pixel types
module PixelFormat
    const RGB = 0
    const BGR = 1
    const RGBX = 2
    const BGRX = 3
    const XBGR = 4
    const XRGB = 5
    const GRAY = 6
    const RGBA = 7
    const BGRA = 8
    const ABGR = 9
    const ARGB = 10
end

# Image operations
module Operations
    const TJXOP_NONE = 0
    const TJXOP_HFLIP = 1
    const TJXOP_VFLIP = 2
    const TJXOP_TRANSPOSE = 3
    const TJXOP_TRANSVERSE = 4
    const TJXOP_ROT90 = 5
    const TJXOP_ROT180 = 6
    const TJXOP_ROT270 = 7
end

module TransformOptions
    const PERFECT = 1
    const TRIM = 2
    const CROP = 4
    const GRAY = 8
    const NOOUTPUT = 16
end

type TJConstArray{T}
    data::Array{T,1}
end
getindex(a::TJConstArray, idx::Int) = a.data[idx+1]

# Red offset (in bytes) for a given pixel format.  This specifies the number
# of bytes that the red component is offset from the start of the pixel.  For
# instance, if a pixel of format TJ_BGRX is stored in <tt>char pixel[]</tt>,
# then the red component will be <tt>pixel[tjRedOffset[TJ_BGRX]]</tt>.

const RedOffset = TJConstArray([0, 2, 0, 2, 3, 1, 0, 0, 2, 3, 1])

# Green offset (in bytes) for a given pixel format.  This specifies the number
# of bytes that the green component is offset from the start of the pixel.
# For instance, if a pixel of format TJ_BGRX is stored in
# <tt>char pixel[]</tt>, then the green component will be
# <tt>pixel[tjGreenOffset[TJ_BGRX]]</tt>.

const GreenOffset = TJConstArray([1, 1, 1, 1, 2, 2, 0, 1, 1, 2, 2])

# Blue offset (in bytes) for a given pixel format.  This specifies the number
# of bytes that the Blue component is offset from the start of the pixel.  For
# instance, if a pixel of format TJ_BGRX is stored in <tt>char pixel[]</tt>,
# then the blue component will be <tt>pixel[tjBlueOffset[TJ_BGRX]]</tt>.

const BlueOffset = TJConstArray([2, 0, 2, 0, 1, 3, 0, 2, 0, 1, 3]);

# Pixel size (in bytes) for a given pixel format.
PixelSize = TJConstArray([3, 3, 4, 4, 4, 4, 1, 4, 4, 4, 4]);

const TJ_NUMSAMP = 5
const TJ_NUMPF = 11

const TJ_NUMXOP = 8
const TJ_YUV = 512
const TJ_BGR = 1

end # module TurboJPEG

type tjscalingfactor
    num::Cint
    denom::Cint
end
type tjregion
    x::Cint
    y::Cint
    w::Cint
    h::Cint
end
type tjtransform
    r::tjregion
    op::Cint
    options::Cint
    data::Ptr{None}
    customFilter::Ptr{Void}
end
typealias tjhandle Ptr{None}


function tjInitCompress()
  ccall( (:tjInitCompress, turbojpeg), tjhandle, (), )
end
function tjCompress2(handle::tjhandle, srcBuf, width, pitch, height,
                     pixelFormat, jpegBuf, jpegSize, jpegSubsamp,
                     jpegQual, flags)
  ccall( (:tjCompress2, turbojpeg), Cint, (tjhandle, Ptr{Cuchar}, Cint, Cint, Cint, Cint, Ptr{Ptr{Cuchar}}, Ptr{Culong}, Cint, Cint, Cint), int32(handle), srcBuf, int32(width), int32(pitch), int32(height), int32(pixelFormat), jpegBuf, jpegSize, int32(jpegSubsamp), int32(jpegQual), int32(flags))
end

function tjBufSize(width::Cint, height::Cint, jpegSubsamp::Cint)
  ccall( (:tjBufSize, turbojpeg), Culong, (Cint, Cint, Cint), width, height, jpegSubsamp)
end
function tjBufSizeYUV(width::Cint, height::Cint, subsamp::Cint)
  ccall( (:tjBufSizeYUV, turbojpeg), Culong, (Cint, Cint, Cint), width, height, subsamp)
end
function tjEncodeYUV2(handle::tjhandle, srcBuf::Ptr{Cuchar}, width::Cint, pitch::Cint, height::Cint, pixelFormat::Cint, dstBuf::Ptr{Cuchar}, subsamp::Cint, flags::Cint)
  ccall( (:tjEncodeYUV2, turbojpeg), Cint, (tjhandle, Ptr{Cuchar}, Cint, Cint, Cint, Cint, Ptr{Cuchar}, Cint, Cint), handle, srcBuf, width, pitch, height, pixelFormat, dstBuf, subsamp, flags)
end
function tjInitDecompress()
  ccall( (:tjInitDecompress, turbojpeg), tjhandle, (), )
end
function tjDecompressHeader2(handle::tjhandle, jpegBuf, jpegSize, width, height, jpegSubsamp)
  ccall( (:tjDecompressHeader2, turbojpeg), Cint, (tjhandle, Ptr{Cuchar}, Culong, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), handle, jpegBuf, jpegSize, width, height, jpegSubsamp)
end
function tjGetScalingFactors(numscalingfactors::Ptr{Cint})
  ccall( (:tjGetScalingFactors, turbojpeg), Ptr{tjscalingfactor}, (Ptr{Cint},), numscalingfactors)
end
function tjDecompress2(handle::tjhandle, jpegBuf::Ptr{Cuchar}, jpegSize::Culong, dstBuf::Ptr{Cuchar}, width::Cint, pitch::Cint, height::Cint, pixelFormat::Cint, flags::Cint)
  ccall( (:tjDecompress2, turbojpeg), Cint, (tjhandle, Ptr{Cuchar}, Culong, Ptr{Cuchar}, Cint, Cint, Cint, Cint, Cint), handle, jpegBuf, jpegSize, dstBuf, width, pitch, height, pixelFormat, flags)
end
function tjDecompressToYUV(handle::tjhandle, jpegBuf::Ptr{Cuchar}, jpegSize::Culong, dstBuf::Ptr{Cuchar}, flags::Cint)
  ccall( (:tjDecompressToYUV, turbojpeg), Cint, (tjhandle, Ptr{Cuchar}, Culong, Ptr{Cuchar}, Cint), handle, jpegBuf, jpegSize, dstBuf, flags)
end
function tjInitTransform()
  ccall( (:tjInitTransform, turbojpeg), tjhandle, (), )
end
function tjTransform(handle::tjhandle, jpegBuf::Ptr{Cuchar}, jpegSize::Culong, n::Cint, dstBufs::Ptr{Ptr{Cuchar}}, dstSizes::Ptr{Culong}, transforms::Ptr{tjtransform}, flags::Cint)
  ccall( (:tjTransform, turbojpeg), Cint, (tjhandle, Ptr{Cuchar}, Culong, Cint, Ptr{Ptr{Cuchar}}, Ptr{Culong}, Ptr{tjtransform}, Cint), handle, jpegBuf, jpegSize, n, dstBufs, dstSizes, transforms, flags)
end
function tjDestroy(handle::tjhandle)
  ccall( (:tjDestroy, turbojpeg), Cint, (tjhandle,), handle)
end
function tjAlloc(bytes::Cint)
  ccall( (:tjAlloc, turbojpeg), Ptr{Cuchar}, (Cint,), bytes)
end
function tjFree(buffer::Ptr{Cuchar})
  ccall( (:tjFree, turbojpeg), None, (Ptr{Cuchar},), buffer)
end
function tjGetErrorStr()
  ccall( (:tjGetErrorStr, turbojpeg), Ptr{Uint8}, (), )
end
function TJBUFSIZE(width::Cint, height::Cint)
  ccall( (:TJBUFSIZE, turbojpeg), Culong, (Cint, Cint), width, height)
end
function TJBUFSIZEYUV(width::Cint, height::Cint, jpegSubsamp::Cint)
  ccall( (:TJBUFSIZEYUV, turbojpeg), Culong, (Cint, Cint, Cint), width, height, jpegSubsamp)
end
