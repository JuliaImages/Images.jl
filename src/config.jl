##########  Configuration   #########

have_imagemagick = false
_, p = readsfrom(`which convert`)
if !Base.wait_success(p)
    warn("ImageMagick utilities not found. Install for more file format support.")
else
    have_imagemagick = true
end

have_winston = false
# # Check whether Winston is installed
# macro usingif(status, sym)
#     if Pkg.installed(string(sym)) != nothing
#         return expr(:toplevel, {expr(:using, {sym}), :($(esc(status)) = true)})
#     end
#     :($(esc(status)) = false)
# end
# 
# @usingif have_winston Winston

# Find a system image viewer
imshow_cmd = ""
if !have_winston
    imshow_cmd_list = ["feh", "gwenview", "open"]
    for thiscmd in imshow_cmd_list
        _, p = readsfrom(`which $thiscmd`)
        if Base.wait_success(p)
            imshow_cmd = thiscmd
            break
        end
    end
    if isempty(imshow_cmd)
        warn("No image viewer found. You will not be able to see images.")
    end
end
