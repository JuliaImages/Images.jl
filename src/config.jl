##########  Configuration   #########

have_imagemagick = false
@unix_only which_cmd = "which"
@windows_only which_cmd = "where"
_, p = readsfrom(`$which_cmd convert`)
wait(p)
if !success(p)
    warn("ImageMagick utilities not found. Install for more file format support.")
else
    have_imagemagick = true
end
