##########  Configuration   #########

have_imagemagick = false
@unix_only begin
_, p = readsfrom(`which convert`)
wait(p)
if !success(p)
    warn("ImageMagick utilities not found. Install for more file format support.")
else
    have_imagemagick = true
end
end
