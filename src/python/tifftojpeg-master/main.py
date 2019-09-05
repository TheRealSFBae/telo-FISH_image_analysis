from PIL import Image
import glob

for name in glob.glob('*.TIF'):
    im = Image.open(name)
    name = str(name).rstrip(".TIF")
    im.save(name + '.jpg', 'JPEG')

for name in glob.glob('*.tiff'):
    im = Image.open(name)
    name = str(name).rstrip(".tiff")
    im.save(name + '.jpg', 'JPEG')

print "Conversion from tif/tiff to jpg completed!"
