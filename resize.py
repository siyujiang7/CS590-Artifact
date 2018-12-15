def Reformat_Image(index):

    from PIL import Image
    ImageFilePath = str(index)+'.jpg'
    image = Image.open(ImageFilePath, 'r')
    image_size = image.size
    width = image_size[0]
    height = image_size[1]

    if(width != height):
        bigside = width if width > height else height
        bigside += 20
        background = Image.new('RGBA', (bigside, bigside), (255, 255, 255, 255))
        offset = (int(round(((bigside - width) / 2), 0)), int(round(((bigside - height) / 2),0)))

        background.paste(image, offset)
        background.save(index+'.png')
        print("Image has been resized !")

    else:
        print("Image is already a square, it has not been resized !")

for i in range(1,18):
    Reformat_Image(str(i))