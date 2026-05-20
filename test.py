from PIL import Image

for src, dst in [("C:/project/mammo-cad/data/interim/inbreast/AllPng/22579847_301f1776aebbf5d2_MG_R_CC_ANON.png", "mammo_CC_small.png"),
                 ("C:/project/mammo-cad/data/interim/inbreast/AllPng/20587346_e634830794f5c1bd_MG_R_ML_ANON.png", "mammo_MLO_small.png")]:
    img = Image.open(src)
    img.thumbnail((1024, 1024))
    img.save(dst, optimize=True)