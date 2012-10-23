#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script generates an animated gif from a list of images, provided a source
directory.  It may be used for automatic gif generation in the Gimp batch mode.
This is useful when creating an animation from a lot of images.

Batch-mode usage example
------------------------
gimp -i -b '(python-fu-makegif RUN-NONINTERACTIVE "source_directory"
"2010-*.png" "animation.gif" 542 585 800)' -b '(gimp-quit 0)'

Parameters
----------
source      : string.
              Source directory of the images.

globpattern : string.
              Pattern of the images to be animated (input to glob).

animname    : string.
              Path and name of the gif to be created.

wid         : integer.
              Width of the gif output, in pixels.

hei         : integer.
              Height of the gif output, in pixels.

delay       : integer.
              Delay between frames, in ms.

NOTE: This script is a Gimp python-fu plug-in.  In order to be used, it must
be placed in  the directory ~/./gimp-x.x/plug-ins, and made executable.
"""

from gimpfu import *
from gimpenums import *
from os import sep
from glob import glob


def Makegif(source, globpattern, animname, wid, hei, delay):
    Files = glob(source + sep + globpattern)  # Getting the path of all images.
    Files.sort()
    image = pdb.gimp_image_new(wid, hei, RGB)
    for ImageFileName in Files:
        try:
            layer = pdb.gimp_file_load_layer(image, ImageFileName)
            pdb.gimp_image_add_layer(image, layer, -1)
            pdb.gimp_message("Added " + ImageFileName + ".")
        except:
            pdb.gimp_message("Could not add " + ImageFileName + ".")
            pass
    # Optimizing for gif.
    pdb.gimp_message("Optimizing image for gif.....................")
    drawable = pdb.gimp_image_get_active_layer(image)
    pdb.plug_in_animationoptimize(image, drawable)
    #  Converting image from RGB to indexed colors.
    pdb.gimp_message("Converting image to indexed colors...........")
    pdb.gimp_image_convert_indexed(image, 0, 0, 255, False, True, -9999)
    GifFilename = source + sep + animname  # Saving gif.
    pdb.gimp_message("Saving animation.............................")
    pdb.file_gif_save(image, drawable, GifFilename,
                      GifFilename, 0, 1, delay, 2)
    pdb.gimp_image_delete(image)
    return

# This is the plugin registration function.
register(
    "python_fu_makegif",  # Name of the plug-in.
    "Creates gif from a series of numbered image files",  # Brief description.
    "Creates an animated gif from a list of images.",  # Detailed description.
    "GIMP",  # Developer.
    "GIMP",  # Organization.
    "June 2012",  # Date.
    "<Toolbox>/MyScripts/Batch gif generator",  # GUI menu location.
    "",  # Types of images accepted (if "", does not require image to run).
    [
        (PF_DIRNAME, "source_dirname", "Source directory", "./figures"),
        (PF_STRING, "pattern", "pattern of the images to be animated", "*"),
        (PF_STRING, "name", "Name of GIF animation to be created", "anim.gif"),
        (PF_INT, "figure_width", "Height of image, in pixels", 500),
        (PF_INT, "figure_height", "Height of image, in pixels", 500),
        (PF_INT, "delay", "Delay between frames, in ms", 500),
    ],  # Input parameters.
    [],  # Outputs.
    Makegif,  # Name of the function Gimp calls when the plug-in is executed.
)

#This should be called after registering the plug-in.
main()
