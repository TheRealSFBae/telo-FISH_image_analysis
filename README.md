###### Author: Angelo Pelonero | Email: [angelo.pelonero@ucsf.edu](mailto:angelo.pelonero@ucsf.edu)
# Automated Telo-FISH nuclear segmentation and analysis

Finally - a way to spend the time you once spent in MetaMorph doing other things for your PI.

## Introduction

This software is written to segment nuclei of interest in telo-FISH fluorescent microscopy images. The general rule system is as follows, _per-image_:

1. Open merged (color, all-channel) .jpg of collected microscopy images
2. Split channels into seperate monochrome versions
3. Select potential nuclear ROIs in DAPI (blue) channel
4. Identify "true" ROIs in GFP (green) channel
5. Measure telomere staining intensity in RFP (red) channel
6. Output .csv of mean gray value telomere measurements and a .png of original image + final nuclear selections

## How to use

**Requirements:**

- Fiji or ImageJ
- Preprocessed .TIF to .jpg images (use bundled .py script), loaded into a single directory

**Use of preprocessing Python3 script:**
1. Open Terminal
2. cd to directory containing color .TIF microscopy images
3. run "python tif2jpg.py"

**Use of ImageJ script:**

1. Open Macro in ImageJ/Fiji
2. Click "run"
    2a. Select input directory (contains color .jps)
    2b. Select output directory (where results will be generated)
    2c. Input appropriate suffix (.jpg)
3. Click run and grab a coffee, this software takes awhile

## Known issues

There are a number of issues with this segmentation software:

- Heterogenous collections of images (with varying intensities/exposures) lead to poor segmentation performance due to hard-coded intensity thresholds for ROI selection


- Homogenous collections of images have variable segmentation performance due to high background intensities, awkward object shapes, or a number of other factors

- Using .TIF files prohibits segmentation. No idea why. This is why batch .TIF -> .jpg conversion is needed

- Setting a limit to ROI size prohibits segmentation. Large selections must be dealt with in postprocessing

- Some nuclei humans can ID as valid choices are skipped no matter how the selection thresholds are manipulated

- "Clumps" of nuclei are often detected as single ROIs

## Fixes and Future Directions

### Fixes:

- Use monochrome single channel images directly from scope instead of splitting color .jpg. This will still require .TIF -> .jpg conversion

- Set GFP selection intensity threshold per image. Collect the background/average intensity and scale by n % for more accurate selection

- Translate from Groovy to Python for easier workflow integration

### Future directions:

- Implement a modified color-gradient weighted distance segmentation approach. Some non-ImageJ possibilities:
    - [Combine approaches from DOIs **10.1016/j.jspc.2013.04.003** and **10.10002/cyto.a.22457** (OpenCV)](https://stackoverflow.com/questions/55471954/how-to-make-color-gradient-weigthed-distance-image-in-opencv-python)
    - [Graydist (MatLab)](https://www.mathworks.com/help/images/ref/graydist.html)
    - [Medial Feature Detector, aka MFT (C++)](http://image.ntua.gr/iva/tools/mfd/)

- Dockerize pipeline for simple distribution

### Kudos!

Big thanks to [Dr. Abel Torres-Espin](https://profiles.ucsf.edu/abel.torresespin) for his ImageJ expertise, the [UCSF Library Data Science Initiative](https://www.library.ucsf.edu/ask-an-expert/data-science/) for supporting development, and [Dr. John Greenland](https://profiles.ucsf.edu/john.greenland) for the automation opportunity.