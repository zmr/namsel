### NOTE: THIS PROJECT IS NO LONGER BEING DEVELOPED. If you are looking for a replacement, Google's Compute [Vision API](https://cloud.google.com/vision) provides fairly accurate OCR on u-chen texts.

# Namsel OCR
An OCR application focused on machine-print Tibetan text

Tested only on Ubuntu 14.04 and higher. 

An overview of the Namsel project can be found in [our article in the journal Himalayan Linguistics](https://escholarship.org/uc/item/6d5781k5).

Check out our library partner for already OCR'd digital text: http://tbrc.org. 

## Install:
```bash
$ bash ubuntu_install.sh
```

This will install required packages, build the cython modules, unpack datasets, and initiate training for the classifiers. Note that training (classify.py) takes up to an hour or more to complete.


## Quickstart

To start, run preprocessing on folder of images:
```bash
$ ./namsel.py preprocess ~/myfolder
```

This will save new, preprocessed image in ~/myfolder/out. Preprocessing will take a few minutes depending on how many CPUs you have available and how many images are in the folder.

Next, run OCR. If pages are "book" (rather than "pecha") - style pages, do the following:

```bash
$ ./namsel.py recognize-volume --page_type=book --format=text ~/myfolder/out
```

To OCR a single page, use the *recognize-page* command and specify a single page file:

```bash
$ ./namsel.py recognize-page --page_type=book --format=text ~/myfolder/out/image-01.tif
```

OCR will run and save the results in a file called ocr_output.txt.

## Preprocessing
Prior to using Namsel, documents in the form of PDFs or images need to be preprocessed. Preprocessing typically involves cleaning up images and putting them in a format Namsel expects.

### Scanning documents
If you are scanning the documents to be OCR'd yourself, here are some tips for improving chances of getting high quality OCR results:
- Scan in black and white
- Scan at a relatively high resolution (400-600 dpi)
- Utilize the scanner's software to align and crop the pages. If your scanner software supports it, deskewing the page and removing empty borders and images can save time later on in the OCR process. (Scantailor, which is mentioned below, won't remove images, but will deskew (rotate) the page and remove empty borders).
- Save images in TIFF format with sensible name (e.g. a sequences of numbers 001.tif, 002.tif, etc)

### Preparing images from PDFs
If your original document is in PDF format, you will need to convert the individual PDF pages to black and white or grayscale jpg, tif, or png images. Black and white tif images are Namsel's preferred format.

There are a variety of tools for converting a PDF to images. The "gs" command from the Ghostscript project is one.

You can invoke gs itself like so:

```bash
$ gs -r600x600 -sDEVICE=tiffg4 -sOutputFile=ocr_%04d.tif -dBATCH -dNOPAUSE mytibetanfile.pdf
```

This will convert all the pages in the pdf to tiff using the "Group 4" compression, which is the most compact form of TIFF compression for black and white images. If your pdf is in grayscale, replace "tiffg4" with "tiffgray." Images will be unpacked at the location where the bash script or gs is run unless you specify otherwise.

### Preparing images using scantailor
[Scantailor](https://github.com/scantailor/scantailor) is an open source project for cleaning up scanned documents and preparing them for OCR. Essentially, it performs 5 core operations:
- Page splitting
- Deskewing
- Content isolation
- Noise removal
- Thresholding (making characters thinner or more bold)

Optionally, it has tools for page dewarping and manual erasing of image content. While not the only tool for image preprocessing, it typically delivers very good results and is easy to use. (A popular alternative is a project called unpaper).

Scantailor has both a graphical and command line interface. The graphical interface is straightforward to use so we won't describe it here. For faster processing on multicore computers, it is ideal to use the command line version of scantailor in order to process pages in parallel. Namsel comes with a utility for batch, multicore processing with Scantailor called "scantailor_multicore.py."

Example:
```bash
python scantailor_multicore.py <my-image-folder> [threshold (optional)]
```

<my-image-folder> is a path to a directory containing tif or jpg images. Threshold controls how bold or thin to make strokes on the page. A threshold higher than 0 makes text more bold or thick. A threshold less than 0 makes it thin. Poorly inked prints sometime benefit from a threshold of 10-30. Low to medium resolution images converted to grayscale from color scans may benefit from thinning or a threshold of -10 to -40. 

For example

```bash
$ python scantailor_multicore.py my-image-folder -20
```

...generates a folder called "out" with images that have been cropped, cleaned, thinned, and deskewed by Scantailor.

Alternatively, you can choose to run scantailor from the the Namsel command line. See the section Preprocessing options below.

### Converting TIFF to Group 4 compression
By default, Scantailor saves images in tiff format using the "lzw" compression format. This format is fine for grayscale and color images, but is unnecessary for black and white images. For black and white tif, convert images to Group 4 (G4) compression if possible. Using the tiffcp utility (part of the libtiff library), convert an entire folder of tiff images like so:

```bash
$ mkdir g4
$ for t in *tif
$ do
$ tiffcp -c g4 $t g4/$t
$ done
```

While it is not necessary to use Group 4 (G4) compression, for projects processing thousands of images, G4 format can greatly decrease the amount of disk storage required for images.

## OCR
###Preprocessing options
In addition to the above parameters, you can also set parameters for the Scantailor application (used with the "preprocessing" command):


  **--layout**
  
The layout of pages that Scantailor can expect. Choices are "single" and "double," referring to scanned images that have up to one or two pages on them.


  **--threshold**
  
The amount of thinning or thickening Scantailor will do. Good values are between -40 and 40. Negative values thin the image, positive values thicken it.


Example command:

```bash
$ ./namsel preprocess --layout=double --st_threshold=-15 /path/to/my-folder-of-tiffs
```

This command will run Scantailor on a folder of tiff-formatted images, command it to split double pages and apply thinning to the characters on the pages.
### Namsel command line options
To run Namsel, simply specify the action you'd like Namsel to take and point the the image or images you would like processed. For example, to OCR a single tif image, run:

```bash
$ ./namsel recognize-page mytibetantextimage.tif
```

For an entire volume:
```bash
./namsel recognize-volume folder-of-tiff-images
```

Other options are "preprocess" and "isolate-lines." Preprocessing is discussed below. "Isolate-lines" runs the Namsel pipeline, but only until the line separation stage and outputs the segmented lines as tif images in a directory called "separated-lines" that is created within the parent directory. (TO BE IMPLEMENTED)

OCR quality can vary widely depending on the runtime configuration being used. Below is a list of tunable configuration parameters that Namsel uses. 


  **--page_type**
  
Choices "book" or "pecha." If not specified, Namsel will attempt to determine the page type based on the length and height of the page.


  **--recognizer**
  
This is the type of recognizer that is used. The options are "hmm" and "probout." Use "hmm" most of the time. If the text you are OCR'ing contains many unusual character combinations and/or many sections requiring complex segmentation, the "probout" recognizer may yield better results.


  **--line_break_method**
  
Options are "line_cut" and "line_cluster." Namsel will attempt to choose for you if this is left unspecified. Generally, "line_cut" works well for book-style pages and "line_cluster" works well for pecha and book-style pages.

  **--break_width** 
  
The value that controls how horizontally-connected stacks will be segmented. A high value (e.g. 4.0) will perform almost no segmentation. A low value (e.g. .5) may severely over-segment characters. Typical good values are 2, 2.5, 3, or 3.5. (Note that some text evade accurate segmentation, in which case there's no "goldlilocks" break-width that will manage to accurately segment wide connected stacks while also avoiding over-segmentation).

  **--segmenter**
  
The type of segmentation strategy to use. "stochastic" is default and is almost always the best option.

 **--low_ink**
 
Default: False. Attempt to compensate for poorly inked texts, particularly cases where the glyphs aren't connected together as part of a single stroke. 

  **--line_cluster_pos**
  
Use with the "line_cluster" line break method. Choices are "top" or "center." Clustering to the center of a line is good for cases where vowels may erroneously get clustered to the above line (on account of being closer to it distance-wise).

  **--postprocess**
  
Run a post-processing step. This is usually an attempt to insert missing tsek characters into the final results of an OCR run. This is highly experimental and can severely mangle otherwise accurate OCR.

 **--detect_o**
 
Detect na-ro vowels prior to segmentation and remove them temporarily. This is useful in cases where the na-ro vowels are long and adversely inflating char-width measurements (measurements that are used to determine how and when to segment horizontally touching stacks).

  **--clear_hr**
  
Identify and remove horizontal rule or line on the top of a page. Set to True when you want to get ride of title or chapter  lines that appear on each page. Note: use with caution if page_type is "pecha"

 **--line_cut_inflation**
 
Rarely used. The number of iterations when dilating text in line cut. Increase this value when need to blob things together. Default is 4 iterations.

### Optional: generate data yourself from fonts
This is strictly optional. The provided datasets already include these datapoints. These commands are also run automatically if you run ubuntu_install.sh.

Install fonts:

```bash
$ cd namsel-ocr
$ sudo apt-get install python-cairo
$ mkdir -p ~/.fonts
$ cp data_generation/fonts/*ttf ~/.fonts/
$ fc-cache -f -v
```

Generate the font-derived datasets:
```bash
$ cd data_generation
$ python font_draw.py
```

### About the name
Namsel is an English rendering of the Tibetan རྣམ་གསལ (transliterated: rnam gsal). Literally translated, རྣམ can be thought of as "thorough" or "in detail" and གསལ can be thought of as "clear", "illuminated." Put together, Namsel can be taken to mean "making clear the details" or "thoroughly illuminating," which are both appropriate descriptions of what the Namsel project is trying to do. Practically speaking, the name was also chosen because it is reasonably easy for non-Tibetan speakers to read and pronounce correctly.
