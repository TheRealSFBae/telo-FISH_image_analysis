

#@ File (label = "Input directory", style = "directory") input
#@ File (label = "Output directory", style = "directory") output
#@ String (label = "File suffix", value = ".jpg") suffix

processFolder(input);

// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(input) {
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		if(File.isDirectory(input + File.separator + list[i]))
			segmenter(input + File.separator + list[i]);
		if(endsWith(list[i], suffix))
			segmenter(input, output, list[i]);
	}
}

// function to select nuclei of interest
function segmenter(input, output, file) {
	open(input + "/" +  file);
	selectWindow(file);
	imageTitle = getTitleStripExtension();
	run("Duplicate...", " ");
	run("Split Channels");
	run("Gaussian Blur...", "sigma=2");
	run("Make Binary");
	run("Watershed");

	roiManager("reset");
	print("\\Clear");
	print("Nucleus,","size,","telo_mean_grey_val");

	// first segmentation step
	// size >30 tries to prevent non-nuclear object selection (filter large selections in postprocessing)
	run("Analyze Particles...", "size=30-Infinity show=Outlines add");

	// switch to green channel
	// -1 is hard-coded into names due to "Duplicate..." - ensure all windows close when done with loop
	selectWindow(imageTitle + "-1.jpg (green)"); 
	
	// loop to recursively select regions of interest
	// enlarge=n enlarges ROIs by n pixels
	// if(mean>n) selects for ROIs with green fluor intensity > n
	for (x=1;x<roiManager("count");x+=1){
		roiManager("select",x);
		run("Enlarge...", "enlarge=10");
		getStatistics(area,mean,min,max,std,histogram);
		if(mean>85){
			selectWindow(imageTitle + "-1.jpg (red)");
			roiManager("select",x);
			getStatistics(area,mean,min,max,std,histogram);
			print(x,",",area,",",mean);
			selectWindow(file);
			roiManager("select", x);
			run("Add Selection...");
			run("Draw", "slice");
		}
	}

	selectWindow(file);
	saveAs("PNG", output + "/" + imageTitle + "_PROCESSED.png");
	close();
	selectWindow(imageTitle + "-1.jpg (green)");
	close();
	selectWindow(imageTitle + "-1.jpg (blue)");
	close();
	selectWindow(imageTitle + "-1.jpg (red)");
	close();
	selectWindow("Drawing of "+ imageTitle + "-1.jpg (blue)");
	close();
	selectWindow("Log");
	saveAs("Text", output + "/" + imageTitle + "_log.csv");

}

// function to strip file extensions from image titles
function getTitleStripExtension() { 
  t = getTitle(); 
  t = replace(t, ".jpg", "");
  t = replace(t, ".JPG", "");
  t = replace(t, ".jpeg", "");
  t = replace(t, ".JPEG", "");
  t = replace(t, ".tif", "");
  t = replace(t, ".TIF", "");           
  t = replace(t, ".tiff", "");
  t = replace(t, ".TIFF", "");       
  t = replace(t, ".lif", "");
  t = replace(t, ".LIF", "");       
  t = replace(t, ".lsm", "");
  t = replace(t, ".LSM", "");     
  t = replace(t, ".czi", "");
  t = replace(t, ".CZI", "");       
  t = replace(t, ".nd2", "");
  t = replace(t, ".ND2", "");     
  return t; 
} 