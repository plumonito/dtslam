These are calibration and flag files to use the City of Sights datasets.

Download the images and videos from the CoS website:
http://studierstube.icg.tugraz.at/handheld_ar/cityofsights.php

For example, to use the robotic arm sequence:
1. Download the Bird's view images from the following link into <BIRDS_FOLDER>.
	http://studierstube.icg.tugraz.at/handheld_ar/cityofsights_data/GTdata/Robo/CS_BirdsView_L0/CS_BirdsView_L0.zip
2. Run the dtslam_desktop -flagfile=<REPO_FOLDER>/data/CityOfSights/calibRobo.txt -DriverDataPath=<BIRDS_FOLDER> -DriverSequenceFormat=Frame_%.5d.jpg
