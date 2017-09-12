# OFTrack

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/Haptein/OFTrack/blob/master/LICENSE)

Animal Tracking with OpenCV

<b>Requirements:</b>
  - Python == 2.7
  - OpenCV => 3.0 with support for FFMPEG
  - Numpy
  - Tkinter

<b>Usage:</b>

You can run OFTrack with:
```
python OFTrack.py
```
and its GUI config utility with:
```
python config.py
```

<b> For best results configure your settings before tracking stuff.</b>
 
Theres also several cli options.
```
OFTrack.py [-h] [-o DES] [-m IMG] [-a] [-ov] [-nv] [-nc] [-nd] [-l SRC]                                                                         
                  [-ht] [-hd]        
                  [input [input ...]]                                      
     
positional arguments:                
  input                 Input files. 

optional arguments:                  
  -h, --help            show this help message and exit
  -o DES, --output DES  Specify output destination
  -m IMG, --mask IMG    Specify a mask image
  -a, --abs             Enable automatic background subtraction based tracking
  -ov, --overlay        Overlay video with trace instead of side by side view
  -nv, --no-video       Disable video file output
  -nc, --no-csv         Disable csv file output
  -nd, --no-display     Disable video display
  -ht, --hide-time      Hide time
  -hd, --hide-distance  Hide distance estimation
  -l SRC, --live SRC    Specify a camera for live video feed, it can be an
                        integer or an ip address
```


  This was originally a fork from [colinlaney/animal-tracking](https://github.com/colinlaney/animal-tracking).

