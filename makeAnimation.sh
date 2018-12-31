#!/bin/bash

#Be sure the paths are loaded
echo 'Creating video from repository:'
echo 'to wmv (ex: output.wmv) output file: ' $1

#gource .git/ –max-files 99999 –disable-progress –stop-at-end -s 0.25 –user-scale 2 –highlight-all-users –output-ppm-stream – | ffmpeg -y -b 3000K -r 60 -f image2pipe -vcodec ppm -i –-vcodec libx264 $2

/cygdrive/c/Program\ Files//Gource/gource.exe -800x600 --stop-position 1.0 --highlight-all-users --hide-filenames --seconds-per-day 0.25 --output-framerate 30 --output-ppm-stream output.ppm

ffmpeg -y -r 30 -f image2pipe -vcodec ppm -i output.ppm  -vcodec wmv1 -r 30 -qscale 0 $1

rm output.ppm
