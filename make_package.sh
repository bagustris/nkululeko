#!/bin/bash

git add nkululeko/*py
for value in augmenting autopredict data feat_extract losses models reporting segmenting utils
do
   git add nkululeko/$value/*.py
done
for data in aesdd androids androids_orig androids_test ased asvp-esd baved cafe clac cmu-mosei crema-d demos ekorpus emns emodb emofilm EmoFilm emorynlp emov-db emovo emozionalmente enterface esd gerparas iemocap jl jtes laughter-types meld mesd mess mlendsnd msp-improv msp-podcast oreau2 portuguese ravdess savee shemo subesco syntact tess thorsten-emotional urdu vivae
do
   git add data/$data/*.py
   git add data/$data/*.md
done
git add data/README.md
git add CHANGELOG.md ini_file.md
source nkululeko/constants.py
git commit -m $VERSION
git tag $VERSION
git push 
git push --tags