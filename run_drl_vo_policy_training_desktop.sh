#!/bin/sh
DIR="${1}"
if [ -d "$DIR" ]; then
  ### Take action if $DIR exists ###
  echo "${DIR} found"
else
  ###  Control will jump here if $DIR does NOT exists ###
  echo "${DIR} not found, creating ${DIR}"
  mkdir ${DIR}
fi
# register gym:
cd ./drl_vo/src/turtlebot_gym/
pip install -e .
cd ../../
# roslaunch training:
roslaunch drl_vo_nav drl_vo_nav_train.launch log_dir:="${DIR}"
