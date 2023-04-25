#!/bin/sh
DIR="${1}"
if [ -d "$DIR" ]; then
  ### Take action if $DIR exists ###
  echo "${DIR} found"
else
  ###  Control will jump here if $DIR does NOT exists ###
  echo "${DIR} not found, creating ${DIR}"
  mkdir $DIR
fi


