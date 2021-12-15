#!/usr/bin/env bash

while read line; do
  file=`echo "$line" | cut -d " " -f 2`
  if [[ "$3" == "sing" ]]; then
      file=`echo "${file/read/sing}"`
  fi
  if [[ ! -d "output/$3" ]]; then
    mkdir -p "output/$3"
  fi
  cp -r "$2"/"$file" output/"$3"/
done < "$1"