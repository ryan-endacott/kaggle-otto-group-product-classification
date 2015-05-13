#!/bin/bash

# Script to copy file contents to clipboard.
# Just pipe the script to xclip.
# For submitting my code in the project report.

full_contents= ""
for filename in *.py; do
  echo "# $filename"
  file_contents=$(<$filename)
  echo $file_contents
  echo ""
  echo ""
done

