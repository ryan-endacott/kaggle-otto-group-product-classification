#!/bin/bash

# Script to copy file contents to clipboard.
# Just pipe the script to xclip.
# For submitting my code in the project report.

for filename in *.py; do
  echo "------------------------------------------"
  echo "# $filename"
  cat $filename
  echo -e $file_contents
  echo ""
  echo ""
done

