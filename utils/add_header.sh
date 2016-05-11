#!/usr/bin/env bash
#
# Add header information for vector model in text format
if [ ! $# -eq 2 ];then
    echo "usage: $0 vector_model_path(in text format) dimension"
    echo
    exit 1
fi

vector_model_path=$1
dimension=$2

# Get word numbers of vector model in text format
word_nums=$(sed -n '$=' ${vector_model_path})

# Write to file in place
sed -i "1i ${word_nums} ${dimension}" ${vector_model_path}
