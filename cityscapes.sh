#!/bin/bash

display_help() {
    echo "This script downloads and sets up the Cityscapes and Cityscapes Foggy datasets."
    echo "It creates a 'datasets' folder in the specified output folder, with 'cityscapes' and 'cityscapes_foggy' subfolders."
    echo "The datasets are restructured as needed for the ML script."
    echo "You should be ready to input your login and password from https://cityscapes-dataset.com."
    echo
    echo "Prerequisites:"
    echo "  - Python and pip should be installed."
    echo "  - The 'cityscapesscripts' package should be installed using 'pip install cityscapesscripts'."
    echo
    echo "Usage: $0 -o <output_folder>"
    echo "Options:"
    echo "  -o, --output <output_folder>   Specify the output folder where the datasets will be created"
    echo "  -h, --help                     Display this help"
}

while getopts ":o:h" opt; do
    case $opt in
        o) output_path=$OPTARG ;;
        h) display_help
           exit 0 ;;
        \?) echo "Invalid option: -$OPTARG" >&2
            display_help
            exit 1 ;;
    esac
done

# Check if the output path argument is provided
if [[ -z $output_path ]]; then
    echo "Output folder not specified."
    display_help
    exit 1
fi

# Set the dataset directory relative to the output path
dataset_dir="$output_path/datasets"

# Store the original working directory
original_dir=$(pwd)

# Create the datasets directory if it doesn't exist
mkdir -p "$dataset_dir"
cd "$dataset_dir"

# Download the dataset using the provided credentials
csDownload

# Download and extract gtFine_trainvaltest.zip if cityscapes/gtFine doesn't exist
if [ ! -d "cityscapes/gtFine" ]; then
    csDownload gtFine_trainvaltest.zip
    mkdir cityscapes
    echo "Unzipping gtFine_trainvaltest.zip... It can take a few minutes..."
    unzip -q gtFine_trainvaltest.zip -d cityscapes/gtFine_trainvaltest
    rm gtFine_trainvaltest.zip
    mv cityscapes/gtFine_trainvaltest/gtFine cityscapes/gtFine
fi

# Download and extract leftImg8bit_trainvaltest.zip if cityscapes/leftImg8bit doesn't exist
if [ ! -d "cityscapes/leftImg8bit" ]; then
    csDownload leftImg8bit_trainvaltest.zip
    echo "Unzipping leftImg8bit_trainvaltest.zip... It can take a few minutes..."
    unzip -q leftImg8bit_trainvaltest.zip -d cityscapes/leftImg8bit_trainvaltest
    cp -r cityscapes/leftImg8bit_trainvaltest/leftImg8bit cityscapes/leftImg8bit
    rm -r cityscapes/leftImg8bit_trainvaltest
    rm leftImg8bit_trainvaltest.zip
fi

if [ ! -d "cityscapes_foggy" ]; then
    mkdir cityscapes_foggy
fi

# Download and extract leftImg8bit_trainvaltest_foggy.zip if cityscapes_foggy/leftImg8bit doesn't exist
if [ ! -d "cityscapes_foggy/leftImg8bit" ]; then
    csDownload leftImg8bit_trainvaltest_foggy.zip
    echo "Unzipping leftImg8bit_trainvaltest_foggy.zip... It can take a few minutes..."
    unzip -q leftImg8bit_trainvaltest_foggy.zip -d cityscapes_foggy/leftImg8bit_trainvaltest_foggy
    cp -r cityscapes_foggy/leftImg8bit_trainvaltest_foggy/leftImg8bit_foggy cityscapes_foggy/leftImg8bit
    rm -r cityscapes_foggy/leftImg8bit_trainvaltest_foggy/
    rm leftImg8bit_trainvaltest_foggy.zip
fi

if [ ! -d "cityscapes_foggy/gtFine" ]; then
    cp -r cityscapes/gtFine cityscapes_foggy/gtFine
fi

if [ ! -d "cityscapes/annotations" ]; then
    cd "$original_dir"
    python cityscapes-to-coco-conversion/main.py --dataset cityscapes --datadir "$dataset_dir/cityscapes" --outdir "$dataset_dir/cityscapes/annotations"
    cd "$dataset_dir"
fi

if [ ! -d "cityscapes_foggy/annotations" ]; then
    cd "$original_dir"
    python cityscapes-to-coco-conversion/main.py --dataset cityscapes --datadir "$dataset_dir/cityscapes_foggy" --outdir "$dataset_dir/cityscapes_foggy/annotations" --file_name_suffix _foggy_beta_0.02
    cd "$dataset_dir"
fi