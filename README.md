# Hands-On-GPU-Programming-with-Python-and-CUDA
Hands-On GPU Programming with Python and CUDA, published by Packt

Important note:
> This book is under active development and is expected to be published in December, 2018

## A Note to Readers
Each chapter's example code is available in the corresponding directory.  (The examples for chapter 1 are under "1", examples for chapter 2 are in "2", and so on.)

## Hardware and Software Requirements
In this text, we will assume that you have a Pascal (2016-era) Nvidia GPU, or later; you should have at least have an entry-level Pascal GTX 1050, or the equivalent.  Generally speaking, you will be able to work through this book with almost any entry-level gaming PC released in 2016 or later that has an Nvidia GPU.  (While the examples haven't been tested on older GPUs, an older 2014-era entry level Maxwell architecture GPU such as a GTX 750 should be sufficient for purposes of this text.)

Both the Windows 10 and Linux Operating Systems provide reasonable environments for CUDA programming, and are both supported by this text.  (Windows 10 is an entirely suitable choice for laptop users for beginning GPU programming, due to the relative ease of the installation of the Nvidia drivers and CUDA environment compared to Linux.)  We would urge Linux users to consider using a Long Term Support (LTS) Ubuntu Linux distribution or any LTS Ubuntu derivatives (e.g., Lubuntu, Xubuntu, Ubuntu Mate, Linux Mint), due to the strong support these distributions receive from Nvidia for both drivers and the CUDA Toolkit.

While we will go over particular development environments in the following chapter, we suggest the Anaconda Python 2.7 distribution (available at https://www.anaconda.com/download/).  In particular, I will be using this Python distribution throughout the text for the examples I will be giving.  Anaconda Python is available for both Windows and Linux, it is very easy to install, and it contains a large number of optimized mathematical, machine learning, and data science related libraries that will come in useful, as well as some very nice pre-packaged Integrated Development Environments (IDE) such as Spyder and Jupyter.  Moreover, Anaconda can be installed on a user-by-user basis, and provides an isolated environment from the system installation of Python.  For these reasons, I suggest you start with the Anaconda Python 2.7 distribution.

For Windows, we would suggest Visual Studio Community Edition 2015 with C++ support, due to its tight integration with both Anaconda and CUDA  (available at https://visualstudio.microsoft.com/vs/older-downloads/).  for Linux, a standard gcc installation along with the Eclipse IDE for C++ from your distribution’s repository should be enough (installation From the Ubuntu bash command line can be performed with: "sudo apt-get update && sudo apt-get install build-essentials && sudo apt-get install eclipse-cdt" ).

It should be noted that any version of the CUDA Toolkit from 9.0 onwards will work for all of the examples in this book, although the examples have only been fully tested with only CUDA 9.2 on Windows and Linux.  (The CUDA Toolkit can be downloaded here: https://developer.nvidia.com/cuda-toolkit.)
