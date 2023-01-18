


# Hands-On GPU Programming with Python and CUDA

<a href="https://www.packtpub.com/application-development/hands-gpu-programming-python-and-cuda?utm_source=github&utm_medium=repository&utm_campaign=9781788993913 "><img src="https://d255esdrn735hr.cloudfront.net/sites/default/files/imagecache/ppv4_main_book_cover/B10306.png" alt="Hands-On GPU Programming with Python and CUDA" height="256px" align="right"></a>

This is the code repository for [Hands-On GPU Programming with Python and CUDA](https://www.packtpub.com/application-development/hands-gpu-programming-python-and-cuda?utm_source=github&utm_medium=repository&utm_campaign=9781788993913 ), published by Packt.

**Explore high-performance parallel computing with CUDA**

## What is this book about?
Hands-On GPU Programming with Python and CUDA hits the ground running: you’ll start by learning how to apply Amdahl’s Law, use a code profiler to identify bottlenecks in your Python code, and set up an appropriate GPU programming environment. You’ll then see how to “query” the GPU’s features and copy arrays of data to and from the GPU’s own memory.

This book covers the following exciting features:
* Launch GPU code directly from Python 
* Write effective and efficient GPU kernels and device functions 
* Use libraries such as cuFFT, cuBLAS, and cuSolver 
* Debug and profile your code with Nsight and Visual Profiler 
* Apply GPU programming to datascience problems 
* Build a GPU-based deep neuralnetwork from scratch 
* Explore advanced GPU hardware features, such as warp shuffling 

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/1788993918) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>

## Instructions and Navigations
All of the code is organized into folders. For example, Chapter02.

The code will look like the following:
```
cublas.cublasDestroy(handle)
print 'cuBLAS returned the correct value: %s' % np.allclose(np.dot(A,x), y_gpu.get())
```

**Following is what you need for this book:**
Hands-On GPU Programming with Python and CUDA is for developers and data scientists who want to learn the basics of effective GPU programming to improve performance using Python code. You should have an understanding of first-year college or university-level engineering mathematics and physics, and have some experience with Python as well as in any C-based programming language such as C, C++, Go, or Java.

With the following software and hardware list you can run all code files present in the book (Chapter 1-12).
### Software and Hardware List
| Chapter  | Software required                    | OS required                         |
| -------- | ------------------------------------ | ----------------------------------- |
| 1-11     | Anaconda 5 (Python 2.7 version)      | Windows, Linux                      |
| 2-11     | CUDA 9.2, CUDA 10.x                  | Windows, Linux                      |
| 2-11     | PyCUDA (latest)                      | Windows, Linux                      |
| 7        | Scikit-CUDA (latest)                 | Windows, Linux                      |
| 2-11     | Visual Studio Community 2015         | Windows                             |
| 2-11     | GCC, GDB, Eclipse                    | Linux                               |


| Chapter  | Hardware required                    | OS required                         |
| -------- | ------------------------------------ | ----------------------------------- |
| 1-11     | 64-bit Intel/AMD PC                  | Windows, Linux                      |
| 1-11     | 4 Gigabytes RAM                      | Windows, Linux                      |
| 2-11     | NVIDIA GPU (GTX 1050 or better)      | Windows, Linux                      |




We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://www.packtpub.com/sites/default/files/downloads/9781788993913_ColorImages.pdf).

### Related products
* Hands-On GPU-Accelerated Computer Vision with OpenCV and CUDA [[Packt]](https://www.packtpub.com/application-development/hands-gpu-accelerated-computer-vision-opencv-and-cuda?utm_source=github&utm_medium=repository&utm_campaign=9781789348293 ) [[Amazon]](https://www.amazon.com/dp/1789348293)

* OpenCV 3 Computer Vision with Python Cookbook [[Packt]](https://www.packtpub.com/application-development/opencv-3-computer-vision-python-cookbook?utm_source=github&utm_medium=repository&utm_campaign=9781788474443 ) [[Amazon]](https://www.amazon.com/dp/1788474449)

## Get to Know the Author
**Dr Brian Tuomanen**
has been working with CUDA and general-purpose GPU programming since 2014. He received his bachelor of science in electrical engineering from the University of Washington in Seattle, and briefly worked as a software engineer before switching to mathematics for graduate school. He completed his PhD in mathematics at the University of Missouri in Columbia, where he first encountered GPU programming as a means for studying scientific problems. Dr. Tuomanen has spoken at the US Army Research Lab about general-purpose GPU programming and has recently led GPU integration and development at a Maryland-based start-up company. He currently works as a machine learning specialist (Azure CSI) for Microsoft in the Seattle area.


### Suggestions and Feedback
[Click here](https://docs.google.com/forms/d/e/1FAIpQLSdy7dATC6QmEL81FIUuymZ0Wy9vH1jHkvpY57OiMeKGqib_Ow/viewform) if you have any feedback or suggestions.



### Download a free PDF

 <i>If you have already purchased a print or Kindle version of this book, you can get a DRM-free PDF version at no cost.<br>Simply click on the link to claim your free PDF.</i>
<p align="center"> <a href="https://packt.link/free-ebook/9781788993913">https://packt.link/free-ebook/9781788993913 </a> </p>