# ADNI-Project

## A fully automated pipeline for PET image analysis aimed at time efficiency and accuracy

- [ADNI-Project](#adni-project)
  - [A fully automated pipeline for PET image analysis aimed at time efficiency and accuracy](#a-fully-automated-pipeline-for-pet-image-analysis-aimed-at-time-efficiency-and-accuracy)
    - [Installation guide](#installation-guide)
    - [Image preprocessing](#image-preprocessing)
    - [Network construction](#network-construction)
    - [Network analysis](#network-analysis)
    - [Notes](#notes)
      - [Downloading the dataset](#downloading-the-dataset)
      - [Parallelization and hardware](#parallelization-and-hardware)
      - [macOS users](#macos-users)

### Installation guide

1. Download and install [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) and [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation)
2. Clone this repository: `git clone https://github.com/raghavprasad13/ADNI-Project.git`
3. In your terminal, navigate to the location you cloned this repository and type `pip3 install -r requirements.txt` to install the Python dependencies
4. Run the pipeline: `./full_pipeline.sh path/to/dataset`

### Image preprocessing

The first step in image preprocessing is extraction of the scans from .zip files and _concatenation of frames_. A PET image is a four-dimensional object, comprising **voxels** (which account for 3 out of the 4 dimensions) over multiple **frames** (which is the 4th dimension, the time dimension). All the PET images in the ADNI database are split up into their constituent frames. This is where it differs from the OASIS3 database. Thus, they need to be concatenated before further processing. This is done using the `fslmerge` tool, part of the [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/) software suite.  
  
Next, similar to the OASIS3 pipeline, we do the following:

- **Registration to MNI space**
- **Smoothing**

Image preprocessing was done using [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) in conjunction with FSL  
The preprocessing of PET images is a computationally intensive task and would have been very hard to accomplish within a reasonable amount of time. To counter this issue, the code execution has been parallelized as far as possible. In addition to the native parallelization code we used [GNU Parallel](https://www.gnu.org/software/parallel/) to achieve maximum parallelization in the image preprocessing stage of the pipeline.

### Network construction

Constructing networks from the preprocessed images required the generation of adjacency matrices. This was accomplished by computing the partial correlation values of intensities in the PET images. These partial correlation values serve as edge weights and constitute the values in the adjacency matrices. This is achieved by `pet_to_network.r` in conjunction with `pet_helper_funcs.py`  
The calculation of partial correlations is a computationally intensive task, mainly due to the pre-calculation of residuals before computing cross-correlation. This calculation was made many times faster with the use of the [ppcor](https://cran.r-project.org/web/packages/ppcor/ppcor.pdf) package for R

### Network analysis

Network analysis is done using the [NetworkX](https://networkx.github.io/) Python library, which enabled the construction of complex brain networks from the adjacency matrices generated by the previous pipeline stage, at a default threshold value of **0.4**  
Next, the global mean percolation centrality value is calculated for each network using the inbuilt function in NetworkX. This has a complexity of $O(n^{3})$ and hence had to be parallelized in order to achieve good throughput throughput.  
`pet_graph_analysis.py` executes this stage of the pipeline

### Notes

#### Downloading the dataset

This pipeline does not contain a stage to download the dataset. This is because, as far as I knpw, ADNI does not have an API to enable downloading datasets. Instead, it provides dataset search and download widgets on its web platform.  
Downloading datasets from the ADNI website is a slightly complicated affair. Using the **Advanced Search** widget is a great option if you know exactly what you're looking for. Working from India, I have found that the download mirrors are quite slow and network issues could force a download _restart_ (not a _resumption_).  
In order to counter network issues, ADNI recommends the use of _Download Managers_, and even lists 3 of them as being tried and tested. I initially used the **Free Download Manager** to download my datasets and I was pleasantly surprised by how much faster my downloads were. However, when I tried to unzip the downloaded file, I found that it was corrupted. And I repeated the download a few times to be sure and even added error handling code to the pipeline for good measure. Thus, I concluded that the fault lay in the download manager. Probably hte reason why it was showing me such high download speeds was because it just wasn't downloading the file completely.
  
#### Parallelization and hardware

A serial processing of the pipeline would be a poor implementation. Thus, the pipeline has been written, keeping in mind the bottlenecks, to achieve a high degree of parallelization. The speed of the pipeline is going to be highly dependent on the hardware it is running on; more precisely, on the speed and number of parallel processing units.  
There is still some room to optimize this further. Moving forward, the code can be modified to make use of GPUs, which have more parallel computing units than a CPU.

#### macOS users

For people trying to run this pipeline on a Mac, you will likely have to turn of **System Integrity Protection (SIP)**. This is because the SIP does not allow the execution of the FreeSurfer and FSL binaries from script.  
You can check the status of SIP on your Mac in the Terminal as follows:

``` bash
$ csrutil status
System Integrity Protection status: enabled.
```

You can disable SIP in the following manner:

1. Boot into Recovery Mode
2. Choose `Utilities` > `Terminal`
3. Type `csrutil disable` and hit return
4. Enter `reboot` and hit return
