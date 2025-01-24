
# Causality-based Therapy plan Recommendation (CTR)
A Python and R implementation of Causality-based Therapy plan Recommendation (CTR) method in paper "Personalised Neoadjuvant Therapy Recommendations for Breast Cancer: A Causal Inference Approach using Multi-omics Data".


# Infrastructure used to run experiments:
* OS: Kali GNU/Linux, version 2024.2. (Linux 5.15.167.4-microsoft-standard-WSL2 x86_64)
* CPU: Intel(R) Core(TM) i7-1255U @ 1.70GHz).
* RAM: 16 GB.

# Dataset
We utilized observational data from the TransNEO
neoadjuvant breast cancer clinical trial with 147 samples (Sammut et al., 2022) and
the ARTemis clinical trial with 75 samples (Earl et al., 2015). Together, these two datasets provided
a merged cohort of 222 samples for our study


# Installation
**Installation requirements for CTR:**

* Python >= 3.12
* numpy 2.2.2
* pandas 2.2.3
* scikit-learn 1.6.1
* scipy 1.15.1
* matplotlib 3.10.0
* seaborn 0.13.2
* R-base = 4.3.3
  * causalTree
  * rpart
 
**Detailed Guidelines for Environment Setup using Conda on Linux**

***1. Create a Conda Environment***
Create a new Conda environment named ctr_env with specific versions of Python and R: (bash)
     
    conda create -n ctr_env python=3.12 r-base=4.3 -c conda-forge -y
    conda activate ctr_env
***2. Install Python Packages:***
     Install essential Python packages using pip: (bash)
     
    pip install numpy pandas scikit-learn scipy matplotlib seaborn
***3. Install R packages:***

   Install the r-devtools package via Conda: (bash)
     
    conda install r-devtools

   Launch R within the environment: (bash)

    R

   In the R session, install the required R packages: (R script)

    # Install the 'causalTree' package from GitHub
    devtools::install_github("susanathey/causalTree")
    
    # Install additional R packages from CRAN
    install.packages(c("dplyr", "graph", "rpart"), repos = "https://cloud.r-project.org", dependencies = TRUE)
    
    # Verify installed R packages
    rownames(installed.packages())

   Exit the R session: (R script)
   
    q()
    



# Reproducing the Paper Results

**1. Run the Causality-based Therapy plan Recommendation (CTR)**

    Rscript do_NEOdata.R
**2. Run 6 baselines with the NEO dataset**

    python do_baselines.py
**3. Generate results in the paper**

    python generate_results.py
