# AI4CHEM: Artificial Intelligence for Experimental Chemistry

This repository provides the complete set of open educational materials for CHEM 5080 — AI for Experimental Chemistry, developed at Washington University in St. Louis. The course introduces machine learning and artificial intelligence to synthetic and experimental chemistry students with no prior programming background.

The course emphasizes chemical context, accessible Python workflows, and realistic experimental datasets. All tutorials run directly in Google Colab with no installation required.

---

## Course Website

The full Jupyter Book, including lecture notes and all Colab notebooks, is available at:

https://zhenglab.wustl.edu/chem5080/

---

## Repository Structure

```
ai4chem/
│
├── book/                     # Jupyter Book source files
├── notebooks/                # Colab-ready tutorial notebooks
├── data/                     # Chemical datasets used in examples
├── figures/                  # Images and figures for the course
├── website_config/           # Configuration for Jupyter Book
└── README.md                 # This file
```

---

## Course Themes

AI4CHEM is organized around five instructional themes:

1. Chemical Data Foundations  
   - Python basics, RDKit, chemical identifiers, data handling  

2. Machine Learning Models  
   - Regression, classification, neural networks, graph models  

3. Patterns in Chemical Space  
   - Dimensionality reduction, clustering, generative models  

4. Vision and Language Intelligence  
   - CNNs for chemical imaging, transformer-based text extraction  

5. AI-Driven Experimentation  
   - Bayesian optimization, active learning, concepts for self-driving labs  

---

## How to Use These Materials

### 1. Run notebooks in Google Colab  
Most notebooks include an "Open in Colab" button. Click to run immediately in the browser.

### 2. Build the Jupyter Book locally  
Install dependencies and build:

```bash
pip install -r requirements.txt
jupyter-book build book/
```

### 3. Adapt for your own teaching  
All materials are open source and can be reused or modified. Suggested uses include:
- Integrating selected modules into existing chemistry courses  
- Running a full semester course in AI for chemistry  
- Using individual notebooks in workshops or research group training  

---

## Citation

If you use these materials, please cite the AI4CHEM course or the associated publication when available. Citation of this repository is also appreciated.

---

## License

All content is distributed under an open-source license (see LICENSE file). Modification and redistribution with attribution are permitted.

---

## Contact

Zhiling "Zach" Zheng  
Department of Chemistry  
Washington University in St. Louis  
