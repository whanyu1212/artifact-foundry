<!-- omit in toc -->
# Curated list of learning resources for DS & MLE
(Working in progress)

a curated list of resources for aspiring data scientists and machine learning engineers

<!-- omit in toc -->
## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Python Version Management](#python-version-management)
  - [Dependency Management](#dependency-management)
  - [Project Templates](#project-templates)
- [Data Structures and Algorithms Resources](#data-structures-and-algorithms-resources)
  - [Books and Courses](#books-and-courses)
  - [Practice Platforms](#practice-platforms)
  - [Applications in DSML](#applications-in-dsml)
- [Software Engineering Resources](#software-engineering-resources)
- [Data Engineering Resources](#data-engineering-resources)
  - [Repositories](#repositories)
  - [Books](#books)
- [Data Science Resources](#data-science-resources)
  - [Books](#books-1)
- [Interview Prep Resources](#interview-prep-resources)
  - [Useful Github Repositories](#useful-github-repositories)
  - [Simplified 90-Day Study Plan (DS/MLE Focus)](#simplified-90-day-study-plan-dsmle-focus)

### Introduction 
**<u>What It Takes to Become a Data Scientist or Machine Learning Engineer</u>**

Becoming a Data Scientist or Machine Learning Engineer requires a <u>blend of technical skills, analytical thinking, and domain expertise</u>. While the two roles overlap in many ways, they serve distinct purposes within the data ecosystem.

Data Scientists focus on extracting actionable insights from data to drive business decisions. They apply statistical analysis, data visualization, and predictive modeling to uncover patterns in complex datasets. Proficiency in Python and other programming tools is essential, complemented by strong foundations in statistics, mathematics, and domain expertise. Today's Data Scientists increasingly leverage Large Language Models (LLMs) as analytical tools—using them for text analysis, automated reporting, and data exploration through natural language interfaces. Effective communication skills are crucial, as Data Scientists routinely translate technical findings into business recommendations for non-technical stakeholders. This communication ability is frequently assessed during interviews, where candidates may be asked to explain complex concepts in accessible terms.


On the other hand, Machine Learning Engineers specialize in designing and deploying scalable machine learning systems. They build robust pipelines that automate data processing and model training, ensuring models perform efficiently in production environments. This role demands strong software engineering skills, system architecture design, proficiency in frameworks like TensorFlow or PyTorch, and a deep understanding of algorithms and data structures. MLEs make critical architectural decisions that balance performance, scalability, cost, and maintainability across the entire ML system lifecycle. Increasingly, expertise with Large Language Models (LLMs) has become essential – including prompt engineering, fine-tuning techniques, RAG (Retrieval-Augmented Generation) implementation, and optimizing LLM deployment for cost and performance. MLEs must stay current with rapidly evolving LLM capabilities and best practices for their responsible integration into production systems.


A **T-shaped skillset** combining deep expertise in a core area with broad knowledge across related domains provides the optimal foundation for success in both data science and machine learning engineering. 

---

### Getting Started

#### Python Version Management
Pyenv is an useful Python version management tool that allows you to install and switch between multiple Python versions on the same system without conflicts. It is excellent for project specific environments and backward compatibility support

<br>

[https://github.com/pyenv/pyenv](https://github.com/pyenv/pyenv)

#### Dependency Management
Effective dependency management is critical in data science and ML projects to ensure <u>**reproducibility**</u>, prevent conflicts, and maintain consistent environments across development and production.

Both `poetry` and `uv` are good choices that attempt to address pip's limitations around dependency resolution, environment isolation and performance (`uv` is written in rust and is blazing fast)

<br>

[https://python-poetry.org/](https://python-poetry.org/)


[https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)

#### Project Templates
Standardized project templates are essential for maintaining consistency and best practices across projects. Cookiecutter is the leading solution for project templates for Python-based data science and ML workflows. Initial investment in setting up appropriate templates may help to reduce technical debt over time.

Potential drawbacks:
- Potential overhead (structures that may be excessive for simple projects)
- Rigidity
- Maintenance requirements (templates need regular updates as best practices evolve over time)

[https://github.com/cookiecutter/cookiecutter](https://github.com/cookiecutter/cookiecutter)

A template that I personally like to use:
[https://github.com/py-pkgs/py-pkgs-cookiecutter](https://github.com/py-pkgs/py-pkgs-cookiecutter)

---

### Data Structures and Algorithms Resources

#### Books and Courses
- [Hello Algo](https://www.hello-algo.com/)
- [NeetCode Course](https://neetcode.io/courses)
- *Data Structures and Algorithms in Python* by Michael T. Goodrich, Roberto Tamassia, and Michael H. Goldwasser (Wiley, 2013) - [Amazon](https://www.amazon.com/Structures-Algorithms-Python-Michael-Goodrich/dp/1118290275)

#### Practice Platforms
- [https://leetcode.com/](https://leetcode.com/)
- [https://neetcode.io/practice](https://neetcode.io/practice)
- [https://www.hackerrank.com/](https://www.hackerrank.com/)
- [https://www.deep-ml.com/problems](https://www.deep-ml.com/problems) (The ML equivalent of LeetCode)
- [https://www.codewars.com](https://www.codewars.com/dashboard) (More suitable for beginners to pick up new languages)

#### Applications in DSML
- **Graph Structures:** LangGraph leverages Directed Acyclic Graphs (DAGs) to orchestrate complex agent workflows and reasoning paths
- **Network Algorithms:** Recommender systems employ graph-based collaborative filtering to model user-item interactions and social connections
- **Optimization Methods:** Feature selection often relies on greedy algorithms to efficiently identify the most predictive variables
- **Dynamic Programming:** Sequence analysis techniques like Dynamic Time Warping use DP to find optimal alignments between temporal sequences
- **Tree Data Structures:** Decision trees and ensemble methods (Random Forests, XGBoost) recursively partition data using binary tree structures
- **Hash Tables:** Vectorization and embedding lookups use efficient hashing for rapid data retrieval in high-dimensional spaces
- **Queue Processing:** Distributed ML systems implement priority queues for task scheduling and workload management

Remember: Those LeetCode grinding sessions weren't just for passing interviews—they're the algorithms secretly powering your ML models too! 😉

---

### Software Engineering Resources
- *Clean Code: A Handbook of Agile Software Craftsmanship* by Robert C. Martin (Prentice Hall, 2008) - [Amazon](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882)
- *Software Engineering for Data Scientists* by Jacqueline Nolis & Emily Robinson (O'Reilly, 2022) - [Amazon](https://www.amazon.com/Software-Engineering-Data-Scientists-applications/dp/1492080799)
- *Fluent Python* by Luciano Ramalho (O'Reilly, 2022) - [Amazon](https://www.amazon.com/Fluent-Python-Concise-Effective-Programming/dp/1492056359)
- *Robust Python* by Patrick Viafore (O'Reilly, 2021) - [Amazon](https://www.amazon.com/Robust-Python-Patrick-Viafore/dp/1098100662)
- *Clean Code in Python* by Mariano Anaya (Packt, 2021) - [Amazon](https://www.amazon.com/Clean-Code-Python-maintainable-efficient/dp/1800560214)
- *Mastering Python Design Patterns* by Kamon Ayeva & Sakis Kasampalis (Packt, 2018) - [Amazon](https://www.amazon.com/Mastering-Python-Design-Patterns-problems/dp/1788837484)
- *Think Python* by Allen B. Downey (O'Reilly, 2015) - [Amazon](https://www.amazon.com/Think-Python-Allen-B-Downey/dp/1491939362)
- *The Pragmatic Programmer* by David Thomas & Andrew Hunt (Addison-Wesley, 2019) - [Amazon](https://www.amazon.com/Pragmatic-Programmer-journey-mastery-Anniversary/dp/0135957052)
- *Refactoring: Improving the Design of Existing Code* by Martin Fowler (Addison-Wesley, 2018) - [Amazon](https://www.amazon.com/Refactoring-Improving-Existing-Addison-Wesley-Signature/dp/0134757599)
- *Crafting Test-Driven Software with Python* by Alessandro Molina (Packt, 2021) - [Amazon](https://www.amazon.com/Crafting-Test-Driven-Software-Python-applications/dp/1838642625)
- [https://pytest-with-eric.com/](https://pytest-with-eric.com/)
- [https://refactoring.guru/](https://refactoring.guru/)

---

### Data Engineering Resources

#### Repositories
- [https://github.com/DataExpert-io/data-engineer-handbook](https://github.com/DataExpert-io/data-engineer-handbook)
- [https://github.com/DataTalksClub/data-engineering-zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp)

#### Books
- *Fundamentals of Data Engineering* by Joe Reis & Matt Housley (O'Reilly, 2022) - [Amazon](https://www.amazon.com/Fundamentals-Data-Engineering-Robust-Systems/dp/1098108302)
- *Data Engineering Design Patterns* by Joe Reis & Matt Housley (O'Reilly, 2023) - [Amazon](https://www.amazon.com/Data-Engineering-Design-Patterns-Grokking/dp/1633439216)
- *Streaming Systems* by Tyler Akidau, Slava Chernyak & Reuven Lax (O'Reilly, 2018) - [Amazon](https://www.amazon.com/Streaming-Systems-Where-Large-Scale-Processing/dp/1491983876)
- *Designing Data-Intensive Applications* by Martin Kleppmann (O'Reilly, 2017) - [Amazon](https://www.amazon.com/Designing-Data-Intensive-Applications-Reliable-Maintainable/dp/1449373321)
- *Financial Data Engineering* by Harvinder Atwal (Packt, 2023) - [Amazon](https://www.amazon.com/Financial-Data-Engineering-manipulate-transformation/dp/1803240997)

**Note:** Focus on mastering fundamental data engineering concepts and skills before specializing in cloud-specific implementations. This foundational knowledge provides the transferable expertise necessary to quickly adapt to any platform-specific requirements in professional settings.

### Data Science Resources

#### Books
- *An Introduction to Statistical Learning (ISLR)* by Gareth James, Daniela Witten, Trevor Hastie, & Robert Tibshirani (Springer, 2021) - [Website](https://www.statlearning.com/)
- *ISLR Labs reproduced in Julia* by JuliaAI (Online Resource) - [Website](https://juliaai.github.io/DataScienceTutorials.jl/info/isl/)
- *The Elements of Statistical Learning* by Trevor Hastie, Robert Tibshirani, & Jerome Friedman (Springer, 2009) - [Amazon](https://www.amazon.com/Elements-Statistical-Learning-Prediction-Statistics/dp/0387848576)
- *Machine Learning with PyTorch and Scikit-Learn* by Sebastian Raschka, Yuxi (Hayden) Liu, & Vahid Mirjalili (Packt, 2022) - [Amazon](https://www.amazon.com/Machine-Learning-PyTorch-Scikit-Learn-learning/dp/1801819319)
- *Causal Inference in Python* by Matheus Facure (O'Reilly, 2023) - [Amazon](https://www.amazon.com/Causal-Inference-Python-Learning-Causality/dp/1098140257)
- *Modern Time Series Forecasting with Python* by Marin Vlastelica & Humphrey Sheil (Packt, 2022) - [Amazon](https://www.amazon.com/Modern-Time-Forecasting-Python-decomposition/dp/1803246723)
- *Transformers for Natural Language Processing* by Denis Rothman (Packt, 2022) - [Amazon](https://www.amazon.com/Transformers-Natural-Language-Processing-Understanding/dp/1803247339)
- *Applied Machine Learning Explainability Techniques* by Alex Thomas (O'Reilly, 2022) - [Amazon](https://www.amazon.com/Applied-Machine-Learning-Explainability-Techniques/dp/1098102401)
- *Interpretable Machine Learning* by Christoph Molnar (Online/Self-published, 2022) - [Website](https://christophm.github.io/interpretable-ml-book/)

### Interview Prep Resources
**Interview Focus:** While familiarity with common tools is helpful, interviews often prioritize assessing your grasp of fundamental concepts, problem-solving abilities, and communication skills. Focus on strengthening your understanding of core statistical principles, algorithms, data structures, and your ability to articulate technical solutions clearly. These transferable skills are more valuable than deep knowledge of any single, rapidly evolving tech stack.

#### Useful Github Repositories
- https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers
- https://github.com/khanhnamle1994/cracking-the-data-science-interview
- https://github.com/jwasham/coding-interview-university (might be a slight overkill for data scientist interview)

#### Simplified 90-Day Study Plan (DS/MLE Focus)

This plan provides a structured approach, adapt it based on your background and target roles.

**Phase 1: Foundations (Days 1-30)**

*   **Week 1-2: Python & Core Libraries:**
    *   Solidify Python fundamentals (data types, control flow, functions, OOP basics).
    *   Master NumPy for numerical operations and Pandas for data manipulation & analysis.
    *   Practice with real-world datasets.
*   **Week 3-4: Data Structures & Algorithms (DS/ML Relevance):**
    *   Review core data structures (Arrays, Hash Tables/Dictionaries, Trees, Graphs).
    *   Focus on algorithms relevant to data processing and ML (Sorting, Searching, basic Graph algorithms).
    *   Practice problems on platforms like LeetCode, focusing on medium difficulty.
*   **Week 4: Statistics & Probability:**
    *   Refresh fundamental concepts: descriptive statistics, probability distributions, hypothesis testing, confidence intervals.
    *   Understand their application in data analysis and model evaluation.

**Phase 2: Core DS/ML (Days 31-60)**

*   **Week 5-6: SQL & Data Wrangling:**
    *   Master SQL for data retrieval and manipulation (Joins, Aggregations, Window Functions).
    *   Practice advanced data cleaning and transformation techniques using Pandas.
*   **Week 7-8: Machine Learning Fundamentals:**
    *   Understand the ML lifecycle (data collection to deployment).
    *   Study supervised vs. unsupervised learning, bias-variance tradeoff, cross-validation.
    *   Learn key evaluation metrics (Accuracy, Precision, Recall, F1-score, AUC, RMSE, MAE).
    *   Explore feature engineering and selection techniques.
*   **Week 9: Common ML Algorithms:**
    *   Deep dive into Linear/Logistic Regression, Decision Trees, Random Forests, SVM, K-Means Clustering.
    *   Understand their assumptions, pros, cons, and implementation (e.g., using Scikit-learn).

**Phase 3: Specialization & Practice (Days 61-90)**

*   **Week 10: Choose Your Path & Deepen Knowledge:**
    *   **Data Scientist:** Focus on Advanced Statistics, A/B Testing, Causal Inference, Data Visualization & Storytelling.
    *   **MLE:** Focus on MLOps (Docker, CI/CD basics, monitoring), ML System Design patterns, introduction to Big Data tools (like Spark basics), Deep Learning fundamentals (if relevant).
*   **Week 11: LLM Applications (Both Roles):**
    *   Understand core LLM concepts (Transformers architecture basics, attention mechanism).
    *   Explore Prompt Engineering techniques for effective interaction.
    *   Learn about Fine-tuning strategies for domain adaptation.
    *   Study Retrieval-Augmented Generation (RAG) patterns.
    *   Consider LLM deployment challenges (cost, latency, ethics).
*   **Week 12: Practice & Projects:**
    *   Intensify practice on coding platforms (SQL, Python, Algorithms).
    *   Work on end-to-end portfolio projects or refine existing ones (potentially incorporating LLMs).
    *   Practice explaining your projects clearly and concisely.
*   **Week 13: Interview Simulation:**
    *   Focus on mock interviews (technical, behavioral, case studies).
    *   Practice explaining complex concepts simply (including LLM concepts if applicable).
    *   Prepare answers for common behavioral questions ("Tell me about a time...", "Why this role?").
    *   Review company-specific requirements and research.
