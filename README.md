<!-- omit in toc -->
# ⚒️ The Artifact Foundry

(Working in progress)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> A personal workshop for deliberate practice in machine learning, AI engineering, and software development.

I treat my career as a craft that requires deliberate practice and a constant return to the basics. Everything here is a result of me getting my hands dirty to understand how things actually work—not just how they are used. This is the structure that I use to keep pace with the field without losing sight of the foundations.


<!-- omit in toc -->
## Table of Contents

- [📚 What's Inside](#-whats-inside)
- [🏗️ Repository Structure](#️-repository-structure)
- [🚀 Getting Started](#-getting-started)
- [🤖 AI for Knowledge Building](#-ai-for-knowledge-building)
  - [Custom Agents](#custom-agents)
  - [Available Skills](#available-skills)
- [🤝 Contributing](#-contributing)
  - [How to Contribute](#how-to-contribute)
  - [Questions or Suggestions?](#questions-or-suggestions)

---

## 📚 What's Inside

This repository spans the full ML/AI engineering stack. Current focus: **classical machine learning** (tree-based methods, ensembles) and **Python fundamentals** (class systems, build tools).

**What's here now:**
- Deep-dive notes on ML algorithms, MLOps, and software engineering
- From-scratch implementations with tests and documentation
- Curated learning resources across all topic areas
- Interview prep cheatsheets for quick reference

**Coming next:**
- Neural networks and deep learning architectures
- Advanced NLP and LLM applications
- More production ML system patterns
- Data structures and algorithms deep dives

**Topic areas:** Foundations • Machine Learning • Deep Learning • ML System Design • AI Engineering • Data Analytics • Software Engineering • Productionization • Interview Prep

Each topic folder contains `notes/`, `snippets/`, and `resources.md` for organized learning.

---

## 🏗️ Repository Structure

This repository follows a deliberate organizational structure:

**Learning Materials** (optimized for understanding):
- Comprehensive explanations with "why" and "how it works internally"
- Real-world context and applications
- Multiple approaches and trade-offs
- Mathematical foundations where relevant

**Interview Prep** (optimized for recall):
- Condensed, no-fluff cheatsheets
- Pattern-focused common question types
- Template solutions and quick lookup tables
- Formulas and algorithms ready for quick reference

**Duplication is intentional** - The same topic can exist in both places with different treatments.

For detailed organization guidelines, see [.claude/ORGANIZATION.md](.claude/ORGANIZATION.md).

---

## 🚀 Getting Started

This repository uses Python 3.10+ with organized dependencies:

```bash
# Clone the repository
git clone https://github.com/whanyu1212/artifact-foundry.git
cd artifact-foundry

# Install dependencies using pip (choose what you need)
pip install -e .                    # Core dependencies only
pip install -e ".[ml]"              # + Machine learning
pip install -e ".[dl]"              # + Deep learning
pip install -e ".[viz]"             # + Visualization
pip install -e ".[notebooks]"       # + Jupyter notebooks
pip install -e ".[dev]"             # + Development tools
pip install -e ".[all]"             # Everything

# Or use uv (faster alternative)
uv pip install -e ".[all]"          # Install all dependencies
```

**Running tests**:
```bash
pytest tests/
```

**Browsing content**:
- Check [machine-learning/notes/](machine-learning/notes/) for ML deep dives
- Explore [machine-learning/snippets/](machine-learning/snippets/) for implementations
- Review [interview-prep/notes/](interview-prep/notes/) for quick references

---

## 🤖 AI for Knowledge Building

This repository leverages **Claude Code** with custom agents and skills for enhanced learning workflows:

### Custom Agents
- **learning-curator**: Organizes notes and resources, ensures consistent structure
- **snippet-reviewer**: Validates code correctness and educational clarity

### Available Skills
- `/add-resource` - Add learning materials to appropriate resources.md files
- `/link-project` - Track implementation projects in PROJECTS.md

See [AGENTS.md](AGENTS.md) for detailed documentation on AI-assisted workflows.

---

## 🤝 Contributing

While this is primarily a personal learning repository, contributions are welcome! If you'd like to collaborate:

### How to Contribute

1. **Resources**: Add valuable learning materials (books, papers, courses, articles)
   - Edit `resources.md` files in the appropriate topic folder
   - Follow the format: `- [Title](link) - Author/Source - Brief description`
   - Keep entries organized by type and alphabetically sorted

2. **Cheatsheets & Notes**: Improve or create study materials
   - Focus on depth and practical understanding over breadth
   - Include code examples, formulas, and real-world applications
   - Cite sources and credit original authors

3. **Projects**: Share implementation examples
   - Add to [PROJECTS.md](PROJECTS.md) with repository links
   - Include context: what you built, why, and key learnings

4. **Code Snippets**: Add practical implementations
   - Place in appropriate topic folders with clear documentation
   - Focus on educational value and correctness
   - Include comments explaining concepts and non-obvious decisions

### Questions or Suggestions?

Feel free to open an issue or submit a pull request. All contributions should maintain the spirit of hands-on learning and deep understanding.

---

<div align="center">

**Built with curiosity, maintained with discipline**

[View Documentation](.claude/) • [Browse Notes](machine-learning/notes/) • [Explore Code](machine-learning/snippets/)

</div>
