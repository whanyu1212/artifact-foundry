# Artifact Foundry - AI Assistant Context

## Project Overview

**The Artifact Foundry** is a personal learning workshop and knowledge archive. This repository treats career development as a craft requiring deliberate practice and constant return to fundamentals. Everything here results from hands-on learning to understand how things actually work—not just how they're used.

### Core Philosophy

- **Understanding over completion**: Deep comprehension beats surface-level coverage
- **Implementation over theory**: Build it to truly understand it
- **Fundamentals over frameworks**: Master the basics, frameworks will follow
- **Quality over quantity**: One well-understood concept beats ten memorized facts

## Repository Structure

### Topic Folders

Each topic folder follows a consistent structure:
- `notes/` - Markdown files with concepts and deep dives
- `snippets/` - Code written while learning (with tests)
- `resources.md` - Curated learning materials (books, articles, courses, papers)

**Main Topic Areas**:
- `foundations/` - Math, statistics, algorithms, data structures, Python fundamentals
- `machine-learning/` - ML algorithms, implementations from scratch
- `deep-learning/` - Neural networks, modern architectures
- `ml-system-design/` - Production ML systems, MLOps
- `ai-engineering/` - LLM applications, prompt engineering
- `data-analytics/` - SQL, data analysis, visualization
- `software-engineering/` - Software design, architecture, best practices
- `productionization/` - Deployment, monitoring, production systems
- `ai-productivity/` - Tools and workflows for AI-assisted development
- `interview-prep/` - Cheatsheets and quick references optimized for interview recall

### Special Files

- `PROJECTS.md` - Links to implementation projects with context
- `.claude/ORGANIZATION.md` - Detailed guide on where different content types belong

### Content Organization Principle

**Separation of Learning vs. Interview Prep**:

- **Topic folders**: Optimized for understanding and application
  - Comprehensive explanations
  - "Why" and "how it works internally"
  - Real-world context and applications
  - Multiple approaches and trade-offs

- **interview-prep/**: Optimized for speed and recall under pressure
  - Condensed, no fluff
  - Pattern-focused (common question types)
  - Template solutions
  - Quick lookup tables and formulas

**Duplication is OK!** The same topic can exist in both places with different treatments (deep dive vs. quick reference).

## Custom Agents

### learning-curator
**When to use**: Adding new resources, organizing notes, reviewing study materials

**What it does**:
- Checks topic folder structure and suggests where new materials should go
- Ensures consistent formatting across similar files
- Cross-references related topics
- Maintains the archive nature of learning materials

**Key behaviors**:
- Keeps snippets well-commented for future reference
- Organizes notes by concept, not by source
- Links related materials across topic areas
- Preserves hands-on, from-scratch learning approach

### snippet-reviewer
**When to use**: After writing implementation snippets or reviewing archived code

**What it does**:
- Reads snippet files and checks if code actually works (runs it if possible)
- Verifies comments explain "why" not just "what"
- Ensures examples show input/output
- Suggests improvements for clarity

**Review checklist**:
- Code is correct and follows best practices
- Well-commented for future reference (explain concepts, not obvious syntax)
- Has clear examples of usage
- Includes edge cases if relevant
- Notes limitations or common pitfalls
- Links to related concepts in other topic folders

**Focus areas**:
- Clarity over cleverness
- Educational value over brevity
- Correct implementation over optimization
- Self-contained examples that work standalone

## Available Skills

### /add-resource
Add new learning resources (books, articles, courses, papers) to the appropriate `resources.md` file.

### /link-project
Add project repository links to `PROJECTS.md` when completing, starting, or sharing hands-on projects.

## Workflow Preferences

### When Adding Notes
1. Choose correct location using the organization decision tree (see `.claude/ORGANIZATION.md`)
2. Include mathematical notation where appropriate
3. Provide practical code examples
4. Cross-reference related topics
5. Maintain archival quality - these are historical records of learning

### When Adding Code Snippets
1. Place in appropriate topic folder's `snippets/` directory
2. Include comprehensive comments explaining concepts
3. Add tests to verify correctness
4. Show clear input/output examples
5. Document edge cases and limitations
6. Make examples self-contained and runnable

### When Adding Resources
1. Use consistent format: `- [Title](link) - Author/Source - Brief description`
2. Organize by type (Books, Papers, Courses, Articles)
3. Keep entries alphabetically sorted within each type
4. Add to appropriate topic's `resources.md`

### Code Style
- **Python**: Follow PEP 8, type hints where helpful for learning
- **Comments**: Explain concepts and non-obvious decisions, not syntax
- **Tests**: Include for all code snippets to verify correctness
- **Documentation**: Focus on educational value and understanding

## Current Focus Areas

Based on recent activity:
- Tree-based machine learning methods and evaluation metrics
- Python class system and OOP fundamentals
- Python build systems and package management
- Decision tree implementations from scratch

## Git Workflow

- Main branch: `main`
- Keep commits focused and descriptive
- Commit messages should describe the learning milestone or addition

## Important Notes

- This is an **archive of learning** - preserve historical context
- Prioritize **understanding** over production-ready code
- Code snippets should be **educational**, not optimized
- **From-scratch implementations** are valued for deep understanding
- Cross-reference related concepts across different topic areas
- Maintain consistent structure across all topic folders
