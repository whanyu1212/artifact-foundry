# Claude Code Setup

Custom agents and skills configured for managing the Artifact Foundry repository.

## Configuration Location

All configurations are in [`.claude/`](../../.claude/)

## Custom Agents

### learning-curator
**Purpose**: Organizes learning materials, ensures consistency across topic folders

**Usage**:
```
Use the learning-curator agent to organize these notes
```

**What it does**:
- Suggests appropriate locations for new materials (notes/, snippets/, resources.md)
- Ensures consistent formatting
- Cross-references related topics
- Maintains the archive structure

### snippet-reviewer
**Purpose**: Reviews code snippets for correctness and educational value

**Usage**:
```
Use the snippet-reviewer to check this implementation
```

**What it does**:
- Verifies code actually works
- Checks comments explain concepts clearly
- Ensures examples are self-contained
- Suggests improvements for clarity

## Custom Skills

### add-resource
**Auto-activates when**: Adding books, courses, articles, papers to resources

**Example**:
```
Add this course to my ML resources: CS229 Stanford Machine Learning
```

**What it does**:
- Determines correct topic folder
- Maintains consistent formatting
- Organizes entries alphabetically by type

### link-project
**Auto-activates when**: Mentioning project repositories

**Example**:
```
I built a transformer implementation at github.com/user/transformer-scratch
```

**What it does**:
- Adds link to PROJECTS.md
- Categorizes project appropriately
- Maintains organized structure

## Key Learnings

- **Agents** are invoked explicitly for complex multi-step workflows
- **Skills** activate automatically when context matches their description
- Both use frontmatter (YAML) for configuration
- Descriptions are critical - they determine when agents/skills are used

## Files Created

- `.claude/agents/learning-curator.md`
- `.claude/agents/snippet-reviewer.md`
- `.claude/skills/add-resource/SKILL.md`
- `.claude/skills/link-project/SKILL.md`
- `.claude/README.md`
