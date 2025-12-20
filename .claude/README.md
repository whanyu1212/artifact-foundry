# Claude Code Configuration

This directory contains custom agents and skills to help manage the Artifact Foundry repository.

## Repository Guidelines

- [ORGANIZATION.md](ORGANIZATION.md) - Complete guide on how to organize content across folders (interview-prep vs topic-specific folders)

## Agents

### learning-curator
Helps organize and curate learning materials. Invoke with:
```
Use the learning-curator agent to organize these notes
```

### snippet-reviewer
Reviews code snippets for correctness and clarity. Invoke with:
```
Use the snippet-reviewer agent to check this code
```

## Skills

### add-resource
Automatically activates when you want to add learning resources:
```
Add this course to my ML resources: [Course Name](link)
```

### link-project
Automatically activates when you mention project repositories:
```
I built a transformer implementation at github.com/user/repo
```

## Learn More

- [Claude Code Documentation](https://github.com/anthropics/claude-code)
- [Agent SDK Documentation](https://github.com/anthropics/claude-agent-sdk)
