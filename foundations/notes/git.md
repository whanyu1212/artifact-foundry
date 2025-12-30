# Git - Version Control Fundamentals

## Overview

Git is a distributed version control system that tracks changes to files. Understanding Git's internal data structures and core concepts is essential for effective version control and collaboration.

## Core Concepts

### The Git Object Model

Git is fundamentally a **content-addressable filesystem** with a version control interface built on top. Everything in Git is stored as objects identified by SHA-1 hashes.

#### Four Object Types

1. **Blob** - Stores file contents
   - Pure content, no metadata (no filename, no permissions)
   - Identified by SHA-1 hash of content
   - Same content = same hash (deduplication)

2. **Tree** - Represents directory structure
   - Contains pointers to blobs and other trees
   - Stores filenames and file modes
   - Like a directory listing

3. **Commit** - Snapshot of your repository at a point in time
   - Points to a tree object (root directory)
   - Contains metadata: author, committer, timestamp, message
   - Points to parent commit(s)
   - Forms a directed acyclic graph (DAG)

4. **Tag** - Named pointer to a specific commit
   - Lightweight tags: just a named reference
   - Annotated tags: full objects with metadata

### The Three States

Files in Git can exist in three states:

1. **Modified** - Changed but not staged
2. **Staged** - Marked to go into next commit
3. **Committed** - Safely stored in local database

These correspond to three areas:

- **Working Directory** - Your current file system
- **Staging Area (Index)** - Preparation area for next commit
- **Repository (.git directory)** - Where Git stores everything

### Branches Are Just Pointers

A branch is simply a **movable pointer to a commit**. That's it.

- `HEAD` is a special pointer that indicates which branch you're on
- `HEAD` is typically a symbolic reference to a branch
- When you commit, the current branch pointer moves forward
- Creating a branch is instant (just creating a 41-byte file)

### Distributed Nature

Every clone is a full repository:
- Complete history
- All branches and tags
- Can work offline
- No single point of failure

## How Git Actually Works

### What Happens When You Commit

```
1. Git takes staged files and creates blob objects (if not already existing)
2. Creates tree objects representing directory structure
3. Creates a commit object pointing to the root tree
4. Moves the current branch pointer to the new commit
5. Hashes determine object identity - same content = same object
```

**Example commit object structure:**
```
commit → tree (snapshot)
         parent (previous commit)
         author
         committer
         message
```

### The DAG (Directed Acyclic Graph)

Git's commit history forms a DAG:
- **Directed**: Commits point to parents (backward in time)
- **Acyclic**: No loops (can't be your own ancestor)
- **Graph**: Can have multiple parents (merges) and multiple children (branches)

```
    A---B---C  (main)
         \
          D---E  (feature)
```

Merging creates a commit with two parents:
```
    A---B---C---F  (main)
         \     /
          D---E  (feature)
```

### Content Addressing and Deduplication

Objects are stored by their SHA-1 hash:
- Same content = same hash = stored once
- Efficient storage across branches
- Integrity checking (can detect corruption)

## Key Operations Explained

### Staging (git add)

When you `git add file.txt`:
1. Git computes the SHA-1 hash of file contents
2. Compresses the content and stores as a blob object in `.git/objects/`
3. Updates the index (staging area) to reference this blob

**Why staging exists**: Allows you to craft precise commits from messy working directory changes.

### Committing (git commit)

When you `git commit`:
1. Creates tree objects from the staging area
2. Creates a commit object pointing to root tree
3. Sets parent to current HEAD
4. Moves the branch pointer forward
5. Content already stored during `git add`

### Branching (git branch)

Creating a branch:
- Creates a new file in `.git/refs/heads/branch-name`
- Contains the SHA-1 hash of a commit
- That's all! Instant operation.

### Merging Strategies

**Fast-forward merge**:
- No divergent changes
- Just moves pointer forward
- No merge commit needed
```
Before:  A---B---C  (main)
              \
               D---E  (feature)

After:   A---B---C---D---E  (main, feature)
```

**Three-way merge**:
- Divergent changes
- Uses common ancestor
- Creates merge commit
```
A---B---C---F  (main)
     \     /
      D---E  (feature)
```

Merge commit F has two parents: C and E

### Rebasing vs Merging

**Merge** preserves history:
- Shows true chronological development
- Creates merge commits
- History can get complex

**Rebase** rewrites history:
- Linear, cleaner history
- Replays commits on new base
- **Never rebase public/shared commits**

```
Before rebase:
    A---B---C  (main)
         \
          D---E  (feature)

After rebase (git rebase main):
    A---B---C  (main)
             \
              D'---E'  (feature)
```

D' and E' are **new commits** (different SHA-1) with same changes.

## The .git Directory

Understanding what's inside `.git/`:

```
.git/
├── HEAD              # Points to current branch
├── config            # Repository configuration
├── description       # For GitWeb
├── hooks/            # Client/server hooks
├── index             # Staging area (binary)
├── info/             # Global exclude patterns
├── objects/          # All git objects (blobs, trees, commits, tags)
│   ├── [0-9a-f][0-9a-f]/  # First 2 chars of SHA-1
│   ├── pack/         # Packed objects (compressed)
│   └── info/         # Object metadata
└── refs/             # Pointers to commits
    ├── heads/        # Local branches
    ├── remotes/      # Remote branches
    └── tags/         # Tags
```

## Remote Operations

### Fetch vs Pull

**Fetch**:
- Downloads objects and refs from remote
- Updates `refs/remotes/origin/*`
- Doesn't modify your working directory or branches
- Safe operation

**Pull**:
- `git fetch` + `git merge`
- Can cause merge conflicts
- Modifies your current branch

### Push

When you `git push`:
1. Git determines which commits remote doesn't have
2. Packages objects into a pack file
3. Sends to remote
4. Remote updates its refs
5. **Fails if not a fast-forward** (unless force pushed)

## Advanced Concepts

### Reflog - Your Safety Net

`git reflog` records when HEAD moves:
- Every commit, checkout, merge, rebase
- Local to your repository
- Typically expires after 90 days
- Can recover "lost" commits

### Detached HEAD State

When HEAD points directly to a commit (not a branch):
```
git checkout <commit-sha>
```
- Can look around and make experimental commits
- Commits not on any branch (will be garbage collected)
- Use `git checkout -b new-branch` to save work

### Cherry-pick

Apply a specific commit to current branch:
```
git cherry-pick <commit-sha>
```
- Creates a **new commit** with same changes
- Different SHA-1 (different parent)
- Useful for applying bug fixes across branches

### Reset vs Revert

**Reset** (rewrites history):
```
git reset --soft HEAD~1   # Move branch pointer, keep changes staged
git reset --mixed HEAD~1  # Move pointer, unstage changes
git reset --hard HEAD~1   # Move pointer, discard changes
```

**Revert** (preserves history):
```
git revert <commit>
```
- Creates a **new commit** that undoes changes
- Safe for public branches

### Stash - Temporary Storage

```
git stash              # Save working directory changes
git stash pop          # Restore and remove from stash
git stash list         # See all stashes
git stash apply stash@{1}  # Apply specific stash
```

Stash is a stack of commits on a special ref (`refs/stash`).

## Git Internals: Plumbing vs Porcelain

Git commands are divided into:

**Porcelain** (user-friendly):
- `git add`, `git commit`, `git push`, etc.
- What you use daily

**Plumbing** (low-level):
- `git hash-object` - Create objects
- `git cat-file` - Examine objects
- `git update-ref` - Modify refs
- Understanding these reveals how Git actually works

### Example: Manual commit creation

```bash
# Create a blob
echo "Hello, Git" | git hash-object -w --stdin
# Returns: 8d0e41234f24b6da002d962a26c2495ea16a425f

# Create a tree (simplified)
git write-tree

# Create a commit
echo "Initial commit" | git commit-tree <tree-sha>

# Update branch reference
git update-ref refs/heads/main <commit-sha>
```

This is what `git add` and `git commit` do under the hood!

## Best Practices Based on Internals

1. **Commit atomic changes**
   - Each commit should be a complete, logical unit
   - Easy to revert, cherry-pick, or understand

2. **Write meaningful commit messages**
   - Commits are documentation
   - First line: imperative mood, 50 chars
   - Body: explain why, not what

3. **Don't rewrite public history**
   - Others may have based work on those commits
   - Causes divergence and conflicts

4. **Use branches liberally**
   - Cheap to create (just a pointer)
   - Isolate features/experiments
   - Easy to delete

5. **Fetch often, push when ready**
   - Stay up to date with remote
   - Resolve conflicts incrementally

6. **Understand what commands do**
   - Know if they rewrite history
   - Know if they're destructive
   - Use `--dry-run` when uncertain

## Common Workflows

### Feature Branch Workflow

```bash
git checkout -b feature-x          # Create feature branch
# ... make changes ...
git add .
git commit -m "Implement feature X"
git checkout main
git pull origin main               # Update main
git merge feature-x                # Merge feature
git push origin main
git branch -d feature-x            # Delete local branch
```

### Rebase Workflow (for clean history)

```bash
git checkout feature-x
git fetch origin
git rebase origin/main             # Replay commits on updated main
# ... resolve conflicts if any ...
git push --force-with-lease        # Update remote feature branch
```

### Fixing Mistakes

**Amend last commit:**
```bash
git commit --amend                 # Rewrites last commit (new SHA-1)
```

**Undo last commit but keep changes:**
```bash
git reset --soft HEAD~1
```

**Discard all local changes:**
```bash
git reset --hard HEAD
git clean -fd                      # Remove untracked files
```

**Recover lost commit:**
```bash
git reflog                         # Find commit SHA
git checkout -b recovery <commit-sha>
```

## Performance Considerations

### Object Packing

Git periodically packs objects:
- Delta compression (stores differences)
- Significantly reduces `.git/objects/` size
- Happens automatically via `git gc`

### Shallow Clones

For large repositories:
```bash
git clone --depth 1 <url>          # Only latest commit
git clone --shallow-since=2024-01-01 <url>  # Commits since date
```
- Faster cloning
- Less disk space
- Limited history

### Sparse Checkout

For monorepos:
```bash
git sparse-checkout init --cone
git sparse-checkout set path/to/directory
```
- Only checkout specific directories
- Full repository history still available

## Key Insights

1. **Git stores snapshots, not diffs**
   - Each commit is a full snapshot
   - Pack files use deltas for efficiency

2. **Branches are cheap**
   - Just 41-byte files with SHA-1 hash
   - No copying of files

3. **Content is king**
   - Same content = same object
   - Integrity via cryptographic hashing

4. **Everything is local first**
   - Commit, branch, merge all happen locally
   - Only push/pull/fetch talk to remote

5. **History is immutable**
   - Commits cannot be changed (new SHA-1)
   - "Rewriting" creates new commits

6. **Understanding the DAG helps**
   - Visualize branch relationships
   - Predict merge outcomes
   - Debug complex histories

## Further Learning

Understanding Git internals makes everything else make sense:
- Why merge conflicts happen
- When to rebase vs merge
- How to recover from mistakes
- Why certain operations are fast/slow

**Recommended deep dive**: Read `.git/` directory contents, experiment with plumbing commands, visualize the object graph.
