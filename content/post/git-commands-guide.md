---
title: Git Commands - Guide
date: 2026-06-09
lastmod: 2026-06-09
description: A comprehensive guide to Git commands and concepts with real-world examples
tags:
  - Notes
  - Git_commands
categories:
  - Learning
draft: false
---

# Git Guide - Complete Reference

## Introduction

Git is a version control system that tracks changes to your code over time. It allows multiple developers to work on the same project without conflicts, and gives you a complete history of every change made to your project.

---

## 1. Initial Setup & Configuration

### Why Configuration Matters

Before making your first commit, Git needs to know who you are. Every commit is stamped with your name and email, creating an audit trail of who changed what.

### First-Time Setup

When you first install Git, set up your identity:

```bash
# Set your name and email globally (for all projects)
git config --global user.name "John Developer"
git config --global user.email "john.dev@example.com"

# Verify it was set correctly
git config --global user.name
git config --global user.email
```

**Real-world example:** On a development team, this ensures everyone's commits are properly attributed. When reviewing history, you'll see "John Developer fixed the login bug" not "Unknown User."

### Understanding Configuration Levels

Git has three configuration levels (from broadest to most specific):

```bash
# 1. SYSTEM - All users on this computer
/etc/gitconfig

# 2. GLOBAL - Your user account on this computer (most common)
~/.gitconfig

# 3. LOCAL - Only this specific project
.git/config
```

More specific settings override broader ones. Example:

```bash
# Set a global default
git config --global user.email "work@company.com"

# For a personal project, use different email (local overrides global)
cd ~/my-personal-project
git config --local user.email "personal@gmail.com"

# Check which email is active here
git config user.email  # Returns: personal@gmail.com
```

### Other Useful Configuration

```bash
# Set the default branch to 'main' for new repositories
git config --global init.defaultBranch main

# Show branch names in terminal prompts
git config --global status.showUntrackedFiles all

# Use VS Code as your default editor for commit messages
git config --global core.editor "code --wait"
```

### View Your Configuration

```bash
# See all global settings
git config --global --list

# See all local settings for current project
git config --local --list

# Open config file in your editor
git config --global --edit
```

---

## 2. Creating Your First Repository

### What is a Repository?

A repository (or "repo") is a project folder that Git watches. It contains all your files plus a hidden `.git` folder that stores the complete history.

### Starting a New Project

```bash
# Create a project folder
mkdir my-awesome-app
cd my-awesome-app

# Initialize Git in this folder
git init

# Verify - you should see a hidden .git folder
ls -la
```

**Real-world example:** Your team decides to build a weather app. You create a folder called `weather-app` and run `git init` to start tracking changes.

### Working with Remote Repositories (GitHub)

```bash
# Clone an existing project from GitHub
git clone https://github.com/username/project-name.git

# This creates a folder called 'project-name' with all files and history
cd project-name
```

---

## 3. Commits - Saving Your Work

### What is a Commit?

A commit is a "save point" in your project. It's a snapshot of your code at a specific moment, with a message explaining what changed.

Think of it like:
- **File save** (Ctrl+S) → Saves current file state
- **Git commit** → Saves entire project state with a description

### The Commit Workflow

```bash
# Step 1: See what changed
git status

# Step 2: Stage files you want to save (choose what goes in the save point)
git add src/App.js
git add src/App.css

# Or add everything
git add .

# Step 3: Create the commit with a message
git commit -m "Add user authentication form"

# Step 4: Verify
git log
```

### Writing Good Commit Messages

**Bad commit message:**
```
git commit -m "stuff"
git commit -m "fixed"
git commit -m "update"
```

**Good commit messages:**
```bash
# Clear, specific, describes what and why
git commit -m "Add password validation for user registration"
git commit -m "Fix bug where login fails on slow networks"
git commit -m "Refactor database queries for better performance"
```

### Real-World Example: A Day's Work

```bash
# Morning: Add feature
git add authentication.js
git commit -m "Implement JWT token validation"

# Midday: Fix a bug
git add api/users.js
git commit -m "Fix endpoint returning 404 on valid user IDs"

# Afternoon: Update documentation
git add README.md
git commit -m "Update installation instructions for Node v18"

# Evening: Review your day's work
git log --oneline
# Output:
# 3f7a9c2 Update installation instructions for Node v18
# 8d2f1a5 Fix endpoint returning 404 on valid user IDs
# 4b6c3e9 Implement JWT token validation
```

### Fixing a Commit Message

If you made a typo, fix the last commit:

```bash
# Before pushing to GitHub
git commit --amend -m "Correct message"
```

---

## 4. Viewing Your Project History

### Git Log - See What Changed

```bash
# See full commit history
git log

# See last 10 commits in compact form (my favorite!)
git log --oneline -10

# See history with visual branch structure
git log --oneline --graph --all

# See history for a specific file
git log -- src/App.js

# See commits by specific author
git log --author="John Developer"

# See commits from last week
git log --since="1 week ago"
```

### Finding When Something Broke

```bash
# Show what changed in a specific commit
git show 3f7a9c2

# Compare two versions
git diff main feature-branch

# Show changes in last 3 commits
git log -p -3
```

### Understanding Commit Hashes

Every commit gets a unique identifier:

```
3f7a9c2a1e8c5d9b4f6e2a1c7b3d8f5e9a1c
```

This is a SHA-1 hash based on:
- Your commit message
- Author name and email
- Date and time
- Parent commit(s)
- Files changed

You typically use just the first 7 characters: `3f7a9c2`

---

## 5. Branches - Working in Parallel

### What is a Branch?

A branch is a separate line of development. It's like creating an alternate version of your project where you can work on new features without affecting the main code.

**Real-world scenario:**
- Main project (main branch) is stable and running in production
- You start working on a new payment feature on a `payment-feature` branch
- Another developer works on mobile optimization on a `mobile-redesign` branch
- Both work independently, then merge back to main

### Create and Switch Branches

```bash
# Create a new branch (stays on current branch)
git branch feature/dark-mode

# Switch to that branch
git switch feature/dark-mode

# Or do both in one command (modern approach)
git switch -c feature/dark-mode

# List all branches (* shows current branch)
git branch

# Output:
#   main
# * feature/dark-mode
```

### Branch Naming Conventions

Good teams use naming patterns:

```bash
git switch -c feature/user-authentication
git switch -c bugfix/login-timeout-issue
git switch -c hotfix/security-patch
git switch -c docs/update-readme
```

### See Branches with Their Status

```bash
# Local branches
git branch

# Remote branches on GitHub
git branch -r

# All branches
git branch -a

# Show last commit on each branch
git branch -v
```

### Real-World Branch Workflow

```bash
# You're working on main (production-ready code)
git switch main

# Your manager asks: "Can you add a feature for bulk user import?"
# Create a feature branch
git switch -c feature/bulk-import

# Make changes
vim upload.js
git add upload.js
git commit -m "Add file upload validation"

# Make more changes
vim bulk_processor.js
git add bulk_processor.js
git commit -m "Implement bulk import processing"

# Feature is done, switch back to main
git switch main

# Now merge the feature into main (see Merge section)
```

---

## 6. Merge - Combining Branches

### When to Merge

Once you've finished a feature on your branch, merge it back into `main` to include it in the project.

### Simple Merge Example

```bash
# Your feature branch is complete
git switch feature/dark-mode

# Final commit
git add theme.css
git commit -m "Complete dark mode implementation"

# Switch to main to receive the changes
git switch main

# Merge feature branch into main
git merge feature/dark-mode

# SUCCESS! Dark mode is now in main
```

### Visualizing a Merge

**Before merge:**
```
main:          A - B - C
               
feature:             D - E
```

**After `git merge feature` (while on main):**
```
main:          A - B - C - F
               \       /
feature:        D - E
```

The `F` is a merge commit that combines both branches.

### Real-World Merge Scenario

```bash
# Team workflow: You finish a feature
git switch main
git pull                           # Get latest from team
git merge feature/user-profiles    # Merge your feature

# If successful:
git push                           # Send to GitHub
git branch -d feature/user-profiles # Clean up old branch

# If there are CONFLICTS (both you and teammate changed same file):
# Git shows <<<<<<< HEAD ... ======= ... >>>>>>> conflicts
# Edit files to resolve conflicts
git add conflicted-file.js
git commit -m "Resolve merge conflicts with profile branch"
```

### Merge Strategies

```bash
# Standard merge (creates merge commit) - RECOMMENDED
git merge feature-branch

# Fast-forward merge (simpler history if no divergence)
git merge --ff-only feature-branch

# Squash merge (combine all commits into one)
git merge --squash feature-branch
```

---

## 7. Rebase - Cleaning Up History

### What is Rebase?

Rebase replays your commits on top of another branch. It's like saying "pretend your branch started from the latest code."

### Merge vs Rebase

Both combine branches, but differently:

**Merge** = "Integrate my work into main" → Creates merge commit
**Rebase** = "Move my work on top of latest main" → Keeps history linear

```bash
# Merge creates: A-B-C-F (where F is merge commit)
#                   \  /
#                   D-E

# Rebase results: A-B-C-D-E (linear, no merge commit)
```

### When to Use Rebase

```bash
# GOOD: Rebase your feature branch before merging (keeps history clean)
git switch feature/notifications
git rebase main
git switch main
git merge feature/notifications  # Now this is fast-forward!

# AVOID: Don't rebase shared/public branches
# (It rewrites history, confusing teammates)
```

### Rebase Example

```bash
# You've been working on feature branch
git switch feature/auth
git log --oneline
# 3e2c1a9 Add password reset email
# 2f1b0a8 Implement two-factor auth
# a9e7d1c Add login validation
# 4b3c2d1 Initial commit (main started here)

# Meanwhile, teammates added stuff to main
git switch main
git log --oneline
# Latest: "Fix critical payment bug"
# "Update API documentation"

# Go back to your branch and rebase onto latest main
git switch feature/auth
git rebase main

# Your commits are now replayed on top of latest main
# History is linear!
git log --oneline
# 5n4m3l2 Add password reset email (NEW HASH)
# 4k3j2i1 Implement two-factor auth (NEW HASH)
# 3j2i1h0 Add login validation (NEW HASH)
# Latest: Fix critical payment bug
# Update API documentation
# ...
```

---

## 8. Git Internal Structure (Optional Deep Dive)

### The .git Folder

Everything Git stores is in this hidden folder:

```bash
.git/
├── objects/        # Your data (commits, files, directories)
├── refs/           # Pointers to commits (branches, tags)
├── HEAD            # Points to current branch
└── config          # Repository settings
```

### Git Objects

Git stores everything as objects:

- **Blob** = File content
- **Tree** = Directory (contains blobs and subtrees)
- **Commit** = Snapshot (points to a tree + metadata)

### Inspecting Objects (Advanced)

```bash
# See what's in a commit
git cat-file -p abc123

# Find commit hashes
git log --oneline

# Explore the internal structure
ls -la .git/objects
```

---

## 9. Real-World Workflows

### Workflow 1: Solo Developer (You + GitHub)

```bash
# Start local
git init
git add .
git commit -m "Initial project setup"

# Create repo on GitHub, then:
git remote add origin https://github.com/you/project.git
git branch -M main
git push -u origin main

# Daily work
git switch -c feature/new-feature
# ... make changes ...
git commit -m "Add new feature"
git push origin feature/new-feature

# On GitHub: Create Pull Request, merge to main
git switch main
git pull origin main
git branch -d feature/new-feature
```

### Workflow 2: Team Developer

```bash
# Clone team project
git clone https://github.com/company/app.git

# Before starting new feature, get latest
git switch main
git pull origin main

# Create feature branch
git switch -c feature/search-optimization

# Work and commit
git add search.js
git commit -m "Optimize search queries"

# Push to GitHub
git push origin feature/search-optimization

# Create Pull Request, await review
# After approval and merge:
git switch main
git pull origin main
```

### Workflow 3: Fixing a Bug

```bash
# You find a bug in production
git switch main

# Create a hotfix branch
git switch -c hotfix/critical-crash

# Make minimal fix
git add crash-handler.js
git commit -m "Fix null reference exception in payment flow"

# Test locally
git push origin hotfix/critical-crash

# Create Pull Request, merge urgently
git switch main
git pull origin main

# Apply same fix to development branch
git switch develop
git merge main
git push origin develop
```

---

## Quick Reference

### Most Common Commands

```bash
# Check status
git status

# Stage changes
git add .

# Commit
git commit -m "message"

# View history
git log --oneline

# Create branch
git switch -c branch-name

# Switch branch
git switch branch-name

# Merge branch
git merge branch-name

# Push to GitHub
git push origin branch-name

# Pull from GitHub
git pull origin branch-name
```

---

## Resources

- [Git Official Documentation](https://git-scm.com/docs)
- [GitHub Learning](https://docs.github.com/en/get-started)
- [Pro Git Book (Free)](https://git-scm.com/book/en/v2) - Excellent reference
- [Git Cheat Sheet](https://github.github.com/training-kit/downloads/github-git-cheat-sheet.pdf)
