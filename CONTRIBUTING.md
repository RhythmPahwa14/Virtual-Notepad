# Contributing to Virtual Notepad

Thank you for your interest in contributing to Virtual Notepad! This document provides guidelines for contributing to the project.

## Team Members

- **Primary Developer**: [Your Name]
- **Collaborator**: [Teammate Name]

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/[username]/VirtualNotepad.git
   cd VirtualNotepad
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Git Workflow

### For Team Members:

1. **Always pull latest changes before starting work:**
   ```bash
   git pull origin main
   ```

2. **Create feature branches for new work:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make commits with descriptive messages:**
   ```bash
   git add .
   git commit -m "Add: Specific description of changes"
   ```

4. **Push your branch:**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request on GitHub for review**

### Commit Message Guidelines:

- `Add: New feature or functionality`
- `Fix: Bug fixes`
- `Update: Modifications to existing features`
- `Remove: Deleted functionality`
- `Docs: Documentation updates`
- `Style: Code formatting changes`

## Code Standards

- Follow PEP 8 for Python code style
- Add comments for complex logic
- Test your changes thoroughly
- Update documentation as needed

## Reporting Issues

Please use GitHub Issues to report bugs or suggest features. Include:
- Clear description of the issue
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- System information (OS, Python version)

## Pull Request Process

1. Ensure your code follows the project standards
2. Update README.md if needed
3. Add your changes to this CONTRIBUTING.md if relevant
4. Request review from team members
5. Address any feedback before merging

## Questions?

Feel free to reach out to team members for any questions about contributing to this project.
