# Contributing to VigiDrive

## Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Format

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `style`: Code style changes (formatting, etc.)
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

### Scopes

- `detection`: Core detection engine
- `auth`: Authentication
- `ui`: User interface
- `database`: Database operations
- `alerts`: Sound alert system
- `analytics`: Analytics dashboard

### Examples

```
feat(detection): implement dual-gate drowsiness logic
fix(database): correct Event timestamp timezone handling
docs(readme): update installation instructions
test(features): add unit tests for EAR/MAR calculation
refactor(alerts): simplify audio playback logic
```

## Pull Request Process

1. Create a feature branch from `development`
2. Make your changes following commit conventions
3. Run tests and linting locally
4. Push and create a PR targeting `development`
5. Wait for review and CI checks to pass
6. Merge after approval
