# CI/CD Pipeline Documentation

This document describes the GitHub Actions CI/CD pipeline for the CoreMusic project.

## Workflows Overview

### 1. CI Workflow (`ci.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual trigger via workflow_dispatch

**Jobs:**

#### Test Job
- **Matrix Strategy**: Tests across Python 3.11, 3.12, and 3.13
- **Platform**: macOS-latest (required for CoreAudio)
- **Steps**:
  1. Checkout code
  2. Set up Python with specified version
  3. Install `uv` package manager
  4. Install project dependencies
  5. Build Cython extension
  6. Run tests (excluding slow tests with `-m "not slow"`)
  7. Run type checking with mypy

#### Lint Job
- **Purpose**: Code quality and formatting checks
- **Steps**:
  1. Install dependencies
  2. Check code formatting with ruff (if configured)

#### Build Job
- **Purpose**: Create distribution packages
- **Dependencies**: Requires test and lint jobs to pass
- **Steps**:
  1. Build source distribution (sdist)
  2. Build wheel distribution
  3. Validate distributions with twine
  4. Upload artifacts for 7 days

#### Test Install Job
- **Purpose**: Verify package installs correctly
- **Dependencies**: Requires build job
- **Steps**:
  1. Download build artifacts
  2. Install wheel package
  3. Test import and basic functionality

**Typical Runtime**: ~5-10 minutes

### 2. Coverage Workflow (`coverage.yml`)

**Triggers:**
- Push to `main` branch
- Pull requests to `main`
- Manual trigger

**Purpose**: Generate and track test coverage metrics

**Steps**:
1. Run tests with coverage collection
2. Upload coverage to Codecov
3. Generate HTML coverage report
4. Create coverage summary in GitHub Actions summary
5. Check coverage threshold (warns if < 75%)

**Artifacts**:
- `coverage-report-{sha}`: HTML coverage report (30 days retention)
- Coverage XML for Codecov integration

**Typical Runtime**: ~5 minutes

### 3. Release Workflow (`release.yml`)

**Triggers:**
- GitHub release published
- Manual trigger with version input

**Environment**: `release` (requires approval for production releases)

**Permissions**:
- `id-token: write` - For trusted PyPI publishing
- `contents: write` - For creating releases

**Steps**:
1. Build all distribution formats (sdist + wheels for Python 3.11, 3.12, 3.13, 3.14)
2. Validate distributions
3. **Test PyPI** (manual trigger): Publish to TestPyPI first
4. **Production** (release trigger): Publish to PyPI using trusted publishing
5. Upload release artifacts
6. Test PyPI installation

**Security**: Uses PyPI trusted publishing (no API tokens needed for releases)

**Typical Runtime**: ~15-20 minutes

### 4. Documentation Workflow (`docs.yml`)

**Triggers:**
- Push to `main` (affecting docs or source code)
- Pull requests modifying docs
- Manual trigger

**Steps**:
1. Build Sphinx documentation
2. Check for broken links
3. Upload documentation artifacts
4. Deploy to GitHub Pages (on main branch pushes)

**Artifacts**:
- `documentation-{sha}`: Built HTML documentation (30 days retention)

**Typical Runtime**: ~3-5 minutes

### 5. Comprehensive Tests Workflow (`comprehensive-tests.yml`)

**Triggers:**
- Weekly schedule (Sundays at 3 AM UTC)
- Manual trigger

**Purpose**: Run all tests including slow AudioUnit tests

**Steps**:
1. Run full test suite (including `@pytest.mark.slow` tests)
2. Generate comprehensive coverage report
3. Create GitHub issue if tests fail

**Timeout**: 2 hours maximum

**Typical Runtime**: ~30-90 minutes (variable due to AudioUnit hangs)

## Dependabot Configuration

Automated dependency updates configured in `.github/dependabot.yml`:

**GitHub Actions**:
- Weekly updates on Mondays
- Max 5 open PRs

**Python Dependencies**:
- Weekly updates on Mondays
- Max 10 open PRs
- Grouped updates for:
  - Dev dependencies (pytest, mypy, sphinx, twine)
  - NumPy/SciPy ecosystem

## Secrets and Variables

### Required Secrets

#### For Release Workflow:
- `TEST_PYPI_API_TOKEN` (optional): Token for TestPyPI uploads
- `GITHUB_TOKEN`: Automatically provided by GitHub Actions

#### For Coverage Workflow:
- `CODECOV_TOKEN` (optional): Codecov integration token

### GitHub Environments

**release** environment:
- Used for production PyPI releases
- Can be configured with protection rules and approvals

## Usage Examples

### Running CI Locally

```bash
# Run the same tests as CI
make test

# Run with coverage like Coverage workflow
make coverage

# Build distributions like Release workflow
make release
```

### Manual Workflow Triggers

#### Trigger Release Workflow:
1. Go to Actions → Release
2. Click "Run workflow"
3. Enter version number (e.g., `0.1.9`)
4. Click "Run workflow"

#### Trigger Comprehensive Tests:
1. Go to Actions → Comprehensive Tests
2. Click "Run workflow"
3. Select branch
4. Click "Run workflow"

## Setting Up for a New Repository

### 1. Enable GitHub Actions
- Actions are enabled by default for public repos
- For private repos: Settings → Actions → Enable Actions

### 2. Configure PyPI Trusted Publishing
1. Go to [PyPI Account Settings](https://pypi.org/manage/account/publishing/)
2. Add a new trusted publisher:
   - **PyPI Project Name**: `coremusic`
   - **Owner**: `shakfu` (or your GitHub username)
   - **Repository**: `coremusic`
   - **Workflow**: `release.yml`
   - **Environment**: `release`

### 3. Create Release Environment
1. Go to Repository Settings → Environments
2. Create environment named `release`
3. (Optional) Add protection rules:
   - Required reviewers
   - Deployment branches (e.g., only `main`)

### 4. Configure Codecov (Optional)
1. Go to [codecov.io](https://codecov.io)
2. Add repository
3. Copy upload token
4. Add as `CODECOV_TOKEN` secret in repository settings

### 5. Enable GitHub Pages (Optional)
1. Settings → Pages
2. Source: GitHub Actions
3. Documentation will be auto-deployed on main branch pushes

## Workflow Status Badges

Add to README.md:

```markdown
[![CI](https://github.com/shakfu/coremusic/actions/workflows/ci.yml/badge.svg)](https://github.com/shakfu/coremusic/actions/workflows/ci.yml)
[![Coverage](https://github.com/shakfu/coremusic/actions/workflows/coverage.yml/badge.svg)](https://github.com/shakfu/coremusic/actions/workflows/coverage.yml)
```

## Troubleshooting

### Tests Fail on CI but Pass Locally
- Check Python version compatibility (CI tests 3.11, 3.12, 3.13)
- Verify macOS-specific dependencies
- Check for timing issues in slow tests

### Release Workflow Fails
- Verify trusted publisher configuration on PyPI
- Check version number format in `pyproject.toml`
- Ensure all distributions pass `twine check`

### Coverage Upload Fails
- Check Codecov token configuration
- Verify coverage.xml is generated
- Review Codecov service status

### Comprehensive Tests Timeout
- AudioUnit tests may hang on certain plugins
- Timeout set to 2 hours to handle this
- Consider adjusting timeout or marking more tests as slow

## Best Practices

1. **Always run tests locally** before pushing: `make test`
2. **Check coverage** before major PRs: `make coverage-html`
3. **Use semantic versioning** for releases (MAJOR.MINOR.PATCH)
4. **Test installations** after PyPI releases
5. **Monitor workflow runs** for failures
6. **Keep dependencies updated** via Dependabot PRs
7. **Review and approve** release workflow runs

## Performance Optimization

- **Caching**: Uses `uv` which has built-in caching
- **Matrix Testing**: Runs Python versions in parallel
- **Selective Testing**: Excludes slow tests from regular CI
- **Artifact Retention**: 7-30 days based on importance

## Monitoring and Alerts

- **Failed Runs**: GitHub sends email notifications
- **Comprehensive Test Failures**: Creates GitHub issues automatically
- **Coverage Degradation**: Warnings in workflow logs if < 75%
- **Dependabot Alerts**: Security vulnerability notifications

## Future Improvements

Potential enhancements:

1. **Performance Benchmarking**: Add benchmark workflow
2. **Multi-OS Testing**: Add Linux/Windows with CoreAudio emulation
3. **Deployment Previews**: Preview docs on PR branches
4. **Slack/Discord Integration**: Notifications for releases
5. **Automated Changelog**: Generate from commit messages
6. **Security Scanning**: CodeQL or Snyk integration
