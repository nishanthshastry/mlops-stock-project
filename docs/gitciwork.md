# Git Workflow, CI/CD Automation, Pull Request Validation and Docker Deployment

## Overview

This document demonstrates the automated Git workflow implemented for the MLOps Trading Platform project.

The repository integrates:

- GitHub Actions
- Continuous Integration (CI)
- Docker Build Automation
- DockerHub Deployment
- Pull Request Validation
- CML Reporting
- Automated Quality Gates

The objective is to ensure every code change is validated before being merged into the main branch.

---

# Architecture

Developer Push
      |
      V
GitHub Actions
      |
      V
CI Pipeline
      |
      V
Docker Build
      |
      V
CML Report
      |
      V
Pull Request Checks
      |
      V
Merge Approval
      |
      V
Main Branch Deployment

---

# 1. Push-Based Workflow Execution

Whenever code is pushed to the development branch:

- CI workflow executes
- Docker image is built
- CML report is generated
- Validation checks run automatically

## Workflow Triggered

![Push Workflow](../reports/cidocs/actions-gitpush.png)

---

## Workflow Status

All workflows completed successfully.

![Workflow Success](../reports/cidocs/actions-list-success.png)

---

# 2. Continuous Integration Validation

The CI workflow validates:

- Ruff linting
- Pytest execution
- Dependency verification
- Build validation

## CI Validation

![CI Validation](../reports/cidocs/actions-gitpush-cml.png)

---

# 3. Docker Build Automation

A Docker image is automatically built after successful validation.

The image contains:

- FastAPI Application
- Model Artifacts
- Monitoring Components
- Retraining Services

---

## DockerHub Repository

Repository:

```text
nshastry00/mlops-stock-api
```

---

## Docker Image Published

![Docker Image](../reports/cidocs/actions-docker-image-gitpush.png)

---

## Docker Tags

Docker tags are generated automatically.

Example:

```text
latest
main
sha-5bc2e91
```

![Docker Tags](../reports/cidocs/actions-docker-tags-gitpush.png)

---

## Docker Image Management

![Docker Image List](../reports/cidocs/actions-docker-imagelist-gitpush.png)

---

# 4. Pull Request Workflow

All development work is merged through Pull Requests.

Benefits:

- Code Review
- Automated Validation
- Quality Gate Enforcement
- Traceability

---

## Pull Request Dashboard

![PR Dashboard](../reports/cidocs/pull-request-dashboard.png)

---

## Pull Request Opened

![Pull Request](../reports/cidocs/pull-request-actions.png)

---

# 5. Pull Request Validation Checks

The following workflows execute automatically:

- CI Pipeline
- Docker Build
- CML Reporting

---

## Validation Running

![PR Checks Running](../reports/cidocs/pull-request-actions-checks.png)

---

## Validation Successful

![PR Checks Successful](../reports/cidocs/pull-request-actions-successchecks.png)

---

# 6. Automated CML Reporting

CML automatically generates a report summarizing:

- Pipeline status
- Model artifacts
- Monitoring outputs
- Figures generated

---

## CML Report

![CML Report](../reports/cidocs/pull-request-actions-pr-comment.png)
![CML Report](../reports/cidocs/pull-request-actions-pr-comment1.png)
![CML Report](../reports/cidocs/pull-request-actions-pr-comment2.png)
![CML Report](../reports/cidocs/pull-request-actions-pr-comment3.png)
![CML Report](../reports/cidocs/pull-request-actions-pr-comment4.png)
![CML Report](../reports/cidocs/pull-request-actions-pr-comment5.png)
![CML Report](../reports/cidocs/pull-request-actions-pr-comment6.png)
![CML Report](../reports/cidocs/pull-request-actions-pr-comment7.png)
![CML Report](../reports/cidocs/pull-request-actions-pr-comment8.png)

---

## Report Continuation

![CML Report Continued](../reports/cidocs/pull-request-actions-pr-commentend.png)

---

# 7. Merge Request Validation

Before merging into main:

All required checks must pass.

---

## Merge Validation

![Merge Validation](../reports/cidocs/merge-request-checks.png)

---

## Merge Validation Success

![Merge Validation Success](../reports/cidocs/merge-request-checks-success.png)

---

## Merge Workflow Status

![Merge Actions](../reports/cidocs/merge-request-actionslist.png)

---

## Merge Workflow Success

![Merge Actions Success](../reports/cidocs/merge-request-actions-success.png)

---

# 8. Docker Deployment After Merge

After merging:

- Main branch workflow executes
- Docker image rebuilds
- DockerHub receives updated image

---

## Docker Deployment

![Docker Deployment](../reports/cidocs/merge-request-dockerimagepush.png)

---

## Main Branch Docker Tags

![Main Docker Tags](../reports/cidocs/merge-request-dockertags.png)

---

# Final Verification

The following components were successfully automated:

- [x] GitHub Actions
- [x] Continuous Integration
- [x] Automated Testing
- [x] Ruff Linting
- [x] Docker Build
- [x] DockerHub Deployment
- [x] Pull Request Validation
- [x] CML Reporting
- [x] Merge Validation
- [x] Main Branch Deployment

---

# Outcome

The project now follows a production-style MLOps workflow where every change:

1. Is validated automatically.
2. Generates reproducible reports.
3. Builds deployable Docker images.
4. Passes quality gates.
5. Is safely merged into main.

This workflow improves reproducibility, reliability, deployment consistency, and overall software quality.