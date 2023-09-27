---
layout: post
title: Deploying GitHub Enterprise Server on Azure the GitOps way
description: A story about GHES, Terraform, GitHub Actions and the GitOps' holy spirit
comments: false
tags: ghes terraform actions gitops azure
minute: 10
toc: true
---


## 0 - Introduction

As a zen apprentice I like to free my mind as much as I can, specially regarding if/where/how my ressources are deployed in the cloud ☁️.

In this post I'll share a simple way to do so, through the example of a GitHub Enterprise Server (GHES) instance that I use for testing purposes. This GHES instance is described as Terraform code and deployed in Azure using GitHub Actions workflows. 

The code can be found [in this repository](https://github.com/ghsioux-octodemo/deploy-ghes-azure-terraform/) which is based on [Azure-Samples/terraform-github-actions](https://github.com/Azure-Samples/terraform-github-actions/), a sample repository that shows how to use GitHub Actions workflows to manage Azure infrastructure with Terraform.

## 1 - The infrastructure
The Terraform code used to describe the target infrastructure - that is, the GHES related resources - is split into several files:

```zsh
❯ tree -Pa '*.tf|*.tfvars|*.hcl' --prune
.
├── .terraform.lock.hcl
├── main.tf
├── outputs.tf
├── providers.tf
├── terraform.tfvars
└── variables.tf
```
Overall this is a typical file structure for a simple Terraform project:
* `main.tf` contains the main Terraform code, where Azure ressources (e.g. storage account, vnet, VM, etc.) are defined to be deployed;
* `providers.tf` references plugins that allow Terraform to interact with cloud providers (here Azure);
* `variables.tf` and `terraform.tfvars` respectively **defines** and **assign custom values** to variables used in `main.tf` to customize the resources;
* `outputs.tf` defines the outputs that the Terraform code will return regarding the deployed resources (e.g. Azure resource group, VM's public IP, FQDN).

The special file `.terraform.lock.hcl` is a lock file ensuring that the same versions of the plugins are used by all collaborators of the repo.


## 2 - The Action workflows

The `.github/workflows` directory contains the GitHub Actions workflows that will work with the Terraform code.

```zsh
❯ tree -Pa '*.yml' --prune
.
└── .github
    └── workflows
        ├── tf-drift.yml
        ├── tf-plan-apply.yml
        └── tf-unit-tests.yml
```

Here are the main characteristics of these workflows:

| Workflow | Triggering events | Description |
|----------|-------|-------------|
| `tf-unit-tests.yml` | `push` on any branch | Runs unit tests and analyze the Terraform code. |
| `tf-plan-apply.yml` | `pull_request` or `push` to `main` branch| Builds the Terraform plan and applies it in Azure. |
| `tf-drift.yml` | `schedule` or manual | Detects drifts between the desired and the current infrastructure. |

A nice thing is that the Terraform state is stored in an Azure storage account, so that it can be shared between the workflows to retrieve the currently deployed infrastructure. For more information, see [the README](https://github.com/ghsioux-octodemo/deploy-ghes-azure-terraform/blob/main/README.md) of the repository.

Another noteworthy point is the use of the [`paths`](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#onpushpull_requestpull_request_targetpathspaths-ignore) keyword in the workflow files, allowing to trigger the workflows only if the Terraform files have been modified, e.g.:

```yaml
on:
  push:
    paths:
    - '**.tf'
    - '**.tfvars' 
```

## 3 - Putting it all together (a.k.a. the GitOps part)

Combining both Terraform and GitHub Actions allows us to embrace the GitOps paradigm. Here is a typical process leveraging this approach:
1. Clone the repo and create a new branch to work on a new feature, or to update a Terraform variable that'll in turn update the GHES deployment;

2. Once the Terraform code has been updated, push the changes to the remote repo and open [a pull request](https://github.com/ghsioux-octodemo/deploy-ghes-azure-terraform/pull/2) towards the `main` branch;

3. The `tf-unit-tests.yml` workflow is triggered to test the Terraform code;
![the unit tests workflow](/assets/images/2023-01-10-ghes-azure-gitops/unit-tests.png "The unit tests workflow")

4. In parallel, the `tf-plan-apply.yml` workflow is triggered to build the Terraform plan and to share it as a comment in the PR conversation, for review purposes;
![the Terraform plan](/assets/images/2023-01-10-ghes-azure-gitops/terraform_plan.png "The Terraform plan")
5. After all the PR [required checks](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches#require-status-checks-before-merging) and [reviews](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/approving-a-pull-request-with-required-reviews) have passed, the PR can be merged into the `main` branch;

6. Once the PR merged, the `tf-plan-apply.yml` workflow is triggered again, and this time it applies the Terraform plan in Azure, effectively deploying the GHES resources;
![the Terraform apply](/assets/images/2023-01-10-ghes-azure-gitops/tf-plan-apply-push.png "The Terraform apply")

At this point, my GitHub Enterprise Server instance is deployed in Azure and ready to be used (or well, to be set up if it's the first boot :)
![GHES first boot](/assets/images/2023-01-10-ghes-azure-gitops/ghes_first_boot.png "GHES first boot")

Now that the deployment is done, we want to ensure the infrastructure is always in line with the Terraform code.
Indeed, unwanted changes might occur in the infrastructure along time, like the VM being deleted manually or the DNS name being updated. This is where the `tf-drift.yml` workflow comes into play.

This workflow is triggered every day to detect drifts between the desired (i.e. the Terraform code) and the currently deployed infrastructure. If drifts are detected, the workflow will open a new issue, and I'll receive a [notification from GitHub](https://docs.github.com/en/account-and-profile/managing-subscriptions-and-notifications-on-github/setting-up-notifications/configuring-notifications#about-custom-notifications):
![the drift issue](/assets/images/2023-01-10-ghes-azure-gitops/drift.png "The drift issue")

## 4 - The end
That's it for today folks, I hope you enjoyed this post and that it will help you to get started with Terraform and GitHub Actions.

As usual, this is more like a PoC and many improvements can be done. Zen masters often say that "the journey is never ending".

As a future work, I'd like to play with [the Terraform provider for GitHub](https://registry.terraform.io/providers/integrations/github/latest/docs) to store GHES configuration (e.g. organizations, policies, teams) as code, and write additional workflows to leverage it. I'll also write another post to go further into the `main` branch protection rules and the unit tests workflow, as it includes some nice [code scanning](https://docs.github.com/en/code-security/code-scanning/automatically-scanning-your-code-for-vulnerabilities-and-errors/about-code-scanning) features.

Oh and by the way, this blog post has been co-written with Copilot 🤖. 

Gasshō! 🙏