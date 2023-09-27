---
layout: post
title: Securing code and its deployment with rules
description: A story about environment and branch protection 
comments: false
tags: actions pull-request administration security
minute: 15
---

Hey shinobis, welcome back to the dojo! Do you remember the previous blog post about [deploying GitHub Enterprise Server on Azure the GitOps way](https://ghsioux.github.io/2023/01/10/ghes-azure-gitops)? In this post, let's go a bit further and see some specific steps that secured the pull-request (PR) process and the resulting infrastructure deployment.

## 1 - The PR Process and its shurikens

In the [repository](https://github.com/ghsioux-octodemo/deploy-ghes-azure-terraform), [a PR  `feature_branch -> main`](https://github.com/ghsioux-octodemo/deploy-ghes-azure-terraform/pull/2) is opened to implement a new feature for the Terraform-coded GHES infrastructure. We've seen previously that [some GitHub Actions workflows](https://github.com/ghsioux-octodemo/deploy-ghes-azure-terraform/tree/main/.github/workflows) run on that PR to unit test/analyze the code and generate the according Terraform plan. If the PR is merged, the Terraform plan is applied to the target cloud environment - here Azure - and the GHES infrastructure is deployed with the new feature.  

I work alone on this repository, and that's for experimentations only, so overall this process is fine. In many cases however, multiple developers and teams collaborate on the same codebase, and that codebase might be deployed to production environments. In such situations, there are some flying shurikens we need to handle well ✦✦✦. 

What if: 

* the new code is pushed directly to the `main` branch without being reviewed & approved?
* the new code is not tested for potential bugs or analyzed for potential vulnerabilities (e.g. an Azure resource with too permissive permissions)?
* the new code is deployed on a production environment without any notification / approval / preparation (e.g. from the SRE team)?

Some cyberninjas say that the risk 0 is just an illusion. But still, let's see how we can mitigate these risks using a mix of branch and environment protection rules.

## 2 - Protecting the branch

[Branch protection rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches) are a great way to maintain the good health of a branch while collaborating. They can be used for instance to ensure that [any new code has been reviewed](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches#require-pull-request-reviews-before-merging) before being merged, or that specific [CI jobs have passed](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches#require-status-checks-before-merging) during the PR process.

In my repo, the `main` branch is the one from where the Terraform code is applied in Azure. I've therefore set up the following rules for that branch:
* a PR is required in order to update the code (i.e. no direct push allowed);
* the code of the PR must be approved by at least one [code owner](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners);
* some jobs of the PR-triggered Actions workflows must pass successfully (they are what we call [required status checks](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches#require-status-checks-before-merging));
* the commits in the PR must be [signed and verified](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification).

![branch protection rules](/assets/images/2023-02-03-securing-branches-and-deployments/protection_rules.png "Branch protection rules")

Each of these rules must be respected before the PR can be merged. All together, they form a nice barrier against the two first shurikens we saw above.

The first and second rules ensure that the code is reviewed by the relevant team/people depending on what's being changed. Here, it could be for example the IaC team having a look at the new Terraform code and the according plan. Such rules normally ensure that the branch is always in a verified and working state. 

The third rule ensures that the ["Unit test"](https://github.com/ghsioux-octodemo/deploy-ghes-azure-terraform/blob/main/.github/workflows/tf-unit-tests.yml#L9-L33), ["Terraform plan"](https://github.com/ghsioux-octodemo/deploy-ghes-azure-terraform/blob/main/.github/workflows/tf-plan-apply.yml#LL30-L122C15) and ["Code Scanning"](https://github.com/ghsioux-octodemo/deploy-ghes-azure-terraform/blob/main/.github/workflows/tf-unit-tests.yml#LL35-L48C26) jobs have run successfully. This allows to __i)__ automatically test and analyze the new code before merging, and __ii)__  generate the Terraform plan and [make it available](https://github.com/ghsioux-octodemo/deploy-ghes-azure-terraform/blob/main/.github/workflows/tf-plan-apply.yml#LL107-L122C15) as [a PR comment](https://github.com/ghsioux-octodemo/deploy-ghes-azure-terraform/pull/2#issuecomment-1376268820) for review purposes. I'll likely talk more about the code scanning part in a future blog post.

We can see these required status checks in different locations, including the PR "Conversation" and "Checks" tab:

![required status checks](/assets/images/2023-02-03-securing-branches-and-deployments/checks.png "Required status checks")

Finally, the last rule adds some extra-security by requiring valid cryptographic signatures on every commits. Such mechanism makes it more difficult to impersonate someone else and push code on its behalf. 
By the way, have you heard about [GitHub vigilant mode](https://docs.github.com/en/authentication/managing-commit-signature-verification/displaying-verification-statuses-for-all-of-your-commits)?  

## 3 - Deployments rhymes with environments

Allright, the PR has successfully passed all the `main` branch protection rules so it is merged. Now the new Terraform code is ready to be deployed, and we have to be sure it will go smoothly. Let's leverage [GitHub Environments](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment) to this end.

Environments are logical deployment targets for a repository. They are used in Actions workflows to customize code deployments through the use of scoped [variables and secrets](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment#environment-secrets). An environment can be protected by some rules, and be associated with a specific branch so only the code of this branch will be allowed to be deployed. 

Here is the `production` environment I've created for my repo:

![environment_secrets_variables](/assets/images/2023-02-03-securing-branches-and-deployments/environment.png "Environment secrets and variables")


This environment is scoped to the `main` branch and will target a given Azure tenant through its `AZURE_CLIENT_ID` secret - used by the "Terraform Apply" job to [OIDC-authenticate in Azure](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-azure). In a more realistic context, we could imagine a second environment named `staging` with its own `AZURE_CLIENT_ID` secret to target another Azure tenant. I've also added a protection rule specifying that the user @ghsioux (🥷) must manually approve the deployment beforehand.

Like this, when the PR is merged and the `main` branch updated, the "Terraform Apply" job is triggered and I have to approve the deployment first:

![deploymend review required](/assets/images/2023-02-03-securing-branches-and-deployments/review_deployment_1.png "deploymend review required")

![deploymend review done](/assets/images/2023-02-03-securing-branches-and-deployments/review_deployment_2.png "deploymend review done")

Once approved, waiting a few minutes for the job to complete, we can see the deployment is active and the GHES URL available: 

![successful_deployment](/assets/images/2023-02-03-securing-branches-and-deployments/deployment.png "Successful deployment")

You get it, with environments and their associated protection rules, the last shuriken isn't a threat anymore.

## 4 - Shurikens to come

Today we have seen how to protect our code from some shurikens, but there are still more to come. For instance, what if sensitive data (e.g. an Azure access key) was pushed to the repo? Or new compliance rules should be enforced on all the repositories of a big organization? We could investigate the use of [secret scanning](https://docs.github.com/en/code-security/secret-scanning/about-secret-scanning) and [required workflows](https://docs.github.com/en/actions/using-workflows/required-workflows) for this. 

Another improvement could be to switch from the [__deploy -> merge__ model to the __branch deploy__ model](https://github.blog/2023-02-02-enabling-branch-deployments-through-issueops-with-github-actions/#understanding-the-branch-deploy-model) for an even safer `main` branch.

But that's another story for another time.

Gasshō! 🙏