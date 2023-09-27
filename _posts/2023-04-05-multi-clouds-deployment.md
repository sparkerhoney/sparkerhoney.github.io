---
layout: post
title: Multi-clouds deployment with OIDC and rollback support
description: A story about (not-only) versioned & passwordless multi-clouds deployment 
comments: false
tags: actions oidc container-app azure aws gcp release reusable-workflows rollback
minute: 30
---

That's it ninjas, spring has arrived, birds are singing and beautiful clouds adorn the sky. It make me remember about these old stories, where some mountain monks could touch multiple of them at once.

Today, we're going to do the same thing, but with different clouds. Let's use our GitHub Actions katas to deploy an application to Azure, AWS and GCP ! ☁️☁️☁️ 


## 1 - General View

We'll deploy the [Spring Petclinic sample application](https://github.com/spring-projects/spring-petclinic) as a container application in the cloud, and this is the [fork repository](https://github.com/ghsioux/multi-cloud-deployment-demo) where we'll work on. The ["🚀 Multi-cloud deployment"](https://github.com/ghsioux/multi-cloud-deployment-demo/blob/main/.github/workflows/multi-cloud-deployment.yml) is the main workflow from where the deployment starts. The overall process is as follows:

1. Build the application using Maven to generate a JAR file [(link to code)](https://github.com/ghsioux/multi-cloud-deployment-demo/blob/main/.github/workflows/multi-cloud-deployment.yml#L45-L72);
2. Build and push a container image ready to run this JAR file [(link to code)](https://github.com/ghsioux/multi-cloud-deployment-demo/blob/main/.github/workflows/multi-cloud-deployment.yml#L74-L111);
3. Run the container image on each cloud provider [(link to code)](https://github.com/ghsioux/multi-cloud-deployment-demo/blob/main/.github/workflows/multi-cloud-deployment.yml#L113-L177).

This dojo session will mainly focus on the last step and some of its specificities. If you've followed its link, you'll have noticed that three [reusable workflows](https://docs.github.com/en/actions/using-workflows/reusing-workflows) - one per cloud provider - are called to this end. This is because I will likely use them in future projects, as deploying container applications to the cloud is a common task. Also, this is a nice way to keep the main workflow as simple as possible.

Overall, the deployment will have the following properties:

| Properties | How |
|----------|------------|
| __Automated__ | The deployment will be performed with GitHub Actions. |
| __Versioned__ | The workflow will be triggered when [a new release is published](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository#creating-a-release), or manually on [specific tags](https://docs.github.com/en/rest/git/tags?apiVersion=2022-11-28#about-git-tags). But not with every tags 🥷👇 |
| __Protected__ | The protection is threefold: <br>{::nomarkdown}<ul class="table-list"><li>{:/}**Who can deploy?** Only repository admins; {::nomarkdown}</li><li>{:/} **What can be deployed?** Only tags matching the regular expression `v*` (and associated releases since releases are based on tags); {::nomarkdown}</li><li>{:/} **Where to deploy?** Only on three specific, protected GitHub environments (named `aws`, `azure` and `gcp` for the cloud providers they'll respectively target).{::nomarkdown}</li></ul>{:/} |
| __Passwordless__ | Authentication to the cloud providers is done using [the OpenID Connect standard](https://openid.net/connect/), hence no need to manage any password. |
| __Geo-distributed__ | The application will be deployed on three different geographical locations: [`ap-southeast-1` for AWS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html#concepts-regions), [`east us` for Azure](https://azure.microsoft.com/en-us/explore/global-infrastructure/geographies/#geographies) and [`europe-west1` for GCP](https://cloud.google.com/compute/docs/regions-zones#available). |  

## 2 - Shaping the clouds

Each of the cloud providers has its own way(s) to deploy container applications, and provides the associated GitHub Actions to do it. Here is a quick recap of the ones I've used:

| Cloud provider | Container Service | GitHub Actions |
|----------|-------|-------------|
| Amazon Web Services | [Elastic Container Service](https://aws.amazon.com/ecs/) | {::nomarkdown}<ul class="table-list"><li>{:/}[aws-actions/amazon-ecs-render-task-definition](https://github.com/marketplace/actions/amazon-ecs-render-task-definition-action-for-github-actions) {::nomarkdown}</li><li>{:/}[aws-actions/amazon-ecs-deploy-task-definition](https://github.com/marketplace/actions/amazon-ecs-deploy-task-definition-action-for-github-actions) {::nomarkdown}</li></ul>{:/} |
| Microsoft Azure | [Azure Container Instances](https://azure.microsoft.com/en-us/services/container-instances/) | [Azure/aci-deploy](https://github.com/marketplace/actions/deploy-to-azure-container-instances) |
| Google Cloud Platform | [Cloud Run](https://cloud.google.com/run) | [google-github-actions/deploy-cloudrun](https://github.com/marketplace/actions/deploy-to-cloud-run) |

Before using these actions in our workflows, we have to set up the cloud infrastructures first. With the aim of keeping this scroll as short as possible, I've detailed this process in the [`doc/infra-setup` folder](https://github.com/ghsioux/multi-cloud-deployment-demo/tree/main/docs/infra-setup) and all the CLI commands are provided. You can also have a look at [the quickstart](https://github.com/ghsioux/multi-cloud-deployment-demo/blob/main/README.md) to get an overall picture.

> 💡 **Bonus!** If you want to try it, but don't want to install the CLI tools on your own machine, there is [a codespaces setup](https://github.com/ghsioux/multi-cloud-deployment-demo/blob/main/.devcontainer/devcontainer.json) ready to be used for that ✨

Once it's done, we'll use these actions in our reusable workflows to deploy the container application:

* For Amazon ([link to step](https://github.com/ghsioux/multi-cloud-deployment-demo/blob/main/.github/workflows/deploy-to-aws-ecs.yml#L68-L83)):
```yaml
      - name: Prepare the Amazon ECS task definition to use our container image 
        id: task-def
        uses: aws-actions/amazon-ecs-render-task-definition@v1
        with:
          task-definition: {% raw %}${{ inputs.ecs-task-definition }}{% endraw %}
          container-name: {% raw %}${{ inputs.container-name }}{% endraw %}
          image: {% raw %}${{ inputs.container-image }}{% endraw %}

      - name: Deploy the Amazon ECS task definition (i.e. deploy the container app)
        id: deploy_task
        uses: aws-actions/amazon-ecs-deploy-task-definition@v1
        with:
          task-definition: {% raw %}${{ steps.task-def.outputs.task-definition }}{% endraw %}
          service: {% raw %}${{ inputs.ecs-service }}{% endraw %}
          cluster: {% raw %}${{ inputs.ecs-cluster }}{% endraw %}
          wait-for-service-stability: true
```

* For Azure ([link to step](https://github.com/ghsioux/multi-cloud-deployment-demo/blob/main/.github/workflows/deploy-to-azure-aci.yml#L63-L71)):
```yaml
      - name: 'Deploy to Azure Container Instances'
        uses: 'azure/aci-deploy@v1'
        with:
          resource-group: {% raw %}${{ inputs.resource-group }}{% endraw %}
          dns-name-label: {% raw %}${{ inputs.deployment-url-prefix }}{% endraw %}
          image: {% raw %}${{ inputs.container-image }}{% endraw %}
          name: {% raw %}${{ inputs.deployment-name }}{% endraw %}
          location: {% raw %}${{ inputs.location }}{% endraw %}
          ports: {% raw %}${{ inputs.ports }}{% endraw %}
```

* For GCP ([link to step](https://github.com/ghsioux/multi-cloud-deployment-demo/blob/main/.github/workflows/deploy-to-gcp-cloudrun.yml#L85-L92)):
```yaml
      - id: 'deploy'
        name: 'Deploy the image to Google Cloud Run'
        uses: 'google-github-actions/deploy-cloudrun@v1'
        with:
          service: {% raw %}${{ inputs.cloudrun-service }}{% endraw %}
          image: {% raw %}${{ inputs.push-to-gar == true && inputs.gar-target-image || inputs.container-image }}{% endraw %}
          region: {% raw %}${{ inputs.region }}{% endraw %}
          flags: {% raw %}${{ inputs.flags }}{% endraw %}
``` 

You can observe that these workflows are called with different regions [[1]](https://github.com/ghsioux/multi-cloud-deployment-demo/blob/main/.github/workflows/multi-cloud-deployment.yml#L130) [[2]](https://github.com/ghsioux/multi-cloud-deployment-demo/blob/main/.github/workflows/multi-cloud-deployment.yml#L152) [[3]](https://github.com/ghsioux/multi-cloud-deployment-demo/blob/main/.github/workflows/multi-cloud-deployment.yml#L173) for each cloud provider, hence making the deployment **geo-distributed**. This is a good practice to ensure high availability and to reduce the risk of downtime. And this is by the way just one of the benefits of the multi-cloud approach.

## 3 - OIDC in Actions

So the clouds are ready and we know exactly how we'll deploy our application. That's neat, but we still have to authenticate beforehand in order to proceed. When it comes to authenticate to cloud providers in GitHub Actions, two good practices pop up:
* __Avoid using passwords__ as they are hard to rotate and can be leaked;
* __Use service accounts__ instead of personal accounts, as they provide better security and control over permissions and access to cloud resources.

And guess what? This is exactly what we gonna do here: we'll use a service account for each cloud provider, and we'll configure them to authenticate using the OpenID Connect standard. 

Similarly to what we did for the cloud infrastructures, I've detailed this process in the [`doc/oidc-setup` folder](https://github.com/ghsioux/multi-cloud-deployment-demo/tree/main/docs/oidc-setup) and the following table summarizes what has been used to this end:


| Cloud provider | Service account construct | OIDC implementation | GitHub Action |
|----------|-------|-------------|---------|
| Amazon Web Services | [IAM role](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html) | [OIDC identity provider](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_providers_create_oidc.html) | [aws-actions/configure-aws-credentials](https://github.com/marketplace/actions/configure-aws-credentials-for-github-actions) |
| Microsoft Azure | [Service principal](https://learn.microsoft.com/en-us/powershell/azure/create-azure-service-principal-azureps?view=azps-9.5.0) | [Workload identity federation](https://learn.microsoft.com/en-us/azure/active-directory/workload-identities/workload-identity-federation?source=recommendations) | [Azure/login](https://github.com/marketplace/actions/azure-login) |
| Google Cloud Platform | [Service account](https://cloud.google.com/iam/docs/service-account-overview) | [Workload identity federation](https://cloud.google.com/iam/docs/workload-identity-federation) | [google-github-actions/auth](https://github.com/marketplace/actions/authenticate-to-google-cloud) |


Once everything is set up, we can authenticate to the cloud providers using OIDC in our reusable workflows:

* For AWS ([link to step](https://github.com/ghsioux/multi-cloud-deployment-demo/blob/main/.github/workflows/deploy-to-aws-ecs.yml#L61-L66)):
```yaml
      - name: Configure AWS credentials using OIDC
        uses: aws-actions/configure-aws-credentials@v2
        with:
          role-to-assume: {% raw %}${{ secrets.oidc-role-to-assume }}{% endraw %}
          role-session-name: workflowrolesession
          aws-region: {% raw %}${{ inputs.aws-region }}{% endraw %}
```

* For Azure ([link to step](https://github.com/ghsioux/multi-cloud-deployment-demo/blob/main/.github/workflows/deploy-to-azure-aci.yml#L56-L61)):
```yaml
      - name: 'Login to Azure using OIDC'
        uses: azure/login@v1
        with:
          client-id: {% raw %}${{ secrets.az_client_id }}{% endraw %}
          tenant-id: {% raw %}${{ secrets.az_tenant_id }}{% endraw %}
          subscription-id: {% raw %}${{ secrets.az_subscription_id }}{% endraw %}
```

* For GCP ([link to step](https://github.com/ghsioux/multi-cloud-deployment-demo/blob/main/.github/workflows/deploy-to-gcp-cloudrun.yml#L63-L69)):
```yaml
      - id: 'auth'
        name: 'Authenticate to Google Cloud using OIDC'
        uses: 'google-github-actions/auth@v1'
        with:
          workload_identity_provider: {% raw %}${{ secrets.workload_identity_provider }}{% endraw %}
          service_account: {% raw %}${{ secrets.service_account }}{% endraw %}
          token_format: 'access_token'
```

You might have noticed that we are still using Actions secrets to pass the OIDC information to the cloud providers. This is just to add an extra layer of security: for instance, if that information was leaked and the OIDC trust relationship was too permissive, an attacker would perhaps be able to access the cloud resources from its own repository.

Here this should not be the case - but we never know! - since during the setup I've restricted the trust relationship to the following OIDC identities:

* for AWS, only the identity `repo:ghsioux/multi-cloud-deployment-demo:environment:aws` (that is, the GitHub environment `aws` of the repository `ghsioux/multi-cloud-deployment-demo`) is trusted;
* similarly, for Azure, only the identity `repo:ghsioux/multi-cloud-deployment-demo:environment:azure` is trusted;
* and for GCP, only the identity `repo:ghsioux/multi-cloud-deployment-demo:environment:gcp` is trusted.

During the authentication process, the identity checks are performed by the cloud providers based on [the token passed by the GitHub OIDC provider](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect#understanding-the-oidc-token).

Allright! We have implemented some mecanisms to make the deployment **geo-distributed** and **passwordless**, now let's make it **protected** and **versioned**.

## 4 - Controlling the version to deploy

The ones with eyes like the hawk will have observed that there is actually [an additional step](https://github.com/ghsioux/multi-cloud-deployment-demo/blob/main/.github/workflows/multi-cloud-deployment.yml#L11-L43) in the main worflow, that I've not listed in the [general view](#1---general-view) above.

To get a bit more context, let's have a look at the very beginning of [the workflow](https://github.com/ghsioux/multi-cloud-deployment-demo/blob/main/.github/workflows/multi-cloud-deployment.yml#L2-L7):

```yaml
name: "🚀 Multi-cloud PetClinic deployment demo"

on:
  workflow_dispatch:
  release:
    types: [published]
```

The workflow will be triggered manually or when a release is published. I'm working alone on this project, but let's imagine a more complex scenario where multiple people collaborate. As a repository administrator, I would possibly want to restrict __i)__ who can trigger/re-run the workflow (e.g. only admin), and __ii__) which tags (and thus associated releases) are allowed to be deployed (e.g. only production tag).

That's why I've added a few checks in the first job of the workflow to ensure that:

```yaml
  prereq_checks:
    name: Ensure this workflow has been triggered by and admin on a protected release tag (v*)
    runs-on: ubuntu-latest
    env:
      TRIGGERING_ACTOR: {% raw %}${{ github.triggering_actor }}{% endraw %}
    steps:
      - name: Fail if the workflow has been triggered manually with a non-release tag (i.e. not using the 'v*' regex)
        run: |
          if [[ "$GITHUB_REF" != refs/tags/v* ]]; then
            echo "Only 'v*' tags and associated releases can be deployed, exiting."
            exit 1
          else
            echo "The workflow has been triggered on a release tag, continuing."
          fi
      - name: Fail if the workflow has been triggered by a non-admin user
        uses: actions/github-script@v6
        with:
          script: |
            const TRIGGERING_ACTOR = process.env.TRIGGERING_ACTOR
            github.rest.repos.getCollaboratorPermissionLevel({
              owner: context.repo.owner,
              repo: context.repo.repo,
              username: TRIGGERING_ACTOR
              }).then(response => {
                if (response.data.permission == 'admin') {
                  core.info('The github actor has admin permission on this repository. Continuing.')
                } else {
                  core.info('Only repository admins can trigger this workflow. Exiting.')
                  process.exit(1)
                }
              })
```

As you can see, only member with the `admin` permission can trigger the workflow, and only tags matching the regexp `v*` are allowed to be deployed. In parallel, I've set up a [tag protection rules](https://docs.github.com/en/enterprise-cloud@latest/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/configuring-tag-protection-rules) to restrict `v*` tags creation to privileged members only. This tag protection rule will also apply when creating a release (i.e. a non-privileged member won't be authorized to create a release with a tag matching `v*`).

If you combine this with the use of the 3 dedicated GitHub environments, which are themselves [bound to the branches](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment#deployment-branches) matching `v*`  (i.e. our protected tags) and protected to require [a manual approval](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment#required-reviewers) before deployment, the overall level of security is fairly satisfying. 

## 5 - The pieces together a.k.a. time to trigger

Congrats, you made it so far, and now comes the time to enjoy and trigger 🔫

Let's say that the developers have been working hard on the application to add new features, and they are ready to deploy the new version to production.

A member with the `admin` permission on the repository will create a new release with a tag matching the `v*` regexp:
![release_creation](/assets/images/2023-04-05-multicloud-deployment/release.png "Release creation")

This will trigger the main workflow, and after the checks have passed, the image is built and pushed to the GitHub container registry. Then the three reusable workflows are called, and since they are working on protected environments, a manual approval is required (e.g. by the SRE team):
![deployments_approval](/assets/images/2023-04-05-multicloud-deployment/deployment_approval.png "Deployments approval")

Once the approval is granted, the workflows will continue and the new version of the application will be deployed to our three cloud providers:
![deployment_succeeded](/assets/images/2023-04-05-multicloud-deployment/deployment_succeeded.png "Deployment succeeded")

Is that all? Could be, but you know, sometimes accidents happen. Let's imagine that's the case, and that the new version of the application is not working as expected. Mayday, mayday, we need to rollback! Nothing more simple, the administrator will just trigger the workflow again, but this time manually, and on the tag matching the previous release:
![rollback](/assets/images/2023-04-05-multicloud-deployment/rollback.png "Rollback triggering")


Phew, we're going back to business! 🎉

## 6 - Next steps

That was a lot of fun, and I hope you enjoyed it as much as I did. As usual, this is not for production use, and there is room for improvement. I'm sure you have some ideas on how to make this even better! 

We could for instance:
* define stricter role permissions and RBAC for our service accounts;
* implement some load-balancing mechanisms (e.g. DNS round-robin, [Traefik](https://doc.traefik.io/traefik/)) to make the application available on a single URL; 
* extend the workflow to implement blue/green deployments, A/B testing or canary release approach.

But the sun is shining, and the flowers are blooming, so let's leave it here for now.

Gasshō 🙏