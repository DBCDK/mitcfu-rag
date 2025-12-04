#! groovy
@Library('ai') _
def workerNode = "ai-t01"
def slackReceivers = "#ai-jenkins-warnings"

pipeline {
	agent { label workerNode }
	options {
		gitLabConnection('gitlab.dbc.dk')
		disableConcurrentBuilds()
	}
	environment {
		PACKAGE="mitcfu-rag"
		DOCKER_TAG = "${env.BRANCH_NAME}-${env.BUILD_NUMBER}"
		GITLAB_PRIVATE_TOKEN = credentials("ai-gitlab-api-token")
	}
	triggers {
	upstream(upstreamProjects: 'Docker-base-python3,Docker-base-python3-bump-trigger', threshold: hudson.model.Result.SUCCESS)
	}
	stages {
		stage("upload wheel package") {
			agent {
				docker {
					label workerNode
					image "docker-dbc.artifacts.dbccloud.dk/build-env:latest"
					alwaysPull true
				}
			}
			when {
				branch "main"
			}
			steps {
				upload()
			}
		}
		stage("docker build") {
			steps {
				buildImage()
			}
		}
		stage("set gitops variables for ai-staging") {
			when {
				branch "main"
			}
			steps {
				script {
					setGitopsVersion("ai-staging", "MITCFU_RAG_1_0_VERSION", "${env.DOCKER_TAG}")
				}
			}
		}
		stage("wait for ai-staging to be ready") {
			agent {
				docker {
					label workerNode
					image "docker-dbc.artifacts.dbccloud.dk/k8s-deploy-env:latest"
					args '-u 0:0'
					alwaysPull true
				}
			}
			environment {
				KUBECONFIG = credentials("kubecert-mi")
				KUBECTL = "kubectl --kubeconfig '${KUBECONFIG}'"
			}
			when {
				branch "main"
			}
			steps {
				script {
					sh """
						$KUBECTL -n ai-staging rollout status deployment/mitfcu-rag-1-0 --timeout=1200s
					"""
				}
			}
		}
		stage("update staging version number") {
			agent {
				docker {
					label workerNode
					image "docker-dbc.artifacts.dbccloud.dk/build-env:latest"
					alwaysPull true
				}
			}
			when {
				branch "main"
			}
			steps {
				dir("deploy") {
					sh 'set-new-version mitcfu-rag-1-0.yml $GITLAB_PRIVATE_TOKEN ai/mitcfu-rag-secrets $DOCKER_TAG -b staging'
				}
				build job: "ai/mitcfu-rag/mitcfu-rag-deployment/staging", wait: true
			}
		}
		stage("set gitops variables for ai-prod") {
			when {
				branch "main"
			}
			steps {
				script {
					setGitopsVersion("ai-prod", "MITCFU_RAG_1_0_VERSION", "${env.DOCKER_TAG}")
				}
			}
		}
		stage("wait for ai-prod to be ready") {
			agent {
				docker {
					label workerNode
					image "docker-dbc.artifacts.dbccloud.dk/k8s-deploy-env:latest"
					args '-u 0:0'
					alwaysPull true
				}
			}
			environment {
				KUBECONFIG = credentials("kubecert-mi")
				KUBECTL = "kubectl --kubeconfig '${KUBECONFIG}'"
			}
			when {
				branch "main"
			}
			steps {
				script {
					sh """
						$KUBECTL -n ai-prod rollout status deployment/mitcfu-rag-1-0 --timeout=1200s
					"""
				}
			}
		}
		stage("update prod version number") {
			agent {
				docker {
					label workerNode
					image "docker-dbc.artifacts.dbccloud.dk/build-env:latest"
					alwaysPull true
				}
			}
			when {
				branch "main"
			}
			steps {
				dir("deploy") {
					sh 'set-new-version mitcfu-rag-1-0.yml $GITLAB_PRIVATE_TOKEN ai/mitcfu-rag-secrets $DOCKER_TAG -b prod'
				}
				build job: "ai/mitcfu-rag/mitcfu-rag-deployment/prod", wait: true
			}
		}
	}
	post {
		unstable { slackSend message: "build became unstable for ${env.JOB_NAME}: ${env.BUILD_URL}", channel: slackReceivers }
		failure { slackSend message: "build failed for ${env.JOB_NAME}: ${env.BUILD_URL}", channel: slackReceivers }
	}
}
