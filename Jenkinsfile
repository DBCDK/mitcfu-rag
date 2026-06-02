#! groovy
@Library('ai') _
def workerNode = "ai-t01"

pipeline {
    agent { label workerNode }
    options {
        gitLabConnection('gitlab.dbc.dk')
        disableConcurrentBuilds()
    }
    environment {
        PACKAGE="science-rag"
        DOCKER_TAG = "${env.BRANCH_NAME}-${env.BUILD_NUMBER}"
        GITLAB_PRIVATE_TOKEN = credentials("ai-gitlab-api-token")
        SLACK_CHANNEL = "${env.BRANCH_NAME == 'main' ? '#ai-jenkins-warnings' : '#ai-jenkins-warnings-debug'}"
    }
    triggers {
	upstream(upstreamProjects: 'Docker-base-python3,Docker-base-python3-bump-trigger', threshold: hudson.model.Result.SUCCESS)
    }
    stages {
		stage("test") {
			agent {
				docker {
					label workerNode
					image "docker-dbc.artifacts.dbccloud.dk/build-env:latest"
					alwaysPull true
				}
			}
			steps {
				uvtest()
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
					setGitopsVersion("ai-staging", "SCIENCE_RAG_1_0_VERSION", "${env.DOCKER_TAG}")
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
						$KUBECTL -n ai-staging rollout status deployment/science-rag-1-0 --timeout=1200s
					"""
				}
			}
		}
		stage("set gitops variables for ai-prod") {
			when {
				branch "main"
			}
			steps {
				script {
					setGitopsVersion("ai-prod", "SCIENCE_RAG_1_0_VERSION", "${env.DOCKER_TAG}")
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
						$KUBECTL -n ai-prod rollout status deployment/science-rag-1-0 --timeout=1200s
					"""
				}
			}
		}
	}
	post {
		success {
			updateGitlabCommitStatus name: 'build', state: 'success'
		}
		unstable {
			updateGitlabCommitStatus name: 'build', state: 'failed'
			slackSend message: "build became unstable for ${env.JOB_NAME}: ${env.BUILD_URL}", channel: env.SLACK_CHANNEL
		}
		failure {
			updateGitlabCommitStatus name: 'build', state: 'failed'
			slackSend message: "build failed for ${env.JOB_NAME}: ${env.BUILD_URL}", channel: env.SLACK_CHANNEL
		}
		fixed {
			updateGitlabCommitStatus name: 'build', state: 'success'
		}
	}
}
