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
        PACKAGE="science-rag"
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
				updateGitlabCommitStatus name: 'build', state: 'running'
				buildImage()
			}
		}
		stage("set gitops variables for ai-staging") {
			when {
                branch "main"
			}
			steps {
				script {
					withCredentials([sshUserPrivateKey(credentialsId: "gitlab-isworker", keyFileVariable: "sshkeyfile")]) {
						env.GIT_SSH_COMMAND = "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -i ${sshkeyfile}"
						nextBuild=sh(returnStdout: true, script: "curl -s ${JENKINS_URL}job/gitops-secrets/job/main/api/json | jq -r .nextBuildNumber")
						sh """
							nix run --refresh git+https://gitlab.dbc.dk/public-de-team/gitops-secrets-set-variables.git \
								ai-staging:SCIENCE_RAG_1_0_VERSION=${env.DOCKER_TAG}
						"""
						waitForGitops("${nextBuild}")
					}
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
					withCredentials([sshUserPrivateKey(credentialsId: "gitlab-isworker", keyFileVariable: "sshkeyfile")]) {
						env.GIT_SSH_COMMAND = "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -i ${sshkeyfile}"
						nextBuild=sh(returnStdout: true, script: "curl -s ${JENKINS_URL}job/gitops-secrets/job/main/api/json | jq -r .nextBuildNumber")
						sh """
							nix run --refresh git+https://gitlab.dbc.dk/public-de-team/gitops-secrets-set-variables.git \
								ai-prod:SCIENCE_RAG_1_0_VERSION=${env.DOCKER_TAG}
						"""
						waitForGitops("${nextBuild}")
					}
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
		unstable { slackSend message: "build became unstable for ${env.JOB_NAME}: ${env.BUILD_URL}", channel: slackReceivers }
		// failure { slackSend message: "build failed for ${env.JOB_NAME}: ${env.BUILD_URL}", channel: slackReceivers }
		failure {
			updateGitlabCommitStatus name: 'build', state: 'failed'
		}
		success { 
			updateGitlabCommitStatus name: 'build', state: 'success'
		}
		fixed {
			updateGitlabCommitStatus name: 'build', state: 'success'
		}

	}
}
