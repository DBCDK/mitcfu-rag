# MitCFU-RAG

RAG-løsning til MitCFU 2025


## How to start the service from the command line
Run `pip install -e .` from the root of the project.

Start the service with the following parameters

`<my-service-name> --port XXXX --my-param ADD_SOMETHING_HERE`

## How to start the service with DOCKER
Build the docker image from the Dockerfile:

`docker build -t $USER/<my-service-name>:test -f Dockerfile .`

Run the docker image:

`docker run -e LOG_FORMAT=text -e ADD_OWN_ENV=my_environment_variable --rm --name $USER-service -p <PORT_NUMBER>:5000 -it $USER/<my-service-name>:test`

`-e LOG_FORMAT=text` gives you log output in text instead of json

`--rm` ensures the docker container is closed down properly after use

If you for example have started the service on the server xpdev-p01 you can reach the service via this url
`http://xpdev-p01:<PORT_NUMBER>`

or locally via this url: 

`localhost:<PORT_NUMBER>`

## Project executables/scripts
### How to run <my_command>
Run `pip install -e .` from the root of the project.

ADD TEXT TO EXPLAIN YOUR COMMAND/SCRIPT

Run

`my_command --with-my params`

## How to run tests for this project
### Unit tests 

### Validation tests

### Performance tests

### Sanity checks before Merge Request


## Artifacts built in this project


## Related Jenkins jobs on is.dbc.dk


## Related artifacts from Artifactory


## Related repositories


## Production version of the service
 
 
## Documentation on confluence
