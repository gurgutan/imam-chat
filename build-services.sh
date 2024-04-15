docker rm $(docker ps -a -q)
docker image prune -q
docker build . -t imam-chat
