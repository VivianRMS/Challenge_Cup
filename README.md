Huawei Challenge Cup 2023: Fatigue Driving Detection
# Use docker for PC running
In docker file, there are following files:
```
├── build_container.sh
├── Dockerfile
├── into_container.sh
├── requirement.txt
├── run_container.sh
└── sources.list
``` 
- build_container.sh is used to trigger Dockerfile for building a container which can be used in docker
- into_container.sh is used if the container (named "fatigue_driver") is run in daemon model or you want to access the container in additional terminal
- run_container.sh is used to start/run the container (named "fatigue_driver") with specific configurations (set in the sh script)
- Dockerfile is the configuration file to make docker container
- sources.list is used to increase the apt install speed by replacing the source in the container with aliyun
- requirement.txt is used to build container's python dependencies

## Preparation
- "docker" is a program to run a light-weighted virtual environment. You can check this for more information. https://www.cnblogs.com/Can-daydayup/p/16472375.html.
- The original "docker" program is installed in the operation system, but not up to date. PLEASE following code to install the docker.
```
sudo apt install curl
curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun
```
## Use method
1. Run build_container.sh if and it will automatically setup the docker image (a running image is called container). The result can be checked by tpying "sudo docker image ls":
```
zhy@zhy-Y9000P:~$sudo docker image ls
REPOSITORY              TAG       IMAGE ID       CREATED        SIZE
pytorch-1.8.0           v1.0      9930548c6bee   21 hours ago   9.05GB
```
2. Run run_container.sh. Your should get into the bash terminal of the container shown as (see the username and PC name is changed to root@SomeNumber):
```
zhy@zhy-Y9000P:~/Desktop/Projects/Challenge_Cup/docker$ ./run_container.sh 

---some prints----

Running docker...
root@863ba7e59727:/fatigue_driver# 
```
3. In the container, run your script on PC. Remember, you need to temporarly remove the dependency of "model_service.pytorch_model_service import PTServingBaseService" in customize_service.py. Comment out the PTServingBaseService and the parent class

```
# from model_service.pytorch_model_service import PTServingBaseService

# class fatigue_driving_detection(PTServingBaseService):
class fatigue_driving_detection():
    def __init__(self, model_name, model_path):
```

4. Run your code by
```
python3 customize_service.py
```

5. If you want to upload it to ModelArts. Remember to uncomment the lines in step 3.


## Visualization in docker
We use OpenCV to show the image. However, docker itself does not have a graphic service, therefore, following configuration need to be set to let docker use the graphic service of the host(Your PC). You need to install the following program and check the script of "run_container.sh" (lines related to X11 and xauth is the one that link the graphic interface between docker container and PC host)

```
sudo apt install x11-xserver-utils
```


# Challenge_Cup
| Branch      | Model name               | Score  | Date     |
| ----------- | ------------------------ | ------ | -------- |
| Baseline    | video_classification_141 | 0.4709 | 20230528 |
| Multithread | model-7230               | 0.4175 | 20230530 |
|             |                          |        |          |