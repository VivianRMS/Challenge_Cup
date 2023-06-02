host_path=$(dirname $(cd "$(dirname $"$0")";pwd))
echo $host_path

sudo docker container rm -f fatigue_driver


xhost +local:docker
XAUTH=/tmp/.docker.xauth
XSOCK=/tmp/.X11-unix

echo "Preparing Xauthority data..."
xauth_list=$(xauth nlist :0 | tail -n 1 | sed -e 's/^..../ffff/')
if [ ! -f $XAUTH ]; then
    if [ ! -z "$xauth_list" ]; then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi


echo "Done."
echo ""
echo "Verifying file contents:"
file $XAUTH
echo "--> It should say \"X11 Xauthority data\"."
echo ""
echo "Permissions:"
ls -FAlh $XAUTH
echo ""
echo "Running docker..."


sudo docker run \
    -it \
    --name fatigue_driver \
    --shm-size=256m \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=$XAUTH \
    -v $XSOCK:$XSOCK \
    -v $XAUTH:$XAUTH \
    -p 1883:1883 -p 5000:5000 \
    -v $host_path:/fatigue_driver \
    pytorch-1.8.0:v1.0
