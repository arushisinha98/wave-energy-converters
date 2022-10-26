sudo docker kill tf_cont
sleep 1
sudo docker run -it --rm -p 8888:8888 -p 6006:6006 --name tf_cont -d -v ~/github/wave-energy-converters/notebooks:/notebooks tf
